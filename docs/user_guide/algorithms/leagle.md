# LEAGLE

LEAGLE 是用于深度 / loop 对照实验的自回归 drafter：保留 DFlash / DSpark 的目标特征融合与每层 KV 注入，使用 EAGLE 式的 token + hidden-state 递推。它是独立算法 `leagle`，原来的 `eagle3`、`dflash`、`dspark` 仍可作为实验基线。

## 仓库位置

以下路径均相对于仓库根目录。

- `src/speculators/models/leagle/`：模型配置、共享 loop、注意力和自回归草稿生成实现。
- `examples/train/leagle_qwen3_4b_700k/`：LEAGLE 的独立 `server.sh` 和 `train.sh`，默认 1 层、1 loop。
- `examples/train/eagle3_qwen3_4b_700k/`：普通 EAGLE-3 的独立 `server.sh` 和 `train.sh`。
- `tests/unit/models/test_leagle.py`：CPU 数值与训练接口测试，使用仓库统一测试和代码检查配置。
- `output/`：训练日志、checkpoint 和复现记录；这两组实验的输出目录已加入忽略规则。

## 架构

给定已经验证的目标上下文，先拼接选定的目标层特征，再做一次 FC + RMSNorm，得到固定上下文 `C`。每个 draft decoder layer 都使用自己的 K/V 投影读取 `C`。

每生成一个候选 token：

1. 将当前已知 token 的 embedding 与上一 token 的 draft hidden state 分别归一化，拼接后用 `input_fc` 映射回 hidden size。第一步的 state 来自最后一个目标位置的融合特征。
2. 对同一组 `num_layers` 层执行 `num_loops` 次，每轮继续更新同一个位置的 hidden state，所有轮次共享权重。
3. 最后一次 loop 后使用冻结的目标 LM head 预测一个 token，将 token 和 hidden state 传给下一个自回归步骤。

这里的 token/state 融合层位于共享 decoder stack 外部；这是一种新的 KV-conditioned EAGLE 变体，并非原版 EAGLE-3 checkpoint 的兼容加载器。当前 decoder 支持 Qwen3，draft 与 target 使用相同 hidden size。

LEAGLE 没有并行 mask-token backbone、Markov/RNN correction head、confidence head 或 confidence scheduling。它直接按顺序生成候选 token。

## 三个独立维度

| 参数 | 含义 | 默认示例 |
|---|---|---|
| `--num-layers` | 具有独立参数的物理 decoder 层数 | 1 |
| `--num-loops` | 每个 token 对同一组层的执行次数 | 1 |
| `--ttt-steps` | 训练时向前展开的 token 步数 | 7 |

有效 decoder 深度 = `num_layers × num_loops`，不包含融合层和输出头。增加 loop 不增加 decoder 参数量，但增加计算量与缓存需求。

循环内位置编号不增加，只有进入下一个 token 时才增加。每个 `(loop, layer)` 使用独立的草稿 KV 历史；目标上下文不会重复追加进草稿缓存。反向传播覆盖所有 loop 和 token steps，没有 detach 或截断。

首批对照可使用 `(层数, loop)` = `(1,1)`、`(2,1)`、`(4,1)`、`(1,2)`、`(1,4)`、`(2,2)`。后四种中的 `(4,1)`、`(1,4)`、`(2,2)` 具有相同有效深度，能比较独立层和共享层。DFlash 的五层对照还需单独训练 `(5,1)`；有效深度相同不等于训练 FLOPs 或实测延迟完全相同。

## 训练

从仓库根目录运行。沿用已经准备好的 Qwen3-4B / 70 万数据、5 epochs、seed 42、3072 packing 长度、0.9 训练比例、32K draft vocabulary、三个目标特征层 `[2,18,33]` 和 TTT=7。默认目标服务使用 NPU 0–1，训练使用 NPU 2–7，端口 8003。脚本中的模型与数据路径需要与训练机器一致。

Qwen3-4B 预设的所有 draft 层使用 full causal attention。脚本根据 `NUM_LAYERS` 自动生成 `--full-attention-indices 0 ... D-1`，每次 loop 复用相同设置。注意力覆盖完整的可见前缀；未来目标特征、其他文档和其他训练轨迹仍然不可见。

### 普通 EAGLE-3 与 LEAGLE 的首轮对照

每个实验分别使用一份 `server.sh` 和一份 `train.sh`。先在一个终端启动目标服务，看到 ready 后，在另一个终端启动训练。服务脚本在前台运行；训练脚本内部使用 `nohup` 后台启动训练，并打印 PID 和日志路径。训练退出不会停止目标服务；停止服务时在服务终端按 Ctrl+C。

两组沿用上述 70 万数据 / 5 epochs 设置，训练配方分别为：

| 参数 | 普通 EAGLE-3 | LEAGLE |
|---|---|---|
| 优化器 | AdamW | AdamW |
| 峰值 LR | `6e-4` | `6e-4` |
| Adam betas | `(0.9, 0.95)` | `(0.9, 0.999)` |
| Weight decay | `0` | `0` |
| 梯度裁剪 max norm | `0.5` | `1.0` |
| Loss | KL | `0.1 CE + 0.9 TV` |
| Loss 实现 | `eager` | `legacy`，与 DSpark 相同 |
| LR scheduler | linear | cosine |
| Warmup 比例 | `0.015` | `0.04` |

普通 EAGLE-3 的 AdamW betas、weight decay 和梯度裁剪取自[作者的 EAGLE3 配置](https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json)，峰值 LR 则按本次实验要求与 LEAGLE 统一为 `6e-4`（作者配置为 `5e-5`）。这里把 `12000 / 800000 = 1.5%` warmup 比例应用到实际 5 epochs 的训练步数，并继续使用仓库的 linear scheduler；这不是原作者 DeepSpeed 训练的完整复现。

LEAGLE 的优化器、loss 与调度参数取自仓库的 [`train_dspark.sh`](../../../examples/train/qwen3_4b_bestarch/train_dspark.sh)。保留 LEAGLE 的层数 / loop、32K 词表、目标特征层和 5 epochs，不引入 DSpark 的 Markov / confidence head 及其辅助损失。`training-recipe=upstream` 保留按词表大小命名的缓存和 hidden-state 恢复流程；loss 显式指定 `legacy`，因此使用与 DSpark 相同的 CE / TV 数值实现。两组 attention 均为 SDPA，loss 的实现选项不改变 attention backend。

两份训练脚本默认开启 `FSDP_SHARD=1` 和 `ACTIVATION_CHECKPOINTING=1`，可分别设为 `0` 关闭。EAGLE3 的重算使用独立缓存，避免反向传播重复追加 TTT KV；LEAGLE 按物理层重算全部共享 loop。FSDP 分片按物理层执行。这里的默认配置尚需在实际 NPU 集群验证显存和速度。

新运行可用环境变量覆盖 `LR`、`WEIGHT_DECAY`、`ADAM_BETA1`、`ADAM_BETA2`、`MAX_GRAD_NORM`、`LOSS_FN`、`LOSS_IMPLEMENTATION`、`EPOCHS`、`SEQ_LENGTH`、`SEED`、`TTT_STEPS`、`TTT_STEP_LOSS_DECAY` 和 scheduler 参数。`LOSS_FN` 也接受仓库已有的加权 JSON，例如 `'{"ce":0.1,"tv":0.9}'`。修改配方时为实验指定新的 `OUTPUT_DIR`，服务端也使用该目录。

```bash
# 只打印最终启动命令，不连接服务、不加载模型、不创建训练输出。
DRY_RUN=1 NUM_LAYERS=2 NUM_LOOPS=2 \
bash examples/train/leagle_qwen3_4b_700k/train.sh

# 已停止的训练继续运行；先用相同 OUTPUT_DIR 重启 server.sh。
RESUME=1 OUTPUT_DIR=./output/leagle_qwen3_4b_700k_5ep_d2_r2_full \
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

`RESUME=1` 从该目录的 `checkpoints/run.yaml` 恢复原架构、loss、LR、优化器（含 betas）、梯度裁剪、scheduler 和显存设置，再加载最新训练进度；不会用脚本的新默认值覆盖原配方。旧 YAML 未记录 betas / 梯度裁剪时，仍采用历史默认值 `(0.9, 0.999)` / `1.0`；旧 Muon 运行继续使用 Muon。切换到本次新配方需要新的输出目录。续训仍保留所有 epoch，可更新服务端口与输出路径。脚本会拒绝向仍有存活训练 PID 的目录启动第二个 trainer。若早期版本已经删除 `run.yaml`，须先从原始命令恢复配置；缺失的配置不会被猜测。

默认保存每个 epoch（五轮对应 `checkpoints/0` 至 `checkpoints/4`），并维护 validation loss 最优的 `checkpoint_best`。最佳 loss 不保证最佳接受长度；可在独立验证集上评估各 epoch。显式启用底层 `--save-best` 时仅清理旧编号 checkpoint，保留 `run.yaml`、`train_command.txt`、patch 和恢复数据。

两组默认一个物理 decoder 层，LEAGLE 默认 `num_loops=1`；两组均显式使用 full attention。普通 EAGLE-3 保留示例的 Llama decoder 和输入归一化设置；LEAGLE 保留 Qwen3 decoder、输入融合和每层 KV 注入。该实验比较结构及训练配方的组合，接受长度差异不能单独归因于结构或 loop；峰值 LR 已统一，若要隔离结构因素，还需统一 loss、betas、梯度裁剪和 scheduler。

```bash
# 普通 EAGLE-3，终端 1：目标特征服务。
bash examples/train/eagle3_qwen3_4b_700k/server.sh

# 普通 EAGLE-3，终端 2：看到服务 ready 后启动训练。
bash examples/train/eagle3_qwen3_4b_700k/train.sh
```

默认设备和端口相同，因此等普通 EAGLE-3 训练完成、手动停止其服务后，再启动 LEAGLE：

```bash
# LEAGLE，终端 1：目标特征服务。
bash examples/train/leagle_qwen3_4b_700k/server.sh

# LEAGLE，终端 2：看到服务 ready 后启动训练。
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

默认输出分别为 `output/eagle3_qwen3_4b_700k_5ep_d1_full/` 和 `output/leagle_qwen3_4b_700k_5ep_d1_r1_full/`。训练日志位于各自的 `logs/`，checkpoint 位于 `checkpoints/`。服务命令带 `--provenance-dir`，vLLM 复现记录保存在对应运行目录的 `server/<时间戳>_<PID>/`；训练的 `train_command.txt` 和 `speculators.patch` 保存在 `checkpoints/`。发布结果时保留这两处复现记录。此前在 `leagle/runs/` 下的运行产物可继续保留；如需使用原路径，在服务和训练脚本中显式设置相同的 `OUTPUT_DIR`。

模型、数据、输出路径和 NPU 分配可在各自脚本配置区或通过同名环境变量修改。服务与训练脚本的 `MODEL`、`OUTPUT_DIR`、`VLLM_PORT`、`NUM_LAYERS` 和 LEAGLE 的 `NUM_LOOPS` 必须一致；两边均使用目标特征层 `[2,18,33]`，服务另加最后一层 teacher slot 36。`VLLM_NPUS` 与 `TRAIN_NPUS` 应使用互不重叠的设备。重复实验时，为两份脚本指定同一个新的 `OUTPUT_DIR`。

测试其他深度和 loop 时，可在两个终端分别指定相同参数，例如：

```bash
NUM_LAYERS=2 NUM_LOOPS=3 bash examples/train/leagle_qwen3_4b_700k/server.sh
NUM_LAYERS=2 NUM_LOOPS=3 bash examples/train/leagle_qwen3_4b_700k/train.sh
```

也可通过统一入口使用 `--speculator-type leagle --num-layers D --num-loops R --ttt-steps 7`，或把对应字段放进训练 YAML 的 `draft` 和 `loss` 组。保存的 checkpoint 包含 `num_loops`。继续训练已有 checkpoint 时不能静默改变这一值。

训练继续复用预处理数据中的 token IDs 和 loss masks，不重新生成回答。脚本不包含真实接受长度评估；评估限制见下节。

训练使用 EAGLE 的 `shift_batch` 对齐和 ground-truth-token TTT：token 输入采用数据中的 token，hidden state 则由 draft 自己逐步递推。目标 KV 的可见范围固定在各自的起始前缀；训练步骤增加不会开放未来目标特征。packed document 边界与 padding 不计入跨边界预测损失。

## 参考推理与验证

`LEagleDraftModel.propose()` 提供单条前缀的真实贪心自回归草稿生成：

```python
model.eval()
tokens, logits = model.propose(
    hidden_states=verified_prefix_features,  # [1, context_len, num_aux * hidden_size]
    anchor_token=bonus_token,               # [1, 1]，目标模型已生成的下一个 token
    num_tokens=7,
)
```

`tokens` 为目标词表 ID，`logits` 对应 draft 词表。方法不会调用目标模型验证；每次验证之后，使用更新后的真实前缀再次调用。当前参考实现每次重新投影目标前缀，没有跨验证轮次的缓存优化。

原有 vLLM EAGLE / DSpark speculator 不能直接执行这种模型。这里的 vLLM 脚本仅用于训练数据的目标特征提取；高性能 serving adapter 仍未接入。完整的单链贪心接受长度评估使用下面的离线入口。普通 validation loss / conditional accuracy 沿用 teacher-forced TTT，不能直接当作真实接受长度。

## 离线接受长度评估

实现位于 `src/speculators_eval/leagle.py` 与 `leagle_decoding.py`，命令入口为 `scripts/evaluate/leagle_offline_eval.py`，启动脚本为 `examples/evaluate/leagle_qwen3_4b_700k.sh`。使用 Transformers 在同一设备加载 full-attention Qwen3 target 和 LEAGLE checkpoint，无需启动 vLLM 服务；默认 target 和 draft 都使用 SDPA、BF16。物理层数和 loop 次数读取 checkpoint，评估不会改变它们。

普通 EAGLE3 使用 `examples/evaluate/eagle3_qwen3_4b_700k.sh`，对应 `scripts/evaluate/eagle3_offline_eval.py`；它与 LEAGLE 共用验证循环。EAGLE3 草稿输入保持训练中的特征/token 错位一位对齐，并使用自己的预测递推。两者与 DSpark 共用 JSONL 处理、选样、prompt 模板和加权统计；完成模板处理后不再额外插入 special tokens。

```bash
DATASETS_ROOT=/path/to/eval_jsonl SPEC_TOKENS=7 \
bash examples/evaluate/eagle3_qwen3_4b_700k.sh

VERIFIER_MODEL=/path/to/Qwen3-4B DRAFT_MODEL=/path/to/dspark/checkpoints/4 \
DATASETS_ROOT=/path/to/eval_jsonl SPEC_TOKENS=7 TEMPERATURE=0 \
bash examples/evaluate/dspark_offline_jsonl.sh
```

DSpark 的 `--spec-tokens` 只限制实际提议数量，不改变已训练的 block 结构；指定值不能超过 checkpoint 支持的槽位数。三者对照时保持数据、seed、thinking、生成长度和两个 attention backend 相同。DSpark 的 `CHECK_GREEDY=1` 适用于 HF target、temperature=0。

从仓库根目录运行，`DATASETS_ROOT` 指向与 DSpark 评估相同的 prompt JSONL 文件或目录，不是训练用的预处理 Arrow 数据。支持 `messages`、`conversations`、`turns`、`prompt` 等已有 DSpark 格式，沿用其样本选择、seed 980406 和默认关闭 thinking 的设置。

```bash
# 先抽少量样本审计：每条生成结果须与 target 单独贪心解码完全一致。
ASCEND_RT_VISIBLE_DEVICES=0 DATASETS_ROOT=/path/to/eval_jsonl \
MAX_SAMPLES=8 CHECK_GREEDY=1 \
bash examples/evaluate/leagle_qwen3_4b_700k.sh

# 正式评估；默认每轮 7 个候选 token、最多生成 512 tokens。
ASCEND_RT_VISIBLE_DEVICES=0 DATASETS_ROOT=/path/to/eval_jsonl \
bash examples/evaluate/leagle_qwen3_4b_700k.sh

# 读取 2 层 / 3 loops 实验的 checkpoint_best。
NUM_LAYERS=2 NUM_LOOPS=3 DATASETS_ROOT=/path/to/eval_jsonl \
bash examples/evaluate/leagle_qwen3_4b_700k.sh

# 多 NPU 按样本分片，每张卡分别加载完整 target + draft。
ASCEND_DEVICES=0,1,2,3 DATASETS_ROOT=/path/to/eval_jsonl \
bash examples/evaluate/leagle_qwen3_4b_700k.sh
```

`NUM_LAYERS` / `NUM_LOOPS` 仅选择默认 checkpoint 路径；指定任意 epoch 时使用 `DRAFT_MODEL=/path/to/checkpoints/4`。可设置 `MODEL`、`OUTPUT_DIR`、`DATASETS`、`MAX_SAMPLES`、`MAX_NEW_TOKENS`、`SPEC_TOKENS`、`ENABLE_THINKING` 和 `RAW_PROMPT_MODE`。CPU 冒烟测试可设置 `DEVICE=cpu DTYPE=float32`，CUDA 可设置 `DEVICE=cuda:0`。多卡参数 `ASCEND_DEVICES` 要求 `DEVICE=npu:0`；它是样本并行，不是张量并行。

评估固定为单链、temperature=0，当前不支持树搜索或随机采样。每轮用 LEAGLE 的真实自回归输出生成最多 K 个候选，target 一次验证，接受连续匹配的最长前缀，并生成拒绝替代 token 或 bonus token。target 的 KV 和特征只保留已验证部分；draft 每轮重新建立上下文和各 `(loop, layer)` 的缓存。

默认结果写入对应训练目录的 `eval_k7_<时间戳>_<PID>/`：

- `summary.csv` / `summary.json`：逐数据集统计，字段与 DSpark 离线评估一致。
- `acceptance_length = 1 + accepted_draft_tokens / proposals`，K=7 时上限 8；`accepted_draft_length` 只计算草稿接受量。
- `position_accept_rates`：每个位置的连续前缀接受次数除以该位置实际被提议次数。另提供 `position_accept_rates_per_round`，以全部提议轮数为分母，与仓库 vLLM `acceptance_at_pos_i` 的分母一致；跨后端仍须核对尾部和 EOS 规则。
- `draft_length`：实际平均提议长度。接近生成预算时会缩短草稿，为 target token 留一位；最后的零草稿 target 轮次也计入 proposals，接受 EOS 时截断统计，与 DSpark 的离线尾部规则相同。
- `artifacts/*.jsonl`：prompt、完整输出 token IDs、各轮提议数和接受数。`--check-greedy` 若发现生成不一致会立即报错。
- `eval_command.txt`、`speculators.patch`、`eval_config.json`：命令、版本、git 状态、运行配置及 checkpoint 架构；发布评估结果时与训练复现记录一并保留。
- `eval_checkpoints.json`、`checkpoint_sha256.txt`、`drafter_checkpoint_sha256.txt`：固定后的 target/draft 路径、epoch、训练 step、配置和权重哈希。开始时解析 `checkpoint_best` 并计算哈希，多进程共用同一份记录；解析后更新 best 指针不会切换正在评估的权重。加载或评估期间文件被删除或替换会报错。哈希计算会增加启动时的磁盘读取开销。

这版用于比较接受长度。日志中的吞吐包含参考实现开销，启用审计还包含 target 单独解码，多卡汇总时间还包含模型加载；这些数值不是 vLLM serving 加速比。逐位置概率分布诊断未计算，结果中相应字段为空并标记 `probability_diagnostics_available=false`。

```bash
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_leagle.py -q
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/evaluate/test_leagle_offline_eval.py -q
```

测试覆盖共享 loop 与展开独立层的前向 / 梯度等价性、缓存长度、packed document 与未来特征隔离、SDPA / eager、词表映射、真实自回归与并行训练轨迹的一致性，以及 activation checkpointing。评估测试额外覆盖全接受、首位 / 中途拒绝后的 KV 和特征回退、EOS、长度边界、不同深度 / loop / 词表的真实小型 Qwen3 生成一致性，以及保存 checkpoint 到 JSONL 结果的完整链路。
