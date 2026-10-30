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

| 参数           | 含义                            | 默认示例 |
| -------------- | ------------------------------- | -------- |
| `--num-layers` | 具有独立参数的物理 decoder 层数 | 1        |
| `--num-loops`  | 每个 token 对同一组层的执行次数 | 1        |
| `--ttt-steps`  | 训练时向前展开的 token 步数     | 7        |

有效 decoder 深度 = `num_layers × num_loops`，不包含融合层和输出头。增加 loop 不增加 decoder 参数量，但增加计算量与缓存需求。

循环内位置编号不增加，只有进入下一个 token 时才增加。每个 `(loop, layer)` 使用独立的草稿 KV 历史；目标上下文不会重复追加进草稿缓存。反向传播覆盖所有 loop 和 token steps，没有 detach 或截断。

首批对照可使用 `(层数, loop)` = `(1,1)`、`(2,1)`、`(4,1)`、`(1,2)`、`(1,4)`、`(2,2)`。后四种中的 `(4,1)`、`(1,4)`、`(2,2)` 具有相同有效深度，能比较独立层和共享层。DFlash 的五层对照还需单独训练 `(5,1)`；有效深度相同不等于训练 FLOPs 或实测延迟完全相同。

## 训练

从仓库根目录运行。沿用已经准备好的 Qwen3-4B / 70 万数据、5 epochs、seed 42、3072 packing 长度、0.9 训练比例和 TTT=7。EAGLE3 使用三个目标特征层 `[2,18,33]`，LEAGLE 使用五个目标特征层 `[1,9,17,25,33]`。两组默认 `DRAFT_VOCAB_SIZE=151936`，使用 Qwen3-4B 的完整词表。默认目标服务使用 NPU 0–1，训练使用 NPU 2–7，端口 8003。脚本中的模型与数据路径需要与训练机器一致。

词表切换无需重新生成预处理数据；启动仍要求数据目录中有 `token_freq.pt`，或与词表大小匹配的 `t2d-151936.npy` / `d2t-151936.npy`。原有 32K 映射不会被当成全词表缓存使用。新训练使用独立的 `fullvocab` 输出目录；如需重新做 32K 对照，可设置 `DRAFT_VOCAB_SIZE=32000` 并另指定 `OUTPUT_DIR`。

Qwen3-4B 预设的所有 draft 层使用 full causal attention。脚本根据 `NUM_LAYERS` 自动生成 `--full-attention-indices 0 ... D-1`，每次 loop 复用相同设置。注意力覆盖完整的可见前缀；未来目标特征、其他文档和其他训练轨迹仍然不可见。

### 普通 EAGLE-3 与 LEAGLE 的首轮对照

每个实验分别使用一份 `server.sh` 和一份 `train.sh`。先在一个终端启动目标服务，看到 ready 后，在另一个终端启动训练。服务脚本在前台运行；训练脚本内部使用 `nohup` 后台启动训练，并打印 PID 和日志路径。训练退出不会停止目标服务；停止服务时在服务终端按 Ctrl+C。

两组沿用上述 70 万数据 / 5 epochs 设置，训练配方分别为：

| 参数              | 普通 EAGLE-3  | LEAGLE                   |
| ----------------- | ------------- | ------------------------ |
| 优化器            | AdamW         | AdamW                    |
| 峰值 LR           | `6e-4`        | `6e-4`                   |
| Adam betas        | `(0.9, 0.95)` | `(0.9, 0.999)`           |
| Weight decay      | `0`           | `0`                      |
| 梯度裁剪 max norm | `0.5`         | `1.0`                    |
| Loss              | KL            | `0.1 CE + 0.9 TV`        |
| Loss 实现         | `eager`       | `legacy`，与 DSpark 相同 |
| LR scheduler      | linear        | cosine                   |
| Warmup 比例       | `0.015`       | `0.04`                   |

普通 EAGLE-3 的 AdamW betas、weight decay 和梯度裁剪取自[作者的 EAGLE3 配置](https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json)，峰值 LR 则按本次实验要求与 LEAGLE 统一为 `6e-4`（作者配置为 `5e-5`）。这里把 `12000 / 800000 = 1.5%` warmup 比例应用到实际 5 epochs 的训练步数，并继续使用仓库的 linear scheduler；这不是原作者 DeepSpeed 训练的完整复现。

LEAGLE 的优化器、loss 与调度参数取自仓库的 [`train_dspark.sh`](../../../examples/train/qwen3_4b_bestarch/train_dspark.sh)。保留 LEAGLE 的层数 / loop、完整词表、目标特征层和 5 epochs，不引入 DSpark 的 Markov / confidence head 及其辅助损失。`training-recipe=upstream` 保留按词表大小命名的缓存和 hidden-state 恢复流程；loss 显式指定 `legacy`，因此使用与 DSpark 相同的 CE / TV 数值实现。两组 attention 均为 SDPA，loss 的实现选项不改变 attention backend。

EAGLE3 训练脚本默认 `FSDP_SHARD=0`、`ACTIVATION_CHECKPOINTING=0`；LEAGLE 仍默认两项均为 `1`。两项均可通过同名环境变量设为 `0` 或 `1`。开启重算时，EAGLE3 使用独立缓存，避免反向传播重复追加 TTT KV；LEAGLE 按物理层重算全部共享 loop。FSDP 分片按物理层执行。这里的默认配置尚需在实际 NPU 集群验证显存和速度。

新运行可用环境变量覆盖 `LR`、`WEIGHT_DECAY`、`ADAM_BETA1`、`ADAM_BETA2`、`MAX_GRAD_NORM`、`LOSS_FN`、`LOSS_IMPLEMENTATION`、`EPOCHS`、`SEQ_LENGTH`、`SEED`、`TTT_STEPS`、`TTT_STEP_LOSS_DECAY` 和 scheduler 参数。`LOSS_FN` 也接受仓库已有的加权 JSON，例如 `'{"ce":0.1,"tv":0.9}'`。修改配方时为实验指定新的 `OUTPUT_DIR`，服务端也使用该目录。

训练脚本通过 `python -m torch.distributed.run` 启动，并将当前仓库的源码路径放在 `PYTHONPATH` 最前面，避免载入环境中安装的旧版 `speculators`。可用 `PYTHON=/path/to/env/bin/python` 指定训练解释器。新版模块入口和旧版 `scripts/train.py` 均支持 `--adam-beta1`、`--adam-beta2`、`--max-grad-norm`（注意最后一个参数的连字符）。

```bash
# 只打印最终启动命令，不连接服务、不加载模型、不创建训练输出。
DRY_RUN=1 NUM_LAYERS=2 NUM_LOOPS=2 \
bash examples/train/leagle_qwen3_4b_700k/train.sh

# 已停止的训练继续运行；先用相同 OUTPUT_DIR 重启 server.sh。
RESUME=1 OUTPUT_DIR=./output/leagle_qwen3_4b_700k_5ep_d2_r2_fullvocab_5hs \
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

`RESUME=1` 从该目录的 `checkpoints/run.yaml` 恢复原架构、loss、LR、优化器（含 betas）、梯度裁剪、scheduler 和显存设置，再加载最新训练进度；不会用脚本的新默认值覆盖原配方。旧 YAML 未记录 betas / 梯度裁剪时，仍采用历史默认值 `(0.9, 0.999)` / `1.0`；旧 Muon 运行继续使用 Muon。切换到本次新配方需要新的输出目录。续训仍保留所有 epoch，可更新服务端口与输出路径。脚本会拒绝向仍有存活训练 PID 的目录启动第二个 trainer。若早期版本已经删除 `run.yaml`，须先从原始命令恢复配置；缺失的配置不会被猜测。

默认保存每个 epoch（五轮对应 `checkpoints/0` 至 `checkpoints/4`），并维护 validation loss 最优的 `checkpoint_best`。最佳 loss 不保证最佳接受长度；可在独立验证集上评估各 epoch。显式启用底层 `--save-best` 时仅清理旧编号 checkpoint，保留 `run.yaml`、`train_command.txt`、patch 和恢复数据。

两组默认一个物理 decoder 层，LEAGLE 默认 `num_loops=1`；两组均显式使用 full attention。普通 EAGLE-3 保留示例的 Llama decoder 和输入归一化设置；LEAGLE 保留 Qwen3 decoder、输入融合和每层 KV 注入。该实验比较结构及训练配方的组合，接受长度差异不能单独归因于结构或 loop；峰值 LR 已统一，若要隔离结构因素，还需统一目标特征层、loss、betas、梯度裁剪和 scheduler。

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

默认输出分别为 `output/eagle3_qwen3_4b_700k_5ep_d1_fullvocab/` 和 `output/leagle_qwen3_4b_700k_5ep_d1_r1_fullvocab_5hs/`，配套服务使用相同路径。两份评估脚本的 checkpoint 路径均手动指定，默认指向各自上述目录。训练日志位于各自的 `logs/`，checkpoint 位于 `checkpoints/`。服务命令带 `--provenance-dir`，vLLM 复现记录保存在对应运行目录的 `server/<时间戳>_<PID>/`；训练的 `train_command.txt` 和 `speculators.patch` 保存在 `checkpoints/`。发布结果时保留这两处复现记录。旧运行不会自动升级词表或特征层；续训时在服务和训练脚本中显式设置相同的原 `OUTPUT_DIR`，并让服务的辅助层列表匹配原 `run.yaml`，训练仍从该文件读取原配方。

模型、数据、输出路径和 NPU 分配可在各自脚本配置区或通过同名环境变量修改。服务与训练脚本的 `MODEL`、`OUTPUT_DIR`、`VLLM_PORT`、`NUM_LAYERS` 和 LEAGLE 的 `NUM_LOOPS` 必须一致；辅助目标层列表也必须匹配：EAGLE3 为 `[2,18,33]`，LEAGLE 为 `[1,9,17,25,33]`。服务另加最后一层 teacher slot 36，用于 loss 的 teacher 输出，不作为额外的 draft 输入特征。`VLLM_NPUS` 与 `TRAIN_NPUS` 应使用互不重叠的设备。重复实验时，为两份脚本指定同一个新的 `OUTPUT_DIR`。

测试其他深度和 loop 时，可在两个终端分别指定相同参数，例如：

```bash
NUM_LAYERS=2 NUM_LOOPS=3 bash examples/train/leagle_qwen3_4b_700k/server.sh
NUM_LAYERS=2 NUM_LOOPS=3 bash examples/train/leagle_qwen3_4b_700k/train.sh
```

也可通过统一入口使用 `--speculator-type leagle --num-layers D --num-loops R --ttt-steps 7`，或把对应字段放进训练 YAML 的 `draft` 和 `loss` 组。保存的 checkpoint 包含 `num_loops`。继续训练已有 checkpoint 时不能静默改变这一值。

训练继续复用预处理数据中的 token IDs 和 loss masks，不重新生成回答。脚本不包含真实接受长度评估；评估限制见下节。

训练使用 EAGLE 的 `shift_batch` 对齐和 ground-truth-token TTT：token 输入采用数据中的 token，hidden state 则由 draft 自己逐步递推。目标 KV 的可见范围固定在各自的起始前缀；训练步骤增加不会开放未来目标特征。packed document 边界与 padding 不计入跨边界预测损失。

### 训练与验证指标

EAGLE3 / LEAGLE 默认日志采用与 DSpark 对应的紧凑指标，不再输出各位置的 `loss_i`、`ce_loss_i`、`tv_loss_i` 和重复的 `full_acc_i` / `cond_acc_i`。训练目标、各步 loss 权重及梯度计算保持原样。

| 指标                                | 含义                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------- |
| `loss`                              | 原训练目标，仍为各 TTT 步加权 loss 之和                                   |
| `full_acc`                          | 所有有效预测位置的 argmax 匹配率，按有效位置数加权                        |
| `position_0_acc` … `position_6_acc` | 各预测步独立的 argmax 匹配率；位置 0 是第一个草稿 token                   |
| `eal`                               | 每条训练轨迹的连续 argmax 匹配长度 + 1，再按轨迹数求平均                  |
| `accept_len`                        | 每条轨迹计算 `1 + Σᵢ ∏ⱼ≤ᵢ αⱼ` 的均值，`α` 为 draft/teacher 概率分布的重叠 |

一条轨迹从有监督的位置开始；mask 中断、文档边界和序列尾部截断其连续接受前缀。TTT=7 时两个长度指标上限为 8；尾部不足七步时，即使全部匹配也可能低于 8。各卡先合并原始分子、分母，再计算均值，空 mask 不贡献长度样本。分布重叠按 token 分块计算 FP32 概率，不保留全序列概率张量或梯度图。

LEAGLE 训练脚本默认 `LOG_FREQ=10`，仅在日志步计算训练接受长度、准确率和候选召回率；验证集每个 batch 都计算完整指标。设为 `LOG_FREQ=1` 可恢复逐步统计，续训也支持此覆盖。所有步骤仍计算相同的主损失与候选 KL，并正常更新参数；没有统计的步骤不会写入虚假的零召回率。直接调用模型时默认保留指标，训练器按日志频率传入 `collect_metrics`。

排查训练耗时时可设置 `PROFILE_FREQ=10`，将已有训练器的分阶段计时透传到新训练或续训。空值沿用配置（新训练默认关闭），`0` 明确关闭；日志中的 `fetch_ms`、`fwd_ms`、`bwd_ms`、`opt_ms` 可区分取数、前向、反向及优化器耗时。采样步会同步设备，本身有额外开销，正式吞吐对比应关闭 profiling，并在相同数据、卡数和计算配置下跳过启动预热。

LEAGLE 使用已知的 TV 损失实现时，直接从其逐位置结果提取 `1 - TV` 供接受长度统计使用，避免再次计算全词表 softmax；其他目标及自定义损失保留原概率重叠计算。该观察只保存 detach 后的逐位置向量，不改变损失值、mask、权重或反向图。训练也复用 draft/target argmax，省去旧指标的重复构造。训练器在新一轮 forward 前释放上一轮梯度，降低激活与梯度同时驻留的显存；不改变优化器更新顺序。

训练器将日志和验证的标量指标按设备及 dtype 分组合并拷回 CPU，避免逐指标 `.item()` 的多次小传输，同时保留整数计数和浮点精度。验证仍按原 batch 顺序在 CPU 累加。每步的有效监督判断在跨卡归约后一次读取本卡状态与有效卡数，保留空监督跳步及梯度缩放规则；非日志步也不再创建仅供日志使用的错误计数张量。这些公共训练器优化同样适用于 EAGLE 等模型。

训练日志使用 `train/…`，验证日志使用 `val/…_epoch`。两种长度均是 teacher-forced 训练代理，不能替代离线真实生成的 `acceptance_length`。当前全词表脚本按完整 teacher 分布统计；旧缩小词表运行仍使用其裁剪后重新归一化的 teacher 分布，不能与全词表接受长度直接比较。`checkpoint_best` 继续按 validation loss 选择。

### 两个训练优化总开关

训练脚本提供两个独立的可选总开关，默认均留空，不改变现有运行配置：

| 环境变量 | CLI | 管理范围 |
| --- | --- | --- |
| `LEAGLE_SAFE_OPTIMIZATIONS` | `--leagle-safe-optimizations` / `--no-leagle-safe-optimizations` | teacher 缓存、监督位置缓存、teacher 命中/缺失索引复用、冻结输出权重的批次内 autocast 转换复用、选择性 checkpoint、mask 缓存、FSDP 前向参数驻留、相同冻结输出头共享、DDP 梯度桶视图 |
| `LEAGLE_NUMERICAL_OPTIMIZATIONS` | `--leagle-numerical-optimizations` / `--no-leagle-numerical-optimizations` | 训练 KV 复用、稀疏注意力、分块 loss、有效位置输出头、候选块对齐、候选梯度 addmm、共享 loss 归一化、主 loss 流式梯度、DDP 批量梯度拷贝，以及候选流式/反向重算路径选择 |

`SAFE` 指在已有 CPU 对照中，相对同一前置计算路径没有发现额外梯度差异，不保证所有 NPU 内核逐元素一致。共享归一化与主 loss 流式梯度放在数值组，是因为其 FP32 优化仍改变运算顺序；BF16 下两项仍按原有规则回退。

- `1`：启用本组满足前置条件的成员；`0`：停用本组。总开关优先于单项设置，无关参数顺序，包括命令行单项覆盖与旧 YAML 中的成员值。
- 留空：新训练使用已有单项配置；续训继承 `run.yaml` 中的总开关及成员。若要从已启用总开关的运行切回单项控制，将对应 YAML 总开关设为 `null`，而非 `false`。数值组的 `false` 是主动关闭，不是释放控制权。
- 安全组不主动启用数值前置条件：未使用分块头时不启用 teacher 缓存和冻结输出权重转换复用；未启用稀疏注意力时不启用选择性 checkpoint / mask 缓存；未启用 activation checkpointing 或 FSDP 时，相应优化自动跳过。
- 数值组明确关闭时，也关闭会隐式引入分块的 teacher 缓存以及依赖稀疏注意力的两项优化。分块 loss 关闭可能显著增加显存。即使安全组开启，也不会绕过此限制。
- 已有候选头时，数值组 `0` 选择 `LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE=1`，用较慢的原生反向重算代替原有流式候选梯度；数值组 `1` 选择流式路径并开启候选 addmm / 块对齐。未启用候选头时跳过这些成员。两组均不创建候选头或修改层数、loop、LR、loss 配方。

最终成员值与总开关一起写入训练的 `run.yaml`，**不写入模型的 `config.json`，不增加 checkpoint 权重**。从完整 checkpoint 初始化时，会根据实际存在的候选头重新解析，并更新运行记录。输出头共享仍要求两份权重完全相同且冻结；数值组中的稀疏注意力仍要求 SDPA，loss 优化仍遵循各自的内置 loss 支持范围。上述总开关仅管理训练执行选项，离线评估开关仍在评估脚本配置；日志降频、指标复用等已合入的公共优化不回滚。

```bash
# 仅允许安全组；停用有数值变化的路径及其依赖，显存需求可能增加。
LEAGLE_SAFE_OPTIMIZATIONS=1 LEAGLE_NUMERICAL_OPTIMIZATIONS=0 \
bash examples/train/leagle_qwen3_4b_700k/train.sh

# 开启两组适用的优化；数值差异和实际加速仍需在目标 NPU 上验证。
LEAGLE_SAFE_OPTIMIZATIONS=1 LEAGLE_NUMERICAL_OPTIMIZATIONS=1 \
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

### 实验优化开关（默认关闭）

下列开关独立于层数、loop 次数和 TTT 长度。普通训练和优化训练共用同一套 TTT 递推、监督与指标逻辑，关闭开关时保持原有计算行为。对应总开关未设置时，训练脚本中的空值表示新训练默认关闭、续训沿用 `run.yaml`；明确设置 `0` 或 `1` 可覆盖续训配置。总开关非空时按上方规则统一解析成员。推理开关每次评估默认关闭。

| 阶段 | 脚本环境变量（设为 `1` 开启）  | CLI 参数                         | 实际行为                                                                                                          |
| ---- | ------------------------------ | -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 训练 | `LEAGLE_REUSE_TRAINING_KV`     | `--leagle-reuse-training-kv`     | 每个物理层只投影一次 target KV，在 TTT 和共享 loop 间复用，保留梯度；投影仍在 decoder 的 forward 内执行           |
| 训练 | `LEAGLE_CHUNKED_LOSS`          | `--leagle-chunked-loss`          | 分块投影完整词表并计算主 loss，反向重算块内中间量，避免保存整序列 logits/probabilities                            |
| 训练 | `LEAGLE_SPARSE_ATTENTION`      | `--leagle-sparse-attention`      | SDPA 按 query 块读取目标前缀和本块轨迹，省去其他轨迹的无效 KV；保留 packed document、padding 和窗口边界           |
| 训练 | `LEAGLE_CANDIDATE_GRAD_ADDMM`  | `--leagle-candidate-grad-addmm`  | 候选头梯度直接以 FP32 `addmm_` 累加，省去逐块 `[V,R]` 梯度临时矩阵；要求已有候选头                                |
| 训练 | `LEAGLE_TEACHER_CACHE`         | `--leagle-teacher-cache`         | 在容量预算内缓存当前 forward 的 teacher 概率与标签，供 TTT 和反向重算复用                                         |
| 训练 | `LEAGLE_CACHE_FROZEN_HEAD_WEIGHTS` | `--leagle-cache-frozen-head-weights` | 分块 loss 在当前 forward/backward 内复用冻结 FP32 输出权重的 BF16/FP16 转换；不修改参数或 checkpoint |
| 训练 | `LEAGLE_CACHE_TEACHER_INDICES` | `--leagle-cache-teacher-indices` | 每个 TTT step 预建 teacher 缓存命中/缺失索引，在各块反向重算中复用；要求 teacher cache |
| 训练 | `LEAGLE_DDP_GRADIENT_AS_BUCKET_VIEW` | `--leagle-ddp-gradient-as-bucket-view` | 将梯度作为 DDP 通信桶的视图，减少额外梯度存储；仅 DDP |
| 训练 | `LEAGLE_DDP_BATCHED_GRAD_COPY` | `--leagle-ddp-batched-grad-copy` | 批量执行梯度到通信桶的拷贝和除法，减少逐参数调度；仅支持该参数的 PyTorch DDP |
| 训练 | `LEAGLE_MASKED_HEAD`           | `--leagle-masked-head`           | 主输出头只投影有监督位置，候选头进一步使用自己的有效 mask；保留原归一化                                           |
| 评估 | `LEAGLE_SPARSE_CANDIDATES`     | `--leagle-sparse-candidates`     | 保留候选 IDs 和分数，在候选集采样；接受概率与残差校正使用真实 q，拒绝后仍可选到候选集外的 target token            |
| 评估 | `LEAGLE_STATIC_KV`             | `--leagle-static-kv`             | 预分配请求内的 verified-prefix KV 和每轮、每个逻辑深度的 rollout KV；全注意力且所有 context 均可见时省去全真 mask |
| 评估 | `LEAGLE_REUSE_ROLLOUT_BUFFERS` | `--leagle-reuse-rollout-buffers` | 请求内跨轮复用每个逻辑深度的 rollout 存储，只追加新增真实前缀 KV；要求 static KV 和默认的前缀复用                 |

每个 CLI 布尔参数均支持对应的 `--no-leagle-…`。训练开关保存在 `run.yaml`，通过模型调用或训练器初始化生效，不增加 checkpoint 权重；评估开关写入 `eval_config.json` 的 `experimental_optimizations`，并传递到每个评估 worker。

`LEAGLE_LOSS_CHUNK_SIZE` / `--leagle-loss-chunk-size` 控制主输出头每块 token 数，`LEAGLE_ATTN_CHUNK_SIZE` / `--leagle-attn-chunk-size` 控制注意力每块 query 数，两者默认 **128**。它们与辅助候选 KL 的 `CANDIDATE_HEAD_CHUNK_SIZE`（默认 32）不同；增大块大小会改变峰值显存和调度开销。主 loss 分块目前支持仓库内置 CE、TV、forward KL 及其加权组合，不接受自定义 callable。分块会在 backward 重算，因此主要用于降低显存，不保证更快。

稀疏训练注意力要求 `sdpa` 和零 attention dropout。KV 复用要求不随序列长度变化的 RoPE。稀疏候选要求 checkpoint 已训练候选头且该头已启用；它不改变候选数。静态 KV 容量按请求长度预留，可能增加常驻显存；每轮的合成草稿缓存仍全部丢弃，跨轮只保留 verifier 确认的前缀。非规则位置或滑动窗口仍构造显式 mask。

新增的输出头优化保持默认关闭，也保存在 `trainer` 配置组。`LEAGLE_TEACHER_CACHE` 和 `LEAGLE_MASKED_HEAD` 自动采用分块输出头，共用 `LEAGLE_LOSS_CHUNK_SIZE`；无需同时开启 `LEAGLE_CHUNKED_LOSS`。teacher 缓存仅支持 eager 的 CE、TV、forward KL 及其组合，CPU/NPU 上也兼容脚本默认的 `legacy`；CUDA 使用缓存时应明确设置 `LOSS_IMPLEMENTATION=eager`。不支持 fused/custom loss 时明确报错，不静默切换损失实现。

`LEAGLE_TEACHER_CACHE_MB` / `--leagle-teacher-cache-mb` 默认 **1024 MiB/卡**，限制常驻 FP32 teacher 概率与 int64 标签的合计容量，临时投影、softmax 和 loss 工作区另计。缓存只覆盖当前 batch 能放下的固定前缀，其余位置仍分块计算，避免有限容量 LRU 在每轮完整遍历中反复失效。缓存不跨 batch、优化器更新或 checkpoint 持久化。全词表 3072 长度的一份完整 FP32 概率约为 1.74 GiB，调大预算前应留出工作区显存。

`LEAGLE_CACHE_FROZEN_HEAD_WEIGHTS` 仅在已启用分块输出头、权重冻结且为 FP32、并使用 BF16/FP16 autocast 时生效。DDP 保留 FP32 参数，而 autocast 不缓存冻结参数，原先每个小块及其反向重算都可能重新转换整张词表矩阵；此选项在当前 loss 图内每份权重只转换一次。共享 draft/teacher 参数时只保留一份转换结果；权重已是 BF16、没有低精度 autocast 或参数可训练时直接使用原权重。它不主动启用分块，也不跨批次缓存。代价是每份不同输出矩阵额外保留低精度副本：Qwen3-4B 全词表约 **742 MiB**，未共享的两份约 **1.45 GiB**，不计入 teacher 概率缓存预算。FSDP 使用 BF16 计算权重时不增加这份副本。

`LEAGLE_CACHE_TEACHER_INDICES` 不改变缓存容量、投影位置和矩阵形状。对于监督子集或压缩后的位置，每个 TTT step 只读取一次短索引向量，并一次性传回打包后的索引；各块 forward/backward 不再反复执行动态 `nonzero`。普通连续前缀查询仍使用原来的算术切片。索引只属于当前 forward 的 loss 图，额外占用少量元数据空间，不计入 teacher 概率缓存预算。CPU 的分块、有效位置、部分命中、候选头、FP32/BF16 autocast 对照保持 loss、梯度和更新一致；NPU 上的同步和索引开销收益需实测。

DDP 下，安全组自动启用 `gradient_as_bucket_view`；数值组请求 `batched_grad_copy`。当前优化器的 `zero_grad()` 保持原行为，不改成持久化梯度或低精度通信。批量拷贝可能改变浮点除法的舍入，因此归入数值组。运行时检查当前 PyTorch 的实际 DDP 参数：不支持的选项会输出明确警告，并在 `run.yaml` 记录为关闭。FSDP 下两项自动跳过；显式单独启用 DDP 参数同时使用 FSDP 会报错。未启动分布式训练时也会关闭 DDP 参数并记录实际配置。

`LEAGLE_DDP_BUCKET_CAP_MB` / `--leagle-ddp-bucket-cap-mb` 可手动设置正整数通信桶容量（MiB）。留空时，新运行不传该参数、保留 PyTorch 默认分桶行为；续训继承原值。数值组设为 `0` 时会恢复默认分桶，避免自定义桶改变归约顺序；数值组设为 `1` 不会擅自调整桶容量。桶越大不保证越快，应根据通信与反向的重叠情况实测。新 DDP 运行示例：

```bash
FSDP_SHARD=0 LEAGLE_SAFE_OPTIMIZATIONS=1 LEAGLE_NUMERICAL_OPTIMIZATIONS=1 \
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

如需试验桶大小，在上面增加 `LEAGLE_DDP_BUCKET_CAP_MB=64`。续训仍从 `run.yaml` 读取 FSDP/DDP 策略，脚本的 `FSDP_SHARD` 环境变量只选择新运行策略。`PROFILE_FREQ=10` 可采样输出取数、前向、反向和优化器分段耗时；计时时会同步设备，测稳定吞吐前应设回 `PROFILE_FREQ=0`。上述控制无需候选头，且不修改层数、loop、TTT、loss 或 LR；实际整体收益取决于目标 NPU 上的瓶颈。

有效位置输出头保持主 loss 的目标位置 mask、候选 KL 的源/目标联合 mask，以及每个 TTT 步的原始分母。LEAGLE 训练器在所有输出头路径都设置 `return_predictions=False`：不记录指标且使用 teacher forcing 时跳过无用的 draft argmax；分块路径也跳过仅用于补齐预测的尾部投影。日志与验证指标仍正常计算。模型直接调用仍默认返回完整预测；自回归训练即使不返回预测，也保留递推所需的预测。空监督路径保留零梯度及正常分布式反向，不把零梯度改成缺失梯度。

候选梯度 `addmm_` 使用 FP32 乘加，改变原来“每块先舍入到 BF16 再累加”的数值边界，但不改变 KL、mask 或 loss 权重。是否更快取决于设备的 FP32 矩阵乘法性能；有效位置选择也可能产生动态索引开销。这两项需要比较训练数值及 NPU 实际耗时。

rollout 存储复用要求 `LEAGLE_STATIC_KV=1`，不能搭配 `--no-draft-kv-reuse` 或自定义 proposer。每轮先丢弃整个合成后缀，再写入新增的真实 target-conditioned KV；被接受的合成状态也不会直接保留。容量按请求上限、每个逻辑深度预留，整个请求结束后释放，不在不同请求之间共享。

以下进一步优化同样默认关闭，不要求增加磁盘存储。训练仍使用 `--on-generate delete`，隐藏状态文件用完即删，不保存跨 epoch 的数据缓存。

| 阶段       | 脚本环境变量（设为 `1` 开启）           | CLI 参数                                  | 实际行为                                                                     |
| ---------- | --------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| 训练       | `LEAGLE_FSDP_KEEP_FORWARD_PARAMS`       | `--leagle-fsdp-keep-forward-params`       | decoder 在 forward 后保留已聚合参数，减少 TTT / loop 重复前向的 all-gather   |
| 训练       | `LEAGLE_TEACHER_CACHE_SUPERVISED`       | `--leagle-teacher-cache-supervised`       | teacher 缓存优先覆盖有监督位置，不用固定序列前缀占满预算；要求 teacher cache |
| 训练       | `LEAGLE_SHARED_LOSS_NORMALIZATION`      | `--leagle-shared-loss-normalization`      | 主 CE/TV/KL 与候选 KL 的 teacher 共用 FP32 归一化结果                        |
| 训练、评估 | `LEAGLE_SHARE_OUTPUT_HEAD`              | `--leagle-share-output-head`              | 完全相同且冻结的 draft / teacher 输出矩阵共享同一个参数                      |
| 评估       | `LEAGLE_STATIC_TARGET_KV`               | `--leagle-static-target-kv`               | target KV 预分配，独立维护可见长度，拒绝后裁短并覆盖旧后缀                   |
| 评估       | `LEAGLE_SELECTIVE_TARGET_HIDDEN_STATES` | `--leagle-selective-target-hidden-states` | target 只保留 draft 使用的 hidden-state 层，避免保留全部层输出               |

FSDP 参数保留要求 `FSDP_SHARD=1`，以额外设备内存换取较少通信。它只改变 decoder forward 后的 reshard 策略，backward 仍由 FSDP 管理，不能消除所有反向 all-gather。层数、loop 次数、loss 或梯度同步方式不变；设备内存紧张时保留关闭。

监督位置缓存从目标位置的 `loss_mask` 与 document 有效性中选取位置，预算不足时只缓存其中一部分，未命中位置仍正常计算。`LEAGLE_TEACHER_CACHE_MB` 同时计入位置映射的 int64 存储；临时索引、投影与 loss 工作区仍另计。建议与 `LEAGLE_MASKED_HEAD=1` 配合，避免为不参与监督的位置计算输出。缓存仅在当前 forward / backward 中存在，不写入磁盘。

共享归一化自动启用分块输出头，支持范围与 teacher cache 相同：eager CE、TV、forward KL，CPU/NPU 也兼容 legacy。仅在 FP32 且没有低精度 autocast 时，主干 logits 的一次 `log_softmax` 及其概率用于多项损失；候选 KL 读取 detached 分布，不向主干或完整输出头反传。BF16/FP16 或低精度 autocast 自动回退到原有分块 loss 运算，teacher cache 同时使用原分布计算方式，保留原生 CE 精度和各 loss 的梯度舍入边界。这些低精度运行没有共享归一化的提速收益，启动日志会提示回退；训练入口仍只支持 FP32/BF16。

输出矩阵共享会检查两份权重已加载、有限、冻结，形状、精度、设备与数值完全一致，任何不满足条件都会报错。训练在设备放置、分布式初始化及 checkpoint 加载之后再次验证，防止续训覆盖共享 teacher。checkpoint 若来自可训练输出头或与当前 target 不一致，应关闭此开关。保存仍只包含原有 draft 输出矩阵，不增加文件。Qwen3-4B 全词表 BF16 的一份 `[151936, 2560]` 矩阵约为 742 MiB；非分片时可省下这一份常驻副本，FSDP 下的每卡收益取决于分片和参数聚合状态。

两项 target 优化仅支持当前 full-attention Qwen3 评估路径。静态 KV 按请求上限预留，可能提高常驻设备内存；注意力只能看到已写入的有效前缀，拒绝后缩短有效长度，下轮写入覆盖被拒绝的后缀。选层保留 HF 编号：slot 0 是 embedding 流，slot N 是最后归一化后的 hidden state；hook 在请求结束或失败时移除。两项均不改变提议预算、接受规则或采样校正。

```bash
# 监督位置缓存；仍按需生成并删除隐藏状态文件。
LEAGLE_TEACHER_CACHE=1 LEAGLE_TEACHER_CACHE_SUPERVISED=1 LEAGLE_MASKED_HEAD=1 \
bash examples/train/leagle_qwen3_4b_700k/train.sh

# 先单独对照这两个训练优化，再考虑组合。
LEAGLE_SHARE_OUTPUT_HEAD=1 bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_SHARED_LOSS_NORMALIZATION=1 bash examples/train/leagle_qwen3_4b_700k/train.sh

# 参数保留需要足够的设备内存；默认关闭。
FSDP_SHARD=1 LEAGLE_FSDP_KEEP_FORWARD_PARAMS=1 \
bash examples/train/leagle_qwen3_4b_700k/train.sh

# 评估 checkpoint 路径继续手动指定。
DRAFT_MODEL=/path/to/checkpoint LEAGLE_STATIC_TARGET_KV=1 \
LEAGLE_SELECTIVE_TARGET_HIDDEN_STATES=1 LEAGLE_SHARE_OUTPUT_HEAD=1 \
bash examples/evaluate/leagle_qwen3_4b_700k.sh
```

建议每次只开启一项进行对比，例如：

```bash
LEAGLE_REUSE_TRAINING_KV=1 bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_CHUNKED_LOSS=1 LEAGLE_LOSS_CHUNK_SIZE=128 bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_SPARSE_ATTENTION=1 LEAGLE_ATTN_CHUNK_SIZE=128 bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_SPARSE_CANDIDATES=1 DRAFT_MODEL=/path/to/candidate-checkpoint bash examples/evaluate/leagle_qwen3_4b_700k.sh
LEAGLE_STATIC_KV=1 DRAFT_MODEL=/path/to/checkpoint bash examples/evaluate/leagle_qwen3_4b_700k.sh
LEAGLE_CANDIDATE_GRAD_ADDMM=1 CANDIDATE_HEAD=1 bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_TEACHER_CACHE=1 LEAGLE_TEACHER_CACHE_MB=1024 bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_MASKED_HEAD=1 bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_STATIC_KV=1 LEAGLE_REUSE_ROLLOUT_BUFFERS=1 DRAFT_MODEL=/path/to/checkpoint bash examples/evaluate/leagle_qwen3_4b_700k.sh
```

这些是实验路径：CPU FP32/BF16 测试覆盖前向、loss、梯度、checkpointing、mask 可见性、贪心一致性和采样校正；本地单进程、双进程 CPU FSDP 已通过 loss/梯度对照，包括部分 teacher 缓存和空监督。NPU 的数值、显存与吞吐尚待实测。块大小、SDPA 后端和梯度累加顺序可能造成浮点差异。稀疏采样改变随机数使用方式，同一 seed 不保证与稠密候选路径逐 token 相同，应比较目标分布与评估统计。对比训练应使用不同 `OUTPUT_DIR`。

### 分块、重算与 mask 复用

以下开关默认关闭，支持对应的 `--no-…`，并写入 `run.yaml`。它们不增加模型参数或磁盘数据缓存。

| 脚本环境变量                    | CLI 参数                          | 行为                                                                                                                          |
| ------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `LEAGLE_ALIGN_CANDIDATE_CHUNKS` | `--leagle-align-candidate-chunks` | 自动采用分块 loss，将实际主块大小设为 `max(LEAGLE_LOSS_CHUNK_SIZE, CANDIDATE_HEAD_CHUNK_SIZE)`；要求候选头                    |
| `LEAGLE_SELECTIVE_CHECKPOINT`   | `--leagle-selective-checkpoint`   | 选中的 decoder 层仅重算 MLP，保留稀疏注意力内部的分块 checkpoint，避免两级重复重算；要求稀疏注意力与 activation checkpointing |
| `LEAGLE_CACHE_ATTENTION_MASKS`  | `--leagle-cache-attention-masks`  | 在当前 forward/backward 内复用相同 TTT 步、窗口和 query 块的可见性 mask；要求稀疏注意力                                       |

候选 KL 在主 loss 块内部执行，所以主块 128 会将候选块 512 限制为最多 128；训练入口会提示这种配置。对齐开关会扩大主块，而不是暂存整序列词表分布。它增加投影、softmax 的临时显存，分块变化也可能改变梯度累加的舍入；保持原 loss、mask、分母、TTT 衰减和候选 KL 权重。

选择性 checkpoint 仍遵循 `ACTIVATION_CHECKPOINTING` 和 `--activation-checkpoint-layers`；未选中的层保持原来的执行方式。注意力的 Q/K/V 及部分层内激活保留到 backward，因此峰值显存可能增加。不要把整个 decoder 的重算开关关闭后仍期待该选项生效。FSDP 参数保留是独立选项。

`LEAGLE_ATTENTION_MASK_CACHE_MB` / `--leagle-attention-mask-cache-mb` 默认 **64 MiB/卡**，只限制缓存持有的 mask 张量字节数；临时工作区和 attention 自身所需的存储另计。缓存满后，其余块按原方式构造，不影响可见性。不同文档、窗口、TTT 步不能混用 mask；缓存不跨 batch，也不会进入 checkpoint。

```bash
# 允许主 loss 块扩大到 512；先确认显存余量。
CANDIDATE_HEAD=1 CANDIDATE_HEAD_CHUNK_SIZE=512 LEAGLE_ALIGN_CANDIDATE_CHUNKS=1 \
bash examples/train/leagle_qwen3_4b_700k/train.sh

# 单独检查重算粒度和 mask 缓存的收益，再决定是否组合。
LEAGLE_SPARSE_ATTENTION=1 ACTIVATION_CHECKPOINTING=1 LEAGLE_SELECTIVE_CHECKPOINT=1 \
bash examples/train/leagle_qwen3_4b_700k/train.sh
LEAGLE_SPARSE_ATTENTION=1 LEAGLE_CACHE_ATTENTION_MASKS=1 LEAGLE_ATTENTION_MASK_CACHE_MB=64 \
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

离线评估默认将采样诊断向量保留在设备上，在请求结束时合并回传；每个请求只构造一次 EOS 集合。原生 LEAGLE greedy 提议不再保留和拼接整轮 draft logits，采样验证仍保留完整所需分数。直接调用 `propose()` 默认仍返回分数；只有 greedy 支持显式 `return_logits=False`。自定义 proposer 的调用接口和 EAGLE 模型本身保持不变。接受规则、EOS 截断和采样随机数调用顺序不变。

### 主 loss 流式梯度与常驻评估 worker

以下两项默认关闭，可单独启用，不改变模型参数结构或已有 checkpoint 的加载方式。

| 脚本环境变量                | CLI 参数                    | 行为                                                                                                 |
| --------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------- |
| `LEAGLE_STREAM_MAIN_LOSS=1` | `--leagle-stream-main-loss` | 自动启用分块主 loss；仅 FP32 预计算 hidden 梯度，BF16/低精度 autocast 自动使用原分块 checkpoint 路径 |
| `PERSISTENT_WORKERS=1`      | `--persistent-workers`      | 多 NPU 离线评估时，每卡只加载一次 target/draft，连续处理所有选定数据集                               |

主 loss 开关支持冻结输出矩阵下的 eager CE、TV、forward KL 及其组合，使用原有 loss 运算计算梯度，保留原 mask、完整序列分母、TTT 衰减和 TV 相等处的梯度规则。梯度预计算仅支持 FP32，可与候选头、teacher cache、mask 筛选、activation checkpointing 与 FSDP 组合。可训练的主输出矩阵、fused loss 与高阶梯度不在支持范围。普通 backward 仍负责 decoder/候选头参数梯度和外层 loss 缩放。

代价是保存每个 TTT 步的 hidden 梯度，额外存储量约为 `min(TTT_steps, L) × B × L × H × dtype_bytes`，不保存整序列词表分布。分块大小继续由 `LEAGLE_LOSS_CHUNK_SIZE` 控制；与候选块对齐同时启用时，遵循对齐后的块大小。

BF16/FP16 或低精度 autocast 下，即使开启此开关，也自动退回原来的分块 checkpoint，不预计算 hidden 梯度。原因是把任意外层 loss 缩放移到低精度舍入之后会改变梯度；默认 BF16 训练因此**没有这项加速收益**。启动日志会提示回退。测试对 BF16/混合精度采用逐元素严格一致的梯度检查，包括 `0.37` 和 `4/3` 缩放；FP32 使用严格浮点误差界限。比较时保持主 loss 块大小一致。NPU 实际数值、显存与吞吐仍需测量。

常驻 worker 仅适用于 `--ascend-devices` 的样本分片评估，要求 target/draft 的 RoPE 频率不随上下文长度变化（支持 default、linear、yarn、llama3）。数据集之间保持屏障，固定每卡的样本分片；每个数据集恢复模型加载完成后的 CPU/设备随机状态，避免温度大于 0 时因随机流延续而改变结果。每份分片仍保存独立的 checkpoint 清单、patch、配置和可单独重放的 `eval_command.txt`。worker 失败会终止本轮并回收已启动的子进程，不读取旧结果冒充成功。

父进程计时仍是实际 wall time：**第一个数据集包含一次模型加载，之后的数据集不再包含加载**，该范围写入 `eval_config.json`。因此它节省整个评估任务的启动时间，并不直接提高单次 draft/target forward 的吞吐；与旧方式比较时，应比较整个数据集集合的总耗时。评估 9 个数据集时，每卡模型加载从 9 次降至 1 次。只产生少量临时控制文件，结束即清理，没有新增 hidden-state 磁盘缓存。

```bash
LEAGLE_STREAM_MAIN_LOSS=1 bash examples/train/leagle_qwen3_4b_700k/train.sh
PERSISTENT_WORKERS=1 DRAFT_MODEL=/path/to/checkpoint bash examples/evaluate/leagle_qwen3_4b_700k.sh
```

### 正确性核查记录（2026-10-30）

本轮在 Windows CPU、Python 3.12.10、PyTorch 2.12.1、Transformers 5.12.1 上核查。训练数值审计使用仓库的小模型 fixture：hidden size 16、词表 24、2 层 × 2 loops、TTT=7、长度 9 的 packed sequence，候选 rank=4；两个随机种子、KL / CE+TV 两种目标、`0.37` / `4/3` 外层 loss 缩放。逐项比较仅改变被测开关及其必要前置配置；全开则与所有实验开关关闭的路径比较。这些数值是该测试矩阵的观测最大值，**不是 Qwen3-4B 或 NPU 上的误差上界，也不能换算成下游精度损失**。

相对梯度误差定义为 `||g_optimized - g_reference||₂ / ||g_reference||₂`。主干和候选头分别统计，避免候选头的偏差被较大的主干梯度掩盖。

| 优化                                       | BF16 / 混合精度核查结果                                 | 判定                                                         |
| ------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| 主 loss 流式梯度                           | 回退后与相同块大小的原路径逐元素一致                    | FP32 才启用预计算                                            |
| 共享 loss 归一化                           | 原实现整体梯度差最高约 0.77%；修正后严格一致            | FP32 才启用共享                                              |
| teacher 缓存、监督位置缓存                 | 对照相同分块路径，loss / 梯度严格一致                   | 测试通过，缓存不跨 batch                                     |
| mask 缓存、选择性 checkpoint、输出矩阵共享 | 对照各自前置路径，loss / 梯度严格一致                   | 测试通过；选择性 checkpoint 的前置稀疏注意力仍有独立数值差异 |
| 训练 target KV 复用                        | 主干相对梯度差最高约 0.13%                              | 仍为默认关闭的实验项，改变梯度合并次序                       |
| 稀疏训练注意力                             | 主干相对梯度差最高约 0.43%                              | 可见性一致，但 SDPA 形状和归约次序变化，不是逐位等价         |
| 分块、候选块对齐、有效位置输出头           | 主干在本样例中一致；候选梯度差最高约 0.26%              | 候选矩阵乘法形状及累加次序变化                               |
| 候选梯度 `addmm_`                          | 主干一致；候选梯度差最高约 0.33%                        | FP32 累加替代 BF16 逐块舍入，保留实验开关                    |
| 全部训练优化组合                           | 修正后整体梯度差最高约 0.42%（BF16）/ 0.44%（混合精度） | **不能宣称全开与基准严格等价**                               |

FP32 的本轮逐项及组合测试，整体相对梯度误差最高约 `2.66e-7`。两个低精度回退路径都有非整数缩放下的逐元素严格检查，不用百分比容差放行。

另外，原有候选头流式辅助梯度本身，与独立 autograd KL 对照时，在单独的候选头测试中仍有最高约 0.53%（BF16）/ 0.56%（混合精度）的相对误差；FP32 约 `1.69e-7`。这是候选头自己的梯度，不是整个模型的梯度。候选输入与 teacher 均 detach，且候选头独立裁剪，因此该辅助损失不直接改变主干梯度；仍不能据此排除候选召回率和接受长度的变化。关闭 `LEAGLE_CANDIDATE_GRAD_ADDMM` 不会关闭原有流式辅助梯度。以上历史全开组合不含下方新增的候选反向重算开关；需要更接近原生 autograd 的候选梯度时，可启用该开关。

实际双进程 CPU FSDP 检查了两个非整数 loss 缩放、空监督 rank、冻结输出矩阵共享，以及保留/重分片参数的两种模式。额外连续执行四个训练步骤，包含部分 rank 无监督、全部 rank 无监督后恢复，比较 loss、梯度和 AdamW 更新后的权重，参数保留开关两侧严格一致。mask、缓存、梯度和候选头的检查与开关继承、续训覆盖和脚本 DRY_RUN 分开验证。

推理覆盖动态/静态 draft 与 target KV、跨轮存储复用、拒绝后覆盖、EOS、短尾、全/缩小词表、所选 hidden-state 层、温度 0/1，以及常驻 worker 的跨数据集 RNG 恢复和失败清理。贪心与独立 target generation 对照；稀疏候选的接受加残差分布恢复完整 target 分布。稀疏候选采样仍不承诺与稠密采样消耗相同随机数。上述 CPU 核查不能替代 NPU 的内核数值、显存和吞吐实测；结构消融应固定数值相关开关，不能把“训练目标相同”写成“训练结果完全相同”。

### 可选候选头反向重算

`LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE=1` / `--leagle-candidate-backward-recompute` 默认关闭，要求模型有候选头。脚本空值在新训练中保持关闭，续训继承 `run.yaml`；显式 `0` / `1` 可覆盖保存配置。它不改变参数名、权重形状、辅助 KL 定义或推理逻辑。

原流式实现先在 forward 计算候选参数梯度，最后乘外层 loss 缩放。这个次序在 BF16 下不同于原生 autograd。新路径把梯度留到 backward，包含实际的外层缩放，再使用原生 `log_softmax` 反向。每个完整 TTT step 的输出矩阵梯度在 FP32 中跨块累积，完成后才转换一次参数精度；中间 activation 梯度仍保留原生低精度转换位置。AMP 的候选权重转换在各 step 间共享，避免改变原生权重梯度的累加精度。

这里必须以完整 TTT step 为单位，不能给主 loss 的每个小块分别建立候选梯度节点。新路径因此独立于主 loss 分块，保留零权重的 masked rows，覆盖 `LEAGLE_CANDIDATE_GRAD_ADDMM` 和 `LEAGLE_MASKED_HEAD` 对候选辅助头的处理；主 loss 的 mask 压缩仍正常生效。候选块大小直接取 `CANDIDATE_HEAD_CHUNK_SIZE`，不再受主 loss 块大小限制。`LEAGLE_ALIGN_CANDIDATE_CHUNKS` 仍可改变主 loss 块大小，但新候选路径不需要它。

代价是额外计算：不复用主 loss 的 dense logits，候选 teacher `W h` 在本路径 forward 和 backward 各计算一次，backward 也重算候选 logits。因此**这是精度对照选项，不能当作提速开关**。保留 detached hidden states、mask 和已有权重引用，不保存整序列 `[L,V]` 激活、不新增磁盘缓存；词表工作区仍受候选块大小限制，backward 使用一组 FP32 `[V,R]` / `[R,H]` 梯度缓冲。

```bash
CANDIDATE_HEAD=1 LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE=1 \
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

CPU 数值对照包含 72 组配置：hidden/词表/rank/长度分别为 `(16,24,4,9)`、`(64,257,16,73)`，两个随机种子、TTT=7、块大小 2/32/128、外层缩放 `0.37` / `4/3`、FP32/BF16/混合精度，并包含无监督 step。对照为独立、不分块的原生 autograd KL，不调用生产梯度公式。

| 候选参数梯度误差          | 原流式路径       | 新反向重算路径     |
| ------------------------- | ---------------- | ------------------ |
| BF16                      | 最高约 0.53%     | 这组测试逐元素一致 |
| FP32 参数 + BF16 autocast | 最高约 0.56%     | 这组测试逐元素一致 |
| FP32                      | 最高约 `4.18e-7` | 最高约 `1.01e-6`   |

单位测试另覆盖负/零外层缩放、部分冻结参数、无监督 batch、候选与主干梯度隔离、组合优化、短序列超过 TTT 的边界，以及保存张量不含词表激活。实际双进程 CPU FSDP 对比独立原生候选 KL，覆盖 FP32/BF16、参数保留开关两种状态、部分/全部 rank 无监督后恢复和连续 AdamW 更新，BF16 梯度及更新权重逐元素一致，FP32 在严格浮点容差内通过。

这些结果不能保证 NPU 或其他 GEMM 形状下逐元素一致，也不消除 KV 复用、稀疏 SDPA 的主干误差。若要将这些误差也排除，应关闭 `LEAGLE_REUSE_TRAINING_KV` 和 `LEAGLE_SPARSE_ATTENTION`，同时关闭依赖后者的 `LEAGLE_SELECTIVE_CHECKPOINT`、`LEAGLE_CACHE_ATTENTION_MASKS`。不得把候选头对照中的一致性解释成全开训练的整体一致性。

## 参考推理与验证

`LEagleDraftModel.propose()` 提供单条前缀的真实自回归草稿生成，默认贪心；传入 `temperature=1` 可按分布采样，并将采样 token 喂回下一步：

```python
model.eval()
tokens, logits = model.propose(
    hidden_states=verified_prefix_features,  # [1, context_len, num_aux * hidden_size]
    anchor_token=bonus_token,               # [1, 1]，目标模型已生成的下一个 token
    num_tokens=7,
)
```

`tokens` 为目标词表 ID，`logits` 对应 draft 词表。方法不会调用目标模型验证；每次验证之后，使用更新后的真实前缀再次调用。上面的直接调用保留逐轮重算的参考路径；下方离线评估入口默认启用已确认前缀的缓存复用。

原有 vLLM EAGLE / DSpark speculator 不能直接执行这种模型。这里的 vLLM 脚本仅用于训练数据的目标特征提取；高性能 serving adapter 仍未接入。完整的单链贪心或采样接受长度评估使用下面的离线入口。普通 validation loss、准确率及训练代理长度沿用 teacher-forced TTT，不能直接当作真实接受长度。

## 离线接受长度评估

实现位于 `src/speculators_eval/leagle.py` 与 `leagle_decoding.py`，命令入口为 `scripts/evaluate/leagle_offline_eval.py`，启动脚本为 `examples/evaluate/leagle_qwen3_4b_700k.sh`。使用 Transformers 在同一设备加载 full-attention Qwen3 target 和 LEAGLE checkpoint，无需启动 vLLM 服务；默认 target 和 draft 都使用 SDPA、BF16。物理层数和 loop 次数读取 checkpoint，评估不会改变它们。

普通 EAGLE3 使用 `examples/evaluate/eagle3_qwen3_4b_700k.sh`，对应 `scripts/evaluate/eagle3_offline_eval.py`；它与 LEAGLE 共用验证循环。EAGLE3 草稿输入保持训练中的特征/token 错位一位对齐，并使用自己的预测递推。两者与 DSpark 共用 JSONL 处理、选样、prompt 模板和加权统计；完成模板处理后不再额外插入 special tokens。

两份离线评估入口默认启用 draft 已确认前缀缓存，无需修改 checkpoint 或重新训练：

- EAGLE3 首轮建立完整前缀 KV，后续轮次只将新增的真实 target 特征与错位一位的 token 输入送入 decoder。上一轮预测状态生成的 KV 全部裁掉，即使对应 token 被接受，也根据真实 target 特征重新建立新增前缀。
- LEAGLE 缓存每个物理层对真实 target 上下文的 K/V 投影，轮内各 token 和共享 loop 复用；下一轮仅融合、投影新确认的上下文。每个 `(loop, layer)` 的草稿轨迹 KV 仍独立且每轮清空，避免将预测状态混入真实 target 上下文。

缓存属于单条生成请求，不跨样本共享；训练 forward、loss、full-attention 可见范围、固定草稿预算和采样验证算法均保持原设置。可在任一评估脚本末尾追加 `--no-draft-kv-reuse` 使用旧参考路径，例如 `bash examples/evaluate/leagle_qwen3_4b_700k.sh --no-draft-kv-reuse`。此开关会传给各 NPU worker，并记录到 `eval_config.json` 的 `draft_kv_reuse_between_rounds` 与 `draft_cache_policy`。随长度改变旧位置旋转频率的 RoPE 配置需要关闭缓存；当前 Qwen3 配置使用的固定频率 RoPE 支持复用。计算批次形状改变可能带来浮点舍入差异，实际 NPU 吞吐需用相同数据和参数测量。

```bash
DATASETS_ROOT=/path/to/eval_jsonl SPEC_TOKENS=7 TEMPERATURE=0 MAX_NEW_TOKENS=512 \
bash examples/evaluate/eagle3_qwen3_4b_700k.sh

VERIFIER_MODEL=/path/to/Qwen3-4B DRAFT_MODEL=/path/to/dspark/checkpoints/4 \
DATASETS_ROOT=/path/to/eval_jsonl SPEC_TOKENS=7 TEMPERATURE=0 \
bash examples/evaluate/dspark_offline_jsonl.sh
```

DSpark 的 `--spec-tokens` 只限制实际提议数量，不改变已训练的 block 结构；指定值不能超过 checkpoint 支持的槽位数。三者对照时保持数据、temperature、seed、thinking、生成长度和两个 attention backend 相同。三者的 `CHECK_GREEDY=1` 均仅适用于 HF target、temperature=0。

从仓库根目录运行，`DATASETS_ROOT` 指向与 DSpark 评估相同的 prompt JSONL 文件或目录，不是训练用的预处理 Arrow 数据。支持 `messages`、`conversations`、`turns`、`prompt` 等已有 DSpark 格式，沿用其样本选择、seed 980406 和默认关闭 thinking 的设置。

EAGLE3 和 LEAGLE 脚本均默认 `TEMPERATURE=1`、`MAX_NEW_TOKENS=2048`、`ASCEND_DEVICES=0,1,2,3,4,5,6,7`，数据目录为 `../../DeepSpec/eval_datasets`，选用 `gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca,arena-hard-v2`。比较结果前应统一这些设置。默认评估使用全部八张卡，需在训练和目标服务退出后运行。

```bash
# 先抽少量样本审计：每条生成结果须与 target 单独贪心解码完全一致。
ASCEND_DEVICES=0 DATASETS_ROOT=/path/to/eval_jsonl \
MAX_SAMPLES=8 TEMPERATURE=0 CHECK_GREEDY=1 \
bash examples/evaluate/leagle_qwen3_4b_700k.sh

# 正式评估；默认温度 1、每轮 7 个候选 token、最多生成 2048 tokens。
ASCEND_DEVICES=0 DATASETS_ROOT=/path/to/eval_jsonl \
bash examples/evaluate/leagle_qwen3_4b_700k.sh

# temperature=1 的采样评估，复用原训练 checkpoint，无需重训。
TEMPERATURE=1 DATASETS_ROOT=/path/to/eval_jsonl \
bash examples/evaluate/leagle_qwen3_4b_700k.sh

TEMPERATURE=1 DATASETS_ROOT=/path/to/eval_jsonl \
bash examples/evaluate/eagle3_qwen3_4b_700k.sh

# 读取 2 层 / 3 loops 实验的 checkpoint_best。
DRAFT_MODEL=./output/leagle_qwen3_4b_700k_5ep_d2_r3_fullvocab_5hs/checkpoints/checkpoint_best \
DATASETS_ROOT=/path/to/eval_jsonl OUTPUT_DIR=eval_leagle_d2_r3_5hs_temp1 \
bash examples/evaluate/leagle_qwen3_4b_700k.sh

# 多 NPU 按样本分片，每张卡分别加载完整 target + draft。
ASCEND_DEVICES=0,1,2,3 DATASETS_ROOT=/path/to/eval_jsonl \
bash examples/evaluate/leagle_qwen3_4b_700k.sh
```

两份评估脚本的 `RUN_DIR` 均手动填写，修改 `NUM_LAYERS` / `NUM_LOOPS` 不会改变评估路径。指定任意 epoch 时使用 `DRAFT_MODEL=/path/to/checkpoints/4`。可设置 `MODEL`、`OUTPUT_DIR`、`DATASETS`、`MAX_SAMPLES`、`MAX_NEW_TOKENS`、`SPEC_TOKENS`、`TEMPERATURE`、`ENABLE_THINKING` 和 `RAW_PROMPT_MODE`。CPU 冒烟测试可设置 `DEVICE=cpu DTYPE=float32`，CUDA 可设置 `DEVICE=cuda:0`；两份脚本均需追加 `--ascend-devices ""`，覆盖默认八卡设置。多卡参数 `ASCEND_DEVICES` 要求 `DEVICE=npu:0`；它是样本并行，不是张量并行。

评估采用单链。`temperature=0` 时 draft/target 取 argmax，接受连续匹配的最长前缀。设置有限的 `temperature>0`（例如脚本默认的 1）时，两者使用同一温度下的 FP32 softmax 分布；默认完整输出头不额外启用 top-k/top-p 截断，可选候选头对 draft 支持集的限制见下节。draft 自回归采样最多 K 个候选；target 一次验证，按 `min(1, p(token)/q(token))` 接受连续前缀，首次拒绝后从归一化的 `max(p-q, 0)` 采样替代 token，全接受时从 target 分布采样 bonus token。缩小词表的 draft 概率映射回完整 target 词表，支持集之外的 q 为零，修正采样可选中这些 token。两种模式下 target 的 KV 和特征都只保留已验证部分；draft 跨轮只复用由真实 target 特征建立的前缀，不保留上一轮的预测状态缓存。当前不使用树搜索。

正温度不能与 `CHECK_GREEDY=1` 同用。采样评估验证目标分布，不要求与 target-only 解码逐 token 相同；相同 seed 下也不保证不同模型或不同分片方式得到相同文本，因为它们消耗随机数的顺序不同。接受 EOS 后不再抽取无用的后续 token。

两份脚本的结果目录均手动指定：EAGLE3 为 `eval_eagle_temp1/`，LEAGLE 为 `eval_leagle_5hs_temp1/`，可修改脚本或设置 `OUTPUT_DIR`；改变温度、层数、loop 次数或 checkpoint 不会自动更改该目录。两者输出文件相同：

- `summary.csv` / `summary.json`：逐数据集统计，字段与 DSpark 离线评估一致。
- `acceptance_length = 1 + accepted_draft_tokens / proposals`，K=7 时上限 8；`accepted_draft_length` 只计算草稿接受量。
- `position_accept_rates`：每个位置的连续前缀接受次数除以该位置实际被提议次数。另提供 `position_accept_rates_per_round`，以全部提议轮数为分母，与仓库 vLLM `acceptance_at_pos_i` 的分母一致；跨后端仍须核对尾部和 EOS 规则。
- `draft_length`：实际平均提议长度。接近生成预算时会缩短草稿，为 target token 留一位；最后的零草稿 target 轮次也计入 proposals，接受 EOS 时截断统计，与 DSpark 的离线尾部规则相同。
- `artifacts/*.jsonl`：prompt、完整输出 token IDs、各轮提议数和接受数；正温度还记录各位置接受概率与分布重叠。`--check-greedy` 若发现贪心生成不一致会立即报错。
- `eval_command.txt`、`speculators.patch`、`eval_config.json`：命令、版本、git 状态、运行配置及 checkpoint 架构；发布评估结果时与训练复现记录一并保留。
- `eval_checkpoints.json`、`checkpoint_sha256.txt`、`drafter_checkpoint_sha256.txt`：固定后的 target/draft 路径、epoch、训练 step、配置和权重哈希。开始时解析 `checkpoint_best` 并计算哈希，多进程共用同一份记录；解析后更新 best 指针不会切换正在评估的权重。加载或评估期间文件被删除或替换会报错。哈希计算会增加启动时的磁盘读取开销。

这版用于比较接受长度。日志中的吞吐包含参考实现开销，启用审计还包含 target 单独解码，多卡汇总时间还包含模型加载；这些数值不是 vLLM serving 加速比。贪心模式保持概率诊断字段为空并标记 `probability_diagnostics_available=false`；正温度模式输出接受概率与分布重叠诊断，并标记为 true。评估配置记录实际温度与验证模式。

### 可选低秩候选输出头（仅 LEAGLE）

训练默认 `CANDIDATE_HEAD=0`，使用完整输出头；设置为 1 时从头创建并训练两个低秩矩阵，直接保存在训练 checkpoint 中。评估默认 `CANDIDATE_HEAD=auto`，按 checkpoint 配置加载并启用，无需转换脚本或另做 SVD。开启候选推理后，每个草稿 token 执行一次候选筛选，物理层数和 loop 次数不改变输出头的调用次数：

| 步骤              | 矩阵与输出维度（单个 token）                           |
| ----------------- | ------------------------------------------------------ |
| 降维              | `h[1,H] × Bᵀ[H,R] → [1,R]`                             |
| 全 draft 词表粗分 | `[1,R] × Aᵀ[R,V] → [1,V]`                              |
| 取候选            | 粗分 top-M，得到候选 ID `[1,M]`                        |
| 精算              | 从原始 `W[V,H]` 取 `W_C[M,H]`，计算 `h × W_Cᵀ → [1,M]` |
| 提议              | 使用精算 logits；候选之外的 draft logits 为 `-inf`     |

原始 `lm_head.weight` 保留且冻结，新增 `candidate_head.down_proj.weight[R,H]` 和 `candidate_head.up_proj.weight[V,R]`。全词表 Qwen3-4B 的 `H=2560`、`V=151936`，默认 `R=256`、`M=128`；这两个值是实验起点。`M` 是每个 token 的候选词数量，与 `SPEC_TOKENS=7` 的草稿序列长度独立。缩小词表模型在自己的 draft 词表上筛选，再用原有 d2t 映射到 target ID。

候选头使用随机初始化，与主干同步训练。主干继续用原始完整输出头计算 CE/TV；候选头增加 `KL(softmax(W·stopgrad(h)) || softmax(A·B·stopgrad(h)))`，teacher 是同一个 drafter hidden state 的完整输出分布。辅助损失不向主干或 W 传梯度，top-k 只用于候选召回统计和推理，不直接反传。主干和候选头分别按 `MAX_GRAD_NORM` 裁剪梯度，避免辅助梯度改变主干的裁剪系数；两者沿用同一 LR 与 scheduler，当前脚本为 AdamW、LR=6e-4。使用 Muon 时，候选输出矩阵归入 AdamW 组。

| 环境变量                     | 默认值 | 训练参数                                   |
| ---------------------------- | ------ | ------------------------------------------ |
| `CANDIDATE_HEAD`             | `0`    | `--candidate-head` / `--no-candidate-head` |
| `CANDIDATE_HEAD_RANK`        | `256`  | `--candidate-head-rank`                    |
| `CANDIDATE_TOP_K`            | `128`  | `--candidate-head-top-k`                   |
| `CANDIDATE_HEAD_LOSS_WEIGHT` | `1.0`  | `--candidate-head-loss-weight`             |
| `CANDIDATE_HEAD_CHUNK_SIZE`  | `32`   | `--candidate-head-chunk-size`              |

YAML 中对应字段放在 `draft` 组。候选 KL 复用同一 TTT 步中主干已经算出的 dense logits，不再为辅助 teacher 重算完整输出投影。每个 token 分块只执行一次低秩前向，并根据 `q-p` 计算两个矩阵的一阶梯度；所有 TTT 步累加到同一组 FP32 缓冲，最终通过一个 autograd 节点传给参数，保留普通梯度同步、loss 权重和裁剪流程。反向不再重算候选头或完整 teacher，不额外保存全序列词表激活，也不在 forward 中调用 backward。该实现仅支持一阶训练梯度。

梯度缓冲大小为 `4 × R × (V+H)` 字节，默认 Qwen3-4B 配置约 151 MiB，不随 token 数或 TTT 步数增加；分块投影、概率和梯度矩阵乘法仍需要临时显存。`CANDIDATE_HEAD_CHUNK_SIZE` 默认保持 32；增大它会减少分块调用，但提高临时显存，实际速度取决于设备。与原实现使用同一全词表 KL、mask、逐步衰减、rank 和候选数；BF16 矩阵乘法形状、梯度累积精度与次序变化可能带来舍入差异。训练仍需原始完整输出头以及新增候选 KL 的计算，推理加速不意味着训练也会比关闭候选头更快。

继续使用原有服务和训练两份脚本。在两个终端中设置相同的 `OUTPUT_DIR`，保证隐藏状态文件路径一致：

```bash
OUTPUT_DIR=./output/leagle_candidate_r256 \
bash examples/train/leagle_qwen3_4b_700k/server.sh

OUTPUT_DIR=./output/leagle_candidate_r256 CANDIDATE_HEAD=1 \
CANDIDATE_HEAD_RANK=256 CANDIDATE_TOP_K=128 \
bash examples/train/leagle_qwen3_4b_700k/train.sh
```

不带候选头的对照实验设置 `CANDIDATE_HEAD=0`，使用另一组匹配的服务/训练输出目录。其余数据、seed、层数、loop、LR 和 epoch 保持一致。同 seed 下候选头初始化不会消耗主干初始化或训练随机数。`RESUME=1` 从原 `run.yaml` 恢复开关及参数；不能在同一次训练的中途增删候选头或改变 rank。新 checkpoint 的权重和配置可直接加载，不支持把已有 dense checkpoint 自动转成候选头模型。

增加三项聚合日志：`draft_loss` 为原主干 loss，`candidate_kl` 为有效 token 上的平均辅助 KL，`candidate_recall` 为候选集包含完整输出头 top-1 的比例。`loss` 是主干 loss 加上加权候选损失，候选损失沿用 TTT 的逐步衰减；普通验证同样报告这些指标。`checkpoint_best` 仍按验证 `loss` 选择，开启候选头时该值包含辅助项。原 `accept_len/eal` 指标仍来自 dense、teacher-forced 轨迹；真实候选接受长度需要离线评估。

同一个训练 checkpoint 可以切换 dense / candidate 推理，评估路径仍手动指定：

```bash
# Dense 对照：明确关闭训练好的候选头。
DRAFT_MODEL=/path/to/checkpoints/4 CANDIDATE_HEAD=0 \
OUTPUT_DIR=eval_leagle_dense bash examples/evaluate/leagle_qwen3_4b_700k.sh

# 默认自动读取 checkpoint 开关、rank 和候选数，温度默认 1。
DRAFT_MODEL=/path/to/checkpoints/4 \
OUTPUT_DIR=eval_leagle_candidate128 bash examples/evaluate/leagle_qwen3_4b_700k.sh

# 改候选数无需重训；改 rank 需要新建训练实验。
DRAFT_MODEL=/path/to/checkpoints/4 CANDIDATE_TOP_K=256 \
OUTPUT_DIR=eval_leagle_candidate256 bash examples/evaluate/leagle_qwen3_4b_700k.sh
```

直接调用评估入口时，默认按 checkpoint 决定是否开启；用 `--no-candidate-head` 强制关闭，`--candidate-head` 强制开启，`--candidate-top-k M` 覆盖候选数。参数传递到每个 NPU worker，`eval_config.json` 记录实际开关、rank、候选数和概率支持集。没有候选头的模型强制开启会报错；缺失、尺寸不匹配或非有限低秩权重也会被拒绝。EAGLE3 不支持此开关。

`temperature>0` 时，q 是候选内的精算 logits 按温度重新归一化后的分布；候选外 q=0，验证使用这一真实 q 做接受和残差修正，因此 target 仍可生成候选集合外的 token。`temperature=0` 时仍用 target 贪心验证纠正候选错误。候选遗漏可能降低接受长度；固定 seed 也不保证与 dense draft 生成相同的采样文本。评估开关只改变提议路径，不改变 checkpoint 内的训练配置。

**这是从头联合训练的实验设置，与 EAGLE4 先训练主干、再冻结主干专门训练候选头的两阶段流程不同。** 候选头跟随训练中不断变化的 hidden-state 分布学习，效果需通过候选召回和实际接受长度验证。本实现保留完整输出矩阵，候选模式额外持有两组低秩权重；采用 PyTorch 的投影、top-k、gather 和散射，仍保留全词表形状的 logits。尚未实测 NPU 训练显存或推理吞吐。

```bash
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_leagle.py -q
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/evaluate/test_leagle_offline_eval.py -q
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/evaluate/test_eagle_inference_cache.py -q
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_leagle_candidate_head.py tests/unit/evaluate/test_leagle_candidate_head.py -q
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_leagle_candidate_loss.py -q
PYTHONPATH=src:hs_connectors/src python -m pytest tests/unit/models/test_leagle_training_diagnostics.py -q
```

测试覆盖共享 loop 与展开独立层的前向 / 梯度等价性、缓存长度、packed document 与未来特征隔离、SDPA / eager、词表映射、真实自回归与并行训练轨迹的一致性，以及 activation checkpointing。评估测试额外覆盖全接受、首位 / 中途拒绝后的 KV 和特征回退、EOS、长度边界、不同深度 / loop / 词表的真实小型 Qwen3 生成一致性，以及保存 checkpoint 到 JSONL 结果的完整链路。
