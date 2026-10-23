# LEAGLE

LEAGLE 是用于深度 / loop 对照实验的自回归 drafter：保留 DFlash / DSpark 的目标特征融合与每层 KV 注入，使用 EAGLE 式的 token + hidden-state 递推。它是独立算法 `leagle`，原来的 `eagle3`、`dflash`、`dspark` 仍可作为实验基线。

## 目录

- `qwen3_4b_700k/server.sh`：目标模型 vLLM 隐藏状态服务。
- `qwen3_4b_700k/train.sh`：70 万数据、5 epochs 的训练脚本。
- `qwen3_4b_700k/run_eagle3.sh`：独立训练普通 EAGLE-3 单层基线，自动管理目标服务。
- `qwen3_4b_700k/run_leagle.sh`：独立训练 LEAGLE 单层、单次 loop 基线，自动管理目标服务。
- `tests/`：CPU 数值与训练接口测试。
- `runs/`：本地运行产物，已加入忽略规则。
- 模型代码按仓库约定位于 `../src/speculators/models/leagle/`。

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
| `--num-loops` | 每个 token 对同一组层的执行次数 | 2 |
| `--ttt-steps` | 训练时向前展开的 token 步数 | 7 |

有效 decoder 深度 = `num_layers × num_loops`，不包含融合层和输出头。增加 loop 不增加 decoder 参数量，但增加计算量与缓存需求。

循环内位置编号不增加，只有进入下一个 token 时才增加。每个 `(loop, layer)` 使用独立的草稿 KV 历史；目标上下文不会重复追加进草稿缓存。反向传播覆盖所有 loop 和 token steps，没有 detach 或截断。

首批对照可使用 `(层数, loop)` = `(1,1)`、`(2,1)`、`(4,1)`、`(1,2)`、`(1,4)`、`(2,2)`。后四种中的 `(4,1)`、`(1,4)`、`(2,2)` 具有相同有效深度，能比较独立层和共享层。DFlash 的五层对照还需单独训练 `(5,1)`；有效深度相同不等于训练 FLOPs 或实测延迟完全相同。

## 训练

从仓库根目录运行。沿用已经准备好的 Qwen3-4B / 70 万数据、5 epochs、seed 42、3072 packing 长度、0.9 训练比例、32K draft vocabulary、三个目标特征层 `[2,18,33]` 和 TTT=7。默认目标服务使用 NPU 0–1，训练使用 NPU 2–7，端口 8003。脚本中的模型与数据路径需要与训练机器一致。

Qwen3-4B 预设的所有 draft 层使用 full causal attention。脚本根据 `NUM_LAYERS` 自动生成 `--full-attention-indices 0 ... D-1`，每次 loop 复用相同设置。注意力覆盖完整的可见前缀；未来目标特征、其他文档和其他训练轨迹仍然不可见。

```bash
# 终端 1：目标特征服务；launcher 自动附加最后一层 teacher slot 36。
NUM_LAYERS=1 NUM_LOOPS=2 bash leagle/qwen3_4b_700k/server.sh

# 终端 2：训练；两份脚本的深度、loop 和输出目录必须一致。
NUM_LAYERS=1 NUM_LOOPS=2 bash leagle/qwen3_4b_700k/train.sh
```

示例输出目录为 `leagle/runs/leagle_qwen3_4b_700k_5ep_d1_r2/`。服务命令带 `--provenance-dir`；训练沿用仓库的 `train_command.txt` 和 `speculators.patch` 记录。首次正式运行前应提交新增模块，使这些文件对应的 Git SHA 包含完整代码。当前没有启动训练任务。

也可通过统一入口使用 `--speculator-type leagle --num-layers D --num-loops R --ttt-steps 7`，或把对应字段放进训练 YAML 的 `draft` 和 `loss` 组。保存的 checkpoint 包含 `num_loops`。继续训练已有 checkpoint 时不能静默改变这一值。

### 普通 EAGLE-3 与 LEAGLE 的首轮对照

`run_eagle3.sh` 与 `run_leagle.sh` 是两份独立脚本，每份只运行自己的实验。共同配方取自 `examples/train/eagle3_qwen3_4b_700k/train.sh`：KL、Muon + AdamW（两组 LR 均为 `1e-4`，weight decay 分别为 `0.1` / `0.01`）、linear scheduler、1% warmup，以及上述 70 万数据 / 5 epochs 设置。这是仓库示例配方，不是原论文超参数复现。

两组固定为一个物理 decoder 层，LEAGLE 固定 `num_loops=1`；两组均显式使用 full attention。普通 EAGLE-3 保留示例的 Llama decoder 和输入归一化设置；LEAGLE 保留 Qwen3 decoder、输入融合和每层 KV 注入。该实验比较整体结构，不单独归因于某一种输入改动。

```bash
# 先预览；不会创建目录、启动服务或训练，也不要求本机有模型和数据。
bash leagle/qwen3_4b_700k/run_eagle3.sh --dry-run
bash leagle/qwen3_4b_700k/run_leagle.sh --dry-run

# 实验 1：普通 EAGLE-3；自行启动服务，训练退出后释放服务。
bash leagle/qwen3_4b_700k/run_eagle3.sh

# 实验 2：LEAGLE d1/r1；在需要时独立启动。
bash leagle/qwen3_4b_700k/run_leagle.sh

# 重复某个实验时，为该脚本指定新输出目录。
OUTPUT_DIR=./leagle/runs/leagle_repeat bash leagle/qwen3_4b_700k/run_leagle.sh
```

默认输出分别为 `leagle/runs/eagle3_qwen3_4b_700k_5ep_d1_full/` 和 `leagle/runs/leagle_qwen3_4b_700k_5ep_d1_r1_full/`。每份脚本独立保存日志、隐藏状态目录、checkpoint、启动命令和脚本快照；vLLM 复现记录写入对应的 `checkpoints/`。脚本在前台等待训练完成，Ctrl+C 会停止它启动的训练和服务，不会自动覆盖或恢复已存在的运行目录。

两份脚本默认使用相同的 NPU 分配和端口，应分别启动、错开运行；如需同时运行，分别配置互不重叠的设备和不同端口。不需要额外启动 `server.sh`。

模型、数据、输出路径和 NPU 分配可在各自脚本配置区或通过同名环境变量独立修改。相对路径统一从仓库根目录解析；训练继续复用预处理数据中的 token IDs 和 loss masks，不重新生成回答。两份脚本只负责训练，不包含真实接受长度评估；评估限制见下节。

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

原有 vLLM EAGLE / DSpark speculator 不能直接执行这种模型。这里的 vLLM 脚本仅用于训练数据的目标特征提取；LEAGLE 的高性能 serving adapter 与端到端接受长度 benchmark 仍需接入。普通 validation loss / conditional accuracy 沿用 teacher-forced TTT，不能直接当作真实接受长度。

```bash
PYTHONPATH=src:hs_connectors/src python -m pytest leagle/tests -q
```

测试覆盖共享 loop 与展开独立层的前向 / 梯度等价性、缓存长度、packed document 与未来特征隔离、SDPA / eager、词表映射、真实自回归与并行训练轨迹的一致性，以及 activation checkpointing。
