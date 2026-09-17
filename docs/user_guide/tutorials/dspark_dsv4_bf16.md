# DSV4-Flash target / BF16 HS 接入（实验性）

这版将 DSV4 target 接到当前 dense DSpark drafter 的训练链路，**不切换到官方
MoE/mHC drafter，不修改 Qwen 的默认计算路径**。当前状态是可供 A3 实机联调的
训练/HS 接口和实验性离线接受长度后端，不是已验证的端到端支持。

## 已实现与边界

- 独立 vLLM architecture 插件；只有 `--dsv4` 启用，旧 `--dsv4-bf16` 为兼容别名。
- 分散取层、teacher HS 修正、冻结 target embedding / head / norm 的加载。
- checkpoint 结构与训练 IO 检查、HS 目录契约检查、单请求 HS 检查脚本。
- 训练 HS 服务可选单机 DP2，检查 TP×DP 设备数；提供多请求并发 HS 接线检查。
- vLLM 0.26.0 / Ascend 0.26.0rc1 的默认 V1 runner 缓存兼容：保留原生 C4/C128/SWA 分组与共享，
  将 HS 独立成缓存组和物理 tensor，并纳入统一 block 池的显存预算及容量检查。
  `--dsv4` 自动启用；只调整缓存规划及 HS tensor 绑定，不改模型结构、取层或训练计算。
  此修复未覆盖强制 `VLLM_USE_V2_MODEL_RUNNER=1` 的启动方式。
- 通用数据入口支持 DSV4 官方服务端编码、已有 token 数据直通及数据来源契约。
- teacher 概率对照工具；显式加载和自动续训均校验 checkpoint 的 target 身份。
- 独立输出目录中的短程训练、验证、保存与恢复验收入口；不替代 A3 实测。
- 保留 `corrGate=0`、Muon + linear、基础 LR `6e-5` 和用户原有特性开关。
- 允许量化 target backbone，由推理后端加载；导出的 HS 仍为 BF16。
- **未提供权重反量化/转换器，也未验证量化 target 的 A3 实机运行。**
- 已实现 opt-in 的 `--target-backend dsv4-vllm` 离线接受长度后端：
  通过 target 服务全前缀重算，不对 V4 压缩注意力状态使用普通 KV cache 回退。
  单机入口默认整块验证，保留逐位置 reference 作为显式对照。
  **尚未进行 A3 实机验证；不用于测量线上吞吐或加速比。**
- 新增单机单入口评估：自动启动本地 target、等待就绪、运行 eval 并回收自己启动的
  子进程；可以分卡，也可在显式授权及设置 target 内存预算后共卡。
- 尚未在真实 A3 上验证启动、HS 导出、teacher logits 或训练收敛。

更新此兼容补丁后，应在 **target 的 vLLM 环境**安装当前 checkout
（`pip install -e . --no-deps`）并重启服务；现有 TP/DP 和训练启动参数无需因此修改。
预算包含 HS 缓存数据；后端的 tensor 对齐开销仍沿用原生处理。
显式 `--num-gpu-blocks-override` 仍继承 vLLM 的强制容量语义，可能超过真实显存，
通常应省略此调试选项，交由显存 profiling 决定容量。

## 权重：`torch_dtype=bf16` 不等于全量 BF16

Preview 或 0731 必须使用各自匹配的 checkpoint、回答、token 数据和 HS，
不混用不同版本或量化方案的 HS，也不复用 Qwen 的缓存。

官方 Preview `deepseek-ai/DeepSeek-V4-Flash` 发布的是 FP4 + FP8 混合权重；
其配置同时包含 `torch_dtype=bfloat16`、`expert_dtype=fp4` 和 FP8 量化配置。
这里不再要求全量 checkpoint 转为 BF16：target 的量化计算由兼容的推理后端负责，
`--dtype bfloat16` 约束运行/HS 路径，不会把磁盘上的 FP4/FP8 权重自动变成全量 BF16。
[官方 Preview 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)

训练端只读取并冻结 target 的 embedding、LM head 和最终 norm；这三个模块必须是
可直接加载的未量化浮点权重。若它们自身被量化，仍会明确报错，需要单独处理这些
权重及其 scale；本补丁不包含局部转换器。不能仅删除量化元数据来通过检查。

vllm-ascend `0.26.0rc1` 的 A3 文档已有 Preview 衍生 W8A8 部署示例；这不等于
官方原始 FP4/FP8 在 A3 上可直接加载。后者仍需后端能力确认与实机验证。
不支持的量化格式将由后端拒绝，不会通过本补丁增加硬件算子支持。
[A3 部署教程](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/models/DeepSeek-V4-Flash.html)

```bash
python scripts/check_dsv4_checkpoint.py /shared/models/dsv4-flash-target
```

检查器只读取 safetensors 头，不加载整个模型，检查：

- Flash 几何参数：43 层、hidden 4096、4 路 HC、vocab 129280。
- shard 索引、张量头结构以及 target embedding/head/final norm 的精确键和尺寸。
- 训练所需的上述 IO 权重为未量化浮点；backbone 允许量化权重与 scale。

**通过检查不代表量化后端支持、scale 使用正确、整个模型权重完整或 A3 算子可用**；
还需完整加载和数值对照。需要主动审计全 BF16 checkpoint 时，仍可使用严格检查器
`python scripts/check_dsv4_bf16.py /path/to/bf16-checkpoint`，但这不是服务和训练的前提。
如另行转换权重，应保留原 checkpoint，在新目录输出并记录来源 revision 与工具版本；
反量化不能恢复量化前丢失的精度。

## HS 约定

训练参数 `--target-layer-ids 1 11 21 30 40` 使用 **HS slot 编号**，对应
0-based decoder block `[0, 10, 20, 29, 39]` 的输出，不是 decoder block 本身编号。
这是本次分散取层实验的选择，不是官方推荐的最优层组合。

每个中间层的四路 residual 在 HC 维上取均值，得到 `[tokens, 4096]`。
最后额外加入 slot 43，专门用于 teacher：

```text
中间层四路 residual -> mean(HC) -> 5 个 drafter 输入槽位
最终四路 residual   -> hc_head -> 第 6 个槽位（final norm 之前）
                                      |
                         trainer: frozen norm -> frozen lm_head
```

不能把最终四路均值当成 teacher；native Ascend 的 auxiliary 输出是均值，而 target
实际输出还经过 `hc_head`。插件保留原 target 的 normalized 输出，只替换导出列表的
最后一项，并在 norm 前复制，避免原地运算污染捕获值。
[对应 Ascend 实现](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc1/vllm_ascend/models/deepseek_v4.py)

传输形状是 `[seq_len, 6, 4096]`，存储 dtype 为 BF16；训练器将前五项展平为 context，第六项送入
已有的 TV loss / correction hidden supervision 路径。
BF16 HS 仍受 target 量化计算的数值影响，不代表与全 BF16 target 的 HS 一致。
训练数据、HS 生成和后续在线部署应保持相同的 target checkpoint 与量化方案。

HS 目录必须是新的空目录。启动器创建 `dspark_dsv4_hs.json`，训练器核对 target
路径、结构签名、HS 层号和格式。不允许将旧 HS 文件补一个新标签后继续使用。
签名基于 config / 量化元数据、张量头、文件大小和修改时间，**不是全量权重内容哈希**。
服务端还记录显式 `--quantization`（或未指定、由后端自动识别），重启时改变该选项
会拒绝复用原 HS 目录。旧版未记录此运行选项的目录也需更换；训练端无需重复传该选项。
两端使用相同的共享绝对路径，训练期间不得改动 checkpoint。
manifest 的存在仅表示启动约定已记录，不代表服务已经成功就绪。

## Drafter 配方

配置：`examples/train/dsv4_flash_dense_config.json`。

保留 5 层 dense Qwen3-style decoder、32 Q / 8 KV heads、head_dim 128、FFN 9728、
SWA 2048、RoPE theta 1e6，以及原有 correction、dynamic conv、selector 等选项。
因当前实现要求 draft/target IO hidden width 一致，hidden 从 Qwen3-4B 的 2560
改为 4096，vocab 改为 129280；**这不是参数量完全相同的模型，也不支持直接复用
Qwen draft 权重**。不把 V4 的 MLA / 压缩注意力几何参数复制给 dense draft。

训练脚本固定 `epochs=10`、`seq_len=3072`、`block_size=7`、`max_anchors=512`、
`correction_gate_bias=0`、`correction_markov_gate_bias=-2`。
优化器保持 Muon + linear，`--lr 6e-5` 对应 AdamW 部分基础 LR；未单独设置时
Muon 基础 LR 为 `6e-4`。这沿用当前实验，不冒充官方 DeepSpec 的 AdamW + cosine。

## A3 启动和实机检查

接口按 vLLM `0.26.0` / vllm-ascend `0.26.0rc1` 编写，并在初始化时校验版本。
镜像 tar 文件名不能证明其内部软件版本；先在容器内核对。

限制：eager、file HS backend、PP=1、PCP=DCP=1，关闭 prefix cache、
chunked prefill、FlashComm1 / SP、DSA-CP。训练 HS 服务支持单机 DP=1 或 2，
底层启动器默认仍为 DP1，当前 server 示例预置为 TP8×DP2；DP2 要求开启 EP、
两个 DP engine 全部位于本机、使用 mp 后端和
内部请求分发。多机 DP、DP>2 和外部负载均衡未开放。整块验证及自动离线评估入口
仍为 DP1。真实 A3 拓扑、HS 导出和数值一致性尚待验证。
不覆盖已有 vLLM/Ascend 安装，也不修改 native V4 / Qwen 注册。

在服务端和训练端的既有环境中安装本 checkout（不要顺带升级 torch/vLLM）：

```bash
pip install -e . --no-deps
pip install -e hs_connectors --no-deps
```

如设置了 `VLLM_PLUGINS` 白名单，需要加入 `speculators_dsv4`，并保留 Ascend
需要的其他插件。不设置白名单时由 vLLM 自动发现。

所有脚本都从仓库根目录运行。两个启动脚本已预置当前双机实验的 checkpoint、
数据/HS 路径、16 个设备及 target 地址 `80.48.17.187:8001`，环境变量可覆盖。
target 与 trainer 必须分别运行在两台机器上，不能直接把两个默认脚本放在同一台。
服务端路径与设备不同时，显式设置以下变量：
脚本与本文保留历史文件名中的 `bf16`，其含义是 BF16 HS 接口，不再要求全量 BF16 权重。

```bash
export MODEL=/shared/models/dsv4-flash-target
export HS_PATH=/shared/hs/dsv4-flash-target-spread-v1
# 按实际设备与内存填写，不在这里假定每台机器的卡数。
export VLLM_NPUS='<target device IDs>'
export TP_SIZE='<target tensor parallel size>'
export DP_SIZE=1  # 可选 2；VLLM_NPUS 的设备数必须等于 TP_SIZE * DP_SIZE。
export VLLM_HOST='<target host internal IP; use 127.0.0.1 for local-only testing>'
# 默认不设置，由后端识别 checkpoint；仅在匹配的 Ascend 量化格式要求时设置：
# export TARGET_QUANTIZATION=ascend
bash examples/train/dspark_dsv4_flash_bf16_server.sh
```

单台 A3 **确实暴露 16 个逻辑设备**时，可选择 `TP8 × DP2 / EP16`：

```bash
export VLLM_NPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TP_SIZE=8
export DP_SIZE=2
export HS_PATH=/shared/hs/dsv4-flash-target-spread-dp2-v1
bash examples/train/dspark_dsv4_flash_bf16_server.sh
```

这里 DP2 在 target **同一台机器内部**，不是两台机器各一个 DP；另一台机器仍
运行 trainer。启动器设置 `--data-parallel-size-local 2`，保留 EP，并在启动前检查
可见设备编号唯一、`设备数=TP×DP`。直接使用 `scripts/launch_vllm.py --dsv4` 时，
也会检查这些条件；DP2 必须显式设置 `ASCEND_RT_VISIBLE_DEVICES`，不能指定
`--headless`、外部 DP rank、Ray 或多机布局。插件在各 worker 中再次检查实际配置。
PP/CP 仍为 1，MoE 的 EP 组大小在此为 TP×DP；EP 不额外乘一遍设备数。
[vLLM 0.26 并行配置](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/parallel.py)

HS 继续使用上游 `ExampleHiddenStatesConnector`：各 DP engine 的 TP rank 0
写出自己调度的请求，通过响应返回文件路径。没有增加第二套 HS 文件格式，teacher
仍是 final norm 前，trainer 参数、drafter 架构和 loss 不变。训练端的多卡 DDP 与
server DP 独立，不需要把 `NUM_TRAIN_NPUS` 改成 2。

更换 TP/DP 布局可能改变浮点计算与显存预算，首次验证建议使用新的 HS 目录，
避免读到旧拓扑缓存。现有 manifest **不绑定 TP/DP 拓扑**；它通过不代表数值或
显存验收通过。上述 16 设备示例也不保证当前量化 checkpoint 一定装得下。

`TARGET_QUANTIZATION` 如非空会原样作为一个 `--quantization` 参数传给后端；
不设置时不添加该参数。`ascend` 不是将任意官方 FP4/FP8 文件转成 Ascend W8A8 的开关。

两台 Atlas A3 可考虑一台 target、一台 trainer，但先按实际量化格式确认单台是否装得下
target 权重与运行时开销；Flash 是 284B 总参数，不能用 13B 激活参数估算权重内存。
脚本没有假定跨机 TP 可用；跨机 TP/EP 需要另行核对通信环境。
[官方模型规模](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

两台机器必须共享 MODEL 和 HS_PATH 的相同绝对路径。将服务端口限制在可信内部网络，
不要直接向公网开放。训练端的 `VLLM_ENDPOINT` 改为 target 主机的内部地址。

服务就绪后，先跑小请求；下面 token IDs 仅为传输测试，不是评估 prompt：

```bash
python scripts/check_dsv4_hs.py \
  --model "$MODEL" --hidden-states-path "$HS_PATH" \
  --vllm-endpoint http://TARGET_INTERNAL_IP:8001/v1 \
  --input-ids 100 200 300 400
```

该脚本检查 token 对齐、`[4,6,4096]` 形状、BF16 和有限值，并保留 HS 文件。
它**不检查 teacher-logit 一致性**。正式训练前还应固定一批 prompt：

1. 对照匹配 checkpoint / 量化方案的可靠参考实现确认 target 输出；如做过转换，
   额外检查转换前后的数值差异。
2. 比较导出的 teacher 经 frozen norm/head 重建的 logits 与服务原始 logits，
   检查 token 对齐和数值差异，而不仅仅看 argmax。
3. 用少量 DSV4 数据跑训练/验证，确认有限 loss、有效梯度和恢复 checkpoint。
4. 用下方实验性离线后端验证接受长度和停止边界，再扩大到完整训练。

DP2 服务还应在单请求检查后运行不同长度的并发请求：

```bash
python scripts/check_dsv4_hs.py \
  --model "$MODEL" --hidden-states-path "$HS_PATH" \
  --vllm-endpoint http://TARGET_INTERNAL_IP:8001/v1 \
  --input-ids 100 200 300 400 --requests 8 --concurrency 2
```

检查器循环使用原输入及其较短前缀，逐请求检查 token、HS 形状、BF16/有限值和
输出文件唯一性，保留所有生成文件。**并发请求通过不证明两个 DP engine 都收到
请求**，还需看服务端各 engine 的请求指标/日志；初始化日志中的 DP rank 0/1
仅证明两个副本初始化，不证明它们都处理过请求。实机要覆盖只有一个副本有任务、
两副本输入长度不同及持续并发，检查 dummy forward、EP 通信和文件写入失败。
这些检查不替代 teacher 概率对照或 DP1/DP2 数值比较，也不是吞吐基准。

`DSV4_EVAL=1` 的普通 HS/reference 服务仍可开启 full-logprob 诊断；自动离线
launcher 和专用 `--dsv4-block-verify` 继续限定 DP1，不随 `DP_SIZE=2` 自动放开。

训练端：

```bash
export DATA_PATH=/shared/data/dsv4-flash-target-arrow
export OUTPUT_DIR=/shared/output/dsv4-flash-target-corrGate0
export TRAIN_NPUS='<training device IDs>'
export NUM_TRAIN_NPUS='<number of visible training devices>'
export VLLM_ENDPOINT=http://TARGET_INTERNAL_IP:8001/v1
bash examples/train/dspark_dsv4_flash_bf16_trainer.sh
```

普通训练通过 nohup 后台运行，输出日志和 PID 写到 `$OUTPUT_DIR/logs`；
TensorBoard 写到 `$OUTPUT_DIR/logs/tensorboard`，脚本打印对应查看/停止命令。
脚本返回只表示已提交后台进程，需要查看日志确认初始化成功；`TRAINING_SMOKE=1`
仍前台运行并传回退出码，以保证 fresh/resume 按顺序执行。
server 脚本前台等待就绪并保持运行，启动超时默认 1800 秒（`VLLM_STARTUP_TIMEOUT`
可覆盖），提前退出会报错；Ctrl+C/退出时仅清理自己创建的进程组。
server 需要 Linux `setsid` 和 `curl`。服务端口应限制在可信网络，保持共享路径一致。

不设置 `OUTPUT_DIR` 时默认使用 `./output/dspark_dsv4_flash_bestArch`；
切换 checkpoint / 量化方案时应使用独立输出目录，不直接续训另一版本的实验。

DATA_PATH 必须是 DSV4 自己的 token IDs/loss masks 和数据 manifest；不能复用 Qwen
Arrow、token_freq 或 vocab mappings。下面的数据入口负责编码已有回答，不生成或
重新采样回答。[官方编码说明](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md#chat-template)

恢复 DSV4 draft 时仍显式提供 `--target-hidden-state-format deepseek_v4_mean_hc_head`；
target checkpoint 和取层不能更换。Qwen 训练无需提供该参数，默认 `standard`。

## 训练数据预处理

仍使用 `scripts/prepare_data.py`，显式加 `--dsv4` 才进入新路径。原始数据可为
`messages` 或 ShareGPT `conversations` 格式；目前只支持文本及可选起始 system，
随后交替 user/assistant。工具调用、多模态和歧义字段会报错，不会猜测 mask。

原始消息通过已启动 target 的 `/tokenize` 使用 vLLM 0.26.0 官方 V4 renderer。
检查服务版本、model root、tokenizer class、HS/checkpoint 契约，并以本地
`tokenizer.json` 对照原始 token 探针；不执行 checkpoint Python 或套用 Qwen 模板。
训练服务脚本已打开只读 `--enable-tokenizer-info-endpoint`。服务仅对可信内网开放。

```bash
python scripts/prepare_data.py \
  --dsv4 --model "$MODEL" \
  --data /shared/data/answers.jsonl \
  --output /shared/data/dsv4-flash-target-arrow \
  --seq-length 3072 --num-preprocessing-workers 1 \
  --disable-thinking \
  --dsv4-tokenizer-endpoint http://TARGET_INTERNAL_IP:8001 \
  --dsv4-served-model-name "$MODEL" \
  --dsv4-hs-manifest "$HS_PATH"
```

tokenizer 地址是服务根 URL，**不带 `/v1`**。自定义了 served-model 别名时，填写
实际别名。若配置鉴权，预处理读取 `OPENAI_API_KEY` 或 `VLLM_API_KEY`。

多轮会按**每个 assistant 回答拆成一个样本**，历史仅作上下文，只监督当前回答的
续写部分（含完整回答的 EOS）。官方 encoder 会移除历史 reasoning；因此每个样本
都严格比较“生成前缀”与“带回答完整序列”的 token 前缀，拒绝不稳定边界，而不是
在整个多轮文本上用正则拼接 mask。超过 `--seq-length` 会截断，截断后没有监督 token
的样本会过滤，所以 `max-samples` 与原始对话数、输出样本数不一定相等。

思考数据使用 `--enable-thinking`，assistant 的思考内容放在独立的 `reasoning`
字段（也识别一致的 `reasoning_content` / `thinking`）。不接受 content 中混入
`<think>` 标签，需先整理到结构化字段；非思考模式遇到非空 reasoning 会报错，
避免静默丢失。没有显式开关时，新 DSV4 原始消息路径默认非思考模式。

输出为普通训练 Arrow：`input_ids`、`loss_mask`、`seq_len`，另外写入
`token_freq.pt` 与 `dspark_dsv4_data.json`。manifest 记录 checkpoint 身份、tokenizer
资产哈希、encoding、thinking、mask 策略及截断配置，训练前会核对。它是数据编码
来源记录，**不是证明回答由 target 生成的凭证，也不是每个 Arrow 内容的完整哈希**。

已由该流程生成的 token 数据可无服务端重打包，不加载普通 chat template：

```bash
python scripts/prepare_data.py --dsv4 --model "$MODEL" \
  --data /shared/data/dsv4-flash-target-arrow \
  --output /shared/data/dsv4-flash-target-arrow-subset \
  --seq-length 3072 --max-samples 64 --num-preprocessing-workers 1
```

默认从输入目录读来源 manifest；独立 token 文件可用 `--dsv4-source-manifest` 指定
它原有的来源契约。**不要给旧 Qwen/来源不明的 token 数据手工补当前 manifest**。
旧 DSV4 Arrow 若没有可审计的来源契约，应回到原始 messages 重编码；这不要求重生成
回答。通用非 DSV4 的预 token 化入口也已修复：已有 IDs/mask 时不再强制加载 processor
或检查 chat template；Qwen 原始消息处理路径保持不变。

## Teacher 概率数值对照

`check_dsv4_hs.py` 只验证 HS 形状/有限值；真正检查训练 teacher 的入口是
`scripts/check_dsv4_teacher.py`。先在启动训练服务时设置 `DSV4_EVAL=1`，允许返回
完整词表的原始 log-probability。只设置客户端参数不会改变已运行服务的配置。

```bash
python scripts/check_dsv4_teacher.py \
  --model "$MODEL" --hidden-states-path "$HS_PATH" \
  --vllm-endpoint http://TARGET_INTERNAL_IP:8001/v1 \
  --served-model-name "$MODEL" --verification-mode reference \
  --input-ids 100 200 300 400 --tail-positions 4 \
  --device npu:0 --norm-dtype float32 --head-dtype bfloat16 \
  --output-json ./output/dsv4-teacher-check.json
```

这里的 IDs 仅作接线测试；验收应换成真实编码 prompt，覆盖不同内容和长度。
校验器只加载已审计的 frozen norm/head，不加载完整 target 或 draft；逐位置比较
teacher HS 重建分布与 native target 分布，报告 TV、KL、log-probability 误差及
argmax 一致率。默认 FP32 norm 与 BF16 投影对齐当前 fresh/config-only 训练的
BF16 autocast 路径；若显式以 BF16 参数加载 draft，应对应调整 `--norm-dtype`。
CPU 默认设备仅用于诊断；验证 A3 数值时显式使用 trainer 的 NPU 和 dtype。

`--positions` 可选零基 HS 行号（行 p 预测 p+1），否则默认尾 4 行；每块默认最多
4 行全词表分布，不物化整个长上下文的 `L × V` 概率张量。`block` 模式可用于已
启动的专用整块评估服务；两种协议必须显式匹配，不会自动降级。

默认阈值 `max-tv=0.02`、`max-logprob-error=0.5`、`min-argmax-agreement=1.0`
只是初筛，不是所有 dtype/量化内核通用的误差保证。退出码 0 表示通过当前阈值，
1 表示数值未通过，2 表示配置/服务/协议错误。少量位置通过不等于完整模型或
量化后端已验收；也不要仅为“通过”而放宽阈值。

## 保存、恢复与 A3 短程验收

新 draft 配置保存 `target_training_contract`，绑定 target 路径/签名、HS 格式、
aux/teacher slots 和 target runtime quantization。显式 `--from-pretrained` 与
save-path 自动续训都在加载权重/优化器之前校验；分布式各 rank 汇总校验错误后
停止。切换 target、取层或量化方案必须开始独立实验。旧缺契约 DSV4 checkpoint
会被明确拒绝，不自动把当前身份补给旧权重；迁移需要另行审计原始来源。
DSV4 `--dry-run` 同样需要有效数据和 HS manifest。

本次还修复了通用恢复路径中的 LR 问题：scheduler 构造时覆盖了已恢复的 optimizer
LR；现在读取 scheduler 状态后同步还原其记录的 LR，再进行首个续训更新。
linear/cosine 公式不变，正常从零训练不受影响。

先启动**普通训练 HS 服务**，准备足够形成 train/val batch 的 DSV4 数据并设置上文
`MODEL`、`DATA_PATH`、`HS_PATH`、`TRAIN_NPUS`、`NUM_TRAIN_NPUS`、`VLLM_ENDPOINT`：

```bash
export SMOKE_ROOT=/shared/output/dsv4-training-smoke
bash examples/train/dspark_dsv4_training_smoke.sh
```

该入口复用现有 trainer 配方，自动运行两个独立进程阶段：默认训练 2 batch + 验证
1 batch、保存 epoch 0，然后自动恢复，再训练 2 batch + 验证 1 batch、保存 epoch 1。
`SMOKE_TRAIN_BATCHES` / `SMOKE_VAL_BATCHES` 可调；不足时受真实 loader 长度限制，
空 train/val 则报错。两阶段都检查有限 loss、非零且有限的梯度、冻结 IO、有效 LR，
并比较恢复前后的模型/optimizer 抽样值、scheduler 状态和 global step。

训练计算、基础 LR、`corrGate=0`、Muon + linear 和 loss 开关保持原配方；验收专用
覆盖为短程 epoch 数、两阶段共用的 scheduler 总步数、零 warmup、每阶段保存及
`num_workers=0`（避免为几步验收预取大量无用 HS 请求）。实际覆盖项写入报告，
不能把短程 loss 当作完整 10-epoch 实验的质量结论。

每次创建新的 `run.*` 目录，不使用/覆盖原 `OUTPUT_DIR` 实验；日志、两个 checkpoint
和 `reports/{fresh,resume}.rank-N.json` 全部保留。张量恢复检查为每个张量前 8 个值的
BF16 规范化抽样，不保证全张量逐位相同或跨进程 RNG 轨迹等同于连续训练。
入口要求 NPU，当前验收封装支持单卡/DDP，不覆盖 FSDP；普通训练的 FSDP 功能未改。
它不自动启动 target，不替代上面的 teacher 数值对照，也不做吞吐/收敛认证。
**本地 CPU 回归通过不代表这些 A3 阶段已经实际运行。**

## DSV4 离线接受长度验证（实验性）

此路径沿用当前 JSONL 离线 eval 的 draft proposal、拒绝采样及接受长度/位置统计口径，
但把 target 的执行替换为匹配 checkpoint 的 vLLM Ascend HS 服务。评估卡只加载
当前 dense drafter 和它需要的 target IO 权重，不再加载整个 284B target。
Qwen 默认仍使用 `--target-backend hf`，原有本地 target 和缓存路径保持不变。

两种模式都采用完整前缀重算，不对 V4 的压缩注意力状态执行 `DynamicCache.crop`：

- `block`：含 `k` 个 draft token 的 proposal 只发送一次 target 请求、做一次完整
  前缀前向。专用 connector 从 target 的真实 LM head 取得尾段 `k+1` 行全词表
  FP32 原始 log-probability，并将所需 BF16 HS 后缀写入同一个 safetensors 文件。
  概率不通过 HTTP 全词表 JSON 传输，也不计算/导出整个前缀的全词表概率。
  第一版默认每轮最多导出 128 行 target 分布（含 bonus 行），超出明确报错，
  防止误配置导出整个长前缀的巨大词表张量。
- `reference`：保留原来的 `k+1` 次完整前缀请求；每次取最后位置的全词表原始
  log-probability，最后一次读取辅助 HS。用于整块路径的正确性对照。

两种模式都使用实际运行的量化 target 概率，不用评估端的 frozen norm/head 重建
概率验收；draft proposal、拒绝采样和位置统计规则不变。整块模式不是增量 KV
缓存方案，仍每轮重新 prefill，不是线上性能实现。改变前向形状可能改变浮点/
量化数值，需对照概率、HS 和采样边界，不能宣称与 reference 或线上逐 token
结果必然逐位相同。模式或服务协议不匹配会报错，不会静默降级为另一种模式。
服务内部仍使用 `extract_hidden_states` 缓存型 drafter；本次优化的是前向次数与
尾段文件导出，不宣称消除了全部 HS 缓存开销。
整块 HS 目录必须支持**同目录 hard-link 原子发布**，推荐本机 POSIX 文件系统。
服务先写临时文件，再以不覆盖已有文件的 hard link 发布完整数据；目录/共享盘不
支持该操作时会明确失败，不降级为非原子写入。只由 TP rank 0 发布，写入失败会
同步通知其他 TP rank，客户端不会在 HTTP 返回前读取半成品。

### 单机单入口（推荐先跑少量样本）

不需要先手动启动服务，也不需要设置 `VLLM_ENDPOINT`、SSH 或第二台机器。
入口是 `scripts/evaluate/run_dsv4_offline_eval.py`，下面的 shell 脚本只是把环境变量
转成参数。它在**同一台主机**启动 target 和 eval 两个子进程，内部通过
`127.0.0.1` HTTP 通信；不是把 284B target 和 drafter 合并到一个进程。
仍需事先准备兼容的 target 权重、已训练的 DSV4 draft、插件及两端依赖。

先用不重叠的卡运行；这里不假定每台 A3 的卡数或 target 所需卡数：

```bash
export VERIFIER_MODEL=/shared/models/dsv4-flash-target
export DRAFT_MODEL=/shared/output/dsv4-flash-target-corrGate0/checkpoint-path
export DATASETS_ROOT=/shared/data/eval-jsonl
export HS_PATH=/shared/hs/dsv4-eval-runs
export OUTPUT_DIR=/shared/eval/dsv4-single-runs
export VLLM_NPUS='<comma-separated target physical device IDs>'
export EVAL_NPU='<one physical device ID outside VLLM_NPUS>'
# TP_SIZE 默认等于 VLLM_NPUS 中的设备数；显式设置也必须与设备数一致。
# export TP_SIZE='<target tensor-parallel size>'
# 非空时原样传给 target；它不负责把权重转换为另一种量化格式。
# export TARGET_QUANTIZATION=ascend
export MAX_SAMPLES=4
export MAX_NEW_TOKENS=64
# 默认 block；设为 reference 会同时切换 target connector 和 eval 客户端。
export VERIFICATION_MODE=block
bash examples/evaluate/dspark_dsv4_single_eval.sh
```

target 默认内存利用率是 `0.9`，只适用于默认分卡配置的启动预算，并不保证模型
一定装得下。`VLLM_NPUS` 与 `EVAL_NPU` 均填写同一主机上的物理设备编号；
eval 子进程会将自己的唯一可见设备作为 `npu:0`。不会把 target TP 组当作 draft
数据并行组；DP 固定为 1，两端都清理继承的分布式 rank 环境变量。

若确实需要 target 和 drafter 共用一张卡，须同时明确开启共卡和提供内存预算：

```bash
export EVAL_NPU='<one physical device ID included in VLLM_NPUS>'
export ALLOW_SHARED_DEVICE=1
export TARGET_MEMORY_UTILIZATION='<explicit target memory fraction between 0 and 1>'
bash examples/evaluate/dspark_dsv4_single_eval.sh
```

只设置 `ALLOW_SHARED_DEVICE=1` 不够，重叠设备而不同时提供内存比例会拒绝启动。
**这是共卡预算开关，不是实机内存安全保证**：target 权重、运行时工作区、KV/压缩
状态，加上 draft、target IO 权重和 PyTorch/NPU 开销都占内存。必须按实际硬件与
量化格式估算并从小请求验证；不能把 `1 - TARGET_MEMORY_UTILIZATION` 直接视为
eval 一定可用的显存，仍可能 OOM 或产生性能干扰。

target 和 eval 可以使用同一主机上不同的 Python 环境：

```bash
export TARGET_PYTHON=/path/to/vllm-ascend-env/bin/python
export EVAL_PYTHON=/path/to/training-env/bin/python
bash examples/evaluate/dspark_dsv4_single_eval.sh
```

控制入口本身只依赖 Python 标准库；两个环境都需按上文安装本 checkout 和相应
依赖、访问相同的模型和 HS 绝对路径。脚本**不会自动进入容器、安装软件或跨机
启动进程**；若使用容器，应从已准备好的同一容器/可见文件系统中运行。

每次运行在 `OUTPUT_DIR` 下创建独立 `run-*` 子目录，在 `HS_PATH` 下创建独立
`hs-*` 子目录。HS 父目录无需为空，但新建的本次子目录必须隔离；不会复用其他
训练或评估的 HS。保留运行日志和 manifest，结果路径以入口打印的位置为准。
target 使用本次独有的 served-model 别名、临时鉴权 token，仅绑定本机回环地址。

入口会等待 target 就绪后才开始评估；target 启动失败、eval 失败、正常结束或
Ctrl+C 时，会回收**本次创建的进程组**，不会停止已有 target 或其他任务。
`SIGKILL`、主机故障等无法执行清理的情况不在此保证范围内，可能留下进程或本次
HS 文件。主动脱离本次进程组的后代也不在进程组清理范围内；A3 实机进程树仍需
验证。先确认任务已停止再检查遗留数据，不要清空 HS 父目录。

常用可选变量：`VLLM_PORT=0` 自动选择本地空闲端口，`STARTUP_TIMEOUT=1800`、
`SHUTDOWN_TIMEOUT=30`、`TARGET_REQUEST_TIMEOUT=120` 均以秒计；
`DSV4_MAX_MODEL_LEN=4096`，`TEMPERATURE=0.0`、`SEED=980406`、
`ENABLE_THINKING=false`、`RAW_PROMPT_MODE=auto`、`VERIFICATION_MODE=block`。
`DATASETS` 可选择子数据集，`KEEP_TARGET_HS=1` 保留本次请求的 HS，
`SKIP_ARTIFACTS=1` 跳过逐样本 artifacts。`DRY_RUN=1` 仅查看计划，不启动 target
和 eval。直接使用 Python 入口时，分别对应 `--keep-target-hs`、`--skip-artifacts`
和 `--dry-run` 等参数。

单入口默认让 target 启用 `--dsv4-block-verify`，客户端启用
`--dsv4-verification-mode block`；`--max-logprobs 0` 关闭不需要的 HTTP 全词表
返回额度，文件内原始概率不受影响。服务强制 `max_num_seqs=1`，只接收携带
整块验证协议且 `max_tokens=1` 的请求，不可混用于训练 HS 采集或普通生成。
入口参数 `--verification-mode reference`（或 shell 的 `VERIFICATION_MODE=reference`）
会同时启动原来的 HS connector 和逐位置客户端，并恢复全词表 HTTP 返回额度。

两种模式都**不改变接受长度统计口径，也不是线上性能评测**。此路径尚未在真实
A3 完成运行验证，首次运行仍应按本文末尾的检查项核对。建议相同 checkpoint、
量化方式、数据与 seed 分别跑 `block` / `reference`，检查概率与 HS 对齐后再扩大样本。

### 手动连接已有服务（保留的高级用法）

已维护独立 HS 服务或需要两台机器时，可以继续使用下面的原有入口；它不会管理
服务生命周期，不需要与单机自动入口同时运行。**手动入口默认 `reference`**，
保持原 HS 服务兼容性；与单机入口默认值不同。

先按上文准备模型、共享目录和插件，使用评估选项启动或重启 target 服务：

```bash
export DSV4_EVAL=1
bash examples/train/dspark_dsv4_flash_bf16_server.sh
```

`DSV4_EVAL=1` 额外设置 `--max-logprobs 129280 --logprobs-mode raw_logprobs
--generation-config vllm`；不设置该变量时，训练 HS 服务的启动参数不变。
Completion 请求实际使用 `logprobs=129280`，而不是 `logprobs=-1`，并要求以 token ID
返回键，避免解码后的字符串重复导致概率丢失。仍然必须启动本仓库的 `--dsv4`
HS bridge，普通 V4 serving 实例不能替代它。
[vLLM Completion 协议](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/completion/protocol.py)

在装有训练依赖的评估机器上，先运行少量样本：

```bash
export VERIFIER_MODEL="$MODEL"
export DRAFT_MODEL=/shared/output/dsv4-flash-target-corrGate0/checkpoint-path
export DATASETS_ROOT=/shared/data/eval-jsonl
export VLLM_ENDPOINT=http://TARGET_INTERNAL_IP:8001/v1
export EVAL_NPU='<one evaluation-only device ID>'
export OUTPUT_DIR=/shared/eval/dsv4-flash-target-reference
export MAX_SAMPLES=4
export MAX_NEW_TOKENS=64
export VERIFICATION_MODE=reference
bash examples/evaluate/dspark_dsv4_offline_eval.sh
```

若手动运行整块模式，必须另外启动专用服务，在 `--` 前同时提供 `--dsv4` 和
`--dsv4-block-verify`，不能把已有训练/reference 服务直接当成 block 服务。例如：

```bash
env -u LOCAL_RANK -u RANK -u WORLD_SIZE \
  ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" \
  python scripts/launch_vllm.py "$MODEL" --dsv4 --dsv4-block-verify \
  --hidden-states-path "$HS_PATH" --target-layer-ids 1 11 21 30 40 -- \
  --tensor-parallel-size "$TP_SIZE" --data-parallel-size 1 \
  --pipeline-parallel-size 1 --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 --max-model-len 4096 \
  --max-num-batched-tokens 4096 --max-num-seqs 1 --block-size 128 \
  --host "$VLLM_HOST" --port 8001 --max-logprobs 0 --generation-config vllm \
  --additional-config '{"enable_flashcomm1": false, "enable_dsa_cp": false}'
```

按实际后端补充 `--quantization`，保持与检查和训练时的量化方案一致，并使用新的
专用 HS 目录。然后设 `VERIFICATION_MODE=block` 再运行手动评估脚本。该服务不
返回全词表 HTTP logprobs，不能供 reference 客户端使用；普通训练 HS 服务也不会
生成整块概率文件，不能供 block 客户端使用。两端必须显式匹配，不自动回退。

示例默认 greedy（`TEMPERATURE=0.0`），可显式设置 `TEMPERATURE=1.0` 测随机采样。
`DSV4_MAX_MODEL_LEN=4096` 必须与服务端实际 `--max-model-len` 对应；评估输入及生成
预算还需为验证候选和 HS 导出的 1 个输出 token 留出空间，不能依靠服务端截断。
`TARGET_REQUEST_TIMEOUT` 默认 120 秒；`SERVED_MODEL_NAME` 只在服务使用自定义别名时设置。
此示例使用一张独立评估卡，不把 target 服务的 TP 卡当作 draft 的数据并行卡。

消息类型 JSONL 由 target 服务的 `/tokenize` renderer 完成聊天编码，使用当前模型的
官方编码方式；无需给 Preview 补造 Jinja 模板。默认 `RAW_PROMPT_MODE=auto`；仅当
文本已经按对应版本的官方协议完整编码时，才使用 `RAW_PROMPT_MODE=raw`。Preview
和 0731 的模板、tokenizer 和 checkpoint 不混用。

结果除既有 `summary.json` / `summary.csv` / artifacts 外，还记录 `eval_backend.json`
以标明 target 后端及运行约定。**本路径禁止 `--measure-base-speedup`**：耗时包含
完整前缀 prefill、HTTP 和共享文件 IO（reference 还有 HTTP 全词表传输），不代表线上 speculative decoding 的
tokens/s 或加速比；不同后端的速度列不能直接比较。

默认只清理本次带独立请求 UUID 的临时 HS，先等异步写入完成，再删除自己的文件；
不会扫描清空整个共享目录。`KEEP_TARGET_HS=1` 对应 `--keep-target-hs`，便于对照
检查，但会迅速占用磁盘。超时或中断可能留下本次孤儿文件，应在确认对应请求已结束
后单独检查；不要删除正在训练或被其他评估使用的 HS。

该后端的本地控制逻辑测试不能替代 A3 验证。首次运行仍需核对完整词表概率、输入
token 对齐、HS 槽位及 dtype，并检查 EOS/生成上限处的输出；现阶段不能声称 DSV4
离线测评已在真实 A3 上跑通。

## 本地测试

不依赖 torch 的控制逻辑测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests/standalone -v
```

包含 checkpoint/manifest/HS 槽位检查、模拟 native runtime 的 hook 和保护条件、
DSV4 block/reference 启动接线与非法组合拒绝、单机进程管理及 Qwen 启动参数回归。
模拟后端不验证真实算子或张量计算。

有完整训练依赖的环境还应执行：

```bash
pytest tests/unit/evaluate/test_dspark_offline_eval.py \
  tests/unit/evaluate/test_dsv4_offline_target.py \
  tests/unit/evaluate/test_dsv4_block_connector.py \
  tests/unit/evaluate/test_dsv4_teacher_parity.py

pytest tests/unit/models/test_dflash_optional_features.py \
  tests/unit/models/test_dspark_core.py \
  tests/unit/train/test_trainer_scheduler.py

pytest tests/unit/train/test_prepare_data.py \
  tests/unit/train/test_dsv4_preprocessing.py \
  tests/unit/train/test_dsv4_training_identity.py \
  tests/unit/train/test_scheduler_resume_lr.py \
  tests/unit/train/test_dsv4_training_smoke.py \
  tests/unit/train/test_dsv4_training_smoke_integration.py
```

离线后端测试使用真实 CPU 张量和模拟服务，覆盖前缀/HS 对齐、拒绝回退、
greedy/T=1 接受与残差采样、多卡参数传递及请求文件清理；不会启动真实 target。

新增 PyTorch 回归覆盖 HS 格式字段不改变 dense backbone 的初始化、context fusion
输出和输入梯度。本地缺少 PyTorch 时不能将其标为已通过。

预处理回归实际写读 Arrow/token_freq，但服务 renderer 用模拟客户端，不启动 V4。
短程集成回归使用 CPU 小模型跑真实 Trainer、Muon/AdamW、linear 和 checkpoint
读写；只适配设备、tiny 模型注册和 Windows 展示 symlink，不是完整 DSpark/NPU/DDP
运行验证。Windows 的独立测试进程需处理库导入中的 POSIX `fcntl` 依赖，不应修改
生产文件锁逻辑或将其解释为平台支持。
