# DSV4-Flash Target / BF16 Hidden States Integration (Experimental)

This integration connects a DSV4 target to the current dense DSpark drafter's
training pipeline. **It does not switch to the official MoE/mHC drafter or change
Qwen's default computation path.** It provides training/hidden-state (HS)
interfaces and an experimental offline acceptance-length backend for A3 hardware
testing, not validated end-to-end support.

## Implemented Features and Limitations

- A separate vLLM architecture plugin, enabled only by `--dsv4`;
  `--dsv4-bf16` remains a compatibility alias.
- Spaced layer selection, corrected teacher HS, and loading of frozen
  target embeddings, LM head, and final norm.
- Checkpoint structure and training IO checks, HS directory contract validation,
  and a single-request HS check script.
- Optional single-host DP2 for the training HS service, TP x DP device-count
  validation, and concurrent multi-request HS integration checks.
- Cache compatibility for the default V1 runner in vLLM 0.26.0 / Ascend 0.26.0rc1:
  native C4/C128/SWA grouping and sharing are preserved. HS gets its own cache
  group and physical tensor, included in the shared block pool's memory budget
  and capacity checks. `--dsv4` enables this automatically. Only cache planning
  and HS tensor binding change, not model structure, layer selection, or training
  computation. This fix does not cover forced `VLLM_USE_V2_MODEL_RUNNER=1` runs.
- A shared data entry point supporting DSV4's official server-side encoding,
  pre-tokenized inputs, and data provenance contracts.
- Teacher probability comparison tools; explicit checkpoint loading and automatic
  resume both validate the checkpoint's target identity.
- Short training, validation, save, and resume checks in isolated output
  directories. These do not replace A3 hardware testing.
- The existing `corrGate=0`, Muon + linear schedule, base LR `6e-5`, and feature
  flags are preserved.
- Quantized target backbones are allowed and loaded by the inference backend;
  exported HS remain BF16.
- **No weight dequantizer/converter is provided, and quantized-target execution
  on A3 hardware has not been validated.**
- An opt-in `--target-backend dsv4-vllm` offline acceptance-length backend that
  recomputes the full prefix through the target service instead of using ordinary
  KV cache rollback on V4's compressed attention state. The single-host launcher
  defaults to block verification, with per-position reference verification
  available explicitly. **This has not been validated on A3 hardware and is not
  intended to measure online throughput or speedup.**
- A single-command, single-host evaluation launcher that starts the local target,
  waits for readiness, runs evaluation, and cleans up its own child processes.
  It supports separate devices, or explicitly authorized device sharing with a
  target memory budget.
- Startup, HS export, teacher logits, and training convergence have not yet been
  validated on real A3 hardware.

After updating this compatibility patch, install the current checkout in the
**target's vLLM environment** (`pip install -e . --no-deps`) and restart the
service. Existing TP/DP and training launch arguments do not need to change for
this fix. The budget includes HS cache data; tensor alignment overhead retains
the backend's native handling. Explicit `--num-gpu-blocks-override` still uses
vLLM's forced-capacity semantics and may exceed physical device memory. Normally,
omit this debugging option and let memory profiling determine cache capacity.

## Weights: `torch_dtype=bf16` Does Not Mean All Weights Are BF16

Use matching checkpoints, responses, token data, and HS for Preview or 0731.
Do not mix HS across versions or quantization schemes, or reuse Qwen caches.

The official Preview `deepseek-ai/DeepSeek-V4-Flash` release uses mixed FP4 + FP8
weights. Its configuration includes `torch_dtype=bfloat16`, `expert_dtype=fp4`,
and FP8 quantization settings. Converting the entire checkpoint to BF16 is no
longer required here: a compatible inference backend handles quantized target
computation. `--dtype bfloat16` controls the runtime/HS path; it does not
automatically convert FP4/FP8 weights on disk into an all-BF16 checkpoint.
[Official Preview configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)

The trainer reads and freezes only the target's embeddings, LM head, and final
norm. These three modules must contain directly loadable, unquantized
floating-point weights. If they are quantized, loading fails explicitly and
their weights and scales must be handled separately. This patch does not include
a converter for those modules. Removing quantization metadata is not a valid
way to pass the checks.

The vllm-ascend `0.26.0rc1` A3 documentation includes a deployment example for a
Preview-derived W8A8 checkpoint. This does not establish that the original
official FP4/FP8 weights load directly on A3; backend support and hardware
validation are still required. The backend rejects unsupported quantization
formats. This patch does not add hardware operator support.
[A3 deployment tutorial](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/models/DeepSeek-V4-Flash.html)

```bash
python scripts/check_dsv4_checkpoint.py /shared/models/dsv4-flash-target
```

The checker reads safetensors headers without loading the whole model. It checks:

- Flash geometry: 43 layers, hidden size 4096, four HC streams, vocabulary 129280.
- Shard indices, tensor header structure, and the exact keys and shapes of the
  target embeddings, LM head, and final norm.
- Unquantized floating-point IO weights for training; quantized backbone weights
  and scales are allowed.

**Passing does not establish quantization-backend support, correct scale usage,
complete model weights, or working A3 operators.** Full loading and numerical
comparison are still required. To audit an all-BF16 checkpoint explicitly, use
`python scripts/check_dsv4_bf16.py /path/to/bf16-checkpoint`; that stricter check
is not a prerequisite for serving or training. If converting weights separately,
keep the original checkpoint, write to a new directory, and record the source
revision and tool versions. Dequantization cannot recover precision already lost
during quantization.

## Hidden-State Contract

The training option `--target-layer-ids 1 11 21 30 40` uses **HS slot indices**,
corresponding to outputs of zero-based decoder blocks `[0, 10, 20, 29, 39]`, not
the decoder block indices themselves. This is the selection used for the current
spread-layer experiment, not an officially recommended optimal combination.

At each selected intermediate layer, the four residual streams are averaged
over the HC dimension, producing `[tokens, 4096]`. Slot 43 is appended separately
for the teacher:

```text
Intermediate four-stream residuals -> mean(HC) -> 5 drafter input slots
Final four-stream residuals        -> hc_head -> slot 6 (before final norm)
                                                    |
                                      trainer: frozen norm -> frozen lm_head
```

Do not use the final four-stream mean as the teacher. Native Ascend auxiliary
outputs are means, while the actual target output also passes through `hc_head`.
The plugin preserves the target's normalized output and replaces only the last
exported entry. It clones that entry before norm to prevent in-place operations
from corrupting the captured value.
[Corresponding Ascend implementation](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc1/vllm_ascend/models/deepseek_v4.py)

The transfer shape is `[seq_len, 6, 4096]`, stored as BF16. The trainer flattens
the first five entries into context and sends the sixth through the existing TV
loss / correction hidden-supervision path. BF16 HS still reflect the numerical
effects of target quantization; they are not necessarily equal to HS from an
all-BF16 target. Keep the same target checkpoint and quantization scheme across
training data, HS generation, and subsequent online deployment.

Start with a new, empty HS directory. The launcher creates `dspark_dsv4_hs.json`,
and the trainer checks the target path, structural signature, HS layer indices,
and format. Do not relabel old HS files to reuse them. The signature covers
configuration/quantization metadata, tensor headers, file sizes, and modification
times; **it is not a hash of all weight contents**. The server also records an
explicit `--quantization` setting, or backend auto-detection when unspecified.
Changing that setting on restart prevents reuse of the HS directory. Directories
from older versions that did not record it must also be replaced; the trainer
does not need to repeat the option. Both sides must use the same shared absolute
paths, and the checkpoint must not change during training. A manifest only
records the launch contract; its existence does not mean the service is ready.

## Drafter Recipe

Configuration: `examples/train/dsv4_flash_dense_config.json`.

The recipe retains a five-layer dense Qwen3-style decoder, 32 Q / 8 KV heads,
head dimension 128, FFN size 9728, SWA 2048, RoPE theta 1e6, and the existing
correction, dynamic convolution, selector, and other options. The current
implementation requires matching draft/target IO hidden widths, so hidden size
changes from Qwen3-4B's 2560 to 4096, and vocabulary size becomes 129280.
**The parameter count is therefore not identical, and Qwen draft weights cannot
be reused directly.** V4's MLA / compressed-attention geometry is not copied
into the dense draft.

The training script pins `epochs=10`, `seq_len=3072`, `block_size=7`,
`max_anchors=512`, `correction_gate_bias=0`, and `correction_markov_gate_bias=-2`.
It retains Muon + linear scheduling. `--lr 6e-5` is the base LR for the AdamW
portion; without a separate override, Muon's base LR is `6e-4`. This preserves
the current experiment, not the official DeepSpec AdamW + cosine recipe.

## A3 Startup and Hardware Checks

The interfaces target vLLM `0.26.0` / vllm-ascend `0.26.0rc1` and validate
versions during initialization. An image tar filename does not establish its
installed software versions; check them inside the container first.

Requirements: eager execution, file HS backend, PP=1, PCP=DCP=1, and disabled
prefix caching, chunked prefill, FlashComm1 / SP, and DSA-CP. The training HS
service supports single-host DP=1 or 2. The underlying launcher still defaults
to DP1; the current server example uses TP8 x DP2. DP2 requires EP, both DP
engines on the same host, the mp backend, and internal request dispatch.
Multi-host DP, DP>2, and external load balancing are not enabled. Block
verification and the automated offline evaluation launcher remain DP1-only.
Real A3 topology, HS export, and numerical agreement still require validation.
The integration does not overwrite an existing vLLM/Ascend installation or
change native V4 / Qwen registration.

Install this checkout in the existing server and trainer environments without
upgrading torch/vLLM as a side effect:

```bash
pip install -e . --no-deps
pip install -e hs_connectors --no-deps
```

If `VLLM_PLUGINS` is set as an allowlist, add `speculators_dsv4` while preserving
other plugins required by Ascend. Otherwise vLLM discovers plugins automatically.

Run all scripts from the repository root. Both launch scripts are preconfigured
with the current two-host experiment's checkpoint, data/HS paths, 16 devices,
and target address `80.48.17.187:8001`; environment variables can override them.
The target and trainer must run on separate hosts with these defaults. Do not
run both default scripts on one host. Set the following variables explicitly
when server paths or devices differ. Historical filenames retain `bf16`, which
now refers to the BF16 HS interface, not a requirement for all-BF16 weights.

```bash
export MODEL=/shared/models/dsv4-flash-target
export HS_PATH=/shared/hs/dsv4-flash-target-spread-v1
# Set these for the actual devices and memory; no device count per host is assumed.
export VLLM_NPUS='<target device IDs>'
export TP_SIZE='<target tensor parallel size>'
export DP_SIZE=1  # Or 2; the VLLM_NPUS device count must equal TP_SIZE * DP_SIZE.
export VLLM_HOST='<target host internal IP; use 127.0.0.1 for local-only testing>'
# Leave unset for backend detection; set only when the matching Ascend format requires it:
# export TARGET_QUANTIZATION=ascend
bash examples/train/dspark_dsv4_flash_bf16_server.sh
```

If one A3 host **actually exposes 16 logical devices**, you can use
`TP8 x DP2 / EP16`:

```bash
export VLLM_NPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TP_SIZE=8
export DP_SIZE=2
export HS_PATH=/shared/hs/dsv4-flash-target-spread-dp2-v1
bash examples/train/dspark_dsv4_flash_bf16_server.sh
```

Here DP2 runs **within the target host**, not as one DP replica on each of two
hosts. The other host still runs the trainer. The launcher sets
`--data-parallel-size-local 2`, retains EP, and checks that visible device IDs
are unique and `device count = TP x DP` before startup. Direct use of
`scripts/launch_vllm.py --dsv4` performs the same checks. DP2 requires an explicit
`ASCEND_RT_VISIBLE_DEVICES`; `--headless`, external DP ranks, Ray, and multi-host
layouts are not allowed. The plugin rechecks the actual configuration in every
worker. PP/CP remain 1, and the MoE EP group size here is TP x DP; EP does not
multiply the device requirement again.
[vLLM 0.26 parallel configuration](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/parallel.py)

HS still use upstream `ExampleHiddenStatesConnector`. TP rank 0 of each DP
engine writes the requests scheduled by that engine and returns the file path
in the response. No second HS file format is introduced. The teacher remains
pre-final-norm, and trainer arguments, drafter architecture, and loss are
unchanged. Trainer-side multi-device DDP is independent of server DP; there is
no need to change `NUM_TRAIN_NPUS` to 2.

Changing TP/DP topology may affect floating-point computation and memory
budgets. Use a fresh HS directory for initial validation to avoid reading
caches from the old topology. The current manifest **does not bind TP/DP
topology**; passing its checks does not validate numerical behavior or memory
capacity. The 16-device example above does not guarantee that the current
quantized checkpoint fits.

If nonempty, `TARGET_QUANTIZATION` is forwarded unchanged as a single
`--quantization` argument; when unset, no such argument is added. `ascend` is
not a switch that converts arbitrary official FP4/FP8 files to Ascend W8A8.

With two Atlas A3 hosts, one can run the target and the other the trainer, but
first confirm that the actual quantization format allows the target weights
and runtime overhead to fit on one host. Flash has 284B total parameters;
13B active parameters must not be used to estimate weight memory. The scripts
do not assume cross-host TP is available; cross-host TP/EP requires separate
communication-environment checks.
[Official model size](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

Both hosts must share the same absolute MODEL and HS_PATH paths. Restrict the
service port to a trusted internal network, not the public internet. Point the
trainer's `VLLM_ENDPOINT` at the target host's internal address.
The trainer script adds that endpoint's hostname/IP and local loopback hosts to
both `NO_PROXY` and `no_proxy`, preserving entries from both existing lists.
Training workers therefore contact the target directly; proxy settings for other
destinations remain unchanged.

Once the service is ready, run a small request first. The token IDs below are
only a transport probe, not an evaluation prompt:

```bash
python scripts/check_dsv4_hs.py \
  --model "$MODEL" --hidden-states-path "$HS_PATH" \
  --vllm-endpoint http://TARGET_INTERNAL_IP:8001/v1 \
  --input-ids 100 200 300 400
```

The script checks token alignment, shape `[4,6,4096]`, BF16 dtype, and finite
values, and retains the HS file. It **does not check teacher-logit agreement**.
Before full training, also use a fixed set of prompts to:

1. Compare target outputs with a reliable reference implementation using the
   same checkpoint and quantization scheme. If weights were converted, also
   check numerical differences before and after conversion.
2. Compare logits reconstructed from exported teacher HS through frozen norm/head
   with the service's raw logits. Check token alignment and numerical differences,
   not just argmax agreement.
3. Run training/validation on a small DSV4 dataset and check finite loss, valid
   gradients, and checkpoint restoration.
4. Validate acceptance length and stopping boundaries with the experimental
   offline backend below before scaling up to full training.

After the single-request check, DP2 services should also run concurrent requests
with different lengths:

```bash
python scripts/check_dsv4_hs.py \
  --model "$MODEL" --hidden-states-path "$HS_PATH" \
  --vllm-endpoint http://TARGET_INTERNAL_IP:8001/v1 \
  --input-ids 100 200 300 400 --requests 8 --concurrency 2
```

The checker cycles through the original input and shorter prefixes. It checks
tokens, HS shape, BF16/finite values, and unique output filenames per request,
retaining all generated files. **Successful concurrent probes do not prove that
both DP engines received requests.** Check per-engine request metrics/logs on
the server. Initialization logs for DP ranks 0/1 only show that both replicas
initialized, not that both processed requests. Hardware testing should cover
work assigned to only one replica, different input lengths on the two replicas,
and sustained concurrency. Check dummy forwards, EP communication, and file
write failures. These checks do not replace teacher probability comparisons or
DP1/DP2 numerical comparisons, and they are not throughput benchmarks.

The ordinary HS/reference service can still enable full-logprob diagnostics
with `DSV4_EVAL=1`. The automated offline launcher and dedicated
`--dsv4-block-verify` service remain DP1-only; `DP_SIZE=2` does not enable DP2
for them.

On the trainer host:

```bash
export DATA_PATH=/shared/data/dsv4-flash-target-arrow
export OUTPUT_DIR=/shared/output/dsv4-flash-target-corrGate0
export TRAIN_NPUS='<training device IDs>'
export NUM_TRAIN_NPUS='<number of visible training devices>'
export VLLM_ENDPOINT=http://TARGET_INTERNAL_IP:8001/v1
bash examples/train/dspark_dsv4_flash_bf16_trainer.sh
```

Normal training runs in the background through nohup, with logs and PID under
`$OUTPUT_DIR/logs`. TensorBoard writes to `$OUTPUT_DIR/logs/tensorboard`; the
script prints commands for viewing output and stopping training. A successful
script return only means the background process was launched: inspect the log
to confirm initialization. `TRAINING_SMOKE=1` still runs in the foreground and
propagates its exit code so fresh/resume stages execute sequentially.
The server script waits for readiness and stays in the foreground. Its default
startup timeout is 1800 seconds, overridable with `VLLM_STARTUP_TIMEOUT`; early
exit is an error. On Ctrl+C/exit, it cleans up only process groups it created.
The server requires Linux `setsid` and `curl`. Restrict its port to a trusted
network and keep shared paths consistent.

If `OUTPUT_DIR` is unset, the default is `./output/dspark_dsv4_flash_bestArch`.
Use a separate output directory when changing checkpoints or quantization
schemes; do not resume an experiment from another version directly.

DATA_PATH must contain token IDs/loss masks matching DSV4. Default strict mode
also requires a data manifest; Arrow prepared by external scripts can use the
explicit compatibility entry point below. Do not reuse Qwen Arrow, token_freq,
or vocabulary mappings. The preprocessing entry point below encodes existing
responses; it does not generate or resample responses.
[Official encoding documentation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md#chat-template)

When restoring a DSV4 draft, still specify
`--target-hidden-state-format deepseek_v4_mean_hc_head`. The target checkpoint
and selected layers must not change. Qwen training does not need this option;
its default is `standard`.

## Training data preprocessing

### Use Arrow data prepared by an external script

If your data was already encoded with the tokenizer, conversation template, and
supervision masks appropriate for the current DSV4 target, but was not produced
by this repository, explicitly add `--dsv4-external-arrow` to the training command.
With the current launcher:

```bash
DSV4_EXTERNAL_ARROW=1 \
DATA_PATH=/mnt/nfs/dataset/arrow_0730_77w_dedup \
bash examples/train/dspark_dsv4_flash_bf16_trainer.sh
```

This path reads an existing Hugging Face `Dataset.save_to_disk` directory, not a
single `.arrow` file or a `DatasetDict`. The `input_ids`, `loss_mask`, and `seq_len`
columns are required. It does not re-tokenize, generate answers, rewrite Arrow,
reorder or filter samples, or fabricate a `dspark_dsv4_data.json` manifest. It
creates only an in-memory tensor view of the IDs and masks for training, excluding
other columns (including messages). Original text stored alongside the tokens
does not trigger another template application. The existing train/validation
split, sampler, and row-index-based HS lookup remain unchanged. In this explicit
external-data mode, each row's IDs and loss mask are right-truncated in memory to
`--total-seq-len` before requesting HS, instead of generating the entire row's HS
and truncating only during batch collation. The source Arrow files and stored
`seq_len` values are not modified. The default Qwen data path is unchanged.

Before training starts, all rows are validated in batches. In multi-device runs,
each rank checks a contiguous range of rows. IDs must be integers within the
vocabulary; masks must have the same length and contain only booleans or numeric
0/1 values (no nulls or NaN/Inf). `seq_len` must be a positive integer equal to the
actual token count. Masks may cover multiple answer spans; a single assistant
suffix is not required. The validator does not change the first mask value or
filter samples whose masks are all zero.

**This flag is your explicit assertion that the external data matches the current
DSV4 target. It does not prove that the tokenizer or supervision semantics are
correct.** Token-range checks cannot identify every dataset encoded for Qwen or
another model. Logs label this mode `USER-ASSERTED`; the launch command and paths
are recorded in the existing `train_command.txt`. The flag is disabled by default,
so the repository's data contract remains required. Even when enabled, a
conflicting existing data contract is rejected, and checks on the HS directory
and target checkpoint are not relaxed.
Use a separate HS directory for each dataset; do not reuse caches from a different
row order or target. A longer existing HS cache can be read as a prefix only when
its token IDs match the required training prefix and its tensor shape/row count
are valid; the cached file is not rewritten. Too-short or mismatched caches fail
explicitly instead of silently skipping samples. Use a new HS directory when
increasing the training length beyond cached prefixes. Fresh HS responses must
match the requested tokens exactly before being cached or deleted.

The server must still allow the longest requested prefix **plus one output token**
used by HS extraction. With `--total-seq-len 3072`, a source row of 5188 tokens now
requests 3072 input tokens plus one output token, fitting `--max-model-len 4096`.
This preserves the causal training prefix, not bitwise reproducibility: changing
target forward shapes or the number of training-noise draws can change numerical
results. If supervision starts beyond the retained prefix, its retained loss mask
is all zero, just as with the previous collation-time truncation; no new sample
filtering is introduced.

### Prepare the repository's data format from raw messages

Continue to use `scripts/prepare_data.py`; the new path requires an explicit
`--dsv4`. Raw data may use `messages` or ShareGPT `conversations` format. Currently,
only text is supported, with an optional initial system message followed by
alternating user/assistant messages. Tool calls, multimodal content, and ambiguous
fields raise errors rather than producing guessed masks.

Raw messages are encoded through the running target's `/tokenize` endpoint using
the official V4 renderer in vLLM 0.26.0. The process checks the server version,
model root, tokenizer class, and HS/checkpoint contracts, and compares raw-token
probes against the local `tokenizer.json`. It neither executes checkpoint Python
code nor applies a Qwen template. The training server script enables the read-only
`--enable-tokenizer-info-endpoint`. Expose this service only on a trusted internal
network.

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

The tokenizer endpoint is the server's root URL, **without `/v1`**. If you set a
served-model alias, provide that exact alias. When authentication is configured,
preprocessing reads `OPENAI_API_KEY` or `VLLM_API_KEY`.

Multi-turn conversations are split into **one sample per assistant answer**.
History is context only; supervision covers the current answer's continuation,
including EOS for a complete answer. The official encoder removes historical
reasoning, so each sample strictly compares the token prefix of the generation
prompt with that of the full sequence containing the answer. Unstable boundaries
are rejected instead of constructing masks with regular expressions over the
entire conversation. Samples exceeding `--seq-length` are truncated, and samples
with no supervised tokens after truncation are filtered out. As a result,
`max-samples`, the original conversation count, and the output sample count may
differ.

For thinking data, use `--enable-thinking` and put assistant reasoning in a
separate `reasoning` field (consistent `reasoning_content` / `thinking` fields are
also recognized). `<think>` tags embedded in content are rejected; first move
them into structured fields. Non-thinking mode rejects nonempty reasoning to
avoid silently discarding it. Without an explicit flag, the new DSV4 raw-message
path defaults to non-thinking mode.

The output is standard training Arrow with `input_ids`, `loss_mask`, and
`seq_len`, plus `token_freq.pt` and `dspark_dsv4_data.json`. The manifest records
checkpoint identity, tokenizer asset hashes, encoding, thinking mode, mask policy,
and truncation settings, which are checked before training. It records data
encoding provenance; **it is neither proof that the target generated the answers
nor a complete hash of every Arrow file's contents**.

Tokenized data produced by this workflow can be repackaged without a server or
loading a conventional chat template:

```bash
python scripts/prepare_data.py --dsv4 --model "$MODEL" \
  --data /shared/data/dsv4-flash-target-arrow \
  --output /shared/data/dsv4-flash-target-arrow-subset \
  --seq-length 3072 --max-samples 64 --num-preprocessing-workers 1
```

By default, the source manifest is read from the input directory. For standalone
token files, use `--dsv4-source-manifest` to point to their existing provenance
contract. **Do not manually attach the current manifest to old Qwen tokens or
tokens of unknown provenance.** If existing DSV4 Arrow data lacks this
repository's contract but its external encoding has been confirmed correct, use
the external Arrow training entry point above. If its provenance is uncertain,
return to the raw messages and re-encode them; regenerating the answers is not
required. The generic non-DSV4 pretokenized entry point has also been fixed: when
IDs and masks already exist, it no longer requires loading a processor or
checking a chat template. Qwen raw-message processing remains unchanged.

## Teacher probability comparison

`check_dsv4_hs.py` checks only HS shapes and finite values. To check the actual
training teacher, use `scripts/check_dsv4_teacher.py`. Set `DSV4_EVAL=1` when
starting the training server to allow it to return raw log-probabilities over the
full vocabulary. Setting client arguments alone does not reconfigure a running
server.

```bash
python scripts/check_dsv4_teacher.py \
  --model "$MODEL" --hidden-states-path "$HS_PATH" \
  --vllm-endpoint http://TARGET_INTERNAL_IP:8001/v1 \
  --served-model-name "$MODEL" --verification-mode reference \
  --input-ids 100 200 300 400 --tail-positions 4 \
  --device npu:0 --norm-dtype float32 --head-dtype bfloat16 \
  --output-json ./output/dsv4-teacher-check.json
```

The IDs above are only for a wiring check. For acceptance testing, replace them
with real encoded prompts covering different content and lengths. The checker
loads only the audited frozen norm/head, not the full target or draft. It compares
the distribution reconstructed from teacher HS with the native target
distribution at each position, reporting TV, KL, log-probability error, and argmax
agreement. The default FP32 norm and BF16 projection match the BF16 autocast path
used by current fresh/config-only training. If you explicitly load the draft's
parameters in BF16, adjust `--norm-dtype` accordingly. The default CPU device is
for diagnostics only; to validate A3 numerics, explicitly use the trainer's NPU
and dtype.

Use `--positions` to select zero-based HS row indices (row p predicts p+1).
Otherwise, the last four rows are checked. By default, each block contains at
most four full-vocabulary distributions, avoiding materialization of an `L × V`
probability tensor for the entire long context. `block` mode can use an already
running dedicated block-evaluation server. The selected protocol must explicitly
match the server; there is no automatic fallback.

The default thresholds `max-tv=0.02`, `max-logprob-error=0.5`, and
`min-argmax-agreement=1.0` are an initial screening, not universal error guarantees
for every dtype or quantization kernel. Exit code 0 means the current thresholds
passed, 1 means numerical checks failed, and 2 means a configuration, server, or
protocol error. Passing at a few positions does not validate the entire model or
quantization backend. Do not loosen thresholds merely to obtain a passing result.

## Saving, resuming, and A3 smoke testing

New draft configurations save a `target_training_contract` binding the target
path/signature, HS format, auxiliary/teacher slots, and target runtime
quantization. Both explicit `--from-pretrained` loading and automatic resume from
the save path validate this contract before loading weights or optimizer state.
Distributed ranks aggregate validation errors before stopping. Changing the
target, selected layers, or quantization scheme requires a separate experiment.
Older DSV4 checkpoints without a contract are explicitly rejected; the current
identity is not automatically attached to old weights. Migration requires a
separate audit of their original provenance.
DSV4 `--dry-run` also requires valid data and an HS manifest. A data manifest is
required by default; explicit `--dsv4-external-arrow` allows it to be absent but
still performs the external data structure checks.

The generic resume path also fixes an LR issue: constructing the scheduler
overwrote the restored optimizer LR. After loading the scheduler state, its
recorded LR is now restored to the optimizer before the first resumed update.
The linear/cosine formulas are unchanged, and normal training from scratch is
unaffected.

First start the **regular training HS server**, prepare enough DSV4 data to form
train/validation batches, and set the variables introduced above: `MODEL`,
`DATA_PATH`, `HS_PATH`, `TRAIN_NPUS`, `NUM_TRAIN_NPUS`, and `VLLM_ENDPOINT`:

```bash
export SMOKE_ROOT=/shared/output/dsv4-training-smoke
bash examples/train/dspark_dsv4_training_smoke.sh
```

This entry point reuses the existing trainer recipe and automatically runs two
stages in separate processes. By default, it trains for two batches, validates on
one batch, and saves epoch 0. It then resumes automatically, trains for another
two batches, validates on one batch, and saves epoch 1.
`SMOKE_TRAIN_BATCHES` / `SMOKE_VAL_BATCHES` are configurable, but the actual loader
length limits each stage when fewer batches are available. Empty train or
validation loaders raise an error. Both stages check finite loss, nonzero finite
gradients, frozen input/output components, and a valid LR. They also compare
sampled model/optimizer values, scheduler state, and global step across resume.

Training computation, base LR, `corrGate=0`, Muon + linear, and loss switches
retain the original recipe. Smoke-test overrides cover the short epoch count,
shared scheduler total steps across both stages, zero warmup, saving after each
stage, and `num_workers=0` (to avoid prefetching many unused HS requests for just a
few test steps). The actual overrides are recorded in the reports. Short-run loss
is not evidence of quality for the full 10-epoch experiment.

Each run creates a new `run.*` directory without using or overwriting the original
`OUTPUT_DIR` experiment. Logs, both checkpoints, and
`reports/{fresh,resume}.rank-N.json` are retained. Tensor restoration checks sample
the first eight values of each tensor after BF16 normalization; they do not
guarantee bitwise equality of entire tensors or an RNG trajectory across processes
identical to uninterrupted training. This entry point requires an NPU. The current
smoke-test wrapper supports a single device and DDP, but does not cover FSDP;
regular training's FSDP support is unchanged. It does not start the target
automatically, replace the teacher numerical comparison above, or certify
throughput or convergence.
**Passing local CPU regression tests does not mean these A3 stages have actually
been run.**

## DSV4 offline acceptance-length validation (experimental)

This path retains the existing JSONL offline evaluator's draft proposals,
rejection sampling, and acceptance-length/per-position statistics, while running
the target through a vLLM Ascend HS service that matches the checkpoint. The
evaluation device loads only the current dense drafter and its required target IO
weights, rather than the entire 284B target. Qwen still defaults to
`--target-backend hf`, with its existing local target and cache paths unchanged.

Both modes recompute the full prefix and never apply `DynamicCache.crop` to V4's
compressed attention state:

- `block`: A proposal containing `k` draft tokens sends one target request and
  runs one full-prefix forward pass. A dedicated connector obtains the final
  `k+1` rows of full-vocabulary FP32 raw log-probabilities from the target's actual
  LM head and writes the required BF16 HS suffix to the same safetensors file.
  Probabilities are not transferred as full-vocabulary HTTP JSON, and
  full-vocabulary probabilities for the entire prefix are neither computed nor
  exported. The initial implementation exports at most 128 target distribution
  rows per round by default, including the bonus row. Exceeding this limit raises
  an error to prevent an accidental export of a huge vocabulary tensor for a long
  prefix.
- `reference`: Retains the original `k+1` full-prefix requests. Each request
  obtains the full-vocabulary raw log-probabilities at its final position, and
  the last request also reads the auxiliary HS. This serves as a correctness
  reference for block mode.

Both modes use probabilities from the running quantized target for acceptance,
not probabilities reconstructed by a frozen norm/head on the evaluation device.
Draft proposals, rejection sampling, and per-position statistics follow the same
rules. Block mode still prefills the prefix on every round; it does not provide
incremental KV caching or an online performance implementation. Changing forward
pass shapes can change floating-point or quantized results. Compare probabilities,
HS, and sampling boundaries; bitwise equality with reference mode or online
token-by-token execution is not guaranteed. A mode or service-protocol mismatch
raises an error without silently falling back to another mode.

The service still uses the cache-based `extract_hidden_states` drafter internally.
This optimization reduces forward passes and exports only the required suffix;
it does not claim to eliminate all HS cache overhead.
The block HS directory must support **atomic publication using hard links within
the same directory**; a local POSIX filesystem is recommended. The service writes
a temporary file, then publishes the complete data through a hard link without
overwriting an existing file. Unsupported directories or shared filesystems fail
explicitly, with no fallback to non-atomic writes. Only TP rank 0 publishes files,
and write failures are synchronized with the other TP ranks. Publication completes
before the HTTP response, so the client does not read a partially written file.

### Single-host entry point (start with a few samples)

You do not need to start the service manually, set `VLLM_ENDPOINT`, use SSH, or
provide a second machine. The entry point is
`scripts/evaluate/run_dsv4_offline_eval.py`; the shell script below simply converts
environment variables into arguments. It starts separate target and evaluation
child processes on **the same host**, communicating over HTTP at `127.0.0.1`.
It does not combine the 284B target and drafter into one process. Compatible target
weights, a trained DSV4 draft, the plugin, and dependencies for both processes
must already be available.

Start with separate devices. This example assumes neither a particular device
count per A3 host nor a particular number of devices required by the target:

```bash
export VERIFIER_MODEL=/shared/models/dsv4-flash-target
export DRAFT_MODEL=/shared/output/dsv4-flash-target-corrGate0/checkpoint-path
export DATASETS_ROOT=/shared/data/eval-jsonl
export HS_PATH=/shared/hs/dsv4-eval-runs
export OUTPUT_DIR=/shared/eval/dsv4-single-runs
export VLLM_NPUS='<comma-separated target physical device IDs>'
export EVAL_NPU='<one physical device ID outside VLLM_NPUS>'
# TP_SIZE defaults to the device count in VLLM_NPUS; an explicit value must match it.
# export TP_SIZE='<target tensor-parallel size>'
# A nonempty value is passed unchanged to the target; it does not convert weights.
# export TARGET_QUANTIZATION=ascend
export MAX_SAMPLES=4
export MAX_NEW_TOKENS=64
# Defaults to block; reference switches both the target connector and eval client.
export VERIFICATION_MODE=block
bash examples/evaluate/dspark_dsv4_single_eval.sh
```

The default target memory utilization is `0.9`. This is a startup budget for the
default configuration with separate devices, not a guarantee that the model fits.
Both `VLLM_NPUS` and `EVAL_NPU` take physical device IDs on the same host. The
evaluation process sees its single visible device as `npu:0`. The target TP group
is not used as a data-parallel group for the draft; DP is fixed at 1, and both
processes clear inherited distributed rank environment variables.

To share a device between the target and drafter, explicitly enable sharing and
provide a memory budget:

```bash
export EVAL_NPU='<one physical device ID included in VLLM_NPUS>'
export ALLOW_SHARED_DEVICE=1
export TARGET_MEMORY_UTILIZATION='<explicit target memory fraction between 0 and 1>'
bash examples/evaluate/dspark_dsv4_single_eval.sh
```

Setting only `ALLOW_SHARED_DEVICE=1` is insufficient: overlapping devices without
an explicit memory fraction are rejected. **This controls the shared-device
budget; it does not guarantee memory safety on real hardware.** Target weights,
runtime workspaces, KV/compressed state, the draft, target IO weights, and
PyTorch/NPU overhead all consume memory. Estimate requirements for the actual
hardware and quantization format, then validate with small requests. Do not assume
that `1 - TARGET_MEMORY_UTILIZATION` is guaranteed to be available to evaluation;
OOM failures or performance interference remain possible.

The target and evaluator can use different Python environments on the same host:

```bash
export TARGET_PYTHON=/path/to/vllm-ascend-env/bin/python
export EVAL_PYTHON=/path/to/training-env/bin/python
bash examples/evaluate/dspark_dsv4_single_eval.sh
```

The controller itself uses only the Python standard library. Both environments
must have this checkout and their dependencies installed as described above, and
must access the same absolute model and HS paths. The script **does not enter
containers, install software, or start processes on another host automatically**.
For containerized use, run it within the same prepared container/filesystem view.

Each run creates a separate `run-*` subdirectory under `OUTPUT_DIR` and an `hs-*`
subdirectory under `HS_PATH`. The HS parent directory need not be empty, but the
new subdirectory must be isolated for this run; HS from other training or
evaluation runs is not reused. Logs and manifests are retained. Use the result
paths printed by the entry point. The target uses a unique served-model alias and
a temporary authentication token for this run, and binds only to the local
loopback address.

Evaluation starts only after the target is ready. On target startup failure,
evaluation failure, normal completion, or Ctrl+C, the controller cleans up **the
process groups created by this run**, without stopping existing targets or other
tasks. Cleanup cannot be guaranteed after `SIGKILL`, host failure, or other events
that prevent cleanup code from running; processes or HS files may remain.
Descendants that deliberately leave the owned process group are also outside
this cleanup guarantee. Process-tree behavior still needs validation on real A3
hardware. Confirm the task has stopped before inspecting leftover data; do not
empty the HS parent directory.

Common optional variables include `VLLM_PORT=0` to select a free local port;
`STARTUP_TIMEOUT=1800`, `SHUTDOWN_TIMEOUT=30`, and `TARGET_REQUEST_TIMEOUT=120`
(all in seconds); `DSV4_MAX_MODEL_LEN=4096`, `TEMPERATURE=0.0`, `SEED=980406`,
`ENABLE_THINKING=false`, `RAW_PROMPT_MODE=auto`, and `VERIFICATION_MODE=block`.
Use `DATASETS` to select subdatasets, `KEEP_TARGET_HS=1` to retain this run's request
HS, and `SKIP_ARTIFACTS=1` to skip per-sample artifacts. `DRY_RUN=1` displays the
plan without starting the target or evaluator. When calling the Python entry point
directly, the corresponding flags include `--keep-target-hs`, `--skip-artifacts`,
and `--dry-run`.

By default, the single entry point enables `--dsv4-block-verify` on the target and
`--dsv4-verification-mode block` on the client. `--max-logprobs 0` disables the
unneeded full-vocabulary HTTP response allowance; raw probabilities in the file
are unaffected. The service enforces `max_num_seqs=1` and accepts only requests
using the block-verification protocol with `max_tokens=1`. It cannot also serve
training HS collection or ordinary generation. Setting
`--verification-mode reference` (or `VERIFICATION_MODE=reference` in the shell)
switches both the original HS connector and the per-position client on, and
restores the full-vocabulary HTTP response allowance.

Both modes **preserve the acceptance-length statistics and are not online
performance benchmarks**. This path has not yet completed runtime validation on
real A3 hardware; follow the checks at the end of this document on the first run.
Run `block` and `reference` with the same checkpoint, quantization, data, and seed,
then check probabilities and HS alignment before increasing the sample count.

### Connecting to an existing service manually (advanced use)

If you already maintain a separate HS service or need two machines, use the
existing entry point below. It does not manage the service lifecycle and does not
need to run alongside the automatic single-host entry point. **The manual entry
point defaults to `reference`** for compatibility with the original HS service;
this differs from the single-host default.

Prepare the models, shared directory, and plugin as described above, then start
or restart the target service with the evaluation options:

```bash
export DSV4_EVAL=1
bash examples/train/dspark_dsv4_flash_bf16_server.sh
```

`DSV4_EVAL=1` additionally sets `--max-logprobs 129280 --logprobs-mode raw_logprobs
--generation-config vllm`. Without this variable, the training HS service's launch
arguments are unchanged. Completion requests use `logprobs=129280`, not
`logprobs=-1`, and require token IDs as response keys so duplicate decoded strings
cannot cause probabilities to be lost. This repository's `--dsv4` HS bridge is
still required; an ordinary V4 serving instance cannot replace it.
[vLLM Completion protocol](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/completion/protocol.py)

Start with a few samples on the evaluation machine, with training dependencies
installed:

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

To run block mode manually, start a separate dedicated service with both `--dsv4`
and `--dsv4-block-verify` before `--`. An existing training/reference service cannot
be used directly as a block service. For example:

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

Add `--quantization` as required by the backend, keeping the quantization scheme
consistent with validation and training, and use a fresh dedicated HS directory.
Then set `VERIFICATION_MODE=block` and run the manual evaluation script. This
service does not return full-vocabulary HTTP logprobs and cannot serve a reference
client. Likewise, an ordinary training HS service does not produce block
probability files and cannot serve a block client. Both ends must explicitly
match; there is no automatic fallback.

The example defaults to greedy decoding (`TEMPERATURE=0.0`). Set
`TEMPERATURE=1.0` explicitly to test stochastic sampling.
`DSV4_MAX_MODEL_LEN=4096` must match the service's actual `--max-model-len`.
The evaluation input and generation budget must also leave room for verification
candidates and the one output token used for HS export; do not rely on server-side
truncation. `TARGET_REQUEST_TIMEOUT` defaults to 120 seconds. Set
`SERVED_MODEL_NAME` only when the service uses a custom alias. This example uses
one dedicated evaluation device; the target service's TP devices are not used for
draft data parallelism.

For message-based JSONL, the target service's `/tokenize` renderer applies the
current model's official chat encoding; no invented Jinja template is needed for
Preview. The default is `RAW_PROMPT_MODE=auto`. Use `RAW_PROMPT_MODE=raw` only when
the text is already fully encoded using the official protocol for that version.
Do not mix Preview and 0731 templates, tokenizers, or checkpoints.

In addition to the existing `summary.json`, `summary.csv`, and artifacts, results
include `eval_backend.json` to record the target backend and runtime contract.
**This path forbids `--measure-base-speedup`.** Elapsed time includes full-prefix
prefill, HTTP, and shared-file IO, plus full-vocabulary HTTP transfer in reference
mode. It does not represent online speculative-decoding tokens/s or speedup;
speed columns from different backends are not directly comparable.

By default, cleanup removes only temporary HS files belonging to this run's unique
request UUIDs, waiting for asynchronous writes to finish before deleting owned
files. It does not scan and empty the shared directory. `KEEP_TARGET_HS=1`
corresponds to `--keep-target-hs` and helps with comparison checks, but can consume
disk space quickly. Timeouts or interruptions may leave orphaned files from this
run. Inspect them individually only after confirming that the corresponding
requests have finished; do not delete HS used by training or other evaluations.

Local control-logic tests for this backend do not replace A3 validation. On the
first run, check full-vocabulary probabilities, input-token alignment, HS slots
and dtype, and outputs at EOS and generation limits. Successful DSV4 offline
evaluation on real A3 hardware has not yet been demonstrated.

## Local tests

Control-logic tests that do not require torch:

```bash
PYTHONPATH=src python -m unittest discover -s tests/standalone -v
```

These cover checkpoint/manifest/HS-slot checks, hooks and guards against a
simulated native runtime, DSV4 block/reference launch wiring and rejection of
invalid combinations, single-host process management, and Qwen launch-argument
regressions. Simulated backends do not validate real operators or tensor
computation.

Environments with the full training dependencies should also run:

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

Offline-backend tests use real CPU tensors and simulated services. They cover
prefix/HS alignment, refusal to fall back, greedy/T=1 acceptance and residual
sampling, multi-device argument forwarding, and request-file cleanup. They do not
start a real target.

The added PyTorch regression checks that the HS-format field does not change
dense-backbone initialization, context-fusion outputs, or input gradients. It
cannot be marked as passed when PyTorch is unavailable locally.

Preprocessing regressions read and write real Arrow/token_freq data, but use a
simulated client for the service renderer and do not start V4. Short integration
regressions run the real Trainer, Muon/AdamW, linear scheduling, and checkpoint
reads/writes with small CPU models. They adapt only device selection, tiny-model
registration, and the display symlink on Windows; they do not validate a full
DSpark/NPU/DDP run. Standalone Windows test processes must handle the POSIX `fcntl`
dependency encountered during library imports. Do not change production file
locking for these tests or interpret the workaround as platform support.
