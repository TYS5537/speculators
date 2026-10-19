# Qwen3-4B: DSpark, bestarch MMuse and DFlash2

Five standalone scripts, matching the original "configuration at the top, command below" format. Edit each script directly. There is no `settings.sh`, `common.sh`, YAML dispatch or shared launch function.

## Configure and run

Run from the repository root. In `server.sh`, set `MODEL`, `VLLM_NPUS`, `VLLM_PORT` and `TARGET_LAYER_IDS`. Start it in a separate terminal and leave it running:

```bash
bash examples/train/qwen3_4b_bestarch/server.sh
```

The defaults are Qwen3-4B, NPUs `0,1`, DP=2, port `8001`, and target layers `1 9 17 25 33`. All four trainers can reuse this one server: it does not need a draft checkpoint or optimizer setting. A compatible server already running does not need to be restarted. The script refuses an occupied endpoint, checks early exit/readiness timeout, records provenance, and stops its own server on exit.

In the training script you want to run, set `MODEL`, `DATA_PATH`, `OUTPUT_DIR`, `VLLM_ENDPOINT`, `TRAIN_NPUS` and `NUM_TRAIN_NPUS`. Both model and target layers must match the server. The device count is explicit, just like the original script: when changing `TRAIN_NPUS`, change `NUM_TRAIN_NPUS` too. Choose **one**:

```bash
# DSpark baseline, AdamW:
bash examples/train/qwen3_4b_bestarch/train_dspark.sh

# MMuse bestarch, legacy Muon/AdamW grouping:
bash examples/train/qwen3_4b_bestarch/train_mmuse_legacy_optimizer.sh

# MMuse bestarch, upstream Muon/AdamW grouping:
bash examples/train/qwen3_4b_bestarch/train_mmuse_upstream_optimizer.sh

# Standalone DFlash2, DSpark training budget with DFlash unary CE:
bash examples/train/qwen3_4b_bestarch/train_dflash2.sh
```

Each training script directly uses `nohup env ... torchrun ...` and prints the log path and PID; no extra `nohup` is needed. All default to NPUs `2,3,4,5,6,7` with six workers. Run them one at a time, waiting for the previous training process to finish. Printing a PID only means the background job was launched; check its log for errors.

Logs, TensorBoard events and checkpoints are separated under `./output/qwen3_4b_bestarch/<variant>/`. For a repeat, edit `OUTPUT_DIR` to a new directory. Existing checkpoints or a training PID file cause refusal, not implicit resumption. In particular, the corrected AdamW DSpark recipe must not resume a previous Muon optimizer state. Training does not own or stop the server.

Settings are literal assignments inside each script, not environment-variable overrides. For DSpark/MMuse, add `--draft-vocab-size` directly to the training command if required; the supplied bestarch did not specify it. Standalone DFlash2 requires the full verifier vocabulary. There are no launcher-specific `DRY_RUN` or `DUMP_CONFIG` modes.

## Optimizers and architecture

| Script         | Optimizer                                            | LR / weight decay                   | Schedule                                    |
| -------------- | ---------------------------------------------------- | ----------------------------------- | ------------------------------------------- |
| DSpark         | AdamW for all trainable parameters, including Markov | 6e-4 / 0                            | Cosine, 4% warmup                           |
| MMuse legacy   | Muon + AdamW; Markov uses Muon                       | Muon 6e-4 / 0.1; AdamW 6e-5 / 0.01  | Original linear schedule, default 1% warmup |
| MMuse upstream | Muon + AdamW; Markov uses AdamW                      | Same group settings as MMuse legacy | Same as MMuse legacy                        |
| DFlash2        | AdamW for all trainable parameters, including selector | 6e-4 / 0                          | Cosine, 4% warmup                           |

DSpark explicitly passes `--optimizer adamw` with no Muon flags. This follows the optimizer family in the [author's implementation](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/utils/optim.py), with LR/decay/warmup from the [public Qwen3-4B recipe](https://github.com/deepseek-ai/DeepSpec/blob/main/config/dspark/dspark_qwen3_4b.py). It still uses the existing Speculators trainer and precision path, not DeepSpec's separate BF16/FP32-master optimizer wrapper. This is not a full reproduction of the paper's training setup.

Both MMuse scripts retain the supplied bestarch: Correction logits mode (768 hidden / rank 256), hidden auxiliary loss and feedback, dual projection, Markov collaboration, context residual, block-position embeddings, gated layer fusion, dynamic convolution, and the greedy top-16 Selector with corrected feedback. No Correction MoE or DFly/KV-projection experiment is enabled.

The two MMuse commands differ only in `--muon-parameter-policy legacy/upstream` and output labels. Moving Markov factors also changes their effective LR and weight decay, so this is a grouping-policy comparison, not an equal-hyperparameter optimizer-algorithm ablation. Selecting upstream grouping does not activate the full upstream training recipe.

## Preserved local training budget

The DSpark and two MMuse trainers retain `--training-recipe legacy`, `--loss-implementation legacy`, the original prepared 700k dataset, sequence length 3072, ten epochs, seed 42, six training devices, one update per packed batch (no gradient accumulation), 90% training / 10% validation, block size 7, at most 512 anchors, five draft layers, SDPA, rank-256 vanilla Markov, confidence head, CE/TV=0.1/0.9, noise std 0.05, and position weighting `exp(-position/7)`. All five draft layers use full attention. DFlash2's matched budget, CE loss and architecture differences are described below.

There is no fixed global batch of 512 sequences. On six devices the packed text-token capacity is 6 * 3072 = 18,432 per update, including padding capacity but excluding synthetic draft anchor positions. The number of original samples depends on their lengths.

These scripts reuse prepared data and existing vocabulary mappings; they do not regenerate datasets. Without explicit vocabulary sizing, existing `d2t.npy`/`t2d.npy` are reused, otherwise the full target vocabulary is used. Preserve the original tokenizer/template/thinking masks and use identical data/mapping files for MMuse comparisons. DSpark versus MMuse is not an architecture-only ablation because optimizer/LR/decay/schedule differ. No pre-refactor bitwise parity is promised.

## MMuse NPU memory troubleshooting

Both MMuse scripts use `TASK_QUEUE_ENABLE=1`. Ascend's level-2 queue mode can
increase peak memory through concurrent workspace use; see the
[Ascend pipeline documentation](https://www.hiascend.com/doc_center/source/en/mindstudio/830/practicalcases/GeneralPerformanceIssue/toolsample6_054.html).
Changing this setting affects scheduling and throughput, not the training
objective, anchor count, optimizer settings or requested precision.

For block size 15 with 512 anchors, add both `--fsdp-shard` and
`--activation-checkpointing` to the training command. They reduce model-state
and decoder-activation memory, but do not bound full-vocabulary loss or
Correction intermediates. With the full 151936-token vocabulary, a single
`[512 * 15, 151936]` FP32 tensor occupies about 4.35 GiB. This is a tensor-size
estimate, not a measurement of an operator's workspace or total peak memory.

If a run reports `NPUWorkspaceAllocator` OOM with asynchronous-stack warnings,
first change `TASK_QUEUE_ENABLE=2` to `1` **inside the script** and restart in a
fresh output directory. An environment prefix outside the script is overwritten
by its literal assignments. Keep block size, anchors, losses and learning rates
unchanged for this retry. If it still fails, set `ASCEND_LAUNCH_BLOCKING=1` inside
the script for a short diagnostic run to locate the failing operator, then return
it to `0` for normal training. Blocking changes execution timing and can hide a
concurrency-related peak; completing that diagnostic run alone does not validate
normal asynchronous training. A traceback surfacing in FSDP allocation or event
handling does not establish that FSDP's communication buffer caused the OOM.

## Optional MMuse forward profiling

The normal training path skips the unused greedy Selector walk in `corrected` teacher forcing and does not compute base diagnostics that would be discarded by the compact logger. Static Selector conditioning, standalone Selector proposals, actual inference rollout, losses and checkpoint parameter names/shapes are unchanged. Full base diagnostics remain available during validation with `--correction-base-diagnostics`. DSpark's implementation is unchanged.

For a diagnostic run, set the optional profiling environment variable before starting either MMuse script:

```bash
MMUSE_PROFILE_FORWARD=50 bash examples/train/qwen3_4b_bestarch/train_mmuse_upstream_optimizer.sh
```

This prints a `MMuse forward profile` line on rank zero every 50 training forwards, beginning with call 0. The stages are `backbone`, `selector_conditioning`, `correction_markov`, `proposal_confidence`, `loss_metrics`, and `auxiliary`, each in milliseconds. The backbone stage includes verifier target-logit reconstruction and the initial draft LM-head projection; Correction includes its final projection and Markov collaboration. `call` is a process-local forward counter, not the saved global step; validation does not advance it. No TensorBoard writer or checkpoint field is added.

Unset the variable or use `MMUSE_PROFILE_FORWARD=0` for normal training (the default). Profiling synchronizes the device between stages and can slow sampled steps; discard startup samples and compare steady-state end-to-end `fwd_ms` with profiling disabled. The stage total excludes the Trainer's post-forward supervision synchronization, so it is not identical to `fwd_ms`. Enabling profiling or changing source files does not update an already-running training process.

## Standalone DFlash2 with DSpark training budget and DFlash CE

`train_dflash2.sh` matches `train_dspark.sh` in target server, prepared 700k dataset, six training NPUs, sequence length 3072, ten epochs and seed 42. It also matches AdamW (LR 6e-4, weight decay 0), cosine scheduling with 4% warmup, five full-attention layers, SDPA, 512 anchors and noise std 0.05. The existing trainer clips gradients at 1.0. The legacy recipe and loss implementation remain explicit for Ascend runs.

DFlash2 reserves slot 0 for the verified anchor, so block size 8 proposes seven tokens, matching DSpark's block-7 budget. Fixed exponential decay keeps gamma 7 for the same prediction-position weights. Only the unary objective changes from CE/TV=0.1/0.9 to CE, following the [DFlash paper, Section 4.2](https://arxiv.org/html/2602.06036v1). The paper's six-epoch, block-16 Qwen3 training setup is not adopted here.

The model remains DFlash2: convolution uses kernel size 2 and group size 16, and the rank-256 selector reranks 16 candidates. Its additional K-way CE loss keeps the local weight 0.1. The total training loss is unary CE plus 0.1 times selector CE, with position weighting; TV is no longer a training-loss term. TV-based acceptance diagnostics may still be reported. DFlash has no selector, and DFlash2's public article does not specify its training loss, so selector CE/weight 0.1 remain experimental. These are the standalone `--conv-*` / `--selector-*` options, not MMuse's `--dflash2-*` extensions.

The run writes to `./output/qwen3_4b_bestarch/dflash2/` and refuses an existing checkpoint/PID directory. DFlash2 always uses the full verifier vocabulary and ignores pruning caches in the shared dataset directory. If DSpark or MMuse uses a pruned vocabulary, that remains an additional difference in the comparison. The selector objective is experimental; this is not a reproduction of an unpublished original training recipe.

The script enables `--save-best`, which saves the best validation-loss checkpoint at `checkpoints/checkpoint_best`. For offline evaluation, use it with the shared evaluator's DFlash2 entry:

```bash
python scripts/evaluate/dflash2_offline_eval.py \
    --verifier-model ../../Qwen3-4B \
    --draft-model ./output/qwen3_4b_bestarch/dflash2/checkpoints/checkpoint_best \
    --datasets-root /path/to/evaluation-jsonl \
    --output-dir ./output/qwen3_4b_bestarch/dflash2/eval \
    --device npu:0 --dtype bfloat16 --draft-attn-impl sdpa \
    --temperature 0 --enable-thinking false
```

Use an available NPU after training has stopped, or adjust the device assignment. When publishing this run, include the shared server's `vllm_command.txt`, `vllm.patch` and `checkpoint_sha256.txt` from the directory printed/configured by `server.sh` alongside the training and evaluation provenance.
