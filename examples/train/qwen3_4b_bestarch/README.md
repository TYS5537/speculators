# Qwen3-4B: DSpark and bestarch MMuse

Four standalone scripts, matching the original "configuration at the top, command below" format. Edit each script directly. There is no `settings.sh`, `common.sh`, YAML dispatch or shared launch function. The upstream examples and Python trainer are unchanged.

## Configure and run

Run from the repository root. In `server.sh`, set `MODEL`, `VLLM_NPUS`, `VLLM_PORT` and `TARGET_LAYER_IDS`. Start it in a separate terminal and leave it running:

```bash
bash examples/train/qwen3_4b_bestarch/server.sh
```

The defaults are Qwen3-4B, NPUs `0,1`, DP=2, port `8001`, and target layers `1 9 17 25 33`. All three trainers can reuse this one server: it does not need a draft checkpoint or optimizer setting. A compatible server already running does not need to be restarted. The script refuses an occupied endpoint, checks early exit/readiness timeout, records provenance, and stops its own server on exit.

In the training script you want to run, set `MODEL`, `DATA_PATH`, `OUTPUT_DIR`, `VLLM_ENDPOINT`, `TRAIN_NPUS` and `NUM_TRAIN_NPUS`. Both model and target layers must match the server. The device count is explicit, just like the original script: when changing `TRAIN_NPUS`, change `NUM_TRAIN_NPUS` too. Choose **one**:

```bash
# DSpark baseline, AdamW:
bash examples/train/qwen3_4b_bestarch/train_dspark.sh

# MMuse bestarch, legacy Muon/AdamW grouping:
bash examples/train/qwen3_4b_bestarch/train_mmuse_legacy_optimizer.sh

# MMuse bestarch, upstream Muon/AdamW grouping:
bash examples/train/qwen3_4b_bestarch/train_mmuse_upstream_optimizer.sh
```

Each training script directly uses `nohup env ... torchrun ...` and prints the log path and PID; no extra `nohup` is needed. All default to NPUs `2,3,4,5,6,7` with six workers. Run them one at a time, waiting for the previous training process to finish. Printing a PID only means the background job was launched; check its log for errors.

Logs, TensorBoard events and checkpoints are separated under `./output/qwen3_4b_bestarch/<variant>/`. For a repeat, edit `OUTPUT_DIR` to a new directory. Existing checkpoints or a training PID file cause refusal, not implicit resumption. In particular, the corrected AdamW DSpark recipe must not resume a previous Muon optimizer state. Training does not own or stop the server.

Settings are literal assignments inside each script, not environment-variable overrides. To customize vocabulary size, add `--draft-vocab-size` directly to the training command if required; the supplied bestarch did not specify it. There are no launcher-specific `DRY_RUN` or `DUMP_CONFIG` modes.

## Optimizers and architecture

| Script         | Optimizer                                            | LR / weight decay                   | Schedule                                    |
| -------------- | ---------------------------------------------------- | ----------------------------------- | ------------------------------------------- |
| DSpark         | AdamW for all trainable parameters, including Markov | 6e-4 / 0                            | Cosine, 4% warmup                           |
| MMuse legacy   | Muon + AdamW; Markov uses Muon                       | Muon 6e-4 / 0.1; AdamW 6e-5 / 0.01  | Original linear schedule, default 1% warmup |
| MMuse upstream | Muon + AdamW; Markov uses AdamW                      | Same group settings as MMuse legacy | Same as MMuse legacy                        |

DSpark explicitly passes `--optimizer adamw` with no Muon flags. This follows the optimizer family in the [author's implementation](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/utils/optim.py), with LR/decay/warmup from the [public Qwen3-4B recipe](https://github.com/deepseek-ai/DeepSpec/blob/main/config/dspark/dspark_qwen3_4b.py). It still uses the existing Speculators trainer and precision path, not DeepSpec's separate BF16/FP32-master optimizer wrapper. This is not a full reproduction of the paper's training setup.

Both MMuse scripts retain the supplied bestarch: Correction logits mode (768 hidden / rank 256), hidden auxiliary loss and feedback, dual projection, Markov collaboration, context residual, block-position embeddings, gated layer fusion, dynamic convolution, and the greedy top-16 Selector with corrected feedback. No Correction MoE or DFly/KV-projection experiment is enabled.

The two MMuse commands differ only in `--muon-parameter-policy legacy/upstream` and output labels. Moving Markov factors also changes their effective LR and weight decay, so this is a grouping-policy comparison, not an equal-hyperparameter optimizer-algorithm ablation. Selecting upstream grouping does not activate the full upstream training recipe.

## Preserved local training budget

All three retain `--training-recipe legacy`, `--loss-implementation legacy`, the original prepared 700k dataset, sequence length 3072, ten epochs, seed 42, six training devices, one update per packed batch (no gradient accumulation), 90% training / 10% validation, block size 7, at most 512 anchors, five draft layers, SDPA, rank-256 vanilla Markov, confidence head, CE/TV=0.1/0.9, noise std 0.05, and position weighting `exp(-position/7)`. All five draft layers use full attention.

There is no fixed global batch of 512 sequences. On six devices the packed text-token capacity is 6 * 3072 = 18,432 per update, including padding capacity but excluding synthetic draft anchor positions. The number of original samples depends on their lengths.

These scripts reuse prepared data and existing vocabulary mappings; they do not regenerate datasets. Without explicit vocabulary sizing, existing `d2t.npy`/`t2d.npy` are reused, otherwise the full target vocabulary is used. Preserve the original tokenizer/template/thinking masks and use identical data/mapping files for MMuse comparisons. DSpark versus MMuse is not an architecture-only ablation because optimizer/LR/decay/schedule differ. No pre-refactor bitwise parity is promised.
