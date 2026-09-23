# Local training configurations

Keep upstream-named examples aligned with the merged upstream baseline. Put this
fork's experimental training settings here instead of maintaining another copy of
the complete server/data/training launcher.

`dspark_qwen3_0_6b_sharegpt.yaml` preserves the former local Qwen3-0.6B recipe:
10 epochs, block size 7, 5 draft layers, 3,072 anchors, learning rate 0.0003,
Markov rank 256, confidence weighting `match-draft`, and CE/TV weights 0.1/0.9.
It explicitly selects `training_recipe: legacy` to retain the original loss,
Markov initialization and Muon grouping/LR policy through the unified entrypoint.
The standard `../../dspark_qwen3_0_6b_sharegpt_online.sh` remains the upstream
5-epoch, 8-token, 3-layer example. There is no separate `_upstream.sh` launcher.

## Run the local experiment

Run the following commands from the repository root. The YAML config controls
training only; it does not launch a target server or prepare a dataset.

Start the target in a separate terminal and wait for it to become ready:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train Qwen/Qwen3-0.6B \
    --target-layer-ids 2 14 25 \
    --provenance-dir ./output/provenance/dspark_qwen3_0_6b_sharegpt_local/server \
    -- --port 8000
```

With that server still running, prepare data using the upstream data entrypoint:

```bash
speculators prepare-data \
    --model Qwen/Qwen3-0.6B \
    --data sharegpt \
    --output ./output/dspark_qwen3_0_6b_sharegpt_local \
    --max-samples 5000 \
    --seq-length 4096 \
    --render-endpoint http://localhost:8000
```

Train with the local configuration on a separate GPU:

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    -m speculators.train \
    --config examples/train/configs/local/dspark_qwen3_0_6b_sharegpt.yaml
```

The `_local` data/checkpoint directory is intentionally separate from the standard
example, avoiding accidental data reuse or checkpoint resumption across recipes.
The upstream preparation path uses server-derived render masks; preserving the
training settings does not promise identical data or numerical results to the old
HF preparation path. For historical comparisons, reuse the exact prepared dataset
with `--data-path` and a separate `--save-path`.

## Customize without another launcher

Copy the YAML for a new experiment, or override individual settings with flags.
Flags take precedence over YAML. For example, `--lr 0.0006` also re-derives the
legacy Muon LR as 0.006 unless `--muon-lr` is explicitly supplied. Likewise, the
legacy decay gamma follows `--block-size` unless explicitly supplied.

`--training-recipe upstream` explicitly opts into upstream loss/optimizer/default
policy while keeping settings already specified in the YAML. Do not change recipes
when resuming optimizer state without checking compatibility; use a fresh save
directory for a new recipe comparison.

To inspect the resolved configuration without loading model weights:

```bash
python -m speculators.train \
    --config examples/train/configs/local/dspark_qwen3_0_6b_sharegpt.yaml \
    --dump-config
```

Keep server model/layers and prepared-data paths consistent with training when
overriding them. Hardware-specific Ascend and MMuse/DSV4 launchers are unchanged
by this example/config separation.
