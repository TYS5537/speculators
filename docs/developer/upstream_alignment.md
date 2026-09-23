# Upstream integration

This merge incorporates upstream Speculators
`824125a36600b5af8fdd1b18286da51ed60f60a8` into the MMuse/DSV4 fork.
It is a Git merge, not a replacement of local architecture code.

## Entry points and defaults

| Entry point | Default recipe | Data preparation |
| --- | --- | --- |
| Existing `python scripts/train.py` flags | Legacy | Existing HF/offline scripts retained |
| `python -m speculators.train` / `speculators train` | Upstream; legacy for MMuse | New typed YAML/CLI supported |
| Typed entry point with `--training-recipe legacy` | Legacy | Existing prepared data remains usable |

`--config`, `--dump-config`, or `--training-recipe` on the old script selects
the typed parser. Specify `--training-recipe legacy` when moving a legacy
DSpark/DFlash experiment to YAML. Explicit flags/YAML values override recipe
defaults. A loaded MMuse checkpoint uses legacy defaults unless the recipe is
explicitly selected; saved architecture and runtime-override validation still apply.

Legacy keeps the prior learning-rate/Muon ratio, DSpark/MMuse block/loss defaults,
Markov initialization (including RNG consumption), optimizer grouping, and local
loss implementations. Upstream selects the new defaults and fused loss interface.
`--loss-implementation eager` explicitly selects the upstream eager implementation;
`legacy` preserves the local stable NLA/pruned-vocabulary path.
Do not change recipes while resuming optimizer state without checking group and
scheduler compatibility.

The original Qwen ShareGPT example remains unchanged. Its upstream counterpart is
`examples/train/dspark_qwen3_0_6b_sharegpt_upstream.sh`.

## Included and preserved

- Included: DFlash2, typed configuration, per-block greedy acceptance counts,
  recovery metadata/circuit breakers, provenance, FP8 and Mooncake connectors,
  and upstream launch/evaluation changes.
- Preserved: MMuse Correction/Selector/backbone collaboration, DSV4 NPU evaluation
  and HTTP hidden-state transfer, legacy HF preprocessing and CLI helpers.
- Preserved: strict token-prefix checks, invalid-cache quarantine, atomic cache
  publication, checkpoint transactions, empty-supervision handling, and exact
  process-RNG restoration.
- A partial epoch stopped by `max_steps` publishes an `interrupted` checkpoint
  at a completed update boundary, before fetching another batch.
- New upstream vocabulary caches use `d2t-N.npy` / `t2d-N.npy`; legacy uses
  `d2t.npy` / `t2d.npy`. Both retain atomic writes and distributed error reporting.

## Environment and validation boundaries

Dependency metadata follows upstream, including Transformers 5 and
pydantic-settings. This is not a guarantee that an existing Ascend/vLLM environment
can upgrade these dependencies in place. Keep the validated NPU environment
separate and check the matching torch/torch-npu/vLLM versions first.

Local verification uses offline CPU regressions, including real tiny-MMuse
forward/backward, checkpoint save/load, and interrupted/uninterrupted comparisons.
It does not validate real NPU/CUDA kernels, Triton fused losses, multi-node
transport, or live vLLM serving. Upstream also changes distributed batch balancing,
so retaining the legacy recipe does not promise bit-identical distributed runs
across this merge.

The merge validation run on 2026-09-23 passed 3,268 CPU test cases and 812
subtests across the legacy/MMuse, upstream, connector/data, and checkpoint suites;
15 cases were skipped and 9 deselected. Windows symlink privileges, distributed
subprocess tests, remote-model downloads, and accelerator-only runs are outside
this local result. Ruff formatting and the lint ratchet passed without increasing
the five pre-existing complexity exceptions.

Always pass `--provenance-dir` when running `scripts/launch_vllm.py`.
