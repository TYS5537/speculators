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

## Post-merge maintenance

This maintenance sequence uses the merged upstream version as its baseline, not the
pre-merge implementation. Model loading now separates config/conversion resolution,
concrete-model dispatch and ordered post-load hooks. It retains upstream external
checkpoint conversion and verifier-owned weight reconstruction alongside legacy
MMuse identity migration. No training recipe, model parameter or checkpoint key
is changed by this refactor.

The MMuse CPU test dependencies live in `tests/mmuse_cpu_requirements.txt`, shared
by CI and the documented local setup. Transformers 5 replaces the incompatible
pre-merge Transformers 4 pin. Standalone tests check the pins against upstream
package requirements; `make test-mmuse` also runs shared model loading, conversion,
checkpoint ownership, typed configuration and recipe compatibility tests.

The model-loading refactor passed 2,586 CPU cases and 816 subtests, with 14
skips and 5 environment-related deselections. This includes tiny-MMuse/DSV4
training and checkpoint-resume regressions. Ruff lint and formatting passed. The
Windows run uses an import-only POSIX-locking stub that fails if called; it does
not execute the Linux CI installer, distributed workers or accelerator services.

The offline evaluator now separates target preflight, model/draft initialization,
DSV4 client construction, architecture reporting and dataset execution. Its main
dispatch function has complexity 2 instead of 17. Parent/worker routing still
follows preflight, and the run's original `ExitStack` owns clients through output
publication. Precision tests now call the real draft-loading helper instead of
extracting statements from the script. Sampling, verification and per-dataset
acceptance/timing calculations are unchanged.

After the evaluator split, the same local regression groups passed 2,594 CPU
cases and 886 subtests, with 14 skips and 5 environment-related deselections.
The new startup/lifetime tests also pass against the pre-split implementation;
Ruff lint and formatting pass with no added exceptions. The same Windows/CPU
and real-service validation boundaries above still apply.

Per-dataset evaluation now separates paired warmup, progress logging and shared
base-speedup reporting from generation. `_evaluate_dataset` has complexity 9
instead of 14. Its wall-clock/synchronized-generation boundaries, random-seed
reset, shard indices, acceptance counters and output formats are unchanged. The
same base-speedup calculation serves local rows and worker aggregation, retaining
the latter's slowest-worker time and zero-base-time guard.

After this dataset split, the same local regression groups passed 2,603 CPU cases
and 917 subtests, with 14 skips and 5 environment-related deselections. The nine
new deterministic dataset tests (31 subtests) pass both before and after the split;
they also run with stdlib unittest without model dependencies. Ruff lint and
formatting pass with no added exceptions. These checks do not measure real-device
performance or replace NPU/vLLM validation.

Draft verification now separates acceptance statistics from accepted-stop lookup.
`verify_draft_tokens` has complexity 8 instead of 11; both new helpers have
complexity 3. Target-call/validation ordering, probability formulas, diagnostic
truncation and replacement/bonus sampling are unchanged. The uniform draw still
covers the full proposal, and the continuation draw is retained even when the
outer decoder discards it after an accepted stop. The outer decoding loop,
cache cropping and model architecture code are untouched by this step.

The 85 new verifier cases passed both before and after the refactor. A separate
1,296-case comparison of the two implementations matched every result field and
the final CPU RNG state exactly, across temperatures, precisions, proposal lengths,
dense/sparse/equal distributions and stop settings. The full local regression groups
then passed 2,688 CPU cases and 917 subtests, with 14 skips and 5 environment-related
deselections. Ruff lint/format and provenance checks pass; the same Windows/CPU and
real-service validation boundaries still apply.

The compatibility response-regeneration script now separates sampling-option
merging, request construction and cached-tool-result pairing from its async loop.
`regenerate_conversation` has complexity 7 instead of 11; the new helpers have
complexities 3, 2 and 3. Request-owned fields, sample commit order, truncation and
input-copy behavior are preserved. Retry/backoff, worker writes, queue cleanup,
resume identity and the separate upstream regeneration CLI are unchanged.

The 42 new regeneration contract cases passed before and after the split; all 140
compatibility/upstream regeneration cases pass. A separate 960-case comparison
matched request payloads, sample rows, truncation/errors, commit ordering and
input mutations between the old and new implementations. The expanded local
regression groups, now also including the upstream regeneration CLI tests, passed
2,784 CPU cases and 917 subtests, with 14 skips and 5 environment-related
deselections. Ruff lint, strict lint, formatting and provenance checks pass.

The tracked local complexity backlog is now empty. Keep the empty baseline and
the no-new-debt gate; this does not mean every upstream function is simple or
that all existing per-file lint exceptions have been removed. Follow-up priorities
are real NPU/vLLM smoke validation and review of the accumulated changes before
a release checkpoint. No live accelerator/service validation was performed by
these CPU-only refactors.
