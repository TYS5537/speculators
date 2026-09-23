# DFlash, DSpark and MMuse

**MMuse** means **Multi-Model fUSE**: multi-model collaboration through adaptive
feature fusion and token-level correction. The canonical model identifier is
`mmuse`, and the Python package is `speculators.models.mmuse`.

The rename does not change architecture switches, training defaults or tensor
names. Existing `muse` checkpoints load as `mmuse` without editing their files;
the next save records the new model identity. `--speculator-type muse` and
`SPECULATOR_TYPE=muse` in the Qwen trainer remain legacy aliases for `mmuse`.
New runs and imports should use `mmuse`, `MMuseDraftModel`, and
`MMuseSpeculatorConfig`. This is a model rename, not a new training recipe.

The model directories separate the baseline algorithms from experimental
architecture extensions:

- `src/speculators/models/dflash/`: DFlash block drafting, verifier context,
  attention and vocabulary handling.
- `src/speculators/models/dspark/`: the DSpark baseline, adding Markov/RNN
  sequential heads and acceptance confidence on top of DFlash.
- `src/speculators/models/mmuse/`: backbone residual and gated-fusion extensions,
  dynamic convolution, candidate Selector, causal Correction and collaboration.

Shared training, checkpointing, numerical safeguards and data contracts are not
architecture extensions and remain in their existing shared modules.

Within MMuse, `selector.py` defines the trainable candidate scorer;
`selector_runtime.py` owns Top-K candidate selection, greedy/global path search,
proposal logits and the restricted-Top-K teacher loss. Its stateless mixin keeps
the existing model methods available without adding a module or changing the
`candidate_selector.*` checkpoint keys. `backbone.py` retains module construction,
initialization and checkpoint checks, as well as the vocabulary mapping shared
by Selector and Correction.

## Training

### Internal tensor interfaces

`runtime_types.py` names the Selector conditioning, static Selector inputs,
initial logit feedback, single Correction step and complete rollout outputs.
Internal consumers use named fields; tuple unpacking keeps its previous order.
These immutable containers hold the original tensor/cache references without
copying, detaching or registering state. Recurrent feedback updates remain local
to the existing rollout loop, with the same tensor lifetimes and operation order.
The public `rollout_correction()` method still returns the plain `(tokens, logits)`
pair. Checkpoint keys and model configuration are unchanged.

### Starting a run

Use `--speculator-type dflash` or `--speculator-type dspark` for the baselines.
Use `--speculator-type mmuse` with the existing extension flags for enhanced runs:

```bash
python scripts/train.py \
  --verifier-name-or-path /path/to/verifier \
  --data-path /path/to/data \
  --save-path /path/to/output \
  --speculator-type mmuse \
  --enable-correction-head \
  --dflash-gated-layer-fusion
```

With the legacy recipe, MMuse inherits DSpark's training defaults (block size 7,
five decoder layers, 10 epochs, CE/TV weights 0.1/0.9). Extensions remain opt-in;
choosing MMuse alone does not silently enable Correction or change its parameters. Extension flags
retain their existing names, including the `dflash-` and `dflash2-` prefixes.
Passing them with a baseline type for a fresh run produces an error directing you
to MMuse. When restoring, the saved model type determines which options apply.

### Training shell examples

Experiment values stay in each `examples/train/` recipe. Two narrowly shared
helpers under `examples/train/common/` remove identical setup and argument lists:

- `ascend_training_env.sh` defines the explicitly called Ascend training preset
  used by the Qwen online/trainer and DSV4 trainer scripts. It does not configure
  standalone target servers, device allocation or proxies.
- `dspark_online_args.sh` builds the baseline training argument array for the two
  full online Qwen DSpark examples. The Ascend caller adds its attention option;
  the trainer-only Qwen MMuse and DSV4 recipes keep their own argument assembly.

Sourcing these helpers only defines functions; it does not change directories,
install traps or launch processes. Callers retain data preparation, readiness
checks, foreground/background execution and cleanup. Qwen's fixed proxy list and
DSV4's endpoint-aware exclusions remain separate, as do DSV4's PID guard and
synchronous smoke execution.

Run recipes from the repository root as before. Helpers are located relative to
the recipe file, without changing how relative data/output paths resolve. Keep
renamed recipe copies beside `common/`; when copying recipes elsewhere, copy the
required `common/` helpers with them. The scripts are no longer standalone files.

### Configuration contract

`MMuseOptions` in `src/speculators/models/mmuse/config.py` owns the extension fields,
including inherited backbone fields. The CLI and training factory obtain their
defaults from that schema; `validate_mmuse_options` applies the same scalar and
cross-option checks in fresh CLI runs, model construction and checkpoint override
handling. Reading a config alone does not construct modules or run these full
cross-option checks. Decoder/vocabulary-dependent checks remain in the modules
that know those dimensions.

`src/speculators/train/legacy_cli.py` owns historical parser construction,
algorithm-specific default resolution and post-parse validation as separate steps.
`scripts/train.py` retains the zero-argument `parse_args()` entry point. The reusable
`parse_train_args(argv)` uses the same argument list for initial parsing and all explicit-option tracking;
omitting `argv` continues to use the process command line. Validation order and
the distinction between parser errors and ordinary validation exceptions are
preserved.

The upstream typed YAML/CLI lives in `src/speculators/train/config/`, while
`src/speculators/train/cli.py` runs the shared training workflow and re-exports
legacy parser helpers for compatibility. The old script selects typed parsing
when `--config`, `--dump-config`, or `--training-recipe` is supplied. Legacy script
arguments are adapted into the same `TrainConfig` without treating parser defaults
as explicit checkpoint overrides. See [upstream integration](../../developer/upstream_alignment.md)
for recipe selection and migration boundaries.

Legacy default resolution dispatches the DSpark paper recipe only for DSpark/MMuse,
then fills shared decoder/normalization/Muon values only where they are `None`.
The recipe uses explicit-option tracking, not comparison with parser defaults:
an explicit value equal to a parser default still wins. Omitted decay gamma is
derived from the resolved block size. Alias normalization and checkpoint override
tracking stay in the finalizer; applying defaults does not enable MMuse extensions.

`src/speculators/train/mmuse_args.py` registers the legacy MMuse backbone/Selector
and Correction CLI options using one defaults mapping supplied by the parser builder.
The two groups retain their original order around DSpark's baseline head options,
including Boolean disable flags and the mutually exclusive greedy/global search
switches. Hidden-state backends register their options when each parser is built.

`src/speculators/train/model_config.py` owns initialization-source conflicts and
checkpoint override policy: which fields must match, which may change at runtime,
and which omitted values are inherited. The training entry point keeps its existing
helper names and passes its logger to the reconciliation helper.

`src/speculators/train/draft_config.py` owns decoder configuration synthesis and
loading: verifier geometry, layer attention types, RoPE/MRoPE adaptation and
explicit decoder-config alignment. The training entry point retains its helper
signatures and logger through thin wrappers; model construction and initialization
source routing live in `src/speculators/train/model_init.py`.

The model-initialization module keeps config-only construction, pretrained weight
loading and fresh training initialization separate. Checkpoint overrides are
validated before constructing a model or loading its weights; saved model identity,
MTP-specific defaults, vocabulary mappings and verifier-weight loading retain their
existing order. The training script delegates through its original helper signatures
and supplies its logger.

`src/speculators/train/vocab_setup.py` owns vocabulary-mapping startup: explicit
files take precedence over cached mappings, then token-frequency regeneration,
then the full verifier vocabulary. The mapping algorithm remains in
`vocab_mapping.py`. Each cache file is published atomically; the pair is not a
two-file transaction. In distributed runs, only rank zero performs mapping I/O
and broadcasts either the mappings or the failure to every rank.

For `--from-pretrained`, parser defaults are not treated as checkpoint settings.
Only explicitly provided fields are checked before loading, then overrides are
merged with the saved config and validated together. Structural changes remain
forbidden; supported runtime-only overrides are applied only after the complete
candidate passes validation. Failed overrides leave the original config and
parsed arguments unchanged. Explicitly disabling a MMuse option still counts as
an override and cannot add MMuse options to a baseline checkpoint.

## Correction execution boundaries

`CausalCorrectionHead.forward` validates embeddings/positions and cache length,
then hidden feedback, then previous-logit inputs. `correction_inputs.py` holds
shape/presence checks without casting, encoding or mutating tensors. Head helpers
encode dense distributions before checking rank width, then assemble features in
the existing projection/addition order. Compact features are already masked;
Selector current-token embeddings are added before the hidden projection, and
hidden feedback stays differentiable. The head retains its causal layer/cache loop
and unchanged three-value return; parameter names and initialization are unchanged.
Construction registers optional logit, feedback and auxiliary-hidden projections
directly on the head, then creates the residual gate. Special position/residual/gate
initialization runs only after all modules exist, preserving seeded weights and RNG
consumption. Disabled projections remain ordinary `None` attributes; construction
helpers add no checkpoint namespace or persistent cache.

`MMuseDraftModel.forward` prepares the backbone, then `training_inputs.py` arranges
teacher-forced block views and predecessor IDs in a `TrainingBlocks` record without
casting, detaching or adding validation. Joint Selector/Correction conditioning
runs before a model dispatch helper chooses parallel teacher forcing or sequential
hidden feedback. Static conditioning uses the
preselected path; corrected conditioning still uses teacher-forced Selector rows
and previous ground-truth tokens in both training and validation. Current-token
embedding lookup is no-grad, while compact previous-distribution encoding retains
its trainable projection. Selector-only proposal handling remains separate.
The parallel path owns active-slot slicing and restoration of the reserved anchor:
with `sample_from_anchor=False`, Correction processes slots 1 onward without
renumbering them, preserves the anchor hidden/logits and pads its Correction state
with zeros. Hidden-output mode retains its final full-block LM-head projection in
the model's training dispatch helper; dual logit/hidden mode projects the corrected
full block in its existing Correction helper. Dispatch returns named logits/states/
hidden fields; `forward` still owns collaboration, validation outputs, confidence,
main metrics and ordered auxiliary losses. The public training return is unchanged.

`parallel_correction.py` holds dense/compact logit conditioning and residual-addition
formulas. Mode selection, no-grad token embedding lookup, Correction/LM-head calls
and causal-state padding remain in the model. Hidden residuals take a fresh slice
of the original block instead of reusing the head's input view, preserving gradient
accumulation order. The parallel helper's existing three-value return is unchanged.

The hidden-feedback path keeps its per-token loop, differentiable corrected-hidden
feedback and cache handoff in the model. `feedback_correction.py` slices each active
slot's conditioning, preferring compact rank features over dense teacher logits.
A reserved anchor seeds hidden feedback but creates no Correction cache entry.
Final logits use a separate model projection helper after all recurrent steps;
base-logit validation remains after the loop, and the three-value return is unchanged.

Autoregressive rollout stays separate. Its input validation and initial dense or
online compact logit feedback preparation use dedicated helpers. A token-embedding
helper handles frozen lookups and online Selector proposals; the loop keeps
collaboration, sampling, attention caches and previous-token selection.
A single-position Correction helper owns the head call, hidden/logit residuals and
projection selection. The private rollout path still permits gradients, while the
public rollout entry point remains no-grad.
Initial-feedback checks retain their position after optional fused base projection.

`rollout_state.py` owns a call-local feedback record, never attached to the model
or saved in checkpoints. It selects static compact, online compact or dense logit
inputs in that order and replaces feedback after sampling, before token mapping.
Logits are detached before reuse/encoding; hidden feedback and the compact encoder
retain their gradient paths. Reserved anchors update hidden feedback only, and
online encoding still runs when static compact inputs take precedence. Replaced
tensors are not retained in a feedback history.

`rollout_validation.py` owns the shape-only Selector token-ID and compact-feature
checks. `_validate_rollout_inputs` keeps the order: model/block dimensions, Selector
IDs, compact features/mask, then base logits. Helpers do not cast, detach, encode
or mutate tensors. Compact feature width remains the Correction head's concern;
features without a static token path remain accepted at this boundary.

`selector_inputs.py` separates static-path token alignment from sparse predecessor
rows. Tokens are mapped to the verifier vocabulary; sparse candidate IDs stay in
the draft vocabulary. The model still selects greedy/global proposal semantics
and detaches proposal logits before shifting. Compact encoding keeps its gradient
path, and reserved-anchor dense-logit checks remain after sparse encoding.

In `selector_runtime.py`, the path dispatcher owns validation, reserved-anchor
slots, detached output copies and the no-grad boundary. Separate greedy/global
helpers consume only active slots. Global search streams one position's K-by-K
lattice at a time, normalizes rows in FP32, backtracks using the existing candidate
index tie-breaking, then scores the realized rows again. Empty active blocks still
return before checking the search mode. Teacher-row scoring/loss remains separate
and differentiable; path selection does not change Top-K candidate order.

Collaboration, confidence and losses consume full-block outputs after the
teacher-forced paths merge. A validation-output helper keeps optional base
diagnostics and no-grad rollout separate from training; a diagnostics-only base
projection does not replace the original base-logit input to rollout. An auxiliary
loss helper adds the Selector objective before hidden alignment, preserving metric
names and detached logging values. Hidden-alignment targets are normalized without
gradients and, for a reserved anchor, shifted before gathering block positions.
Logged-metric filtering remains at the end of `forward`.

Changes to one execution path should retain the other paths' gradient boundaries
and projection counts. `make test-mmuse` automatically
includes all `test_mmuse_*.py` modules in the model and training unit-test directories:
Correction/caching/fusion, metrics, optional backbone features, parallel and rollout
contracts, and configuration checks. Shared activation-checkpointing, CLI/draft
initialization, RoPE configuration, vocabulary startup and real training-resume
checks are also included. Vocabulary startup tests cover atomic file publication
and two-rank CPU Gloo success and failure when that backend is available.

## Checkpoints and evaluation

Generic `SpeculatorModelConfig.from_pretrained` and
`SpeculatorModel.from_pretrained` resolve the model type from the saved config.
Model loading keeps three separate responsibilities in `src/speculators/model.py`:
`_resolve_pretrained_config` handles upstream external-checkpoint conversion and
legacy model identity migration; `from_pretrained` dispatches the concrete model
and delegates weight loading to Transformers; `_finalize_pretrained_load` restores
vocabulary mappings, verifier weights and model-specific missing weights in that
order. Loading diagnostics keep the same object and are returned only after those
hooks succeed. Errors still stop subsequent hooks; checkpoint keys, weight
ownership and public arguments are unchanged.

The existing `scripts/evaluate/dspark_offline_eval.py` and both debug probes accept
DSpark and MMuse without renaming the scripts or changing baseline evaluation.

Enhanced legacy **DSpark** configs are automatically migrated to the MMuse model
identity with a warning. Tensor names and architecture settings are preserved;
the next save records `speculators_model_type: mmuse`. Baseline DSpark and DFlash
configs keep their original identities. This is not a new training recipe or a
conversion of weights.

Enhanced legacy **DFlash** configs fail with an explicit migration error: merely
renaming them to MMuse could change their training/loss semantics. Use the
pre-split implementation for those checkpoints or start a fresh MMuse run.

For new code, import extensions from `speculators.models.mmuse`; the baseline
packages no longer provide their former internal extension-module paths.
