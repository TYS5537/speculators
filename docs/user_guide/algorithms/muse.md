# DFlash, DSpark and Muse

The model directories separate the baseline algorithms from experimental
architecture extensions:

- `src/speculators/models/dflash/`: DFlash block drafting, verifier context,
  attention and vocabulary handling.
- `src/speculators/models/dspark/`: the DSpark baseline, adding Markov/RNN
  sequential heads and acceptance confidence on top of DFlash.
- `src/speculators/models/muse/`: backbone residual and gated-fusion extensions,
  dynamic convolution, candidate Selector, causal Correction and collaboration.

Shared training, checkpointing, numerical safeguards and data contracts are not
architecture extensions and remain in their existing shared modules.

Within Muse, `selector.py` defines the trainable candidate scorer;
`selector_runtime.py` owns Top-K candidate selection, greedy/global path search,
proposal logits and the restricted-Top-K teacher loss. Its stateless mixin keeps
the existing model methods available without adding a module or changing the
`candidate_selector.*` checkpoint keys. `backbone.py` retains module construction,
initialization and checkpoint checks, as well as the vocabulary mapping shared
by Selector and Correction.

## Training

Use `--speculator-type dflash` or `--speculator-type dspark` for the baselines.
Use `--speculator-type muse` with the existing extension flags for enhanced runs:

```bash
python scripts/train.py \
  --verifier-name-or-path /path/to/verifier \
  --data-path /path/to/data \
  --save-path /path/to/output \
  --speculator-type muse \
  --enable-correction-head \
  --dflash-gated-layer-fusion
```

Muse inherits DSpark's training defaults (block size 7, five decoder layers,
10 epochs, CE/TV weights 0.1/0.9). Extensions remain opt-in; choosing Muse alone
does not silently enable Correction or change its parameters. Extension flags
retain their existing names, including the `dflash-` and `dflash2-` prefixes.
Passing them with a baseline type for a fresh run produces an error directing you
to Muse. When restoring, the saved model type determines which options apply.

### Training shell examples

Experiment values stay in each `examples/train/` recipe. Two narrowly shared
helpers under `examples/train/common/` remove identical setup and argument lists:

- `ascend_training_env.sh` defines the explicitly called Ascend training preset
  used by the Qwen online/trainer and DSV4 trainer scripts. It does not configure
  standalone target servers, device allocation or proxies.
- `dspark_online_args.sh` builds the baseline training argument array for the two
  full online Qwen DSpark examples. The Ascend caller adds its attention option;
  the trainer-only Qwen Muse and DSV4 recipes keep their own argument assembly.

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

`MuseOptions` in `src/speculators/models/muse/config.py` owns the extension fields,
including inherited backbone fields. The CLI and training factory obtain their
defaults from that schema; `validate_muse_options` applies the same scalar and
cross-option checks in fresh CLI runs, model construction and checkpoint override
handling. Reading a config alone does not construct modules or run these full
cross-option checks. Decoder/vocabulary-dependent checks remain in the modules
that know those dimensions.

`src/speculators/train/cli.py` owns parser construction, algorithm-specific default
resolution and post-parse validation as separate steps. `scripts/train.py` retains
the zero-argument `parse_args()` entry point. The reusable `parse_train_args(argv)`
uses the same argument list for initial parsing and all explicit-option tracking;
omitting `argv` continues to use the process command line. Validation order and
the distinction between parser errors and ordinary validation exceptions are
preserved.

`src/speculators/train/muse_args.py` registers the Muse backbone/Selector and
Correction CLI options using one defaults mapping supplied by the parser builder.
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
parsed arguments unchanged. Explicitly disabling a Muse option still counts as
an override and cannot add Muse options to a baseline checkpoint.

## Correction execution boundaries

`MuseDraftModel.forward` prepares the backbone, then delegates joint
Selector/Correction conditioning to a dedicated helper before choosing parallel
teacher forcing or sequential hidden feedback. Static conditioning uses the
preselected path; corrected conditioning still uses teacher-forced Selector rows
and previous ground-truth tokens in both training and validation. Current-token
embedding lookup is no-grad, while compact previous-distribution encoding retains
its trainable projection. Selector-only proposal handling remains separate.
The parallel path owns active-slot slicing and restoration of the reserved anchor:
with `sample_from_anchor=False`, Correction processes slots 1 onward without
renumbering them, preserves the anchor hidden/logits and pads its Correction state
with zeros. Hidden-output mode retains its final full-block LM-head projection in
`forward`; dual logit/hidden mode projects the corrected full block in the helper.

Autoregressive rollout stays separate. Its input validation and initial dense or
online compact logit feedback preparation use dedicated helpers; the token loop
keeps Selector conditioning, collaboration, sampling, caches and feedback updates.
A single-position Correction helper owns the head call, hidden/logit residuals and
projection selection. The private rollout path still permits gradients, while the
public rollout entry point remains no-grad.
Initial-feedback checks retain their position after optional fused base projection.

Collaboration, confidence and losses consume full-block outputs after the
teacher-forced paths merge. A validation-output helper keeps optional base
diagnostics and no-grad rollout separate from training; a diagnostics-only base
projection does not replace the original base-logit input to rollout. An auxiliary
loss helper adds the Selector objective before hidden alignment, preserving metric
names and detached logging values. Hidden-alignment targets are normalized without
gradients and, for a reserved anchor, shifted before gathering block positions.
Logged-metric filtering remains at the end of `forward`.

Changes to one execution path should retain the other paths' gradient boundaries
and projection counts. `make test-muse` automatically
includes all `test_muse_*.py` modules in the model and training unit-test directories:
Correction/caching/fusion, metrics, optional backbone features, parallel and rollout
contracts, and configuration checks. Shared activation-checkpointing, CLI/draft
initialization, RoPE configuration, vocabulary startup and real training-resume
checks are also included. Vocabulary startup tests cover atomic file publication
and two-rank CPU Gloo success and failure when that backend is available.

## Checkpoints and evaluation

Generic `SpeculatorModelConfig.from_pretrained` and
`SpeculatorModel.from_pretrained` resolve the model type from the saved config.
The existing `scripts/evaluate/dspark_offline_eval.py` and both debug probes accept
DSpark and Muse without renaming the scripts or changing baseline evaluation.

Enhanced legacy **DSpark** configs are automatically migrated to the Muse model
identity with a warning. Tensor names and architecture settings are preserved;
the next save records `speculators_model_type: muse`. Baseline DSpark and DFlash
configs keep their original identities. This is not a new training recipe or a
conversion of weights.

Enhanced legacy **DFlash** configs fail with an explicit migration error: merely
renaming them to Muse could change their training/loss semantics. Use the
pre-split implementation for those checkpoints or start a fresh Muse run.

For new code, import extensions from `speculators.models.muse`; the baseline
packages no longer provide their former internal extension-module paths.
