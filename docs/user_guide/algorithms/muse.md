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

### Configuration contract

`MuseOptions` in `src/speculators/models/muse/config.py` owns the extension fields,
including inherited backbone fields. The CLI and training factory obtain their
defaults from that schema; `validate_muse_options` applies the same scalar and
cross-option checks in fresh CLI runs, model construction and checkpoint override
handling. Reading a config alone does not construct modules or run these full
cross-option checks. Decoder/vocabulary-dependent checks remain in the modules
that know those dimensions.

For `--from-pretrained`, parser defaults are not treated as checkpoint settings.
Only explicitly provided fields are checked before loading, then overrides are
merged with the saved config and validated together. Structural changes remain
forbidden; supported runtime-only overrides are applied only after the complete
candidate passes validation. Failed overrides leave the original config and
parsed arguments unchanged. Explicitly disabling a Muse option still counts as
an override and cannot add Muse options to a baseline checkpoint.

## Correction execution boundaries

`MuseDraftModel.forward` prepares the backbone and optional Selector inputs, then
uses separate paths for parallel teacher forcing and sequential hidden feedback.
The parallel path owns active-slot slicing and restoration of the reserved anchor:
with `sample_from_anchor=False`, Correction processes slots 1 onward without
renumbering them, preserves the anchor hidden/logits and pads its Correction state
with zeros. Hidden-output mode retains its final full-block LM-head projection in
`forward`; dual logit/hidden mode projects the corrected full block in the helper.

Autoregressive rollout stays separate. Its input validation and initial dense or
online compact logit feedback preparation use dedicated helpers; the token loop
owns generation, caches and subsequent feedback updates. The private rollout path
still permits gradients, while the public rollout entry point remains no-grad.
Initial-feedback checks retain their position after optional fused base projection.

Collaboration, confidence and losses consume full-block outputs after the
teacher-forced paths merge. Changes to one execution path should retain the other
paths' gradient boundaries and projection counts. `make test-muse` includes parallel
Correction contracts, rollout/input-preparation checks, model and configuration
contracts, and training-resume checks.

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
