"""Architecture recipe defaults and explicit-option precedence at the train CLI."""

import argparse

import pytest

from speculators.train import cli

_RECIPE_VALUES = {
    "block_size": 8,
    "dflash_decay_gamma": 4.0,
    "epochs": 20,
    "loss_fn": "kl_div",
    "num_layers": 1,
}
_SHARED_FIELDS = ("draft_arch", "norm_before_fc", "norm_output", "muon_lr")


def _namespace(algorithm):
    return argparse.Namespace(
        speculator_type=algorithm,
        **_RECIPE_VALUES,
        **dict.fromkeys(_SHARED_FIELDS),
        lr=0.003,
        sample_from_anchor=None,
        enable_correction_head=False,
        _provided_model_config_dests={"sample_from_anchor"},
    )


def _provided(mask, fields):
    return {name for index, name in enumerate(fields) if mask & (1 << index)}


@pytest.mark.parametrize("algorithm", ["dspark", "mmuse"])
@pytest.mark.parametrize("explicit_mask", range(32))
def test_recipe_preserves_every_subset_of_explicit_parser_default_values(
    algorithm, explicit_mask
):
    args = _namespace(algorithm)
    provided = _provided(explicit_mask, _RECIPE_VALUES)
    original_provided = provided.copy()
    provenance = args._provided_model_config_dests
    expected = {
        "block_size": 7,
        "dflash_decay_gamma": 7.0,
        "epochs": 10,
        "loss_fn": '{"ce": 0.1, "tv": 0.9}',
        "num_layers": 5,
    }
    expected.update({name: _RECIPE_VALUES[name] for name in provided})
    if "dflash_decay_gamma" not in provided:
        expected["dflash_decay_gamma"] = float(expected["block_size"])

    assert cli._apply_training_defaults(args, provided) is None

    assert {name: getattr(args, name) for name in expected} == expected
    assert type(args.dflash_decay_gamma) is float
    assert args._provided_model_config_dests is provenance
    assert provenance == {"sample_from_anchor"}
    assert provided == original_provided
    assert args.sample_from_anchor is None
    assert args.enable_correction_head is False
    snapshot = vars(args).copy()
    cli._apply_training_defaults(args, provided)
    assert vars(args) == snapshot


@pytest.mark.parametrize(
    "algorithm", ["eagle3", "dflash", "dspark", "mmuse", "peagle", "mtp"]
)
@pytest.mark.parametrize("explicit_mask", range(16))
def test_shared_defaults_fill_only_none_preserving_false_and_zero(
    algorithm, explicit_mask
):
    args = _namespace(algorithm)
    eagle = algorithm == "eagle3"
    overrides = {
        "draft_arch": "qwen3" if eagle else "llama",
        "norm_before_fc": not eagle,
        "norm_output": not eagle,
        "muon_lr": 0.0,
    }
    explicit = _provided(explicit_mask, _SHARED_FIELDS)
    for name in explicit:
        setattr(args, name, overrides[name])
    expected = {
        "draft_arch": "llama" if eagle else "qwen3",
        "norm_before_fc": eagle,
        "norm_output": eagle,
        "muon_lr": 10 * args.lr,
    }
    expected.update({name: overrides[name] for name in explicit})
    # Shared defaults use None checks, not membership in the recipe's provided set.
    cli._apply_training_defaults(args, set(_RECIPE_VALUES))
    assert {name: getattr(args, name) for name in _SHARED_FIELDS} == expected
    assert {name: getattr(args, name) for name in _RECIPE_VALUES} == _RECIPE_VALUES


class _RecordingNamespace(argparse.Namespace):
    def __setattr__(self, name, value):
        if "assignments" in vars(self):
            self.assignments.append(name)
        super().__setattr__(name, value)


@pytest.mark.parametrize("algorithm", ["dspark", "mmuse", "eagle3", "dflash"])
def test_recipe_assignments_still_precede_shared_defaults_in_order(algorithm):
    args = _RecordingNamespace(**vars(_namespace(algorithm)))
    args.assignments = []
    cli._apply_training_defaults(args, set())
    recipe = list(_RECIPE_VALUES) if algorithm in {"dspark", "mmuse"} else []
    assert args.assignments == recipe + list(_SHARED_FIELDS)


@pytest.mark.parametrize("explicit_block", [False, True])
def test_gamma_conversion_failure_still_stops_before_remaining_defaults(
    monkeypatch, explicit_block
):
    args = _namespace("mmuse")
    if explicit_block:
        args.block_size = "invalid-block"
    else:
        monkeypatch.setattr(cli, "DSPARK_PAPER_BLOCK_SIZE", "invalid-block")
    before = vars(args).copy()
    provided = {"block_size"} if explicit_block else set()
    with pytest.raises(ValueError, match="could not convert string to float"):
        cli._apply_training_defaults(args, provided)
    assert vars(args) == {**before, "block_size": "invalid-block"}


def test_paper_recipe_reads_live_constants_without_freezing_them(monkeypatch):
    monkeypatch.setattr(cli, "DSPARK_PAPER_BLOCK_SIZE", 13)
    monkeypatch.setattr(cli, "DSPARK_PAPER_EPOCHS", 2)
    monkeypatch.setattr(cli, "DSPARK_PAPER_LOSS_FN", "ce")
    monkeypatch.setattr(cli, "DSPARK_PAPER_NUM_LAYERS", 3)
    args = _namespace("mmuse")
    cli._apply_training_defaults(args, set())
    assert (
        args.block_size,
        args.dflash_decay_gamma,
        args.epochs,
        args.loss_fn,
        args.num_layers,
    ) == (13, 13.0, 2, "ce", 3)


@pytest.mark.parametrize("algorithm", ["dspark", "mmuse", "muse"])
@pytest.mark.parametrize("spelling", ["split", "equals", "abbreviated"])
def test_parser_tracks_explicit_raw_defaults_in_all_supported_spellings(
    monkeypatch, algorithm, spelling
):
    monkeypatch.setattr("sys.argv", ["train.py", "--invalid-ambient-argument"])
    argv = ["--verifier-name-or-path", "unused", "--speculator-type", algorithm]
    for name, value in _RECIPE_VALUES.items():
        flag = "--" + name.replace("_", "-")
        if spelling == "abbreviated":
            flag = flag[:-1]
        argv.extend([f"{flag}={value}"] if spelling == "equals" else [flag, str(value)])
    argv.extend(["--no-norm-before-fc", "--no-norm-output", "--lr=0", "--muon-lr=0"])
    snapshot = argv.copy()
    args = cli.parse_train_args(argv)
    assert {name: getattr(args, name) for name in _RECIPE_VALUES} == _RECIPE_VALUES
    assert args.norm_before_fc is args.norm_output is False
    assert args.lr == args.muon_lr == 0.0
    assert args.speculator_type == ("mmuse" if algorithm == "muse" else algorithm)
    assert args._provided_model_config_dests == {"block_size"}
    assert argv == snapshot


@pytest.mark.parametrize("algorithm", ["dspark", "mmuse", "muse"])
@pytest.mark.parametrize(
    "source",
    [
        [],
        ["--from-pretrained", "unused-checkpoint"],
        ["--draft-config", "unused-decoder"],
    ],
)
@pytest.mark.parametrize("block", [None, 4])
def test_cli_defaults_do_not_claim_checkpoint_overrides_or_enable_features(
    algorithm, source, block
):
    argv = [
        "--verifier-name-or-path",
        "unused",
        "--speculator-type",
        algorithm,
        "--no-sample-from-anchor",
        "--no-enable-confidence-head",
        *source,
    ]
    expected_provided = {"sample_from_anchor", "enable_confidence_head"}
    if block is not None:
        argv.extend(["--block-size", str(block)])
        expected_provided.add("block_size")
    args = cli.parse_train_args(argv)
    assert args.block_size == (7 if block is None else block)
    assert args.dflash_decay_gamma == float(args.block_size)
    assert args.num_layers == 5
    assert args._provided_model_config_dests == expected_provided
    assert args.sample_from_anchor is False
    assert args.enable_confidence_head is False
    assert args.enable_correction_head is False
    assert args.dflash2_candidate_selector is False
    assert args.dflash_gated_layer_fusion is False
