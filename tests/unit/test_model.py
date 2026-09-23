"""
Unit tests for the model module in the Speculators library.
"""

import tempfile
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators import (
    SpeculatorModel,
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
    reload_schemas,
)
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.mtp.core import MTPDraftModel
from speculators.models.peagle.core import PEagleDraftModel
from speculators.proposals import GreedyTokenProposalConfig

# ===== Test Helper Classes =====


@SpeculatorModelConfig.register("test_speculator_model")
class SpeculatorModelTestConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["test_speculator_model"] = "test_speculator_model"
    test_param: int = 123


@SpeculatorModel.register("test_speculator")
class SpeculatorTestModel(SpeculatorModel):
    config_class = SpeculatorModelTestConfig  # type: ignore[misc]

    def __init__(self, config: SpeculatorModelTestConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.test_module = nn.Linear(10, 10)
        self.post_init()  # type: ignore[attr-defined]

    def forward(self, *args, **kwargs):
        # Simple implementation for testing
        return {"logits": torch.randn(1, 10, 1000)}

    @classmethod
    def from_training_args(cls, verifier_config, **kwargs):
        """Create model from training arguments."""
        config = SpeculatorModelTestConfig(
            speculators_config=SpeculatorsConfig(
                algorithm="test_speculator",
                proposal_methods=[GreedyTokenProposalConfig()],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs.get("verifier_name_or_path")
                ),
            )
        )
        return cls(config=config)

    @staticmethod
    def get_trainer_kwargs(**kwargs):
        """Get training and validation kwargs."""
        return {}, {}


# Reload registries to include test classes
reload_schemas()


@pytest.fixture
def speculator_model_test_config():
    return SpeculatorModelTestConfig(
        test_param=456,
        speculators_config=SpeculatorsConfig(
            algorithm="test_algorithm",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None,
                architectures=["TestModel"],
            ),
        ),
    )


# ===== SpeculatorModel Class Attributes Tests =====


@pytest.mark.smoke
def test_speculator_model_class_attributes():
    assert SpeculatorModel.auto_package == "speculators.models"
    assert SpeculatorModel.registry_auto_discovery is True
    assert SpeculatorModel.config_class == SpeculatorModelConfig
    assert SpeculatorModel.base_model_prefix == "model"
    assert SpeculatorModel.main_input_name == "input_ids"


# ===== SpeculatorModel Registry Tests =====


@pytest.mark.smoke
def test_speculator_model_registry_contains_test_model():
    assert SpeculatorModel.registry is not None
    assert "test_speculator" in SpeculatorModel.registry
    assert SpeculatorModel.registry["test_speculator"] == SpeculatorTestModel


@pytest.mark.smoke
def test_speculator_model_registered_model_class_from_config(
    speculator_model_test_config,
):
    model_class = SpeculatorModel.registered_model_class_from_config(
        speculator_model_test_config
    )
    assert model_class == SpeculatorTestModel


@pytest.mark.sanity
def test_speculator_model_registered_model_class_from_config_invalid():
    with pytest.raises(
        TypeError, match="Expected config to be an instance of SpeculatorModelConfig"
    ):
        SpeculatorModel.registered_model_class_from_config("invalid_config")  # type: ignore[arg-type]

    config = SpeculatorModelConfig(
        speculators_model_type="test_speculator_model",
        test_param=456,
        speculators_config=SpeculatorsConfig(
            algorithm="test_algorithm",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="test/verifier",
                architectures=["TestModel"],
            ),
        ),
    )

    with pytest.raises(
        TypeError,
        match="Received a SpeculatorModelConfig instance but expected a subclass",
    ):
        SpeculatorModel.registered_model_class_from_config(config)

    class UnregisteredConfig(SpeculatorModelConfig):
        speculators_model_type: Literal["unregistered"] = "unregistered"

    config = UnregisteredConfig(
        speculators_config=SpeculatorsConfig(
            algorithm="test_algorithm",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="test/verifier",
                architectures=["TestModel"],
            ),
        )
    )

    with pytest.raises(
        ValueError, match="No registered model class found for config type"
    ):
        SpeculatorModel.registered_model_class_from_config(config)


# # ===== SpeculatorModel Initialization Tests =====


@pytest.mark.smoke
def test_speculator_model_initialization(speculator_model_test_config):
    model = SpeculatorTestModel(speculator_model_test_config)
    assert model.config == speculator_model_test_config


@pytest.mark.sanity
def test_speculator_model_initialization_invalid():
    # No config
    with pytest.raises(
        ValueError, match="Config must be provided to initialize a SpeculatorModel"
    ):
        SpeculatorModel(config=None)  # type: ignore[abstract, arg-type]

    # Invalid config type
    with pytest.raises(
        TypeError, match="Expected config to be an instance of SpeculatorModelConfig"
    ):
        SpeculatorModel(  # type: ignore[abstract]
            config="invalid_config",  # type: ignore[arg-type]
        )


# ===== SpeculatorModel from_pretrained Tests =====


@pytest.mark.smoke
def test_speculator_model_from_pretrained_config(speculator_model_test_config):
    state_dict = SpeculatorTestModel(speculator_model_test_config).state_dict()  # type: ignore[attr-defined]
    model = SpeculatorModel.from_pretrained(
        None, config=speculator_model_test_config, state_dict=state_dict
    )
    assert isinstance(model, SpeculatorTestModel)
    assert model.test_module is not None
    assert model.test_module.weight.abs().sum() > 0  # Ensure weights are initialized
    assert isinstance(model.config, SpeculatorModelTestConfig)
    assert model.config.speculators_model_type == "test_speculator_model"
    assert model.config.test_param == 456


@pytest.mark.smoke
def test_speculator_model_from_pretrained_local_marshalling(
    speculator_model_test_config,
):
    original_model = SpeculatorTestModel(speculator_model_test_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the model to a local directory
        original_model.save_pretrained(tmpdir)  # type: ignore[attr-defined]

        # Load the model from the local directory
        loaded_model = SpeculatorModel.from_pretrained(tmpdir)

        assert isinstance(loaded_model, SpeculatorTestModel)
        assert loaded_model.test_module is not None
        assert (
            pytest.approx(
                (loaded_model.test_module.weight - original_model.test_module.weight)
                .detach()
                .abs()
                .sum()
            )
            == 0
        )
        assert isinstance(loaded_model.config, SpeculatorModelTestConfig)
        assert loaded_model.config.speculators_model_type == "test_speculator_model"
        assert loaded_model.config.test_param == 456


@pytest.mark.smoke
def test_from_pretrained_loading_info_keeps_post_load_hooks(
    speculator_model_test_config,
    monkeypatch,
):
    original_model = SpeculatorTestModel(speculator_model_test_config)
    loaded_vocab = []
    loaded_verifier = []

    def load_vocab_mappings(self, t2d, d2t):
        loaded_vocab.append((self, t2d, d2t))

    def load_verifier_weights(self):
        loaded_verifier.append(self)

    monkeypatch.setattr(
        SpeculatorTestModel,
        "load_vocab_mappings",
        load_vocab_mappings,
        raising=False,
    )
    monkeypatch.setattr(
        SpeculatorTestModel,
        "load_verifier_weights",
        load_verifier_weights,
        raising=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_model.save_pretrained(tmpdir)  # type: ignore[attr-defined]
        t2d = torch.tensor([True])
        d2t = torch.tensor([0])
        loaded_model, loading_info = SpeculatorTestModel.from_pretrained(
            tmpdir,
            t2d=t2d,
            d2t=d2t,
            output_loading_info=True,
        )

    assert isinstance(loaded_model, SpeculatorTestModel)
    assert isinstance(loading_info, dict)
    assert len(loaded_vocab) == 1
    assert loaded_vocab[0][0] is loaded_model
    assert loaded_vocab[0][1] is t2d
    assert loaded_vocab[0][2] is d2t
    assert loaded_verifier == [loaded_model]


@pytest.fixture
def hf_loader(monkeypatch):
    """Record the concrete HF load without downloading or initializing weights."""
    loader = Mock()
    monkeypatch.setattr(
        PreTrainedModel,
        "from_pretrained",
        classmethod(loader),
    )
    return loader


@pytest.mark.parametrize("loader", [SpeculatorModel, SpeculatorTestModel])
@pytest.mark.parametrize("requested_info", [False, True])
@pytest.mark.parametrize("hf_returns_info", [False, True])
@pytest.mark.parametrize(
    "hooks",
    [(), ("vocab",), ("verifier",), ("missing",), ("vocab", "verifier", "missing")],
)
def test_pretrained_dispatch_preserves_options_and_hook_order(
    speculator_model_test_config,
    monkeypatch,
    hf_loader,
    tmp_path,
    loader,
    requested_info,
    hf_returns_info,
    hooks,
):
    config = speculator_model_test_config
    model = SpeculatorTestModel(config)
    info = {"missing_keys": ["optional.weight"], "unexpected_keys": []}
    hf_loader.return_value = (model, info) if hf_returns_info else model
    calls = []
    t2d, d2t = torch.tensor([True]), torch.tensor([0])
    hook_specs = {
        "vocab": ("load_vocab_mappings", (t2d, d2t)),
        "verifier": ("load_verifier_weights", ()),
        "missing": (
            "_prepare_missing_checkpoint_weights",
            (info if hf_returns_info else {},),
        ),
    }
    for name in hooks:
        method, _ = hook_specs[name]
        monkeypatch.setattr(
            model,
            method,
            lambda *args, name=name: calls.append((name, args)),
            raising=False,
        )
    options = {
        "cache_dir": tmp_path / "cache",
        "ignore_mismatched_sizes": True,
        "force_download": True,
        "local_files_only": True,
        "token": "unit-test-token",
        "revision": "test-revision",
        "use_safetensors": False,
        "weights_only": False,
        "dtype": torch.float32,
    }
    constructor_arg = object()
    result = loader.from_pretrained(
        tmp_path,
        constructor_arg,
        config=config,
        t2d=t2d,
        d2t=d2t,
        verifier="conversion-only-verifier",
        output_loading_info=requested_info,
        **options,
    )
    hf_loader.assert_called_once_with(
        SpeculatorTestModel,
        tmp_path,
        constructor_arg,
        config=config,
        output_loading_info=True,
        **options,
    )
    assert [name for name, _ in calls] == list(hooks)
    for name, args in calls:
        expected = hook_specs[name][1]
        assert len(args) == len(expected)
        if name == "vocab":
            assert args[0] is t2d
            assert args[1] is d2t
        elif name == "missing":
            assert args[0] == expected[0]
            if hf_returns_info:
                assert args[0] is info
    if requested_info:
        assert result[0] is model
        assert result[1] == (info if hf_returns_info else {})
        if hf_returns_info:
            assert result[1] is info
    else:
        assert result is model


@pytest.mark.parametrize("failing_hook", [0, 1, 2])
def test_pretrained_hook_failure_stops_subsequent_work(
    speculator_model_test_config, monkeypatch, hf_loader, failing_hook
):
    model = SpeculatorTestModel(speculator_model_test_config)
    hf_loader.return_value = (model, {})
    calls = []
    failure = ValueError("incompatible checkpoint")

    def hook(index, *args):
        calls.append(index)
        if index == failing_hook:
            raise failure

    for index, name in enumerate(
        (
            "load_vocab_mappings",
            "load_verifier_weights",
            "_prepare_missing_checkpoint_weights",
        )
    ):
        monkeypatch.setattr(
            model,
            name,
            lambda *args, index=index: hook(index, *args),
            raising=False,
        )
    with pytest.raises(ValueError, match="incompatible checkpoint") as raised:
        SpeculatorModel.from_pretrained("unused", config=speculator_model_test_config)
    assert raised.value is failure
    assert calls == list(range(failing_hook + 1))


@pytest.mark.parametrize("loader", [SpeculatorModel, SpeculatorTestModel])
@pytest.mark.parametrize("external", [False, True])
def test_pretrained_config_resolution_converts_only_external_checkpoints(
    speculator_model_test_config, monkeypatch, hf_loader, tmp_path, loader, external
):
    config = speculator_model_test_config
    model = SpeculatorTestModel(config)
    hf_loader.return_value = (model, {})
    checkpoint = tmp_path / "original"
    converted = str(tmp_path / "converted")
    raw_config = {"dflash_config": {}} if external else config.to_dict()
    inspect_config = Mock(return_value=(raw_config, {}))
    load_config = Mock(return_value=config)
    convert = Mock(return_value=converted)
    monkeypatch.setattr(PretrainedConfig, "get_config_dict", inspect_config)
    monkeypatch.setattr(SpeculatorModelConfig, "from_pretrained", load_config)
    monkeypatch.setattr(
        "speculators.convert.entrypoints.maybe_convert_external_checkpoint", convert
    )
    options = {
        "cache_dir": tmp_path / "cache",
        "force_download": False,
        "local_files_only": True,
        "token": "unit-test-token",
        "revision": "test-revision",
    }
    result = loader.from_pretrained(checkpoint, verifier="target", **options)
    assert result is model
    inspect_config.assert_called_once_with(checkpoint, cache_dir=options["cache_dir"])
    if external:
        convert.assert_called_once_with(
            checkpoint,
            verifier="target",
            cache_dir=options["cache_dir"],
            config_dict=raw_config,
        )
    else:
        convert.assert_not_called()
    resolved_path = converted if external else checkpoint
    load_config.assert_called_once_with(resolved_path, **options)
    assert hf_loader.call_count == 1
    assert hf_loader.call_args.args == (SpeculatorTestModel, resolved_path)
    assert hf_loader.call_args.kwargs["config"] is config


def test_pretrained_explicit_config_skips_detection_and_preserves_state_dict(
    speculator_model_test_config, monkeypatch, hf_loader
):
    model = SpeculatorTestModel(speculator_model_test_config)
    hf_loader.return_value = (model, {})
    unexpected_read = Mock(
        side_effect=AssertionError("explicit config must not read files")
    )
    monkeypatch.setattr(PretrainedConfig, "get_config_dict", unexpected_read)
    state_dict = model.state_dict()
    assert (
        SpeculatorModel.from_pretrained(
            None, config=speculator_model_test_config, state_dict=state_dict
        )
        is model
    )
    unexpected_read.assert_not_called()
    assert hf_loader.call_args.kwargs["state_dict"] is state_dict


@pytest.mark.smoke
def test_speculator_model_from_pretrained(
    speculator_model_test_config,
):
    state_dict = SpeculatorTestModel(speculator_model_test_config).state_dict()  # type: ignore[attr-defined]
    model = SpeculatorModel.from_pretrained(
        None, config=speculator_model_test_config, state_dict=state_dict
    )
    assert isinstance(model, SpeculatorTestModel)
    assert isinstance(model.config, SpeculatorModelTestConfig)
    assert model.config.speculators_model_type == "test_speculator_model"
    assert model.config.test_param == 456


@pytest.mark.sanity
def test_speculator_model_from_pretrained_invalid(speculator_model_test_config):
    with pytest.raises(
        ValueError,
        match="Either `config` or `pretrained_model_name_or_path` must be provided",
    ):
        SpeculatorModel.from_pretrained(None)

    with pytest.raises(
        ValueError,
        match="Either `pretrained_model_name_or_path` or `state_dict` must be provided",
    ):
        SpeculatorModel.from_pretrained(None, config=speculator_model_test_config)

    with pytest.raises(
        TypeError, match="Expected config to be an instance of SpeculatorModelConfig"
    ):
        SpeculatorModel.from_pretrained("test/path", config="invalid_config")

    with pytest.raises(OSError, match="'path/does/not/exist'."):
        SpeculatorModel.from_pretrained(
            "path/does/not/exist", config=speculator_model_test_config
        )


# # ===== SpeculatorModel Forward Method Tests =====


@pytest.mark.smoke
def test_speculator_model_forward_concrete(
    speculator_model_test_config,
):
    model = SpeculatorTestModel(speculator_model_test_config, verifier=None)
    result = model.forward()

    assert "logits" in result
    assert result["logits"].shape == (1, 10, 1000)


@pytest.mark.smoke
def test_speculator_model_forward_abstract(speculator_model_test_config):
    model = SpeculatorModel(  # type: ignore[abstract]
        speculator_model_test_config, verifier=None, verifier_attachment_mode=None
    )

    with pytest.raises(
        NotImplementedError, match="The forward method is only supported on concrete"
    ):
        model.forward()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_class",
    [Eagle3DraftModel, DFlashDraftModel, PEagleDraftModel, MTPDraftModel],
)
def test_save_ignore_keys_are_ignored_on_load_missing(model_class):
    """Weights excluded from saved checkpoints (e.g. verifier_lm_head, which is
    reloaded from the verifier via load_verifier_weights) must also be ignored when
    missing on load. Otherwise loading an initialized/trained checkpoint flags the
    absent key as missing.
    """
    save_ignore = set(getattr(model_class, "_keys_to_ignore_on_save", None) or [])
    load_missing_ignore = set(
        getattr(model_class, "_keys_to_ignore_on_load_missing", None) or []
    )

    not_ignored_on_load = save_ignore - load_missing_ignore
    assert not not_ignored_on_load, (
        f"{model_class.__name__} excludes {sorted(not_ignored_on_load)} from saved "
        "checkpoints but does not list them in _keys_to_ignore_on_load_missing; "
        "loading a checkpoint will raise on the absent key(s)."
    )
