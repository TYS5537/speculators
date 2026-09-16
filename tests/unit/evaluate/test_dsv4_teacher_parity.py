"""Real CPU tensors/safetensors; fake target transport, never an A3 parity claim."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from speculators_dsv4 import contract, offline, parity
from speculators_dsv4.block_protocol import BLOCK_PROTOCOL_VERSION, BLOCK_REQUEST_KEY


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "target"
    root.mkdir()
    config = {"hidden_size": 4, "vocab_size": 5, "rms_norm_eps": 1e-6}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "embed.weight": torch.ones(5, 4, dtype=torch.bfloat16),
        "head.weight": torch.tensor(
            [
                [1, 0, 0, -1],
                [0, 1, -1, 0],
                [-1, 1, 0, 0],
                [0, -1, 1, 0],
                [0, 0, -1, 1],
            ],
            dtype=torch.bfloat16,
        ),
        "norm.weight": torch.tensor([0.5, 1, 1.5, 2], dtype=torch.bfloat16),
    }
    save_file(tensors, str(root / "model.safetensors"))
    # Tiny geometry only; keep all real safetensors/index/IO/signature checks.
    monkeypatch.setattr(contract, "validate_config", lambda config: None)
    report = contract.inspect_checkpoint(root)
    teacher = parity.load_teacher_head(report)
    hidden = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [-1, 2, -3, 4],
            [2, -4, 1, -3],
            [3, 1, -2, 4],
            [-2, -1, 4, 3],
        ],
        dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        root=root,
        report=report,
        teacher=teacher,
        tensors=tensors,
        hidden=hidden,
        directory=tmp_path / "hs",
    )


@pytest.fixture
def service(checkpoint, monkeypatch):
    case = checkpoint
    case.calls = []
    case.paths = []
    case.packet_hook = lambda packet: None
    case.response_hook = lambda response: None
    contract.ensure_manifest(
        case.directory, contract.make_manifest(case.report, [1, 11]), create=True
    )

    def create(**kwargs):
        case.calls.append(kwargs)
        prefix = kwargs["prompt"]
        request_id = kwargs["extra_body"]["request_id"]
        path = case.directory / f"cmpl-{request_id}-0.safetensors"
        case.paths.append(path)
        hidden = torch.stack(
            [
                -case.hidden[: len(prefix)],
                case.hidden[: len(prefix)].roll(1, -1),
                case.hidden[: len(prefix)],
            ],
            dim=1,
        )
        block = (
            kwargs["extra_body"].get("kv_transfer_params", {}).get(BLOCK_REQUEST_KEY)
        )
        transfer = {"hidden_states_path": str(path)}
        top = None
        if block:
            start, hidden_start = block["logits_start"], block["hidden_start"]
            packet = {
                "token_ids": torch.tensor(prefix, dtype=torch.int64),
                "verification_metadata": torch.tensor(
                    [BLOCK_PROTOCOL_VERSION, len(prefix), start, hidden_start]
                ),
                "layer_ids": torch.tensor([1, 11, 43]),
                "logprobs": case.teacher.logprobs(case.hidden[start : len(prefix)]),
                "hidden_states": hidden[hidden_start:].contiguous(),
            }
            transfer["dsv4_block_verify_version"] = BLOCK_PROTOCOL_VERSION
        else:
            packet = {"token_ids": torch.tensor(prefix), "hidden_states": hidden}
            top = [
                {
                    f"token_id:{index}": value
                    for index, value in enumerate(
                        case.teacher.logprobs(
                            case.hidden[len(prefix) - 1 : len(prefix)]
                        )[0].tolist()
                    )
                }
            ]
        case.packet_hook(packet)
        save_file(packet, str(path))
        response = SimpleNamespace(
            id=f"cmpl-{request_id}",
            model="teacher-target",
            choices=[
                SimpleNamespace(
                    prompt_token_ids=prefix, logprobs=SimpleNamespace(top_logprobs=top)
                )
            ],
            kv_transfer_params=transfer,
        )
        case.response_hook(response)
        return response

    # Same on-disk safetensors as real transfers, without connector dependencies.
    connectors = ModuleType("hs_connectors")
    connectors.FileTransfer = lambda directory: SimpleNamespace(
        get_generated=load_file,
        delete=lambda handle: Path(handle).unlink(),
    )
    transfer_module = ModuleType("hs_connectors.transfer")
    transfer_module.wait_for_lock = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "hs_connectors", connectors)
    monkeypatch.setitem(sys.modules, "hs_connectors.transfer", transfer_module)
    case.client = SimpleNamespace(completions=SimpleNamespace(create=create))
    case.target = offline.DSV4OfflineTarget.from_contract(
        case.report,
        [1, 11],
        hidden_states_path=case.directory,
        client=case.client,
        model_name="teacher-target",
        max_model_len=64,
        verification_mode="block",
    )
    return case


def test_frozen_head_matches_trainer_rounding(checkpoint):
    case = checkpoint
    values = case.hidden.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    # Actual trainer: parameters are FP32, Qwen3RMSNorm output stays FP32,
    # but the Linear runs under BF16 autocast.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = torch.nn.functional.linear(
            normalized * case.tensors["norm.weight"].float(),
            case.tensors["head.weight"].float(),
        )
    expected = torch.log_softmax(logits.float(), dim=-1)
    torch.testing.assert_close(
        case.teacher.logprobs(case.hidden), expected, rtol=0, atol=0
    )
    assert case.teacher.head_weight.dtype == torch.bfloat16
    assert case.teacher.norm_weight.dtype == torch.float32
    assert not case.teacher.head_weight.requires_grad


def test_norm_parameter_dtype_changes_rounding_under_autocast():
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(5, 4, generator=generator).bfloat16()
    norm = torch.randn(4, generator=generator).bfloat16()
    head = torch.randn(6, 4, generator=generator).bfloat16()
    outputs = []
    for dtype in (torch.float32, torch.bfloat16):
        checker = parity.FrozenTeacherHead(norm.to(dtype), head, 1e-6)
        inputs = hidden.to(dtype)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            normalized = (
                inputs.float()
                * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + 1e-6)
            ).to(dtype) * norm.to(dtype)
            reference = torch.nn.functional.linear(normalized, head.float())
        expected = torch.log_softmax(reference.float(), -1)
        actual = checker.logprobs(hidden)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        outputs.append(actual)
    assert not torch.equal(*outputs)


@pytest.mark.parametrize("mode", ["reference", "block"])
def test_roundtrip_probabilities_and_teacher_alignment(service, mode):
    case = service
    case.target.verification_mode = mode
    result = parity.check_teacher_parity(
        case.target,
        case.teacher,
        [0, 1, 2, 3, 4, 0],
        positions=[0, 2, 3, 5],
        position_chunk_size=2,
    )
    assert result["passed"]
    assert result["max_tv"] == 0
    assert result["argmax_agreement"] == 1
    assert [row["predicts_position"] for row in result["rows"]] == [1, 3, 4, 6]
    assert len(case.calls) == (4 if mode == "reference" else 3)
    assert all(not path.exists() for path in case.paths)
    if mode == "block":
        assert [
            call["extra_body"]["kv_transfer_params"][BLOCK_REQUEST_KEY]["logits_start"]
            for call in case.calls
        ] == [0, 2, 5]


def test_keep_artifacts(service):
    service.target.keep_hidden_states = True
    parity.check_teacher_parity(
        service.target, service.teacher, [0, 1], tail_positions=1
    )
    assert service.paths[0].is_file()


@pytest.mark.parametrize(
    "corruption",
    [
        "teacher_slot",
        "off_by_one",
        "nan",
        "wrong_token",
        "wrong_dtype",
        "wrong_slot_id",
    ],
)
def test_bad_hs_or_packet_fails_closed(service, corruption):
    def corrupt(packet):
        if corruption == "teacher_slot":
            packet["hidden_states"][:, -1] = packet["hidden_states"][:, 0]
        elif corruption == "off_by_one":
            packet["hidden_states"] = packet["hidden_states"].roll(1, 0)
        elif corruption == "nan":
            packet["hidden_states"][0, -1, 0] = math.nan
        elif corruption == "wrong_token":
            packet["token_ids"][0] = 4
        elif corruption == "wrong_dtype":
            packet["hidden_states"] = packet["hidden_states"].float()
        else:
            packet["layer_ids"][-1] = 42

    service.packet_hook = corrupt
    if corruption in {"teacher_slot", "off_by_one"}:
        result = parity.check_teacher_parity(service.target, service.teacher, [0, 1, 2])
        assert not result["passed"]
        assert result["threshold_violations"]
    else:
        with pytest.raises(ValueError):
            parity.check_teacher_parity(service.target, service.teacher, [0, 1, 2])
    assert all(not path.exists() for path in service.paths)


def test_wrong_epsilon_is_detected(service):
    wrong = parity.FrozenTeacherHead(
        service.teacher.norm_weight, service.teacher.head_weight, 10.0
    )
    result = parity.check_teacher_parity(service.target, wrong, [0, 1, 2])
    assert not result["passed"]
    assert result["max_tv"] > 0.02


@pytest.mark.parametrize("mode", ["reference", "block"])
def test_wrong_alias_is_rejected_without_deleting_unowned_artifact(service, mode):
    service.target.verification_mode = mode
    service.response_hook = lambda response: setattr(
        response, "model", "different-target"
    )
    with pytest.raises(ValueError, match="model alias"):
        parity.check_teacher_parity(service.target, service.teacher, [0, 1])
    assert service.paths[0].exists()


def test_no_protocol_fallback(service):
    service.response_hook = lambda response: response.kv_transfer_params.pop(
        "dsv4_block_verify_version"
    )
    with pytest.raises(ValueError, match="protocol version"):
        parity.check_teacher_parity(service.target, service.teacher, [0, 1])
    assert len(service.calls) == 1
    assert service.paths[0].exists()


def test_manifest_is_required(checkpoint):
    with pytest.raises(ValueError, match="Missing"):
        offline.DSV4OfflineTarget.from_contract(
            checkpoint.report,
            [1, 11],
            hidden_states_path=checkpoint.directory,
            client=object(),
            model_name="target",
            max_model_len=64,
        )


def test_checkpoint_changed_since_audit_fails(checkpoint):
    changed = dict(checkpoint.tensors)
    changed["head.weight"] = changed["head.weight"] + 1
    save_file(changed, str(checkpoint.root / "model.safetensors"))
    with pytest.raises(ValueError, match="changed"):
        parity.load_teacher_head(checkpoint.report)


def test_nonfinite_weight_fails(checkpoint):
    changed = dict(checkpoint.tensors)
    changed["head.weight"][0, 0] = math.nan
    save_file(changed, str(checkpoint.root / "model.safetensors"))
    with pytest.raises(ValueError, match="non-finite"):
        parity.load_teacher_head(checkpoint.report)


def test_logit_constant_offset_is_ignored():
    logits = torch.tensor([[1.0, 2.0, 4.0]])
    rows = parity.compare_rows(
        torch.log_softmax(logits, -1),
        torch.log_softmax(logits + 42, -1),
        [7],
        parity.ParityThresholds(),
    )
    assert rows[0]["max_abs_logprob_error"] == 0


def test_argmax_threshold_independent_of_tv(service):
    wrong = parity.FrozenTeacherHead(
        service.teacher.norm_weight,
        -service.teacher.head_weight,
        service.teacher.epsilon,
    )
    result = parity.check_teacher_parity(
        service.target,
        wrong,
        [0, 1, 2],
        thresholds=parity.ParityThresholds(
            max_tv=1, max_logprob_error=100, min_argmax_agreement=1
        ),
    )
    assert not result["passed"]
    assert result["threshold_violations"] == []
    assert result["argmax_mismatch_positions"]


@pytest.mark.parametrize("tokens", [[True], [-1], [5], []])
def test_invalid_input_no_request(service, tokens):
    with pytest.raises(ValueError):
        parity.check_teacher_parity(service.target, service.teacher, tokens)
    assert not service.calls


def _cli():
    path = Path(__file__).resolve().parents[3] / "scripts/check_dsv4_teacher.py"
    spec = importlib.util.spec_from_file_location("teacher_parity_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("passed", "expected"), [(True, 0), (False, 1)])
def test_cli_emits_report_and_nonzero_on_threshold_failure(
    service, monkeypatch, tmp_path, passed, expected
):
    cli = _cli()
    openai_module = ModuleType("openai")

    class Client:
        models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="teacher-target")])
        )

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    openai_module.OpenAI = Client
    httpx_module = ModuleType("httpx")
    httpx_module.Client = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "httpx", httpx_module)
    monkeypatch.setattr(
        cli.DSV4OfflineTarget, "from_contract", lambda *args, **kwargs: service.target
    )
    monkeypatch.setattr(
        cli, "check_teacher_parity", lambda *args, **kwargs: {"passed": passed}
    )
    output = tmp_path / "report.json"
    rc = cli.main(
        [
            "--model",
            str(service.root),
            "--hidden-states-path",
            str(service.directory),
            "--served-model-name",
            "teacher-target",
            "--input-ids",
            "0",
            "1",
            "--output-json",
            str(output),
        ]
    )
    assert rc == expected
    assert json.loads(output.read_text())["passed"] == passed


def test_cli_protocol_error_is_nonzero(monkeypatch):
    cli = _cli()

    def fail(args):
        raise ValueError("bad packet")

    monkeypatch.setattr(cli, "run", fail)
    assert (
        cli.main(
            ["--model", "target", "--hidden-states-path", "hs", "--input-ids", "0"]
        )
        == 2
    )
