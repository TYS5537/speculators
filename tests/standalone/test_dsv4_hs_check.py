"""Dependency-free tests for the HS concurrency probe (not actual DP/NPU tests)."""

# ruff: noqa: PT009, PT027 -- Keep the suite runnable without pytest/torch.

import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "dsv4_hs_check", Path(__file__).resolve().parents[2] / "scripts/check_dsv4_hs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HSCheckTests(unittest.TestCase):
    def test_concurrent_different_length_requests_and_unique_handles(self):
        barrier = threading.Barrier(2, timeout=5)

        def check_one(tokens):
            barrier.wait()
            return {"hidden_states_file": f"hs-{len(tokens)}", "tokens": tokens}

        result = MODULE.check_requests([1, 2, 3, 4], 4, 2, check_one)
        self.assertEqual(
            [item["tokens"] for item in result], [[1, 2, 3, 4], [1, 2, 3], [1, 2], [1]]
        )

    def test_single_request_keeps_original_tokens(self):
        check_one = Mock(return_value={"hidden_states_file": "hs-1"})
        MODULE.check_requests([1, 2], 1, 1, check_one)
        check_one.assert_called_once_with([1, 2])

    def test_duplicate_handles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same output file"):
            MODULE.check_requests(
                [1, 2], 2, 2, lambda _: {"hidden_states_file": "same"}
            )

    def test_request_failure_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "engine failure"):
            MODULE.check_requests(
                [1, 2], 2, 2, Mock(side_effect=RuntimeError("engine failure"))
            )

    def test_invalid_probe_counts(self):
        for requests, concurrency in ((0, 1), (-1, 2), (2, 0), (1, -1)):
            with (
                self.subTest(requests=requests, concurrency=concurrency),
                self.assertRaises(ValueError),
            ):
                MODULE.check_requests([1], requests, concurrency, Mock())

    def test_decode_probe_uses_greedy_multiple_tokens_and_requires_completion(self):
        client = Mock()
        response = SimpleNamespace(usage=SimpleNamespace(completion_tokens=4))
        client.completions.create.return_value = response
        self.assertIs(
            MODULE.request_decode_probe(
                client, "target", [1, 2], max_tokens=4, timeout=120
            ),
            response,
        )
        client.completions.create.assert_called_once_with(
            model="target",
            prompt=[1, 2],
            max_tokens=4,
            temperature=0,
            extra_body={"return_token_ids": True, "ignore_eos": True},
            timeout=120,
        )
        for invalid in (None, SimpleNamespace(completion_tokens=1)):
            response.usage = invalid
            with self.assertRaisesRegex(ValueError, "Decode probe requested"):
                MODULE.request_decode_probe(
                    client, "target", [1, 2], max_tokens=4, timeout=120
                )

    def test_payload_checks_tokens_shape_dtype_and_finiteness(self):
        torch = SimpleNamespace(bfloat16="bf16", isfinite=Mock())
        torch.isfinite.return_value.all.return_value.item.return_value = True
        hidden = SimpleNamespace(shape=(2, 6, 4096), dtype="bf16")
        tokens = Mock()
        tokens.tolist.return_value = [1, 2]
        payload = {"token_ids": tokens, "hidden_states": hidden}
        self.assertIs(MODULE.validate_payload(payload, [1, 2], 6, torch), hidden)
        for invalid in (
            None,
            {**payload, "hidden_states": SimpleNamespace(shape=(1, 6, 4096))},
            {
                **payload,
                "hidden_states": SimpleNamespace(shape=(2, 6, 4096), dtype="fp32"),
            },
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.validate_payload(invalid, [1, 2], 6, torch)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            MODULE.validate_payload(payload, [2, 1], 6, torch)
        torch.isfinite.return_value.all.return_value.item.return_value = False
        with self.assertRaisesRegex(ValueError, "finite BF16"):
            MODULE.validate_payload(payload, [1, 2], 6, torch)

    def test_rank_header_reaches_normal_and_decode_clients_and_is_reported(self):
        # Exercise main with fake dependencies, not a live engine/DP setup.
        for rank in (None, 0, 2, 3, 9):
            for max_tokens in (1, 4):
                with (
                    self.subTest(rank=rank, max_tokens=max_tokens),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    handle = str(Path(directory) / "hs.safetensors")
                    client = Mock()
                    client.completions.create.return_value = SimpleNamespace(
                        usage=SimpleNamespace(completion_tokens=max_tokens)
                    )
                    context = MagicMock()
                    context.__enter__.return_value = client
                    openai = SimpleNamespace(OpenAI=Mock(return_value=context))
                    torch = SimpleNamespace(bfloat16="bf16", isfinite=Mock())
                    torch.isfinite.return_value.all.return_value.item.return_value = (
                        True
                    )
                    hidden = Mock(shape=(2, 6, 4096), dtype="bf16")
                    mean = hidden.float.return_value.square.return_value.mean
                    mean.return_value.sqrt.return_value.tolist.return_value = [1.0] * 6
                    tokens = Mock()
                    tokens.tolist.return_value = [1, 2]
                    transfer = Mock()
                    transfer.get_generated.return_value = {
                        "token_ids": tokens,
                        "hidden_states": hidden,
                    }
                    generator = Mock(return_value=handle)
                    extract = Mock(return_value=handle)
                    modules = {
                        "openai": openai,
                        "torch": torch,
                        "hs_connectors": SimpleNamespace(
                            FileTransfer=Mock(return_value=transfer)
                        ),
                        "speculators.data_generation.vllm_client": SimpleNamespace(
                            generate_hidden_states=generator, extract_output=extract
                        ),
                    }
                    arguments = [
                        "check_dsv4_hs.py",
                        "--model",
                        "target",
                        "--hidden-states-path",
                        directory,
                        "--input-ids",
                        "1",
                        "2",
                        "--probe-max-tokens",
                        str(max_tokens),
                    ]
                    if rank is not None:
                        arguments += ["--data-parallel-rank", str(rank)]
                    output = io.StringIO()
                    with (
                        patch.dict(sys.modules, modules),
                        patch.object(sys, "argv", arguments),
                        patch.object(
                            MODULE,
                            "inspect_checkpoint",
                            return_value={"config": {"vocab_size": 1000}},
                        ),
                        patch.object(MODULE, "make_manifest", return_value={}),
                        patch.object(MODULE, "ensure_manifest"),
                        redirect_stdout(output),
                    ):
                        MODULE.main()
                    options = openai.OpenAI.call_args.kwargs
                    result, _ = json.JSONDecoder().raw_decode(output.getvalue())
                    if rank is None:
                        self.assertNotIn("default_headers", options)
                        self.assertNotIn("requested_data_parallel_rank", result)
                    else:
                        self.assertEqual(
                            options["default_headers"],
                            {"X-data-parallel-rank": str(rank)},
                        )
                        self.assertEqual(result["requested_data_parallel_rank"], rank)
                    if max_tokens == 1:
                        generator.assert_called_once_with(
                            client,
                            "target",
                            {"input_ids": [1, 2]},
                            timeout=120,
                            max_retries=0,
                        )
                        client.completions.create.assert_not_called()
                    else:
                        generator.assert_not_called()
                        client.completions.create.assert_called_once()
                        extract.assert_called_once_with(
                            client.completions.create.return_value, [1, 2]
                        )

    def test_negative_rank_rejected_before_loading_dependencies(self):
        arguments = [
            "check_dsv4_hs.py",
            "--model",
            "target",
            "--hidden-states-path",
            "/shared/hs",
            "--input-ids",
            "1",
            "--data-parallel-rank",
            "-1",
        ]
        error = io.StringIO()
        with (
            patch.object(sys, "argv", arguments),
            patch.object(MODULE, "inspect_checkpoint") as inspect,
            redirect_stderr(error),
            self.assertRaises(SystemExit) as raised,
        ):
            MODULE.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--data-parallel-rank must be nonnegative", error.getvalue())
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
