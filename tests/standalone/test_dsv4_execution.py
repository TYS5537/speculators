"""Execution-policy checks, not ACL graph or NPU integration tests."""

# ruff: noqa: PT009, PT027 -- Also runnable without pytest/torch.

import json
import unittest
from enum import Enum, IntEnum
from types import SimpleNamespace

from speculators_dsv4.execution import (
    configure_execution_args,
    validate_execution_config,
)


class CompileMode(IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


class GraphMode(Enum):
    NONE = 0
    FULL_DECODE_ONLY = (2, 0)
    FULL_AND_PIECEWISE = (2, 1)


def config(eager=True, graph=GraphMode.NONE, asynchronous=False, compile_mode=0):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=SimpleNamespace(mode=compile_mode, cudagraph_mode=graph),
        scheduler_config=SimpleNamespace(async_scheduling=asynchronous),
    )


class ExecutionTests(unittest.TestCase):
    def test_eager_and_graph_with_independent_async_options(self):
        for mode in ("eager", "full-decode-only"):
            for flag in (None, "--async-scheduling", "--no-async-scheduling"):
                with self.subTest(mode=mode, flag=flag):
                    args = ["--max-num-seqs", "4", *([flag] if flag else [])]
                    configure_execution_args(args, mode)
                    self.assertEqual(args[:2], ["--max-num-seqs", "4"])
                    self.assertEqual(args.count(flag or "--no-async-scheduling"), 1)
                    self.assertIn(
                        "--enforce-eager" if mode == "eager" else "--no-enforce-eager",
                        args,
                    )
                    self.assertEqual(
                        json.loads(args[args.index("--compilation-config") + 1]),
                        {
                            "mode": 0,
                            "cudagraph_mode": (
                                "NONE" if mode == "eager" else "FULL_DECODE_ONLY"
                            ),
                        },
                    )

    def test_conflicting_eager_overrides_rejected_without_mutation(self):
        for mode, flag in (
            ("eager", "--no-enforce-eager"),
            ("full-decode-only", "--enforce-eager"),
            ("full-decode-only", "--enforce_eager"),
        ):
            args = [flag]
            with (
                self.subTest(mode=mode, flag=flag),
                self.assertRaisesRegex(ValueError, "conflicts"),
            ):
                configure_execution_args(args, mode)
            self.assertEqual(args, [flag])

    def test_matching_eager_flags_not_duplicated(self):
        for mode, flag in (
            ("eager", "--enforce-eager"),
            ("full-decode-only", "--no-enforce-eager"),
        ):
            args = [flag]
            configure_execution_args(args, mode)
            self.assertEqual(args.count(flag), 1)

    def test_compilation_overrides_cannot_bypass_policy(self):
        for option in (
            "--compilation-config",
            "--compilation_config",
            "-cc",
            "-cc.mode",
            "-cc.cudagraph_mode",
            "--optimization-level",
            "--optimization_level",
            "-O",
            "--compilation-config.mode",
            "--compilation_config.cudagraph_mode",
        ):
            for args in ([option, "3"], [f"{option}=3"]):
                before = args.copy()
                with (
                    self.subTest(args=args),
                    self.assertRaisesRegex(ValueError, "owns"),
                ):
                    configure_execution_args(args, "full-decode-only")
                self.assertEqual(args, before)

    def test_block_verification_stays_eager_and_synchronous(self):
        for mode, flags in (
            ("full-decode-only", []),
            ("eager", ["--async-scheduling"]),
            ("eager", ["--async_scheduling"]),
        ):
            with (
                self.subTest(mode=mode, flags=flags),
                self.assertRaisesRegex(ValueError, "block verification"),
            ):
                configure_execution_args(flags, mode, block_verify=True)
        args = []
        configure_execution_args(args, "eager", block_verify=True)
        self.assertIn("--enforce-eager", args)
        self.assertIn("--no-async-scheduling", args)

    def test_runtime_resolved_modes(self):
        for asynchronous in (False, True):
            for eager, graph, expected in (
                (True, GraphMode.NONE, "eager"),
                (False, GraphMode.FULL_DECODE_ONLY, "full-decode-only"),
                (False, "FULL_DECODE_ONLY", "full-decode-only"),
            ):
                with self.subTest(eager=eager, graph=graph, asynchronous=asynchronous):
                    self.assertEqual(
                        validate_execution_config(
                            config(eager, graph, asynchronous, CompileMode.NONE)
                        ),
                        (expected, asynchronous),
                    )

    def test_runtime_rejects_downgraded_and_unsupported_graphs(self):
        for runtime in (
            config(False, GraphMode.NONE),
            config(False, GraphMode.FULL_AND_PIECEWISE),
            config(True, GraphMode.FULL_DECODE_ONLY),
            config(
                False, GraphMode.FULL_DECODE_ONLY, compile_mode=CompileMode.VLLM_COMPILE
            ),
            config(True, compile_mode=CompileMode.VLLM_COMPILE),
        ):
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                validate_execution_config(runtime)

    def test_block_runtime_rejects_graph_or_async(self):
        for runtime in (
            config(asynchronous=True),
            config(False, GraphMode.FULL_DECODE_ONLY),
        ):
            with self.assertRaisesRegex(ValueError, "block verification"):
                validate_execution_config(runtime, block_verify=True)
        self.assertEqual(
            validate_execution_config(config(), block_verify=True), ("eager", False)
        )


if __name__ == "__main__":
    unittest.main()
