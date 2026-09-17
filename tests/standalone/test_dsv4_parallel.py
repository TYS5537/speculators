"""Topology/control tests only: no vLLM processes or NPU collectives are run."""

# ruff: noqa: PT009, PT027 -- Keep the suite runnable without pytest/torch.

import unittest
from types import SimpleNamespace

from speculators_dsv4.parallel import configure_parallel_args, validate_parallel_config


class ParallelTests(unittest.TestCase):
    def test_dp1_defaults_preserve_launch_arguments(self):
        arguments = ["--tensor-parallel-size", "8"]
        configure_parallel_args(arguments, {})
        self.assertEqual(arguments, ["--tensor-parallel-size", "8"])

    def test_dp2_topology_and_spellings(self):
        for arguments in (
            ["--tensor-parallel-size", "8", "--data-parallel-size", "2"],
            ["-tp", "8", "-dp", "2", "-dpb", "mp", "-ep"],
            ["--tensor_parallel_size=8", "--data_parallel_size=2"],
        ):
            with self.subTest(arguments=arguments):
                if "-ep" not in arguments:
                    arguments.append("--enable-expert-parallel")
                configure_parallel_args(
                    arguments,
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                )
                self.assertEqual(arguments[-2:], ["--data-parallel-size-local", "2"])

    def test_explicit_local_size_is_not_duplicated(self):
        arguments = ["-tp", "1", "-dp", "2", "-dpl", "2", "--enable-expert-parallel"]
        original = list(arguments)
        configure_parallel_args(arguments, {"ASCEND_RT_VISIBLE_DEVICES": "2,4"})
        self.assertEqual(arguments, original)

    def test_invalid_visible_devices(self):
        for devices in (
            None,
            "",
            "0",
            "0,0",
            "0,1,2",
            "0,1,",
            "0, 1",
            "-1,0",
            "00,1",
            "a,b",
        ):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                environment = (
                    {} if devices is None else {"ASCEND_RT_VISIBLE_DEVICES": devices}
                )
                configure_parallel_args(
                    ["-dp", "2", "--enable-expert-parallel"], environment
                )

    def test_unsupported_topologies(self):
        for extra in (
            ["-dp", "3"],
            ["-tp", "0"],
            ["-dpl", "1"],
            ["--nnodes", "2"],
            ["--node-rank", "1"],
            ["-n", "2"],
            ["-r", "1"],
            ["--data-parallel-backend", "ray"],
            ["--distributed-executor-backend", "external_launcher"],
            ["--data-parallel-external-lb"],
            ["--data-parallel-hybrid-lb"],
            ["--data-parallel-multi-port-external-lb"],
            ["-dpb", "ray"],
            ["-dpe"],
            ["-dph"],
            ["-dpm"],
            ["-dpn", "0"],
            ["-dpr", "0"],
            ["--data-parallel-rank", "0"],
            ["--data-parallel-start-rank", "0"],
            ["--headless"],
            ["-pp", "2"],
            ["-pcp", "2"],
            ["-dcp", "2"],
            ["--no-enable-expert-parallel"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configure_parallel_args(
                    ["-dp", "2", "--enable-expert-parallel", *extra],
                    {"ASCEND_RT_VISIBLE_DEVICES": "0,1"},
                )

    def test_dp2_does_not_enable_block_evaluation(self):
        with self.assertRaisesRegex(ValueError, "block verification"):
            configure_parallel_args(
                ["-dp", "2", "--enable-expert-parallel"],
                {"ASCEND_RT_VISIBLE_DEVICES": "0,1"},
                block_verify=True,
            )

    def test_worker_runtime_rechecks_resolved_topology(self):
        good = {
            "data_parallel_size": 2,
            "data_parallel_size_local": 2,
            "enable_expert_parallel": True,
            "distributed_executor_backend": "mp",
        }
        validate_parallel_config(SimpleNamespace(**good, data_parallel_rank=1))
        for name, value in (
            ("data_parallel_size_local", 1),
            ("nnodes", 2),
            ("node_rank", 1),
            ("data_parallel_external_lb", True),
            ("data_parallel_hybrid_lb", True),
            ("data_parallel_backend", "ray"),
            ("distributed_executor_backend", "ray"),
            ("enable_expert_parallel", False),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_parallel_config(SimpleNamespace(**{**good, name: value}))


if __name__ == "__main__":
    unittest.main()
