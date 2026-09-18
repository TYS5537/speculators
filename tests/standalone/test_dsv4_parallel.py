"""Topology/control tests only: no vLLM processes or NPU collectives are run."""

# ruff: noqa: PT009, PT027 -- Keep the suite runnable without pytest/torch.

import unittest
from types import SimpleNamespace

from speculators_dsv4.parallel import configure_parallel_args, validate_parallel_config


class ParallelTests(unittest.TestCase):
    def test_dp1_defaults_preserve_launch_arguments(self):
        arguments = ["--tensor-parallel-size", "8"]
        topology = configure_parallel_args(arguments, {})
        self.assertEqual(arguments, ["--tensor-parallel-size", "8"])
        self.assertEqual(topology.data_parallel_size_local, 1)
        self.assertFalse(topology.headless)

    def test_dp1_visible_count_does_not_follow_local_size(self):
        for local_size in (0, 1, 2):
            with self.subTest(local_size=local_size):
                arguments = ["-tp", "8", "-dpl", str(local_size)]
                original = list(arguments)
                configure_parallel_args(
                    arguments,
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(8)))},
                )
                self.assertEqual(arguments, original)
                with self.assertRaisesRegex(ValueError, "visible device count"):
                    configure_parallel_args(
                        arguments,
                        {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                    )

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

    @staticmethod
    def dp4_arguments():
        return [
            "-tp",
            "8",
            "-dp",
            "4",
            "-dpl",
            "2",
            "-ep",
            "-dpa",
            "192.0.2.10",
            "-dpp",
            "13345",
        ]

    def test_dp4_head_and_worker_share_global_topology(self):
        for extra, headless, start in (
            ([], False, None),
            (["--data-parallel-start-rank", "0"], False, 0),
            (["--headless", "--data-parallel-start-rank", "2"], True, 2),
        ):
            with self.subTest(extra=extra):
                arguments = [*self.dp4_arguments(), *extra]
                original = list(arguments)
                topology = configure_parallel_args(
                    arguments,
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                )
                self.assertEqual(arguments, original)
                self.assertEqual(topology.data_parallel_size, 4)
                self.assertEqual(topology.data_parallel_size_local, 2)
                self.assertEqual(topology.headless, headless)
                self.assertEqual(topology.data_parallel_start_rank, start)

    def test_dp4_flexible_argument_spellings(self):
        topology = configure_parallel_args(
            [
                "--tensor_parallel_size=8",
                "--data_parallel_size=4",
                "--data_parallel_size_local=2",
                "--enable_expert_parallel",
                "--data_parallel_address=target-head.example",
                "--data_parallel_rpc_port=13345",
                "--headless",
                "--data_parallel_start_rank=2",
            ],
            {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
        )
        self.assertTrue(topology.headless)
        self.assertEqual(topology.data_parallel_address, "target-head.example")

    def test_dp4_device_count_uses_local_not_global_size(self):
        for count in (8, 32):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "TP \\* local DP"),
            ):
                configure_parallel_args(
                    self.dp4_arguments(),
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(count)))},
                )
        with self.assertRaisesRegex(ValueError, "ASCEND_RT_VISIBLE_DEVICES"):
            configure_parallel_args(self.dp4_arguments(), {})

    def test_dp4_requires_explicit_local_size_address_and_port(self):
        for option in ("-dpl", "-dpa", "-dpp"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                arguments = self.dp4_arguments()
                index = arguments.index(option)
                del arguments[index : index + 2]
                configure_parallel_args(
                    arguments,
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                )

    def test_dp4_rejects_invalid_rank_role_pairings(self):
        for extra in (
            ["--headless"],
            ["--headless", "-dpr", "0"],
            ["--headless", "-dpr", "1"],
            ["--headless", "-dpr", "3"],
            ["--headless", "-dpr", "4"],
            ["-dpr", "-1"],
            ["-dpr", "1"],
            ["-dpr", "2"],
            ["-dpr", "4"],
            ["-dpn", "0"],
            ["--headless", "-dpr", "2", "-dpn", "2"],
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "DSV4 DP4"),
            ):
                configure_parallel_args(
                    [*self.dp4_arguments(), *extra],
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                )

    def test_dp4_rejects_other_topologies(self):
        for extra in (
            ["-dpl", "1"],
            ["-dpl", "4"],
            ["-n", "2"],
            ["-r", "1"],
            ["-dpb", "ray"],
            ["--distributed-executor-backend", "ray"],
            ["--distributed-executor-backend", "external_launcher"],
            ["-dpe"],
            ["-dph"],
            ["-dpm"],
            ["-pp", "2"],
            ["-pcp", "2"],
            ["-dcp", "2"],
            ["--no-enable-expert-parallel"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configure_parallel_args(
                    [*self.dp4_arguments(), *extra],
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                )

    def test_dp4_rejects_unusable_shared_endpoint(self):
        for extra in (
            ["-dpa", ""],
            ["-dpa", "localhost"],
            ["-dpa", "LOCALHOST."],
            ["-dpa", "127.0.0.1"],
            ["-dpa", "127.1.2.3"],
            ["-dpa", "0.0.0.0"],  # noqa: S104 -- Rejected, never bound.
            ["-dpa", "::1"],
            ["-dpa", "::"],
            ["-dpa", "224.0.0.1"],
            ["-dpa", "http://192.0.2.10"],
            ["-dpa", "192.0.2.10:13345"],
            ["-dpa", "192.0.2.10 "],
            ["-dpp", "0"],
            ["-dpp", "-1"],
            ["-dpp", "65536"],
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "explicit, non-loopback"),
            ):
                configure_parallel_args(
                    [*self.dp4_arguments(), *extra],
                    {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                )

    def test_dp4_does_not_enable_block_evaluation(self):
        with self.assertRaisesRegex(ValueError, "block verification"):
            configure_parallel_args(
                self.dp4_arguments(),
                {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(16)))},
                block_verify=True,
            )

    def test_dp4_runtime_accepts_every_native_engine_rank(self):
        # vLLM resolves --data-parallel-address to data_parallel_master_ip.
        # headless/start-rank are launch-only; they are not ParallelConfig fields.
        good = {
            "tensor_parallel_size": 8,
            "data_parallel_size": 4,
            "data_parallel_size_local": 2,
            "data_parallel_master_ip": "192.0.2.10",
            "data_parallel_rpc_port": 13345,
            "enable_expert_parallel": True,
            "distributed_executor_backend": "mp",
        }
        for rank in range(4):
            for local_rank in (None, rank % 2):
                with self.subTest(rank=rank, local_rank=local_rank):
                    validate_parallel_config(
                        SimpleNamespace(
                            **good,
                            data_parallel_rank=rank,
                            data_parallel_rank_local=local_rank,
                        )
                    )
        for name, value in (
            ("data_parallel_size_local", 1),
            ("data_parallel_size_local", 4),
            ("nnodes", 2),
            ("node_rank", 1),
            ("data_parallel_external_lb", True),
            ("data_parallel_hybrid_lb", True),
            ("data_parallel_backend", "ray"),
            ("distributed_executor_backend", "ray"),
            ("enable_expert_parallel", False),
            ("data_parallel_master_ip", "127.0.0.1"),
            ("data_parallel_rpc_port", 0),
            ("data_parallel_rank", -1),
            ("data_parallel_rank", 4),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_parallel_config(SimpleNamespace(**{**good, name: value}))
        for rank, local_rank in (
            (0, 1),
            (1, 0),
            (2, 1),
            (3, 0),
            (0, -1),
            (1, 2),
            (2, 2),
            (3, 3),
            (None, 0),
        ):
            with (
                self.subTest(rank=rank, local_rank=local_rank),
                self.assertRaisesRegex(ValueError, "data_parallel_rank_local"),
            ):
                validate_parallel_config(
                    SimpleNamespace(
                        **good,
                        data_parallel_rank=rank,
                        data_parallel_rank_local=local_rank,
                    )
                )


if __name__ == "__main__":
    unittest.main()
