"""End-to-end tests for backend arg auto-registration.

Verifies that backend-contributed train args survive the full
``resolve -> flatten`` and ``dump_yaml -> --config`` pipelines.
"""

import argparse
import dataclasses
import textwrap

import pytest
from hs_connectors.transfer import MooncakeBackend

from speculators.train.config.schema import TrainConfig


@pytest.mark.parametrize("dsv4", [False, True])
def test_mooncake_train_and_export_configs_preserve_role_defaults(monkeypatch, dsv4):
    monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "fixture-host")
    parser = argparse.ArgumentParser()
    MooncakeBackend.add_train_args(parser)
    args = parser.parse_args(
        [
            "--mooncake-namespace",
            "fixture-run",
            "--mooncake-protocol",
            "rdma",
            "--mooncake-device-name",
            "fixture-verbs",
            "--mooncake-global-segment-gib",
            "0.25",
        ]
    )
    args.dsv4 = dsv4
    args.target_hidden_state_format = "deepseek_v4_mean_hc_head" if dsv4 else None
    args.mooncake_writer_threads = 7
    transfer = MooncakeBackend.from_train_args(args, "unused")
    trainer = dataclasses.asdict(transfer.store.config)
    exporter = MooncakeBackend.build_kv_transfer_config(args)[
        "kv_connector_extra_config"
    ]["mooncake"]
    assert exporter == {**trainer, "num_writer_threads": 7}
    assert trainer["num_writer_threads"] == 4
    assert trainer["initialize_accelerator"] is not dsv4
    assert trainer["global_segment_size"] == 256 * 1024**2
    assert trainer["local_hostname"] == "fixture-host"
    assert trainer["device_name"] == "fixture-verbs"
    assert trainer["pinned_pool_bytes"] == 0
    assert trainer["use_copy_events"] is False


def test_mooncake_copy_optimizations_are_explicit_producer_options(monkeypatch):
    monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "fixture-host")
    parser = argparse.ArgumentParser()
    MooncakeBackend.add_launch_args(parser)
    args = parser.parse_args(
        ["--mooncake-pinned-pool-mib", "256", "--mooncake-copy-events"]
    )
    config = MooncakeBackend.build_kv_transfer_config(args)[
        "kv_connector_extra_config"
    ]["mooncake"]
    assert config["pinned_pool_bytes"] == 256 * 1024**2
    assert config["use_copy_events"] is True
    args.mooncake_pinned_pool_mib = -1
    with pytest.raises(ValueError, match="nonnegative"):
        MooncakeBackend.build_kv_transfer_config(args)


def test_http_training_args_survive_resolution_and_yaml_roundtrip():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "m",
            "--hidden-states-backend",
            "http",
            "--hs-http-endpoints",
            "http://first:8002",
            "http://second:8002",
            "--hidden-states-path",
            "/local/downloads",
            "--hs-http-timeout",
            "180",
        ]
    )
    flat = cfg.flatten()
    assert flat["hs_http_endpoints"] == ["http://first:8002", "http://second:8002"]
    assert flat["hs_http_timeout"] == 180
    assert flat["hs_http_max_file_bytes"] == 512 * 1024**2
    assert (
        TrainConfig.from_flat(flat).flatten()["hs_http_endpoints"]
        == flat["hs_http_endpoints"]
    )
    assert "DSV4_HS_HTTP_TOKEN" not in cfg.dump_yaml()


def test_mooncake_args_survive_from_sources():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "10.0.0.1:50051"
    assert flat["mooncake_protocol"] == "tcp"  # default backfilled


def test_mooncake_defaults_backfilled_for_selected_backend():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "127.0.0.1:50051"
    assert flat["mooncake_metadata_server"] == "P2PHANDSHAKE"
    assert flat["mooncake_protocol"] == "tcp"
    assert flat["mooncake_global_segment_gib"] == 4.0
    assert flat["mooncake_local_buffer_gib"] == 2.0


def test_file_backend_defaults_backfilled():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "file"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["hidden_states_path"] is None  # FileBackend default


def test_fp8_backend_defaults_backfilled():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "fp8"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["hidden_states_path"] is None  # FP8Backend default (shared with file)


def test_resolve_fp8_backend_hidden_states_path():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "m",
            "--hidden-states-backend",
            "fp8",
            "--hidden-states-path",
            "/data/hs-fp8",
        ]
    )
    flat = cfg.flatten()
    assert flat["hidden_states_path"] == "/data/hs-fp8"


def test_unselected_backend_args_not_backfilled():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "file"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert "mooncake_master" not in flat


def test_backend_args_provenance():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    assert cfg.provenance["mooncake_master"] == "flag"
    assert "mooncake_protocol" not in cfg.provenance  # default, no provenance entry


def test_backend_args_in_yaml_roundtrip(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
        train:
          verifier:
            verifier_name_or_path: m
          data:
            hidden_states_backend: mooncake
          backend:
            mooncake_master: "10.0.0.1:50051"
            mooncake_protocol: rdma
        """)
    )

    cfg = TrainConfig.from_sources(
        cli={}, config_path=str(yaml_file), argv=["train.py"]
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "10.0.0.1:50051"
    assert flat["mooncake_protocol"] == "rdma"
    assert cfg.provenance["mooncake_master"] == "yaml"
    assert cfg.provenance["mooncake_protocol"] == "yaml"


def test_backend_cli_beats_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
        train:
          verifier:
            verifier_name_or_path: m
          data:
            hidden_states_backend: mooncake
          backend:
            mooncake_master: "yaml-addr:50051"
        """)
    )

    cfg = TrainConfig.from_sources(
        cli={},
        config_path=str(yaml_file),
        backend_cli={"mooncake_master": "cli-addr:50051"},
        argv=["train.py"],
    )
    assert cfg.flatten()["mooncake_master"] == "cli-addr:50051"
    assert cfg.provenance["mooncake_master"] == "flag"


def test_dump_yaml_includes_backend_block():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    yaml_str = cfg.dump_yaml()
    assert "backend:" in yaml_str
    assert "mooncake_master" in yaml_str


def test_resolve_with_mooncake_args():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "m",
            "--hidden-states-backend",
            "mooncake",
            "--mooncake-master",
            "10.0.0.1:50051",
        ]
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "10.0.0.1:50051"
    assert flat["mooncake_protocol"] == "tcp"


def test_resolve_file_backend_hidden_states_path():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "m",
            "--hidden-states-path",
            "/data/hs",
        ]
    )
    flat = cfg.flatten()
    assert flat["hidden_states_path"] == "/data/hs"


def test_from_flat_roundtrip_with_backend_args():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    cfg2 = TrainConfig.from_flat(flat)
    assert cfg2.flatten()["mooncake_master"] == "10.0.0.1:50051"
