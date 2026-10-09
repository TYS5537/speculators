"""Run-scoped DSV4 contracts and training samples in the Mooncake memory store.

TCP/RDMA here transfers host tensors, not NPU-direct memory. This module remains
dependency-light so launcher validation does not initialize an accelerator/store.
"""

import math
import os
import re
from uuid import uuid4

from speculators_dsv4.training_contract import validate_training_contract


def validate_namespace(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}", value) is None
    ):
        raise ValueError(
            "DSV4 Mooncake requires a fresh --mooncake-namespace "
            "(8-64 safe characters) on every host"
        )
    return value


def contract_identity(contract):
    identity = validate_training_contract(contract)
    identity.pop("model_path")
    return identity


def contract_key(namespace, rank):
    return f"dsv4hs:{validate_namespace(namespace)}:contract:{rank}"


def sample_metadata(namespace, contract):
    return {
        "dsv4_namespace": validate_namespace(namespace),
        "dsv4_identity": contract_identity(contract),
    }


class Dsv4MooncakeTransfer:
    strict_errors = True
    preserve_failed_artifacts = True
    allow_model_name_relocation = True

    def __init__(
        self, store, *, namespace, target_dp_size, expected_contract=None, timeout=120.0
    ):
        self.namespace = validate_namespace(namespace)
        if type(target_dp_size) is not int or target_dp_size not in {1, 2, 4}:
            raise ValueError("DSV4 Mooncake target DP size must be 1, 2 or 4")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("HS transfer timeout must be finite and positive")
        self.store = store
        self.target_dp_size = target_dp_size
        self.timeout = timeout
        self.expected_contract = expected_contract
        self._pid = None
        self._contract = None
        self._pending = set()
        self._completed = {}

    def setup(self):
        if self._pid == os.getpid():
            return
        if self.store.is_setup:
            raise RuntimeError("Do not fork or share a live Mooncake training client")
        self.store.setup()
        try:
            contracts = []
            for rank in range(self.target_dp_size):
                value = self.store.get_json(
                    contract_key(self.namespace, rank), timeout=self.timeout
                )
                if (
                    value.get("namespace") != self.namespace
                    or value.get("dp_rank") != rank
                    or value.get("dp_size") != self.target_dp_size
                ):
                    raise ValueError(
                        "DSV4 Mooncake producer topology/namespace mismatch"
                    )
                contracts.append(validate_training_contract(value.get("contract")))
            expected = self.expected_contract or contracts[0]
            if any(
                contract_identity(value) != contract_identity(expected)
                for value in contracts
            ):
                raise ValueError("DSV4 Mooncake producer training contract mismatch")
            self._contract = contracts[0]
            self._pid = os.getpid()
        except Exception:
            self.store.close()
            raise

    def read_contract(self):
        self.setup()
        return validate_training_contract(self._contract)

    def request_kwargs(self):
        self.setup()
        request_id = "hsmc-" + uuid4().hex
        self._pending.add(request_id)
        return {"request_id": request_id, "max_retries": 0}

    def get_cached(self, file_idx):  # noqa: ARG002 -- Transfer interface.
        return None

    def cache(self, handle, file_idx):  # noqa: ARG002 -- Transfer interface.
        raise ValueError("DSV4 Mooncake requires --on-generate delete")

    def _owned(self, handle):
        pattern = (
            rf"dsv4hs:{re.escape(self.namespace)}:([0-9]+):"
            r"cmpl-(hsmc-[0-9a-f]{32})-0(?:-[0-9a-f]{8})?"
        )
        matched = re.fullmatch(pattern, handle) if isinstance(handle, str) else None
        if (
            matched is None
            or int(matched[1]) >= self.target_dp_size
            or matched[2] not in self._pending
        ):
            raise ValueError(
                "Mooncake HS handle does not belong to this training request/run"
            )
        return matched[2]

    def get_generated(self, handle):
        self.setup()
        request_id = self._owned(handle)
        if handle in self._completed:
            raise ValueError("Mooncake HS handle was already consumed")
        sample = self.store.get_sample(
            handle,
            timeout=self.timeout,
            expected_metadata=sample_metadata(self.namespace, self._contract),
        )
        self._completed[handle] = request_id
        return sample

    def delete(self, handle):
        self._owned(handle)
        if handle not in self._completed:
            raise ValueError("Only successfully read Mooncake HS may be deleted")
        self.store.delete_sample(handle)
        self._pending.remove(self._completed.pop(handle))

    def close(self):
        self.store.close()
        self._pid = None
