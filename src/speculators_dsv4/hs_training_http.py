"""Authenticated, host-routed training HS downloads without a shared filesystem.

This duck-typed HiddenStatesTransfer keeps contract preflight independent of torch.
Each DataLoader process initializes its own clients; only successful, owned downloads
can be deleted. Failed source artifacts remain available for diagnosis.
"""

import os
import re
import tempfile
import time
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request
from uuid import uuid4

from speculators_dsv4.hs_http import (
    DEFAULT_MAX_FILE_BYTES,
    TRAINING_REQUEST_PREFIX,
    HttpHiddenStates,
    _remote_path,
    validate_endpoint,
    validate_filename,
    validate_token,
)
from speculators_dsv4.hs_profile import profile_file, profile_stage
from speculators_dsv4.training_contract import (
    contract_identity,
    validate_training_contract,
)


def _load_tensors(path):
    from safetensors.torch import (  # noqa: PLC0415
        load_file,
    )

    return load_file(path)


class HttpTrainingTransfer:
    strict_errors = True
    preserve_failed_artifacts = True
    allow_model_name_relocation = True

    def __init__(
        self,
        endpoints,
        token,
        directory,
        *,
        timeout=120.0,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        expected_contract=None,
    ):
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        self.endpoints = tuple(validate_endpoint(item) for item in endpoints)
        if not self.endpoints or len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("Provide distinct HS HTTP endpoints for all target hosts")
        self.token = validate_token(token)
        self.directory = Path(directory)
        self.timeout = timeout
        self.max_file_bytes = max_file_bytes
        self.expected_contract = (
            validate_training_contract(expected_contract)
            if expected_contract is not None
            else None
        )
        self._clients = []
        self._pid = None
        self._contract = None
        self._pending = set()
        self._completed = {}

    def __getstate__(self):
        # urllib openers and request ownership never cross worker processes.
        state = dict(self.__dict__)
        state.update(_clients=[], _pid=None, _pending=set(), _completed={})
        return state

    def setup(self):
        if self._pid == os.getpid() and self._clients:
            if self.expected_contract is not None:
                self._check_contract(self._contract, self.expected_contract)
            return
        clients = [
            HttpHiddenStates(
                endpoint,
                self.token,
                self.directory,
                timeout=self.timeout,
                max_file_bytes=self.max_file_bytes,
                training=True,
            )
            for endpoint in self.endpoints
        ]
        contracts = [
            validate_training_contract(client.read_manifest()) for client in clients
        ]
        first = contracts[0]
        for contract in contracts[1:]:
            self._check_contract(contract, first)
        if self.expected_contract is not None:
            self._check_contract(first, self.expected_contract)
        if self._contract is not None:
            self._check_contract(first, self._contract)
        self._contract = first
        self._clients = clients
        self._pid = os.getpid()
        self._pending = set()
        self._completed = {}

    @staticmethod
    def _check_contract(actual, expected):
        if contract_identity(actual) != contract_identity(expected):
            raise ValueError(
                "DSV4 HS HTTP training contract mismatch; all target hosts must "
                "match checkpoint, layers, dtype and runtime quantization"
            )

    def read_contract(self):
        self.setup()
        return validate_training_contract(self._contract)

    def request_kwargs(self):
        self.setup()
        request_id = TRAINING_REQUEST_PREFIX + uuid4().hex
        self._pending.add(request_id)
        # A retried inference request is not an idempotent HS download. Retries
        # here could overwrite the nonce's immutable source or leak a second HS.
        return {"request_id": request_id, "max_retries": 0}

    def get_cached(self, file_idx):  # noqa: ARG002 -- Transfer interface.
        return None

    def cache(self, handle, file_idx):  # noqa: ARG002 -- Transfer interface.
        raise ValueError("HTTP HS transport does not support --on-generate cache")

    def _owned(self, handle):
        path = _remote_path(handle)
        name = validate_filename(path.name, training=True)
        request_id = re.fullmatch(
            r"cmpl-(hstrain-[0-9a-f]{32})-0(?:-[0-9a-f]{8})?\.safetensors", name
        ).group(1)
        if request_id not in self._pending:
            raise ValueError(
                "HS HTTP handle does not belong to an active training request"
            )
        clients = [
            client for client in self._clients if client.remote_directory == path.parent
        ]
        if not clients:
            raise ValueError(
                "HS HTTP handle is outside every verified target directory"
            )
        return name, request_id, clients

    @staticmethod
    def _probe(client, name, timeout):
        request = Request(  # noqa: S310 -- Fixed authenticated route, no redirects/proxies.
            client.endpoint + "/v1/probe/" + name,
            headers={"Authorization": "Bearer " + client.token},
        )
        try:
            with client.opener.open(request, timeout=timeout) as response:
                if response.status != HTTPStatus.OK:
                    raise ValueError("Invalid HS HTTP ownership probe response")
                return True
        except HTTPError as error:
            status = error.code
            error.close()
            if status == HTTPStatus.NOT_FOUND:
                return False
            if status == HTTPStatus.CONFLICT:
                return True  # A writer owns the nonce; downloading waits for it.
            raise ValueError(
                f"HS HTTP ownership probe failed with status {status}"
            ) from None

    def _route(self, name, clients):
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No target host published this training HS request")
            owners = [
                client for client in clients if self._probe(client, name, remaining)
            ]
            if len(owners) > 1:
                raise ValueError(
                    "Ambiguous HS HTTP artifact: nonce exists on multiple hosts"
                )
            if owners:
                return owners[0]
            time.sleep(min(0.1, remaining))

    def get_generated(self, handle):
        self.setup()
        name, request_id, clients = self._owned(handle)
        if handle in self._completed:
            raise ValueError("Training HS handle has already been downloaded")
        with profile_stage("http_route_ms"):
            client = self._route(name, clients)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="hstrain-", dir=self.directory
        ) as temporary:
            with profile_stage("http_download_ms"):
                local = client._download_with_retries(name, Path(temporary))  # noqa: SLF001 -- Shared transport implementation.
            profile_file(local)
            with profile_stage("http_load_ms"):
                sample = _load_tensors(local)
        self._completed[handle] = (client, name, request_id)
        return sample

    def delete(self, handle):
        self.setup()
        if handle not in self._completed:
            raise ValueError("Only successfully downloaded training HS may be deleted")
        client, name, request_id = self._completed[handle]
        client._delete(name)  # noqa: SLF001 -- Shared transport implementation.
        del self._completed[handle]
        self._pending.remove(request_id)

    def close(self):
        """No persistent sockets; do not delete failed remote artifacts on close."""
