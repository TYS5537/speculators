"""Opt-in, authenticated HS downloads for DSV4 evaluation without shared storage.

The target still writes ordinary safetensors. Only the hshttp- request namespace
is exposed by the sidecar; training caches and ordinary requests are off limits.
"""

import hashlib
import json
import logging
import math
import re
import tempfile
import time
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

PROTOCOL_VERSION = 1
REQUEST_PREFIX = "hshttp-"
FILE_PATTERN = re.compile(r"cmpl-hshttp-[0-9a-f]{32}-0(?:-[0-9a-f]{8})?\.safetensors")
MAX_MANIFEST_BYTES = 1024 * 1024
DEFAULT_MAX_FILE_BYTES = 512 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MIN_TOKEN_CHARS = 32
PRINTABLE_ASCII_MIN = 33
PRINTABLE_ASCII_MAX = 126
logger = logging.getLogger(__name__)


def validate_token(token):
    if (
        not isinstance(token, str)
        or len(token) < MIN_TOKEN_CHARS
        or not token.isascii()
        or any(
            not PRINTABLE_ASCII_MIN <= ord(char) <= PRINTABLE_ASCII_MAX
            for char in token
        )
    ):
        raise ValueError(
            "DSV4_HS_HTTP_TOKEN must contain at least 32 printable ASCII characters"
        )
    return token


def validate_endpoint(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("HS HTTP endpoint must be a URL")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(char.isspace() or not char.isprintable() for char in endpoint)
    ):
        raise ValueError(
            "HS HTTP endpoint must be an http(s) URL without credentials/query/fragment"
        )
    return endpoint.rstrip("/")


def validate_filename(name):
    if not isinstance(name, str) or FILE_PATTERN.fullmatch(name) is None:
        raise ValueError("Only HTTP evaluation request artifacts may be transferred")
    return name


def _remote_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Missing remote HS path")
    # Clients need not use the server's OS or mount its directory.
    path = (
        PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)
    )
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Remote HS paths must be absolute without traversal")
    return path


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, *args, **kwargs):  # noqa: ARG002 -- urllib hook.
        response.close()
        raise ValueError("HS HTTP redirects are disabled to protect credentials")


class HttpHiddenStates:
    def __init__(
        self,
        endpoint,
        token,
        directory,
        *,
        timeout=120.0,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
    ):
        self.endpoint = validate_endpoint(endpoint)
        self.token = validate_token(token)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("HS HTTP timeout must be finite and positive")
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("HS HTTP file limit must be positive")
        self.timeout = timeout
        self.max_file_bytes = max_file_bytes
        self.directory = Path(directory)
        self.remote_directory = None
        # Never forward the bearer token to environment-configured proxies.
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    @staticmethod
    def new_request_id():
        return REQUEST_PREFIX + uuid4().hex

    def _open(self, route, *, method="GET"):
        deadline = time.monotonic() + self.timeout
        request = Request(  # noqa: S310 -- Explicit http(s), fixed routes, no redirects.
            self.endpoint + route,
            method=method,
            headers={"Authorization": "Bearer " + self.token},
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for a complete target HS artifact"
                )
            try:
                return self.opener.open(request, timeout=remaining)
            except HTTPError as error:
                status = error.code
                error.close()
                if status != HTTPStatus.CONFLICT:
                    raise ValueError(
                        f"HS HTTP {method} failed with status {status}"
                    ) from None
                time.sleep(min(0.1, remaining))

    def validate_manifest(self, expected):
        with self._open("/v1/manifest") as response:
            data = response.read(MAX_MANIFEST_BYTES + 1)
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError("HS HTTP manifest exceeds size limit")
        info = json.loads(data)
        if (
            not isinstance(info, dict)
            or type(info.get("protocol_version")) is not int
            or info["protocol_version"] != PROTOCOL_VERSION
        ):
            raise ValueError("Unsupported HS HTTP protocol")
        actual = info.get("manifest")
        if not isinstance(actual, dict):
            raise ValueError("Missing HS HTTP manifest")
        actual = dict(actual)
        # Same policy as the file consumer: validate checkpoint/HS identity, while
        # the target owns its runtime quantization override.
        actual.pop("runtime_quantization", None)
        if actual != expected:
            raise ValueError("DSV4 HS contract/target mismatch over HTTP")
        self.remote_directory = _remote_path(info.get("hidden_states_path"))

    def _filename(self, handle, request_id):
        path = _remote_path(handle)
        name = validate_filename(path.name)
        pattern = rf"cmpl-{re.escape(request_id)}-0(?:-[0-9a-f]{{8}})?\.safetensors"
        if path.parent != self.remote_directory or re.fullmatch(pattern, name) is None:
            raise ValueError(
                "Remote HS handle does not belong to this request/directory"
            )
        return name

    def _download(self, name, destination):
        with self._open("/v1/files/" + name) as response:
            try:
                length = int(response.headers["Content-Length"])
            except (TypeError, ValueError):
                raise ValueError(
                    "HS HTTP response needs a valid Content-Length"
                ) from None
            digest = response.headers.get("X-HS-SHA256", "")
            if not 0 < length <= self.max_file_bytes or not re.fullmatch(
                r"[0-9a-f]{64}", digest
            ):
                raise ValueError(
                    "HS HTTP response exceeds size limit or has no checksum"
                )
            checksum = hashlib.sha256()
            received = 0
            deadline = time.monotonic() + self.timeout
            with destination.open("xb") as stream:
                while received < length:
                    if time.monotonic() > deadline:
                        raise TimeoutError("HS HTTP download exceeded its time budget")
                    chunk = response.read(min(CHUNK_BYTES, length - received))
                    if not chunk:
                        raise ValueError("Incomplete HS HTTP download")
                    stream.write(chunk)
                    checksum.update(chunk)
                    received += len(chunk)
            if checksum.hexdigest() != digest:
                raise ValueError("HS HTTP checksum mismatch")

    def _delete(self, name):
        with self._open("/v1/files/" + name, method="DELETE"):
            pass

    @contextmanager
    def artifact(self, handle, request_id, *, keep=False, download=True):
        name = self._filename(handle, request_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="hshttp-", dir=self.directory
        ) as temporary:
            path = Path(temporary) / name
            if download:
                # A failed/truncated transfer retains the remote file for diagnosis.
                self._download(name, path)
            try:
                yield path if download else None
            finally:
                if not keep:
                    try:
                        self._delete(name)
                    except (OSError, ValueError):
                        # Cleanup must not mask a model/protocol error. Only our
                        # dedicated request namespace may be cleaned up manually.
                        logger.warning(
                            "HS HTTP cleanup failed for %s; remote artifact retained",
                            name,
                        )
