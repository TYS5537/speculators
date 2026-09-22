"""Small HS sidecar for trusted LAN/VPN use; never a general directory server."""

import argparse
import hashlib
import hmac
import json
import logging
import os
import stat
from contextlib import contextmanager, suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import MANIFEST
from speculators_dsv4.hs_http import (
    CHUNK_BYTES,
    DEFAULT_MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    PROTOCOL_VERSION,
    validate_filename,
    validate_token,
)

logger = logging.getLogger(__name__)
MAX_PORT = 65535


@contextmanager
def _regular_file(path, limit):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Only ordinary files without symlinks/hard links are allowed")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > limit
            or (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(
                "Artifact must be a bounded regular file without hard links"
            )
        yield stream


@contextmanager
def _writer_lock(path):
    lock = Path(str(path) + ".lock")
    if lock.is_symlink():
        raise ValueError("Symlink locks are not allowed")
    if not lock.exists():
        yield
        return
    # Existing training servers use POSIX advisory locks. Never remove their lock
    # merely because an HTTP request timed out or a downloader disconnected.
    if os.name != "posix":
        raise BlockingIOError("A producer lock exists")
    import fcntl  # noqa: PLC0415 -- Only target-side POSIX writers need this.

    with _regular_file(lock, MAX_MANIFEST_BYTES) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


class HiddenStatesServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address, directory, token, *, max_file_bytes=DEFAULT_MAX_FILE_BYTES
    ):
        self.directory = Path(directory).resolve(strict=True)
        self.token = validate_token(token)
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("HS HTTP file limit must be positive")
        self.max_file_bytes = max_file_bytes
        self.manifest = self.read_manifest()
        super().__init__(address, _Handler)

    def read_manifest(self):
        with _regular_file(self.directory / MANIFEST, MAX_MANIFEST_BYTES) as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict) or manifest.get("format") != HS_FORMAT:
            raise ValueError("Start a matching DSV4 HS target before the HTTP sidecar")
        return manifest

    def check_manifest(self):
        if self.read_manifest() != self.manifest:
            raise ValueError(
                "Target HS contract changed; restart the sidecar explicitly"
            )


class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 closes connections, including rejected requests with unread bodies.
    server_version = "DSV4-HS/1"

    def setup(self):
        self.request.settimeout(10)
        super().setup()

    def log_message(self, *args):  # noqa: ARG002 -- BaseHTTPRequestHandler hook.
        # No access log containing bearer tokens, paths or training data.
        return

    def do_GET(self):
        self._dispatch(delete=False)

    def do_DELETE(self):
        self._dispatch(delete=True)

    def _empty(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _dispatch(self, *, delete):
        authorization = self.headers.get("Authorization", "")
        if not hmac.compare_digest(
            authorization.encode(), ("Bearer " + self.server.token).encode()
        ):
            self._empty(HTTPStatus.UNAUTHORIZED)
            return
        try:
            self.server.check_manifest()
            if self.path == "/v1/manifest" and not delete:
                self._manifest()
            elif self.path.startswith("/v1/files/"):
                name = validate_filename(self.path.removeprefix("/v1/files/"))
                self._artifact(self.server.directory / name, delete=delete)
            else:
                self._empty(HTTPStatus.NOT_FOUND)
        except (BlockingIOError, FileNotFoundError):
            self._empty(HTTPStatus.CONFLICT)
        except ValueError:
            self._empty(HTTPStatus.BAD_REQUEST)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # Leave artifacts intact after an interrupted GET.
            return
        except OSError:
            self._empty(HTTPStatus.INTERNAL_SERVER_ERROR)

    def _manifest(self):
        payload = json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "hidden_states_path": str(self.server.directory),
                "manifest": self.server.manifest,
            }
        ).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _artifact(self, path, *, delete):
        with _writer_lock(path):
            # Missing files may belong to a queued writer, not a completed
            # deletion. Return 409 through _dispatch and let the client retry.
            with _regular_file(path, self.server.max_file_bytes) as stream:
                if not delete:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
                    length = stream.tell()
                    stream.seek(0)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(length))
                    self.send_header("X-HS-SHA256", digest.hexdigest())
                    self.end_headers()
                    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                        self.wfile.write(chunk)
            if delete:
                path.unlink()
        if delete:
            # Only this HTTP evaluation request's now-unlocked producer lock.
            Path(str(path) + ".lock").unlink(missing_ok=True)
            self._empty(HTTPStatus.NO_CONTENT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-states-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args()
    if not 1 <= args.port <= MAX_PORT:
        parser.error("Port must be between 1 and 65535")
    token = validate_token(os.environ.get("DSV4_HS_HTTP_TOKEN"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    with HiddenStatesServer(
        (args.host, args.port),
        args.hidden_states_path,
        token,
        max_file_bytes=args.max_file_bytes,
    ) as server:
        logger.info(
            "HS HTTP sidecar listening on %s:%s; "
            "use only a trusted LAN/VPN or TLS proxy",
            args.host,
            args.port,
        )
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
