"""Parallel commands survive one access-token refresh.

Several ``comfy`` processes sharing one stored sign-in each hold the same stale
access token. The auth server keeps a single live access token and rotates it
with the refresh token, so a refresh spent by every process in turn would
invalidate the token a peer just installed and fail its retried request. Only
the first 401 may spend the refresh token; the rest adopt what it stored.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROCESSES = 6
REQUESTS_PER_PROCESS = 4


class _AuthServer:
    """One live access token, rotated together with the refresh token it came from."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.access = "access-0"
        self.refresh = "refresh-0"
        self.rotations = 0
        self.refused: list[str] = []
        self.api_calls = 0

    def handler(self) -> type[BaseHTTPRequestHandler]:
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _answer(self, status: int, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self) -> None:
                form = urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
                with state.lock:
                    if form.get("refresh_token") != [state.refresh]:
                        self._answer(400, {"error": "invalid_grant", "error_description": "reuse detected"})
                        return
                    state.rotations += 1
                    state.access = f"access-{state.rotations}"
                    state.refresh = f"refresh-{state.rotations}"
                    self._answer(
                        200,
                        {
                            "access_token": state.access,
                            "refresh_token": state.refresh,
                            "token_type": "Bearer",
                            "expires_in": 900,
                        },
                    )

            def do_GET(self) -> None:
                with state.lock:
                    state.api_calls += 1
                    if self.headers.get("Authorization") != f"Bearer {state.access}":
                        state.refused.append(self.headers.get("Authorization") or "")
                        self._answer(401, {"message": "invalid auth token"})
                        return
                self._answer(200, {"id": "r1", "status": "building"})

        return Handler


_WORKER = """
import sys, time
from pathlib import Path
from comfy_cli.builder_api import BuilderClient

base_url, go, count = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
client = BuilderClient.from_session(base_url)
while not go.exists():
    time.sleep(0.005)
for _ in range(count):
    client.get_release("r1")
    time.sleep(0.05)
"""


def test_parallel_commands_survive_one_token_refresh(tmp_path, monkeypatch) -> None:
    # Given a stored sign-in whose access token the server no longer accepts
    server = _AuthServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"
    secrets = tmp_path / "secrets.json"
    monkeypatch.setenv("COMFY_SECRETS_PATH", str(secrets))
    from comfy_cli.auth import store

    store.save_cloud_session(
        base_url=base_url,
        resource=base_url,
        client_id="comfy-cli",
        scope="",
        access_token="access-stale",
        refresh_token=server.refresh,
        token_type="Bearer",
        expires_at=int(time.time()) + 900,
    )
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    go = tmp_path / "go"

    # When six processes start their requests together
    try:
        workers = [
            subprocess.Popen(
                [sys.executable, str(script), base_url, str(go), str(REQUESTS_PER_PROCESS)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for _ in range(PROCESSES)
        ]
        time.sleep(1.5)  # every worker is built and waiting on the signal
        go.touch()
        outputs = [(worker.wait(timeout=60), worker.stdout.read()) for worker in workers]
    finally:
        httpd.shutdown()

    # Then every process completes, and the refresh token was spent once
    failures = [output for code, output in outputs if code != 0]
    assert not failures, failures[0][-2000:]
    assert server.rotations == 1
