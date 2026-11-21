"""`comfy deploy models add`, `ls` and `rm`, with comfy-deploy, the builder and the bucket faked at their HTTP
layers, and the pointer the two library uploads print first."""

from __future__ import annotations

import _thread
import copy
import hashlib
import http.client
import io
import json
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
from email.message import Message
from pathlib import Path
from urllib.parse import urlsplit

import jsonschema
import pytest
import requests
from blake3 import blake3
from deploy_up_support import FakeBuilder, deployment
from typer.testing import CliRunner

from comfy_cli import builder_api
from comfy_cli.builder_api import BuilderClient, PartsUpload
from comfy_cli.caller import Caller
from comfy_cli.cmdline import app
from comfy_cli.command import deploy_models
from comfy_cli.command.assets_library import DEPLOYMENT_POINTER
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api import DeployClient

_CONTENT = b"0123456789"
# The builder's plan for these ten bytes in these tests: pieces of 4, 4 and 2.
_PLAN = {1: (0, 4), 2: (4, 4), 3: (8, 2)}
_LATER = "2999-01-01T00:00:00Z"
_EXPIRED = "2000-01-01T00:00:00Z"
_ENTRY: JsonObject = {
    "id": "model-1",
    "type": "loras",
    "filename": "lora.safetensors",
    "status": "adding",
    "sizeBytes": 10,
}
_MODELS = "/v1/deployments/dep-1/models"
_GONE = "gone"
_DAY = 86400.0
_REMOVE_TEXT = (
    "Remove lora.safetensors? It is deleted from this deployment's disk, and jobs that name it will fail. A job "
    "already running may still finish, and a worker that already loaded it may keep using it for a few minutes."
)


def _parts(*numbers: int, sig: str = "a", expires: str = _LATER) -> list[JsonObject]:
    return [
        {
            "partNumber": number,
            "offsetBytes": _PLAN[number][0],
            "sizeBytes": _PLAN[number][1],
            "uploadUrl": f"https://storage.test/part-{number}?sig={sig}",
            "expiresAt": expires,
        }
        for number in numbers
    ]


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for the presigned url", response=self)


class Storage:
    """The bucket behind the presigned links. A URL in ``refuse`` answers 403 once, as an expired link does."""

    def __init__(self, log: list[str], refuse: set[str] | None = None) -> None:
        self.log = log
        self.refuse = set(refuse or ())
        self.puts: list[tuple[str, bytes, dict, int | None]] = []
        self.landed: set[int] = set()
        self._lock = threading.Lock()

    def __call__(self, url, data=None, headers=None, timeout=None, allow_redirects=True):
        size = len(data) if hasattr(data, "__len__") else None
        body = b"".join(iter(lambda: data.read(3), b""))
        with self._lock:
            self.log.append(f"put {url}")
            self.puts.append((url, body, dict(headers or {}), size))
            if url in self.refuse:
                self.refuse.discard(url)
                return _Response(403)
            name = urlsplit(url).path.rsplit("/", 1)[-1]
            if name.startswith("part-"):
                self.landed.add(int(name.removeprefix("part-")))
        return _Response(200)

    def bodies(self) -> dict[str, bytes]:
        return {url: body for url, body, _, _ in self.puts}


class Builder:
    """comfy-builder's blob routes; a refresh answers fresh links for the pieces the bucket does not hold yet."""

    def __init__(self, log: list[str], answer: JsonObject, storage: Storage) -> None:
        self.log = log
        self.answer = answer
        self.storage = storage
        self.bodies: list[JsonObject | None] = []

    def __call__(self, url, _target, *, method="GET", body=None, **_kwargs):
        path = urlsplit(url).path
        self.log.append(f"builder {method} {path}")
        self.bodies.append(body)
        if (method, path) == ("POST", "/v1/blobs"):
            return 201, copy.deepcopy(self.answer)
        if (method, path) == ("POST", "/v1/blobs/blob-1/parts"):
            missing = [number for number in _PLAN if number not in self.storage.landed]
            return 200, {"blobId": "blob-1", "parts": _parts(*missing, sig="fresh")}
        if (method, path) == ("POST", "/v1/blobs/blob-1/complete"):
            return 200, {"blobId": "blob-1", "sizeBytes": len(_CONTENT)}
        pytest.fail(f"unexpected builder call: {method} {url}")


class Control:
    """comfy-deploy's models routes, and the deployment list a name resolves against."""

    def __init__(
        self,
        log: list[str],
        *,
        refusal: tuple[int, JsonObject] | None = None,
        statuses: tuple[str, ...] = ("ready",),
        models: list[JsonObject] | None = None,
    ) -> None:
        self.log = log
        self.refusal = refusal
        self.statuses = list(statuses)
        self.models = models
        self.calls: list[tuple[str, str, JsonObject | None]] = []

    def __call__(self, url, _target, *, method="GET", body=None, **_kwargs):
        path = urlsplit(url).path
        self.log.append(f"deploy {method} {path}")
        self.calls.append((method, path, body))
        if (method, path) == ("POST", _MODELS):
            if self.refusal is not None:
                status, payload = self.refusal
                stream = io.BytesIO(json.dumps(payload).encode())
                raise urllib.error.HTTPError(url, status, "refused", Message(), stream)
            return 202, copy.deepcopy(_ENTRY)
        if (method, path) == ("GET", _MODELS):
            if self.models is not None:
                return 200, {"models": copy.deepcopy(self.models)}
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            if status == _GONE:
                return 200, {"models": []}
            failure = {"failure": "copy to volume vol-9 failed: quota"} if status == "failed" else {}
            return 200, {"models": [{**_ENTRY, "status": status, **failure}]}
        if (method, path) == ("DELETE", f"{_MODELS}/model-1"):
            return 202, {**_ENTRY, "status": "removing"}
        if (method, path) == ("GET", "/v1/deployments"):
            return 200, {"deployments": [deployment("dep-1", name="production")]}
        pytest.fail(f"unexpected comfy-deploy call: {method} {url}")

    def count(self, method: str, path: str) -> int:
        return sum(1 for called, at, _ in self.calls if (called, at) == (method, path))


def _install(
    monkeypatch, tmp_path: Path, control: Control, answer: JsonObject | None = None, storage: Storage | None = None
) -> tuple[Path, Builder | None]:
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", control)
    clients = (FakeBuilder(), DeployClient("https://control.test", "tok"))
    monkeypatch.setattr(deploy_models, "_command_clients", lambda: clients)
    monkeypatch.setattr(deploy_models, "_blob_client", lambda: BuilderClient("https://builder.test", "tok"))
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] > _DAY:
            pytest.fail("still waiting after a day of polls")

    monkeypatch.setattr(deploy_models, "_sleep", sleep)
    monkeypatch.setattr(deploy_models, "_clock", lambda: now[0], raising=False)
    builder = None
    if answer is not None:
        builder = Builder(control.log, answer, storage or Storage(control.log))
        monkeypatch.setattr("comfy_cli.builder_api.request_json", builder)
    if storage is not None:
        monkeypatch.setattr("requests.put", storage)
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)
    return model, builder


def _invoke(*args: str, pretty: bool = False):
    output = "--no-json" if pretty else "--json"
    return CliRunner().invoke(app, [output, "deploy", "models", *args], env={"COLUMNS": "400", "NO_COLOR": "1"})


def _envelope(result) -> JsonObject:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _validate(data: JsonObject) -> None:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / "deploy_models.json"
    jsonschema.Draft202012Validator(json.loads(path.read_text(encoding="utf-8"))).validate(data)


def _in_pieces() -> JsonObject:
    return {"blobId": "blob-1", "deduplicated": False, "parts": _parts(1, 2, 3)}


# --- add ------------------------------------------------------------------------


def test_add_adds_before_the_pieces_land_then_joins_them_and_waits_until_ready(tmp_path, monkeypatch) -> None:
    # Given a builder that takes uploads in pieces, and an entry that reads adding, then ready
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", "ready")), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then the upload asks for pieces and re-use, with both hashes
    assert result.exit_code == 0, result.output
    assert builder is not None
    assert builder.bodies[0] == {
        "kind": "model",
        "filename": "lora.safetensors",
        "sha256": hashlib.sha256(_CONTENT).hexdigest(),
        "blake3": blake3(_CONTENT).hexdigest(),
        "sizeBytes": 10,
        "reuse": True,
        "parts": True,
    }
    # And the file is added before any piece moves, every piece lands once with its own bytes, then they are joined
    assert control.calls[0] == ("POST", _MODELS, {"blobId": "blob-1", "type": "loras"})
    puts = [index for index, line in enumerate(log) if line.startswith("put ")]
    complete = log.index("builder POST /v1/blobs/blob-1/complete")
    assert log.index(f"deploy POST {_MODELS}") < min(puts) and max(puts) < complete
    assert storage.bodies() == {
        "https://storage.test/part-1?sig=a": b"0123",
        "https://storage.test/part-2?sig=a": b"4567",
        "https://storage.test/part-3?sig=a": b"89",
    }
    assert all(headers == {"x-goog-if-generation-match": "0"} for _, _, headers, _ in storage.puts)
    assert sorted(size for _, _, _, size in storage.puts) == [2, 4, 4]
    # And it reads the entry until it is ready
    assert control.count("GET", _MODELS) == 2
    assert complete < log.index(f"deploy GET {_MODELS}")
    data = _envelope(result)["data"]
    assert data["model"]["status"] == "ready"
    assert data["uploaded"] is True
    _validate(data)


def test_add_prints_each_status_it_waits_through(tmp_path, monkeypatch) -> None:
    # Given
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", "ready")), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", pretty=True)

    # Then
    assert result.exit_code == 0, result.output
    assert "lora.safetensors: adding" in result.stdout
    assert "lora.safetensors: ready" in result.stdout
    assert "Added loras/lora.safetensors to deployment dep-1; it is ready." in result.stdout


def test_no_wait_returns_once_the_pieces_are_joined(tmp_path, monkeypatch) -> None:
    # Given
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    # Then
    assert result.exit_code == 0, result.output
    assert "builder POST /v1/blobs/blob-1/complete" in log
    assert control.count("GET", _MODELS) == 0
    assert _envelope(result)["data"]["model"]["status"] == "adding"


def test_links_about_to_expire_are_swapped_for_fresh_ones_in_one_ask(tmp_path, monkeypatch) -> None:
    # Given links for pieces 2 and 3 that have run out
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    answer = {"blobId": "blob-1", "deduplicated": False, "parts": [*_parts(1), *_parts(2, 3, expires=_EXPIRED)]}
    model, _ = _install(monkeypatch, tmp_path, control, answer, storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    # Then one ask brings fresh links, and the two pieces go up on them, never on the ones that ran out
    assert result.exit_code == 0, result.output
    assert log.count("builder POST /v1/blobs/blob-1/parts") == 1
    bodies = storage.bodies()
    assert bodies["https://storage.test/part-2?sig=fresh"] == b"4567"
    assert bodies["https://storage.test/part-3?sig=fresh"] == b"89"
    assert not {"https://storage.test/part-2?sig=a", "https://storage.test/part-3?sig=a"} & set(bodies)
    # Piece 1 goes on whichever link it held when it started; the refresh may have come first.
    assert {(urlsplit(url).path, body) for url, body in bodies.items()} == {
        ("/part-1", b"0123"),
        ("/part-2", b"4567"),
        ("/part-3", b"89"),
    }
    assert storage.landed == {1, 2, 3}


def test_a_refused_link_is_sent_again_once_on_a_fresh_one(tmp_path, monkeypatch) -> None:
    # Given the bucket refusing piece 2's first link, as it refuses an expired one
    log: list[str] = []
    control, storage = Control(log), Storage(log, refuse={"https://storage.test/part-2?sig=a"})
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    # Then
    assert result.exit_code == 0, result.output
    sent = [url for url, _, _, _ in storage.puts if "part-2" in url]
    assert sent == ["https://storage.test/part-2?sig=a", "https://storage.test/part-2?sig=fresh"]
    assert storage.bodies()["https://storage.test/part-2?sig=fresh"] == b"4567"
    assert storage.landed == {1, 2, 3}
    assert "builder POST /v1/blobs/blob-1/complete" in log


def test_a_builder_without_pieces_gets_one_put_and_the_add_once_it_ends(tmp_path, monkeypatch) -> None:
    # Given a builder that ignores `parts` and answers one link
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    answer = {"blobId": "blob-1", "deduplicated": False, "uploadUrl": "https://storage.test/whole?sig=a"}
    model, _ = _install(monkeypatch, tmp_path, control, answer, storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    # Then
    assert result.exit_code == 0, result.output
    assert storage.bodies() == {"https://storage.test/whole?sig=a": _CONTENT}
    assert log.index("put https://storage.test/whole?sig=a") < log.index(f"deploy POST {_MODELS}")
    assert not any("/parts" in line or "/complete" in line for line in log)


def test_bytes_the_workspace_already_holds_are_not_sent_again(tmp_path, monkeypatch) -> None:
    # Given
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, {"blobId": "blob-1", "deduplicated": True}, storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    # Then
    assert result.exit_code == 0, result.output
    assert storage.puts == []
    assert control.calls[0] == ("POST", _MODELS, {"blobId": "blob-1", "type": "loras"})
    assert not any("/complete" in line for line in log)
    assert _envelope(result)["data"]["uploaded"] is False


def test_a_404_says_adding_is_not_on_and_sends_no_bytes(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy where the route is off: its 404 is an unknown route's, with no code
    log: list[str] = []
    control, storage = Control(log, refusal=(404, {"message": "Not Found"})), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_models_unavailable"
    assert error["message"] == "Adding models to a deployment is not on for this workspace yet."
    assert error["hint"] == (
        f"To upload it to the workspace library instead: comfy assets library upload {shlex.quote(str(model))} "
        "--type loras"
    )
    assert storage.puts == []
    assert not any("/complete" in line for line in log)


def test_a_404_prints_the_not_on_line_then_the_library_line(tmp_path, monkeypatch) -> None:
    # Given
    log: list[str] = []
    control, storage = Control(log, refusal=(404, {"message": "Not Found"})), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", pretty=True)

    # Then
    assert result.exit_code == 1
    not_on = result.output.index("Adding models to a deployment is not on for this workspace yet.")
    assert not_on < result.output.index("To upload it to the workspace library instead: comfy assets library upload")
    assert storage.puts == []


@pytest.mark.parametrize(
    ("path", "shown"),
    [
        (r"C:\models\lora.safetensors", r"C:\models\lora.safetensors"),
        (r"C:\My Models\lora.safetensors", r'"C:\My Models\lora.safetensors"'),
    ],
)
def test_on_windows_the_library_hint_quotes_the_path_the_way_cmd_reads_it(monkeypatch, path: str, shown: str) -> None:
    # Given a Windows shell, where single quotes are part of the file name
    monkeypatch.setattr(sys, "platform", "win32")

    # When
    error = deploy_models.ModelsUnavailableError("dep-1", Path(path), "loras")

    # Then the path is in double quotes only when it needs them, and never in single quotes
    assert (
        error.hint == f"To upload it to the workspace library instead: comfy assets library upload {shown} --type loras"
    )
    assert "'" not in error.hint


def test_on_windows_the_rm_and_ls_hints_quote_the_way_cmd_reads_them(monkeypatch) -> None:
    # Given a Windows shell and a file name with a space in it
    monkeypatch.setattr(sys, "platform", "win32")
    entry = {**_ENTRY, "filename": "my lora.safetensors", "status": "failed"}

    # When
    not_added = deploy_models.ModelNotAddedError("dep-1", entry)
    twice = deploy_models.ModelNotFoundError("dep-1", "my lora.safetensors", [entry, entry])

    # Then
    assert not_added.hint == 'run `comfy deploy models rm dep-1 "my lora.safetensors"`, then add it again'
    assert "`comfy deploy models ls dep-1 --json`" in twice.hint


@pytest.mark.parametrize(
    "path",
    [
        r"C:\My Models\$(whoami).safetensors",
        r"C:\models\%USERPROFILE%.safetensors",
        r"C:\models\lo`ra.safetensors",
        r'C:\models\lo"ra.safetensors',
        r"C:\models\lora!.safetensors",
        "C:\\models\\it\u2019s.safetensors",
    ],
)
def test_on_windows_a_path_a_shell_would_expand_is_a_placeholder_in_the_library_hint(monkeypatch, path: str) -> None:
    # Given a Windows shell, which may be CMD or PowerShell, and a path one of them would expand or end a quote on
    monkeypatch.setattr(sys, "platform", "win32")

    # When
    error = deploy_models.ModelsUnavailableError("dep-1", Path(path), "loras")

    # Then the command shows where the path goes, and the path itself is named after it, outside the command
    assert error.hint == (
        "To upload it to the workspace library instead: comfy assets library upload <file> --type loras, "
        f"where <file> is {path}"
    )


def test_on_windows_a_server_file_name_a_shell_would_expand_is_a_placeholder_in_the_rm_hint(monkeypatch) -> None:
    # Given a Windows shell and a file name, from comfy-deploy, holding a PowerShell variable
    monkeypatch.setattr(sys, "platform", "win32")
    entry = {**_ENTRY, "filename": "lora$env:USERNAME.safetensors", "status": "failed"}

    # When
    not_added = deploy_models.ModelNotAddedError("dep-1", entry)
    missing = deploy_models.ModelNotFoundError("dep-$x", "lora.safetensors", [])

    # Then the commands hold placeholders, and the message and details still name the values
    assert not_added.hint == "run `comfy deploy models rm dep-1 <file>`, then add it again"
    assert str(not_added).startswith("lora$env:USERNAME.safetensors was not added")
    assert not_added.details["model"]["filename"] == "lora$env:USERNAME.safetensors"
    assert missing.hint == "run `comfy deploy models ls <deployment>` to see its model files"
    assert "deployment dep-$x" in str(missing)


def test_elsewhere_the_library_hint_keeps_posix_shell_quoting(monkeypatch) -> None:
    # Given a POSIX shell
    monkeypatch.setattr(sys, "platform", "linux")
    model = Path("/models/my lora.safetensors")

    # When
    error = deploy_models.ModelsUnavailableError("dep-1", model, "loras")

    # Then
    assert error.hint == (
        "To upload it to the workspace library instead: "
        "comfy assets library upload '/models/my lora.safetensors' --type loras"
    )


def test_elsewhere_a_path_with_a_subexpression_is_shown_single_quoted_not_as_a_placeholder(monkeypatch) -> None:
    # Given a POSIX shell, where single quotes keep every character literal
    monkeypatch.setattr(sys, "platform", "linux")
    model = Path("/models/$(whoami) `id` %HOME%.safetensors")

    # When
    error = deploy_models.ModelsUnavailableError("dep-1", model, "loras")

    # Then
    upload = shlex.join(["comfy", "assets", "library", "upload", str(model), "--type", "loras"])
    assert error.hint == f"To upload it to the workspace library instead: {upload}"


def _pwsh_reads(command: str, pwsh: str) -> subprocess.CompletedProcess[str]:
    """``command`` pasted into PowerShell, with ``comfy`` a function printing the arguments it got as JSON."""
    script = (
        "function pwned { 'PWNED' }\n"
        "function comfy { ConvertTo-Json -Compress -InputObject @($args | ForEach-Object { [string]$_ }) }\n"
        f"{command}\n"
    )
    return subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=60
    )


@pytest.mark.parametrize(
    "path",
    [
        r"C:\models\lora.safetensors",
        r"C:\My Models\lora.safetensors",
        r"C:\My Models\lora (v2) [best] {x} @home, #1 + co's & more; ~=.safetensors",
    ],
)
def test_powershell_reads_the_windows_library_hint_path_literally(monkeypatch, path: str) -> None:
    # Given PowerShell's own parser
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is not on PATH")
    monkeypatch.setattr(sys, "platform", "win32")

    # When the suggested command is pasted into it
    error = deploy_models.ModelsUnavailableError("dep-1", Path(path), "loras")
    command = error.hint.removeprefix("To upload it to the workspace library instead: ")
    read = _pwsh_reads(command, pwsh)

    # Then comfy gets the path as it is
    assert read.returncode == 0, read.stderr
    assert json.loads(read.stdout) == ["assets", "library", "upload", path, "--type", "loras"]


def test_powershell_runs_nothing_a_windows_hint_path_names(monkeypatch) -> None:
    # Given PowerShell's own parser and a path holding a subexpression
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is not on PATH")
    monkeypatch.setattr(sys, "platform", "win32")
    entry = {**_ENTRY, "filename": "$(pwned).safetensors", "status": "failed"}

    # When the suggested commands are pasted into it
    library = deploy_models.ModelsUnavailableError("dep-1", Path(r"C:\My Models\$(pwned).safetensors"), "loras")
    upload = library.hint.removeprefix("To upload it to the workspace library instead: ").split(", where ")[0]
    rm = deploy_models.ModelNotAddedError("dep-1", entry).hint.split("`")[1]

    # Then the subexpression never runs
    for command in (upload, rm):
        read = _pwsh_reads(command, pwsh)
        assert "PWNED" not in read.stdout + read.stderr


@pytest.mark.parametrize("server_code", ["NOT_FOUND", "UPLOAD_NOT_FOUND"])
def test_a_404_comfy_deploy_names_stays_its_own_error(tmp_path, monkeypatch, server_code: str) -> None:
    # Given
    log: list[str] = []
    refusal = (404, {"error": server_code, "message": "No model upload with that id is in this workspace."})
    control, storage = Control(log, refusal=refusal), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_not_found"
    assert error["details"]["server_code"] == server_code
    assert storage.puts == []


def test_a_409_prints_comfy_deploys_sentence_and_sends_no_bytes(tmp_path, monkeypatch) -> None:
    # Given
    log: list[str] = []
    sentence = "The release's models and the added files need more storage than this deployment can be given."
    control = Control(log, refusal=(409, {"error": "MODEL_STORAGE_EXCEEDED", "message": sentence}))
    storage = Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_conflict"
    assert error["message"] == sentence
    assert error["details"]["server_code"] == "MODEL_STORAGE_EXCEEDED"
    assert storage.puts == []


def test_a_builder_without_pieces_sends_the_whole_file_before_adding_is_refused(tmp_path, monkeypatch) -> None:
    # Given a builder that answers one link, and a comfy-deploy where adding models is off
    log: list[str] = []
    control, storage = Control(log, refusal=(404, {"message": "Not Found"})), Storage(log)
    answer = {"blobId": "blob-1", "deduplicated": False, "uploadUrl": "https://storage.test/whole?sig=a"}
    model, _ = _install(monkeypatch, tmp_path, control, answer, storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then the refusal is the same, but only after the whole file went up, which the docs now say
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_models_unavailable"
    assert storage.bodies() == {"https://storage.test/whole?sig=a": _CONTENT}
    assert log.index("put https://storage.test/whole?sig=a") < log.index(f"deploy POST {_MODELS}")


def test_a_file_removed_after_the_check_is_a_file_error_not_a_crash(tmp_path, monkeypatch) -> None:
    # Given a file that is gone by the time it is hashed
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    real = deploy_models.file_digests

    def removed_first(path: Path):
        path.unlink()
        return real(path)

    monkeypatch.setattr(deploy_models, "file_digests", removed_first)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_asset_upload_failed"
    assert error["details"] == {"local_path": str(model), "reason": "file_not_found"}
    assert storage.puts == []
    assert control.count("POST", _MODELS) == 0


def test_a_file_removed_between_pieces_is_a_file_error_and_is_not_joined(tmp_path, monkeypatch) -> None:
    # Given pieces sent one at a time, and a file that is gone once the first piece lands
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())
    storage = Storage(log)

    def put_then_remove(url, **kwargs):
        answer = storage(url, **kwargs)
        model.unlink(missing_ok=True)
        return answer

    monkeypatch.setattr("requests.put", put_then_remove)
    monkeypatch.setattr(builder_api, "_PARTS_AT_ONCE", 1)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_asset_upload_failed"
    assert error["details"]["reason"] == "file_not_found"
    assert [url for url, _, _, _ in storage.puts] == ["https://storage.test/part-1?sig=a"]
    assert not any("/complete" in line for line in log)


def test_a_piece_whose_file_read_fails_is_a_file_error_not_a_dropped_connection() -> None:
    # Given a piece reader over a file that fails to read
    class Broken(io.BytesIO):
        def read(self, *_args) -> bytes:
            raise PermissionError(13, "Permission denied")

    reader = builder_api._PieceReader(Broken(), 4, None, threading.Event(), Path("lora.safetensors"))

    # When / Then: not an OSError, so requests passes it through instead of calling it a dropped connection
    with pytest.raises(builder_api.BlobFileError) as raised:
        reader.read(4)
    assert not isinstance(raised.value, OSError)
    assert isinstance(raised.value.error, PermissionError)


@pytest.mark.parametrize(
    "dropped",
    [http.client.RemoteDisconnected("Remote end closed connection without response"), http.client.IncompleteRead(b"")],
)
def test_a_control_plane_answer_cut_short_is_a_server_error_not_a_crash(tmp_path, monkeypatch, dropped) -> None:
    # Given a comfy-deploy whose answer to the add is cut short while it is read
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    def cut_short(url, target, *, method="GET", body=None, **kwargs):
        if (method, urlsplit(url).path) == ("POST", _MODELS):
            raise dropped
        return control(url, target, method=method, body=body, **kwargs)

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", cut_short)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_server_error"
    assert error["message"]
    assert storage.puts == []


def test_a_connection_reset_quoting_a_signed_link_is_a_server_error_without_the_signature(
    tmp_path, monkeypatch
) -> None:
    # Given the bucket's connection reset with a message that quotes the presigned link
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())

    def reset(url, **_kwargs):
        raise ConnectionResetError(f"connection reset sending to {url}")

    monkeypatch.setattr("requests.put", reset)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_server_error"
    assert "https://storage.test/part-" in error["message"]
    assert "sig=" not in result.output


def test_a_file_that_fails_while_waited_on_exits_without_its_failure_text(tmp_path, monkeypatch) -> None:
    # Given
    log: list[str] = []
    control, storage = Control(log, statuses=("failed",)), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_failed"
    assert error["message"] == "lora.safetensors was not added to this deployment. Remove it and add it again."
    assert "quota" not in result.output


def test_a_file_that_never_settles_ends_the_wait_saying_it_may_still_be_adding(tmp_path, monkeypatch) -> None:
    # Given a file that reads adding on every poll
    log: list[str] = []
    control, storage = Control(log, statuses=("adding",)), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 75, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_wait_timeout"
    assert error["message"] == "lora.safetensors was still adding after 30 minutes of waiting; it may still be adding"
    assert error["hint"] == "run `comfy deploy models ls dep-1` to see when it is ready"
    assert error["details"]["waitedSeconds"] == 1800.0
    assert error["details"]["model"]["status"] == "adding"
    assert control.count("GET", _MODELS) == 900


def test_a_file_removed_while_waited_on_says_so_and_hints_only_to_add_it_again(tmp_path, monkeypatch) -> None:
    # Given a file that leaves the listing while it is being added
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", _GONE)), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    # Then
    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_removed"
    assert error["message"] == "lora.safetensors is no longer on deployment dep-1; it was removed while being added"
    add = shlex.join(["comfy", "deploy", "models", "add", "dep-1", str(model), "--type", "loras"])
    assert error["hint"] == f"add it again: `{add}`"
    assert " rm " not in error["hint"]


# --- ls ---------------------------------------------------------------------------

_LISTED: list[JsonObject] = [
    {**_ENTRY, "status": "ready"},
    {
        "id": "model-2",
        "type": "checkpoints",
        "filename": "base.safetensors",
        "status": "failed",
        "sizeBytes": 2048,
        "failure": "copy to volume vol-9 failed: quota",
    },
]


def test_ls_prints_folder_name_size_and_status_and_says_what_to_do_with_a_failed_file(tmp_path, monkeypatch) -> None:
    # Given
    _install(monkeypatch, tmp_path, Control([], models=_LISTED))

    # When
    result = _invoke("ls", "dep-1", pretty=True)

    # Then
    assert result.exit_code == 0, result.output
    assert "loras  lora.safetensors  10 B  ready" in result.stdout
    assert "checkpoints  base.safetensors  2.0 KB  failed" in result.stdout
    assert "base.safetensors was not added to this deployment. Remove it and add it again." in result.stdout
    assert "quota" not in result.output


def test_ls_json_lists_the_entries_without_their_failure_text(tmp_path, monkeypatch) -> None:
    # Given
    _install(monkeypatch, tmp_path, Control([], models=_LISTED))

    # When
    result = _invoke("ls", "dep-1")

    # Then
    assert result.exit_code == 0, result.output
    envelope = _envelope(result)
    assert envelope["command"] == "deploy models ls"
    data = envelope["data"]
    assert data["deploymentId"] == "dep-1"
    assert [entry["filename"] for entry in data["models"]] == ["lora.safetensors", "base.safetensors"]
    assert all("failure" not in entry for entry in data["models"])
    _validate(data)


def test_a_deployment_is_named_the_way_other_deploy_commands_name_it(tmp_path, monkeypatch) -> None:
    # Given deployment dep-1, named production, of Build "example"
    control = Control([], models=[])
    _install(monkeypatch, tmp_path, control)

    # When
    result = _invoke("ls", "example/production")

    # Then
    assert result.exit_code == 0, result.output
    assert _envelope(result)["data"]["deploymentId"] == "dep-1"
    assert control.count("GET", _MODELS) == 1


# --- rm ---------------------------------------------------------------------------


def _promptable(monkeypatch, answer: bool, asked: list[str]) -> None:
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", agentic=False, source_env=None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: asked.append(question) or answer)


def test_rm_asks_once_with_the_remove_text_and_deletes_once(tmp_path, monkeypatch) -> None:
    # Given
    control, asked = Control([], models=_LISTED), []
    _install(monkeypatch, tmp_path, control)
    _promptable(monkeypatch, True, asked)

    # When
    result = _invoke("rm", "dep-1", "lora.safetensors", pretty=True)

    # Then
    assert result.exit_code == 0, result.output
    assert asked == [_REMOVE_TEXT]
    assert control.count("DELETE", f"{_MODELS}/model-1") == 1


def test_rm_declined_deletes_nothing(tmp_path, monkeypatch) -> None:
    # Given
    control, asked = Control([], models=_LISTED), []
    _install(monkeypatch, tmp_path, control)
    _promptable(monkeypatch, False, asked)

    # When
    result = _invoke("rm", "dep-1", "lora.safetensors", pretty=True)

    # Then
    assert result.exit_code == 0, result.output
    assert len(asked) == 1
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_rm_yes_skips_the_question_prints_the_text_and_deletes_once(tmp_path, monkeypatch) -> None:
    # Given
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda _question: pytest.fail("asked despite --yes"))

    # When
    result = _invoke("rm", "dep-1", "lora.safetensors", "--yes", pretty=True)

    # Then
    assert result.exit_code == 0, result.output
    assert _REMOVE_TEXT in " ".join(result.stdout.split())
    assert control.count("DELETE", f"{_MODELS}/model-1") == 1


def test_rm_json_payload_names_the_entry_it_removes(tmp_path, monkeypatch) -> None:
    # Given
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)

    # When
    result = _invoke("rm", "dep-1", "loras/lora.safetensors", "--yes")

    # Then
    assert result.exit_code == 0, result.output
    data = _envelope(result)["data"]
    assert data == {"deploymentId": "dep-1", "model": {**_ENTRY, "status": "removing"}}
    _validate(data)


def test_rm_without_yes_where_nothing_can_answer_is_refused(tmp_path, monkeypatch) -> None:
    # Given
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)

    # When
    result = _invoke("rm", "dep-1", "lora.safetensors")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_remove_needs_confirm"
    assert error["details"]["modelId"] == "model-1"
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_rm_of_a_file_the_deployment_does_not_hold_is_refused(tmp_path, monkeypatch) -> None:
    # Given
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)

    # When
    result = _invoke("rm", "dep-1", "other.safetensors", "--yes")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_model_not_found"
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_rm_of_a_name_in_two_folders_hints_with_the_deployment(tmp_path, monkeypatch) -> None:
    # Given the same file name in two folders
    twice = [{**_ENTRY, "status": "ready"}, {**_ENTRY, "id": "model-2", "type": "checkpoints", "status": "ready"}]
    control = Control([], models=twice)
    _install(monkeypatch, tmp_path, control)

    # When
    result = _invoke("rm", "dep-1", "lora.safetensors", "--yes")

    # Then the hint is a command that runs as written
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_not_found"
    assert "`comfy deploy models ls dep-1 --json`" in error["hint"]
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_the_model_names_given_to_the_models_commands_are_not_tracked(tmp_path, monkeypatch) -> None:
    # Given analytics recording every event, and a private model file name
    events: list[tuple[str, JsonObject]] = []
    monkeypatch.setattr(
        "comfy_cli.tracking.track_event", lambda name, properties=None, **_kw: events.append((name, properties or {}))
    )
    control, asked = Control([], models=[{**_ENTRY, "filename": "client-secret.safetensors"}]), []
    _install(monkeypatch, tmp_path, control)
    _promptable(monkeypatch, False, asked)

    # When the removal is declined, and the file is listed
    removed = _invoke("rm", "dep-1", "client-secret.safetensors", pretty=True)
    listed = _invoke("ls", "dep-1")

    # Then the name never leaves the machine, while the positional syntax still works
    assert removed.exit_code == 0, removed.output
    assert listed.exit_code == 0, listed.output
    assert len(asked) == 1
    started = dict(events)
    assert started["deploy:rm_cmd"]["file"] == "<redacted>"
    assert started["deploy:ls_cmd"] == {"deployment": "dep-1"}
    assert "client-secret" not in json.dumps(events)


def test_the_file_given_to_models_add_is_not_tracked(tmp_path, monkeypatch) -> None:
    # Given analytics recording every event
    events: list[tuple[str, JsonObject]] = []
    monkeypatch.setattr(
        "comfy_cli.tracking.track_event", lambda name, properties=None, **_kw: events.append((name, properties or {}))
    )
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    # When
    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    # Then
    assert result.exit_code == 0, result.output
    assert dict(events)["deploy:add_cmd"] == {"deployment": "dep-1", "folder": "loras", "wait": False}
    assert str(tmp_path) not in json.dumps(events)


# --- stopping an upload in pieces -------------------------------------------------------


def _parts_upload() -> PartsUpload:
    return PartsUpload("blob-1", False, builder_api._blob_parts({"parts": _parts(1, 2, 3)}), None)


@pytest.fixture
def default_sigint():
    """Ctrl-C as Python's default handler raises it, whatever an earlier test installed."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def test_ctrl_c_stops_the_pieces_still_reading_the_file(tmp_path, monkeypatch, default_sigint) -> None:
    # Given pieces that keep reading their file until something stops them, and Ctrl-C once they started
    log: list[str] = []
    monkeypatch.setattr("comfy_cli.builder_api.request_json", Builder(log, _in_pieces(), Storage(log)))
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)
    started, begun, stopped = threading.Event(), [], []

    def slow_put(url, data=None, **_kwargs):
        begun.append(url)
        if not started.is_set():
            started.set()
            _thread.interrupt_main()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                data.read(1)
            except Exception:
                stopped.append(url)
                raise
            time.sleep(0.01)
        return _Response(200)

    monkeypatch.setattr("requests.put", slow_put)
    upload = _parts_upload()

    # When
    began = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        BuilderClient("https://builder.test", "tok").upload_blob_parts(upload.blob_id, upload.parts, model)

    # Then every piece that began stopped reading, well before it would have finished, and nothing was joined
    assert time.monotonic() - began < 4
    assert begun
    assert sorted(stopped) == sorted(begun)
    assert "builder POST /v1/blobs/blob-1/complete" not in log


def test_ctrl_c_is_raised_without_waiting_on_a_piece_waiting_for_its_answer(
    tmp_path, monkeypatch, default_sigint
) -> None:
    # Given a piece whose bytes are all sent and whose answer never comes, and Ctrl-C then
    log: list[str] = []
    monkeypatch.setattr("comfy_cli.builder_api.request_json", Builder(log, _in_pieces(), Storage(log)))
    monkeypatch.setattr(builder_api, "_CANCEL_GRACE_SECONDS", 0.2, raising=False)
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)
    answer, threads = threading.Event(), []

    def hung_put(url, data=None, **_kwargs):
        b"".join(iter(lambda: data.read(3), b""))
        if not threads:
            threads.append(threading.current_thread())
            _thread.interrupt_main()
        answer.wait(10)
        return _Response(200)

    monkeypatch.setattr("requests.put", hung_put)
    upload = _parts_upload()

    # When
    began = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            BuilderClient("https://builder.test", "tok").upload_blob_parts(upload.blob_id, upload.parts, model)
        waited = time.monotonic() - began
    finally:
        answer.set()

    # Then it is raised within the grace, and the stuck PUT's thread cannot hold the process open at exit
    assert waited < 5
    assert threads[0].daemon


# --- the library uploads ---------------------------------------------------------------


@pytest.mark.parametrize("command", [["assets", "library", "upload"], ["model", "upload"]])
def test_the_library_uploads_point_at_deploy_models_first_and_still_upload(tmp_path, monkeypatch, command) -> None:
    # Given
    uploads: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "comfy_cli.command.assets_library.run_upload", lambda *args, **kwargs: uploads.append((args, kwargs))
    )
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)

    # When
    args = ["--no-json", *command, str(model), "--type", "loras"]
    result = CliRunner().invoke(app, args, env={"COLUMNS": "400", "NO_COLOR": "1"})

    # Then
    assert result.exit_code == 0, result.output
    assert DEPLOYMENT_POINTER == (
        "To add a model to a deployment, use: comfy deploy models add <deployment> <file> --type <folder>"
    )
    assert DEPLOYMENT_POINTER in result.stdout
    assert len(uploads) == 1
    assert uploads[0][0] == (model, "loras")
