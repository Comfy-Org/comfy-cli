from __future__ import annotations

import _thread
import copy
import hashlib
import http.client
import io
import json
import os
import shlex
import signal
import sys
import threading
import time
import urllib.error
from email.message import Message
from pathlib import Path, PurePosixPath
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
    def __init__(self, log: list[str], refuse: set[str] | None = None, answers: dict[str, int] | None = None) -> None:
        self.log = log
        self.refuse = set(refuse or ())
        self.answers = dict(answers or {})
        self.redirects: list[bool] = []
        self.puts: list[tuple[str, bytes, dict, int | None]] = []
        self.landed: set[int] = set()
        self._lock = threading.Lock()

    def __call__(self, url, data=None, headers=None, timeout=None, allow_redirects=True):
        size = len(data) if hasattr(data, "__len__") else None
        body = b"".join(iter(lambda: data.read(3), b""))
        with self._lock:
            self.log.append(f"put {url}")
            self.puts.append((url, body, dict(headers or {}), size))
            self.redirects.append(allow_redirects)
            if url in self.answers:
                return _Response(self.answers[url])
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


def test_add_adds_before_the_pieces_land_then_joins_them_and_waits_until_ready(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", "ready")), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

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
    assert control.count("GET", _MODELS) == 2
    assert complete < log.index(f"deploy GET {_MODELS}")
    data = _envelope(result)["data"]
    assert data["model"]["status"] == "ready"
    assert data["uploaded"] is True
    _validate(data)


def test_add_prints_each_status_it_waits_through(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", "ready")), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", pretty=True)

    assert result.exit_code == 0, result.output
    assert "lora.safetensors: adding" in result.stdout
    assert "lora.safetensors: ready" in result.stdout
    assert "Added loras/lora.safetensors to deployment dep-1; it is ready." in result.stdout


def test_no_wait_returns_once_the_pieces_are_joined(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    assert result.exit_code == 0, result.output
    assert "builder POST /v1/blobs/blob-1/complete" in log
    assert control.count("GET", _MODELS) == 0
    assert _envelope(result)["data"]["model"]["status"] == "adding"


def test_links_about_to_expire_are_swapped_for_fresh_ones_in_one_ask(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    answer = {"blobId": "blob-1", "deduplicated": False, "parts": [*_parts(1), *_parts(2, 3, expires=_EXPIRED)]}
    model, _ = _install(monkeypatch, tmp_path, control, answer, storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    assert result.exit_code == 0, result.output
    assert log.count("builder POST /v1/blobs/blob-1/parts") == 1
    bodies = storage.bodies()
    assert bodies["https://storage.test/part-2?sig=fresh"] == b"4567"
    assert bodies["https://storage.test/part-3?sig=fresh"] == b"89"
    assert not {"https://storage.test/part-2?sig=a", "https://storage.test/part-3?sig=a"} & set(bodies)
    assert {(urlsplit(url).path, body) for url, body in bodies.items()} == {
        ("/part-1", b"0123"),
        ("/part-2", b"4567"),
        ("/part-3", b"89"),
    }
    assert storage.landed == {1, 2, 3}


def test_a_refused_link_is_sent_again_once_on_a_fresh_one(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log, refuse={"https://storage.test/part-2?sig=a"})
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    assert result.exit_code == 0, result.output
    sent = [url for url, _, _, _ in storage.puts if "part-2" in url]
    assert sent == ["https://storage.test/part-2?sig=a", "https://storage.test/part-2?sig=fresh"]
    assert storage.bodies()["https://storage.test/part-2?sig=fresh"] == b"4567"
    assert storage.landed == {1, 2, 3}
    assert "builder POST /v1/blobs/blob-1/complete" in log


def test_a_piece_sent_again_on_a_fresh_link_is_counted_once_in_the_progress(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    storage = Storage(log, refuse={"https://storage.test/part-2?sig=a"})
    monkeypatch.setattr("comfy_cli.builder_api.request_json", Builder(log, _in_pieces(), storage))
    monkeypatch.setattr("requests.put", storage)
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)
    counted: list[int] = []
    upload = _parts_upload()

    BuilderClient("https://builder.test", "tok").upload_blob_parts(upload.blob_id, upload.parts, model, counted.append)

    assert len([url for url, _, _, _ in storage.puts if "part-2" in url]) == 2
    assert sum(counted) == len(_CONTENT)


def _pieces_against(monkeypatch, tmp_path: Path, storage: Storage, answer: JsonObject) -> tuple:
    log = storage.log
    monkeypatch.setattr("comfy_cli.builder_api.request_json", Builder(log, answer, storage))
    monkeypatch.setattr("requests.put", storage)
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)
    plan = builder_api._blob_parts(answer)
    return log, lambda: BuilderClient("https://builder.test", "tok").upload_blob_parts("blob-1", plan, model)


def test_a_piece_storage_already_holds_counts_as_landed(tmp_path, monkeypatch) -> None:
    storage = Storage([], answers={"https://storage.test/part-2?sig=a": 412})
    log, upload = _pieces_against(monkeypatch, tmp_path, storage, _in_pieces())

    upload()

    assert [url for url, _, _, _ in storage.puts if "part-2" in url] == ["https://storage.test/part-2?sig=a"]
    assert "builder POST /v1/blobs/blob-1/parts" not in log
    assert "builder POST /v1/blobs/blob-1/complete" in log


def test_a_piece_put_answered_with_a_redirect_is_refused_not_followed(tmp_path, monkeypatch) -> None:
    storage = Storage([], answers={"https://storage.test/part-2?sig=a": 307})
    log, upload = _pieces_against(monkeypatch, tmp_path, storage, _in_pieces())

    with pytest.raises(requests.HTTPError, match="redirected \\(307\\); refusing to follow"):
        upload()

    assert storage.redirects and not any(storage.redirects)
    assert "builder POST /v1/blobs/blob-1/complete" not in log


def test_the_first_failed_piece_stops_the_pieces_still_queued(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder_api, "_PARTS_AT_ONCE", 1)
    storage = Storage([], answers={"https://storage.test/part-1?sig=a": 500})
    log, upload = _pieces_against(monkeypatch, tmp_path, storage, _in_pieces())

    with pytest.raises(requests.HTTPError, match="500"):
        upload()

    assert [url for url, _, _, _ in storage.puts] == ["https://storage.test/part-1?sig=a"]
    assert "builder POST /v1/blobs/blob-1/complete" not in log


def test_a_refresh_that_leaves_out_a_landed_piece_drops_its_old_link(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder_api, "_PARTS_AT_ONCE", 1)
    storage = Storage([])
    storage.landed.add(3)
    answer = {"blobId": "blob-1", "deduplicated": False, "parts": [*_parts(1), *_parts(2, 3, expires=_EXPIRED)]}
    log, upload = _pieces_against(monkeypatch, tmp_path, storage, answer)

    upload()

    assert sorted(url for url, _, _, _ in storage.puts) == [
        "https://storage.test/part-1?sig=a",
        "https://storage.test/part-2?sig=fresh",
    ]
    assert "builder POST /v1/blobs/blob-1/complete" in log


def test_an_add_answer_with_no_entry_is_a_server_error_even_without_waiting(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(DeployClient, "add_deployment_model", lambda self, *_args: {})

    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    assert result.exit_code == 1, result.output
    assert _envelope(result)["error"]["code"] == "deploy_server_error"


@pytest.mark.parametrize("size", [None, "four"])
def test_a_piece_plan_with_a_bad_size_is_a_server_error_without_its_links(tmp_path, monkeypatch, size) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    answer = _in_pieces()
    answer["parts"][1]["sizeBytes"] = size
    model, _ = _install(monkeypatch, tmp_path, control, answer, storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_server_error"
    assert "sig=" not in json.dumps(error)
    assert storage.puts == []
    assert control.count("POST", _MODELS) == 0


def test_a_builder_refusing_the_credential_says_to_sign_in_again(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control)

    def refused(url, _target, **_kwargs):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", Message(), io.BytesIO(b"{}"))

    monkeypatch.setattr("comfy_cli.builder_api.request_json", refused)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_not_signed_in"
    assert error["hint"] == "replace COMFY_BUILDER_TOKEN with a fresh Cloud JWT"


@pytest.mark.parametrize(
    ("refused_path", "status", "answer"),
    [
        ("/v1/blobs", 400, {"error": "INVALID_REQUEST", "message": "sizeBytes must be at least 1"}),
        ("/v1/blobs", 413, {"error": "TOO_LARGE", "message": "a model upload is at most 200 GiB"}),
        ("/v1/blobs/blob-1/complete", 409, {"error": "PARTS_MISSING", "message": "piece 3 has not landed"}),
    ],
)
def test_a_builder_refusal_prints_the_builders_sentence_not_a_network_fault(
    tmp_path, monkeypatch, refused_path: str, status: int, answer: JsonObject
) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    def refusing(url, target, *, method="GET", body=None, **kwargs):
        if (method, urlsplit(url).path) == ("POST", refused_path):
            raise urllib.error.HTTPError(url, status, "refused", Message(), io.BytesIO(json.dumps(answer).encode()))
        return builder(url, target, method=method, body=body, **kwargs)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", refusing)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "build_builder_error"
    assert error["message"] == f"builder call failed ({status}): {answer['error']}: {answer['message']}"
    assert error["details"]["status"] == status
    assert "network" not in error["hint"]
    assert "HTTP Error" not in result.output


def test_a_builder_without_pieces_gets_one_put_and_the_add_once_it_ends(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    answer = {"blobId": "blob-1", "deduplicated": False, "uploadUrl": "https://storage.test/whole?sig=a"}
    model, _ = _install(monkeypatch, tmp_path, control, answer, storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    assert result.exit_code == 0, result.output
    assert storage.bodies() == {"https://storage.test/whole?sig=a": _CONTENT}
    assert log.index("put https://storage.test/whole?sig=a") < log.index(f"deploy POST {_MODELS}")
    assert not any("/parts" in line or "/complete" in line for line in log)


def test_bytes_the_workspace_already_holds_are_not_sent_again(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, {"blobId": "blob-1", "deduplicated": True}, storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    assert result.exit_code == 0, result.output
    assert storage.puts == []
    assert control.calls[0] == ("POST", _MODELS, {"blobId": "blob-1", "type": "loras"})
    assert not any("/complete" in line for line in log)
    assert _envelope(result)["data"]["uploaded"] is False


def test_a_404_says_adding_is_not_on_and_sends_no_bytes(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, refusal=(404, {"message": "Not Found"})), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(sys, "platform", "linux")

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

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
    log: list[str] = []
    control, storage = Control(log, refusal=(404, {"message": "Not Found"})), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", pretty=True)

    assert result.exit_code == 1
    not_on = result.output.index("Adding models to a deployment is not on for this workspace yet.")
    assert not_on < result.output.index("To upload it to the workspace library instead: comfy assets library upload")
    assert storage.puts == []


def test_on_windows_the_hints_print_placeholders_and_name_the_real_values(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    entry = {**_ENTRY, "filename": "my lora.safetensors", "status": "failed"}

    file = Path(r"C:\My Models\lora.safetensors")
    library = deploy_models.ModelsUnavailableError("dep-1", file, "loras")
    not_added = deploy_models.ModelNotAddedError("dep-1", entry)
    removed = deploy_models.ModelRemovedError("dep-1", {**_ENTRY, "status": "adding"}, Path("lora.safetensors"), "x")

    assert library.hint == (
        "To upload it to the workspace library instead: comfy assets library upload <file> --type <folder>, "
        f"where <file> is {file} and <folder> is loras"
    )
    assert not_added.hint == (
        "run `comfy deploy models rm <deployment> <id>`, wait until `comfy deploy models ls <deployment>` no longer "
        "lists it, then add it again"
    )
    assert str(not_added).startswith("my lora.safetensors was not added")
    assert removed.hint == "add it again: `comfy deploy models add <deployment> <file> --type <folder>`"
    assert "deployment dep-1" in str(removed)


def test_elsewhere_the_library_hint_keeps_posix_shell_quoting(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    model = PurePosixPath("/models/my lora.safetensors")

    error = deploy_models.ModelsUnavailableError("dep-1", model, "loras")

    assert error.hint == (
        "To upload it to the workspace library instead: "
        "comfy assets library upload '/models/my lora.safetensors' --type loras"
    )


def test_elsewhere_a_path_with_a_subexpression_is_shown_single_quoted_not_as_a_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    model = PurePosixPath("/models/$(whoami) `id` %HOME%.safetensors")

    error = deploy_models.ModelsUnavailableError("dep-1", model, "loras")

    upload = shlex.join(["comfy", "assets", "library", "upload", str(model), "--type", "loras"])
    assert error.hint == f"To upload it to the workspace library instead: {upload}"


def test_a_file_named_like_an_option_is_given_as_a_path_in_the_hints(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    file = Path("-draft.safetensors")
    shown = os.path.join(os.curdir, "-draft.safetensors")

    unavailable = deploy_models.ModelsUnavailableError("dep-1", file, "loras")
    removed = deploy_models.ModelRemovedError("dep-1", {**_ENTRY, "status": "adding"}, file, "loras")

    assert f"comfy assets library upload {shown} --type loras" in unavailable.hint
    assert f"comfy deploy models add dep-1 {shown} --type loras" in removed.hint
    assert " -draft" not in unavailable.hint + removed.hint


def test_a_file_named_like_an_option_in_the_add_hint_parses_as_the_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.chdir(tmp_path)
    Path("-draft.safetensors").write_bytes(_CONTENT)
    removed = deploy_models.ModelRemovedError("dep-1", {**_ENTRY, "status": "adding"}, Path("-draft.safetensors"), "x")
    seen: list[Path] = []
    monkeypatch.setattr(deploy_models, "add", lambda file, *_args, **_kwargs: seen.append(file) or {})

    words = shlex.split(removed.hint.split("`")[1])
    result = CliRunner().invoke(app, ["--json", *words[1:], "--no-wait"])

    assert seen == [Path("./-draft.safetensors")], result.output


def test_a_listing_carries_no_uploaded_flag_by_the_schema() -> None:
    _validate({"deploymentId": "dep-1", "models": []})
    with pytest.raises(jsonschema.ValidationError):
        _validate({"deploymentId": "dep-1", "models": [], "uploaded": True})


@pytest.mark.parametrize("extra", [{"uploaded": True}, {}])
def test_a_payload_carrying_both_models_and_model_is_refused_by_the_schema(extra: JsonObject) -> None:
    _validate({"deploymentId": "dep-1", "models": [_ENTRY]})
    _validate({"deploymentId": "dep-1", "model": _ENTRY, "uploaded": True})
    _validate({"deploymentId": "dep-1", "model": _ENTRY})
    with pytest.raises(jsonschema.ValidationError):
        _validate({"deploymentId": "dep-1", "models": [], "model": _ENTRY, **extra})


@pytest.mark.parametrize("server_code", ["NOT_FOUND", "UPLOAD_NOT_FOUND"])
def test_a_404_comfy_deploy_names_stays_its_own_error(tmp_path, monkeypatch, server_code: str) -> None:
    log: list[str] = []
    refusal = (404, {"error": server_code, "message": "No model upload with that id is in this workspace."})
    control, storage = Control(log, refusal=refusal), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_not_found"
    assert error["details"]["server_code"] == server_code
    assert storage.puts == []


def test_a_409_prints_comfy_deploys_sentence_and_sends_no_bytes(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    sentence = "The release's models and the added files need more storage than this deployment can be given."
    control = Control(log, refusal=(409, {"error": "MODEL_STORAGE_EXCEEDED", "message": sentence}))
    storage = Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_conflict"
    assert error["message"] == sentence
    assert error["details"]["server_code"] == "MODEL_STORAGE_EXCEEDED"
    assert storage.puts == []


def test_a_builder_without_pieces_sends_the_whole_file_before_adding_is_refused(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, refusal=(404, {"message": "Not Found"})), Storage(log)
    answer = {"blobId": "blob-1", "deduplicated": False, "uploadUrl": "https://storage.test/whole?sig=a"}
    model, _ = _install(monkeypatch, tmp_path, control, answer, storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_models_unavailable"
    assert storage.bodies() == {"https://storage.test/whole?sig=a": _CONTENT}
    assert log.index("put https://storage.test/whole?sig=a") < log.index(f"deploy POST {_MODELS}")


def test_a_file_removed_after_the_check_is_a_file_error_not_a_crash(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    real = deploy_models.file_digests

    def removed_first(path: Path):
        path.unlink()
        return real(path)

    monkeypatch.setattr(deploy_models, "file_digests", removed_first)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_asset_upload_failed"
    assert error["details"] == {"local_path": str(model), "reason": "file_not_found"}
    assert storage.puts == []
    assert control.count("POST", _MODELS) == 0


def test_a_file_removed_between_pieces_is_a_file_error_and_is_not_joined(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())
    storage = Storage(log)
    monkeypatch.setattr("requests.put", storage)
    monkeypatch.setattr(builder_api, "_PARTS_AT_ONCE", 1)
    real_open = Path.open

    def removed_once_a_piece_landed(path: Path, *args, **kwargs):
        if storage.landed and path == model:
            model.unlink(missing_ok=True)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", removed_once_a_piece_landed)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_asset_upload_failed"
    assert error["details"]["reason"] == "file_not_found"
    assert [url for url, _, _, _ in storage.puts] == ["https://storage.test/part-1?sig=a"]
    assert not any("/complete" in line for line in log)


def test_a_piece_whose_file_read_fails_is_a_file_error_not_a_dropped_connection() -> None:
    class Broken(io.BytesIO):
        def read(self, *_args) -> bytes:
            raise PermissionError(13, "Permission denied")

    reader = builder_api._PieceReader(Broken(), 4, None, threading.Event(), Path("lora.safetensors"))

    with pytest.raises(builder_api.BlobFileError) as raised:
        reader.read(4)
    assert not isinstance(raised.value, OSError)
    assert isinstance(raised.value.error, PermissionError)


def test_a_piece_whose_file_ends_before_the_piece_does_is_a_file_error() -> None:
    reader = builder_api._PieceReader(io.BytesIO(b"01"), 4, None, threading.Event(), Path("lora.safetensors"))

    assert reader.read(0) == b""
    assert reader.read(4) == b"01"
    with pytest.raises(builder_api.BlobFileError):
        reader.read(4)


def test_a_piece_that_reads_to_its_end_then_ends_normally() -> None:
    reader = builder_api._PieceReader(io.BytesIO(b"0123"), 4, None, threading.Event(), Path("lora.safetensors"))

    assert reader.read(4) == b"0123"
    assert reader.read(4) == b""


@pytest.mark.parametrize(
    "dropped",
    [http.client.RemoteDisconnected("Remote end closed connection without response"), http.client.IncompleteRead(b"")],
)
def test_a_control_plane_answer_cut_short_is_a_server_error_not_a_crash(tmp_path, monkeypatch, dropped) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    def cut_short(url, target, *, method="GET", body=None, **kwargs):
        if (method, urlsplit(url).path) == ("POST", _MODELS):
            raise dropped
        return control(url, target, method=method, body=body, **kwargs)

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", cut_short)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_server_error"
    assert error["message"]
    assert storage.puts == []


def test_a_connection_reset_quoting_a_signed_link_is_a_server_error_without_the_signature(
    tmp_path, monkeypatch
) -> None:
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())

    def reset(url, **_kwargs):
        raise ConnectionResetError(f"connection reset sending to {url}")

    monkeypatch.setattr("requests.put", reset)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_server_error"
    assert "https://storage.test/part-" in error["message"]
    assert "sig=" not in result.output


def _refused_removal(monkeypatch, control: Control) -> None:
    def refusing(url, target, *, method="GET", body=None, **kwargs):
        if method == "DELETE":
            control.log.append(f"deploy DELETE {urlsplit(url).path} 500")
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", Message(), io.BytesIO(b"{}"))
        return control(url, target, method=method, body=body, **kwargs)

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", refusing)


def _dropped_piece(monkeypatch, raised: BaseException) -> None:
    def put(url, **_kwargs):
        raise raised

    monkeypatch.setattr("requests.put", put)


def test_a_send_that_fails_takes_the_entry_off_the_deployment_again(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())
    monkeypatch.setattr(sys, "platform", "linux")
    _dropped_piece(monkeypatch, ConnectionResetError("connection reset"))

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_server_error"
    assert control.count("DELETE", f"{_MODELS}/model-1") == 1
    assert error["hint"] == (
        "lora.safetensors was taken off the deployment again: wait until `comfy deploy models ls dep-1` no longer "
        "lists it, then add it again"
    )


def test_a_file_error_after_the_add_takes_the_entry_off_and_says_so_instead_of_add_it_again(
    tmp_path, monkeypatch
) -> None:
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())
    monkeypatch.setattr(sys, "platform", "linux")
    _dropped_piece(monkeypatch, builder_api.BlobFileError(model, FileNotFoundError(2, "No such file")))

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_asset_upload_failed"
    assert control.count("DELETE", f"{_MODELS}/model-1") == 1
    assert error["hint"].startswith("lora.safetensors was taken off the deployment again: wait until")


def test_a_failed_send_whose_removal_fails_names_the_rm_command(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())
    monkeypatch.setattr(sys, "platform", "linux")
    _dropped_piece(monkeypatch, builder_api.BlobFileError(model, FileNotFoundError(2, "No such file")))
    _refused_removal(monkeypatch, control)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    assert f"deploy DELETE {_MODELS}/model-1 500" in log
    assert _envelope(result)["error"]["hint"] == (
        "lora.safetensors stays on the deployment reading adding: remove it with "
        "`comfy deploy models rm dep-1 model-1`, wait until `comfy deploy models ls dep-1` no longer lists "
        "it, then add it again"
    )


def _join_raises(monkeypatch, builder: Builder | None, raised: BaseException) -> None:
    assert builder is not None

    def joining(url, target, **kwargs):
        if urlsplit(url).path == "/v1/blobs/blob-1/complete":
            builder.log.append("builder POST /v1/blobs/blob-1/complete")
            raise raised
        return builder(url, target, **kwargs)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", joining)


_MAY_BE_STORED = (
    "lora.safetensors stays on the deployment, as the builder may have stored it: if "
    "`comfy deploy models ls dep-1` shows it failed, or still adding after 30 minutes, remove it with "
    "`comfy deploy models rm dep-1 model-1`, wait until `comfy deploy models ls dep-1` no longer lists it, then add "
    "it again"
)


@pytest.mark.parametrize(
    "raised",
    [
        TimeoutError("timed out"),
        ConnectionResetError("connection reset"),
        urllib.error.HTTPError(
            "https://builder.test/v1/blobs/blob-1/complete", 503, "Unavailable", Message(), io.BytesIO(b"")
        ),
    ],
    ids=["timeout", "reset", "503"],
)
def test_a_join_whose_outcome_is_unknown_keeps_the_entry_and_says_how_to_check_it(
    tmp_path, monkeypatch, raised
) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(sys, "platform", "linux")
    _join_raises(monkeypatch, builder, raised)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    assert "builder POST /v1/blobs/blob-1/complete" in log
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_server_error"
    assert error["hint"] == _MAY_BE_STORED


def test_a_join_the_builder_refuses_takes_the_entry_off_the_deployment_again(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(sys, "platform", "linux")
    refusal = json.dumps({"code": "PART_MISSING", "message": "piece 2 of 3 of upload blob-1 is not uploaded"})
    _join_raises(
        monkeypatch,
        builder,
        urllib.error.HTTPError(
            "https://builder.test/v1/blobs/blob-1/complete", 409, "Conflict", Message(), io.BytesIO(refusal.encode())
        ),
    )

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    assert control.count("DELETE", f"{_MODELS}/model-1") == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_builder_error"
    assert "lora.safetensors was taken off the deployment again" in error["hint"]


def _refused_join(monkeypatch, builder: Builder | None) -> None:
    refusal = json.dumps({"code": "PART_MISSING", "message": "piece 2 of 3 of upload blob-1 is not uploaded"})
    _join_raises(
        monkeypatch,
        builder,
        urllib.error.HTTPError(
            "https://builder.test/v1/blobs/blob-1/complete", 409, "Conflict", Message(), io.BytesIO(refusal.encode())
        ),
    )


def test_a_join_the_builder_refuses_reports_no_finished_upload_in_json(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    _refused_join(monkeypatch, builder)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    types = [event["type"] for event in events if event.get("schema") == "event/1"]
    assert "upload_progress" in types
    assert "upload_complete" not in types
    assert _envelope(result)["error"]["code"] == "build_builder_error"


def test_a_join_the_builder_refuses_prints_no_uploaded_line(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    _refused_join(monkeypatch, builder)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", pretty=True)

    assert result.exit_code == 1, result.output
    shown = " ".join((result.stdout + result.stderr).split())
    assert "1/1 lora.safetensors" in shown
    assert "lora.safetensors: uploaded" not in shown
    assert "builder call failed (409): piece 2 of 3 of upload blob-1 is not uploaded" in shown


def _removal_forbidden(monkeypatch, control: Control) -> None:
    def forbidding(url, target, *, method="GET", body=None, **kwargs):
        if method == "DELETE":
            pytest.fail(f"Ctrl-C sent a removal that can wait on the sign-in lock: DELETE {urlsplit(url).path}")
        return control(url, target, method=method, body=body, **kwargs)

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", forbidding)


def test_ctrl_c_during_the_pieces_sends_no_removal_and_names_the_rm_by_id(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control = Control(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces())
    monkeypatch.setattr(sys, "platform", "linux")
    _dropped_piece(monkeypatch, KeyboardInterrupt())
    _removal_forbidden(monkeypatch, control)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", pretty=True)

    assert result.exit_code == 130, result.output
    assert not any(" DELETE " in line for line in log)
    assert (
        "lora.safetensors stays on the deployment reading adding: remove it with "
        "`comfy deploy models rm dep-1 model-1`" in " ".join(result.output.split())
    )
    assert "builder POST /v1/blobs/blob-1/complete" not in log


def test_ctrl_c_during_the_join_sends_no_removal_and_says_the_builder_may_have_stored_it(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, builder = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(sys, "platform", "linux")
    _join_raises(monkeypatch, builder, KeyboardInterrupt())
    _removal_forbidden(monkeypatch, control)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", pretty=True)

    assert result.exit_code == 130, result.output
    assert not any(" DELETE " in line for line in log)
    assert _MAY_BE_STORED in " ".join(result.output.split())


def test_a_send_that_fails_before_any_entry_exists_removes_nothing(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control = Control(log)
    answer = {"blobId": "blob-1", "deduplicated": False, "uploadUrl": "https://storage.test/whole?sig=a"}
    model, _ = _install(monkeypatch, tmp_path, control, answer)
    _dropped_piece(monkeypatch, ConnectionResetError("connection reset"))

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    assert control.count("POST", _MODELS) == 0
    assert not any(" DELETE " in line for line in log)


def test_a_file_that_fails_while_waited_on_exits_without_its_failure_text(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("failed",)), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_failed"
    assert error["message"] == "lora.safetensors was not added to this deployment. Remove it and add it again."
    assert "quota" not in result.output


def test_a_file_that_never_settles_ends_the_wait_saying_it_may_still_be_adding(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("adding",)), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 75, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_wait_timeout"
    assert error["message"] == "lora.safetensors was still adding after 30 minutes of waiting; it may still be adding"
    assert error["hint"] == "run `comfy deploy models ls dep-1` to see when it is ready"
    assert error["details"]["waitedSeconds"] == 1800.0
    assert error["details"]["model"]["status"] == "adding"
    assert control.count("GET", _MODELS) == 900


def _unanswered_reads(monkeypatch, control: Control, failing: int) -> list[int]:
    reads = [0]

    def flaky(url, target, *, method="GET", body=None, **kwargs):
        if (method, urlsplit(url).path) == ("GET", _MODELS):
            reads[0] += 1
            if 1 < reads[0] <= 1 + failing:
                control.log.append(f"deploy GET {_MODELS} 502")
                raise urllib.error.HTTPError(url, 502, "Bad Gateway", Message(), io.BytesIO(b"{}"))
        return control(url, target, method=method, body=body, **kwargs)

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", flaky)
    return reads


def test_a_read_the_deploy_api_does_not_answer_is_retried_while_waiting(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", "adding", "ready")), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    _unanswered_reads(monkeypatch, control, failing=2)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 0, result.output
    assert log.count(f"deploy GET {_MODELS} 502") == 2
    assert _envelope(result)["data"]["model"]["status"] == "ready"


def test_reads_unanswered_for_the_whole_window_end_the_wait_not_the_file(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("adding",)), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(sys, "platform", "linux")
    _unanswered_reads(monkeypatch, control, failing=1000)

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 75, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_wait_timeout"
    assert error["message"].startswith("the deploy API stopped answering while waiting on lora.safetensors (")
    assert error["message"].endswith("); it may still be adding")
    assert error["hint"] == "run `comfy deploy models ls dep-1` to see when it is ready"
    assert error["details"]["model"]["status"] == "adding"
    assert 1 < log.count(f"deploy GET {_MODELS} 502") < 10


def test_a_file_removed_while_waited_on_says_so_and_hints_only_to_add_it_again(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", _GONE)), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(sys, "platform", "linux")

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_removed"
    assert error["message"] == "lora.safetensors is no longer on deployment dep-1; it was removed while being added"
    add = shlex.join(["comfy", "deploy", "models", "add", "dep-1", str(model), "--type", "loras"])
    assert error["hint"] == f"add it again: `{add}`"
    assert " rm " not in error["hint"]


def test_a_file_whose_removal_starts_while_waited_on_says_to_wait_until_it_is_gone_before_adding_again(
    tmp_path, monkeypatch
) -> None:
    log: list[str] = []
    control, storage = Control(log, statuses=("adding", "removing")), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)
    monkeypatch.setattr(sys, "platform", "linux")

    result = _invoke("add", "dep-1", str(model), "--type", "loras")

    assert result.exit_code == 1, result.output
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_removed"
    assert error["message"] == (
        "lora.safetensors is being removed from deployment dep-1; its removal was started while it was being added"
    )
    add = shlex.join(["comfy", "deploy", "models", "add", "dep-1", str(model), "--type", "loras"])
    assert error["hint"] == f"wait until `comfy deploy models ls dep-1` no longer lists it, then add it again: `{add}`"
    assert error["details"]["model"]["status"] == "removing"
    assert " rm " not in error["hint"]


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
    _install(monkeypatch, tmp_path, Control([], models=_LISTED))

    result = _invoke("ls", "dep-1", pretty=True)

    assert result.exit_code == 0, result.output
    assert "loras  lora.safetensors  10 B  ready" in result.stdout
    assert "checkpoints  base.safetensors  2.0 KB  failed" in result.stdout
    assert "base.safetensors was not added to this deployment. Remove it and add it again." in result.stdout
    assert "quota" not in result.output


def test_ls_json_lists_the_entries_without_their_failure_text(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, tmp_path, Control([], models=_LISTED))

    result = _invoke("ls", "dep-1")

    assert result.exit_code == 0, result.output
    envelope = _envelope(result)
    assert envelope["command"] == "deploy models ls"
    data = envelope["data"]
    assert data["deploymentId"] == "dep-1"
    assert [entry["filename"] for entry in data["models"]] == ["lora.safetensors", "base.safetensors"]
    assert all("failure" not in entry for entry in data["models"])
    _validate(data)


def test_a_deployment_is_named_the_way_other_deploy_commands_name_it(tmp_path, monkeypatch) -> None:
    control = Control([], models=[])
    _install(monkeypatch, tmp_path, control)

    result = _invoke("ls", "example/production")

    assert result.exit_code == 0, result.output
    assert _envelope(result)["data"]["deploymentId"] == "dep-1"
    assert control.count("GET", _MODELS) == 1


def _promptable(monkeypatch, answer: bool, asked: list[str]) -> None:
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", agentic=False, source_env=None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: asked.append(question) or answer)


def test_rm_asks_once_with_the_remove_text_and_deletes_once(tmp_path, monkeypatch) -> None:
    control, asked = Control([], models=_LISTED), []
    _install(monkeypatch, tmp_path, control)
    _promptable(monkeypatch, True, asked)

    result = _invoke("rm", "dep-1", "lora.safetensors", pretty=True)

    assert result.exit_code == 0, result.output
    assert asked == [_REMOVE_TEXT]
    assert control.count("DELETE", f"{_MODELS}/model-1") == 1


def test_rm_declined_deletes_nothing(tmp_path, monkeypatch) -> None:
    control, asked = Control([], models=_LISTED), []
    _install(monkeypatch, tmp_path, control)
    _promptable(monkeypatch, False, asked)

    result = _invoke("rm", "dep-1", "lora.safetensors", pretty=True)

    assert result.exit_code == 0, result.output
    assert len(asked) == 1
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_rm_yes_skips_the_question_prints_the_text_and_deletes_once(tmp_path, monkeypatch) -> None:
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda _question: pytest.fail("asked despite --yes"))

    result = _invoke("rm", "dep-1", "lora.safetensors", "--yes", pretty=True)

    assert result.exit_code == 0, result.output
    assert _REMOVE_TEXT in " ".join(result.stdout.split())
    assert control.count("DELETE", f"{_MODELS}/model-1") == 1


def test_rm_json_payload_names_the_entry_it_removes(tmp_path, monkeypatch) -> None:
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)

    result = _invoke("rm", "dep-1", "loras/lora.safetensors", "--yes")

    assert result.exit_code == 0, result.output
    data = _envelope(result)["data"]
    assert data == {"deploymentId": "dep-1", "model": {**_ENTRY, "status": "removing"}}
    _validate(data)


def test_rm_without_yes_where_nothing_can_answer_is_refused(tmp_path, monkeypatch) -> None:
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)

    result = _invoke("rm", "dep-1", "lora.safetensors")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_remove_needs_confirm"
    assert error["details"]["modelId"] == "model-1"
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_rm_of_a_file_the_deployment_does_not_hold_is_refused(tmp_path, monkeypatch) -> None:
    control = Control([], models=_LISTED)
    _install(monkeypatch, tmp_path, control)

    result = _invoke("rm", "dep-1", "other.safetensors", "--yes")

    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_model_not_found"
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_rm_of_a_name_in_two_folders_hints_with_the_deployment(tmp_path, monkeypatch) -> None:
    twice = [{**_ENTRY, "status": "ready"}, {**_ENTRY, "id": "model-2", "type": "checkpoints", "status": "ready"}]
    control = Control([], models=twice)
    _install(monkeypatch, tmp_path, control)

    result = _invoke("rm", "dep-1", "lora.safetensors", "--yes")

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_model_not_found"
    assert "`comfy --json deploy models ls dep-1`" in error["hint"]
    assert control.count("DELETE", f"{_MODELS}/model-1") == 0


def test_the_command_the_two_folders_hint_gives_runs_and_lists_their_ids(tmp_path, monkeypatch) -> None:
    twice = [{**_ENTRY, "status": "ready"}, {**_ENTRY, "id": "model-2", "type": "checkpoints", "status": "ready"}]
    control = Control([], models=twice)
    _install(monkeypatch, tmp_path, control)
    monkeypatch.setattr(sys, "platform", "linux")
    hint = _envelope(_invoke("rm", "dep-1", "lora.safetensors", "--yes"))["error"]["hint"]

    words = shlex.split(hint.split("`")[1])
    result = CliRunner().invoke(app, words[1:], env={"COLUMNS": "400", "NO_COLOR": "1"})

    assert result.exit_code == 0, result.output
    assert [entry["id"] for entry in _envelope(result)["data"]["models"]] == ["model-1", "model-2"]


def test_the_failed_file_hint_removes_it_by_id_where_two_folders_hold_its_name(tmp_path, monkeypatch) -> None:
    failed = {**_ENTRY, "status": "failed"}
    twice = [{**_ENTRY, "id": "model-2", "type": "checkpoints", "status": "ready"}, failed]
    control = Control([], models=twice)
    _install(monkeypatch, tmp_path, control)
    monkeypatch.setattr(sys, "platform", "linux")
    hint = deploy_models.ModelNotAddedError("dep-1", failed).hint

    words = shlex.split(hint.split("`")[1])
    result = CliRunner().invoke(app, ["--json", *words[1:], "--yes"], env={"COLUMNS": "400", "NO_COLOR": "1"})

    assert result.exit_code == 0, result.output
    assert _envelope(result)["data"]["model"]["id"] == "model-1"
    assert "no longer lists it, then add it again" in hint


def test_the_model_names_given_to_the_models_commands_are_not_tracked(tmp_path, monkeypatch) -> None:
    events: list[tuple[str, JsonObject]] = []
    monkeypatch.setattr(
        "comfy_cli.tracking.track_event", lambda name, properties=None, **_kw: events.append((name, properties or {}))
    )
    control, asked = Control([], models=[{**_ENTRY, "filename": "client-secret.safetensors"}]), []
    _install(monkeypatch, tmp_path, control)
    _promptable(monkeypatch, False, asked)

    removed = _invoke("rm", "dep-1", "client-secret.safetensors", pretty=True)
    listed = _invoke("ls", "dep-1")

    assert removed.exit_code == 0, removed.output
    assert listed.exit_code == 0, listed.output
    assert len(asked) == 1
    started = dict(events)
    assert started["deploy:rm_cmd"]["file"] == "<redacted>"
    assert started["deploy:ls_cmd"] == {"deployment": "dep-1"}
    assert "client-secret" not in json.dumps(events)


def test_the_file_given_to_models_add_is_not_tracked(tmp_path, monkeypatch) -> None:
    events: list[tuple[str, JsonObject]] = []
    monkeypatch.setattr(
        "comfy_cli.tracking.track_event", lambda name, properties=None, **_kw: events.append((name, properties or {}))
    )
    log: list[str] = []
    control, storage = Control(log), Storage(log)
    model, _ = _install(monkeypatch, tmp_path, control, _in_pieces(), storage)

    result = _invoke("add", "dep-1", str(model), "--type", "loras", "--no-wait")

    assert result.exit_code == 0, result.output
    assert dict(events)["deploy:add_cmd"] == {"deployment": "dep-1", "folder": "loras", "wait": False}
    assert str(tmp_path) not in json.dumps(events)


def _parts_upload() -> PartsUpload:
    return PartsUpload("blob-1", False, builder_api._blob_parts({"parts": _parts(1, 2, 3)}), None)


@pytest.fixture
def default_sigint():
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def test_ctrl_c_stops_the_pieces_still_reading_the_file(tmp_path, monkeypatch, default_sigint) -> None:
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

    began = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        BuilderClient("https://builder.test", "tok").upload_blob_parts(upload.blob_id, upload.parts, model)

    assert time.monotonic() - began < 4
    assert begun
    assert sorted(stopped) == sorted(begun)
    assert "builder POST /v1/blobs/blob-1/complete" not in log


def test_ctrl_c_is_raised_without_waiting_on_a_piece_waiting_for_its_answer(
    tmp_path, monkeypatch, default_sigint
) -> None:
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

    began = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            BuilderClient("https://builder.test", "tok").upload_blob_parts(upload.blob_id, upload.parts, model)
        waited = time.monotonic() - began
    finally:
        answer.set()

    assert waited < 5
    assert threads[0].daemon


def test_a_failed_piece_is_raised_without_waiting_on_a_piece_waiting_for_its_answer(tmp_path, monkeypatch) -> None:
    log: list[str] = []
    monkeypatch.setattr("comfy_cli.builder_api.request_json", Builder(log, _in_pieces(), Storage(log)))
    monkeypatch.setattr(builder_api, "_CANCEL_GRACE_SECONDS", 0.2, raising=False)
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)
    answer, hung = threading.Event(), threading.Event()

    def put(url, data=None, **_kwargs):
        b"".join(iter(lambda: data.read(3), b""))
        if "part-1" in url:
            hung.wait(5)
            return _Response(500)
        hung.set()
        answer.wait(10)
        return _Response(200)

    monkeypatch.setattr("requests.put", put)
    upload = _parts_upload()

    began = time.monotonic()
    try:
        with pytest.raises(requests.HTTPError):
            BuilderClient("https://builder.test", "tok").upload_blob_parts(upload.blob_id, upload.parts, model)
        waited = time.monotonic() - began
    finally:
        answer.set()

    assert waited < 5
    assert "builder POST /v1/blobs/blob-1/complete" not in log


@pytest.mark.parametrize("command", [["assets", "library", "upload"], ["model", "upload"]])
def test_the_library_uploads_point_at_deploy_models_first_and_still_upload(tmp_path, monkeypatch, command) -> None:
    uploads: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "comfy_cli.command.assets_library.run_upload", lambda *args, **kwargs: uploads.append((args, kwargs))
    )
    model = tmp_path / "lora.safetensors"
    model.write_bytes(_CONTENT)

    args = ["--no-json", *command, str(model), "--type", "loras"]
    result = CliRunner().invoke(app, args, env={"COLUMNS": "400", "NO_COLOR": "1"})

    assert result.exit_code == 0, result.output
    assert DEPLOYMENT_POINTER == (
        "To add a model to a deployment, use: comfy deploy models add <deployment> <file> --type <folder>"
    )
    assert DEPLOYMENT_POINTER in result.stdout
    assert len(uploads) == 1
    assert uploads[0][0] == (model, "loras")
