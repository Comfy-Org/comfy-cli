from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import jsonschema
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer
from comfy_cli.target import Target


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    renderer = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    renderer.mode = OutputMode.JSON
    set_renderer(renderer)


class _ChunkedResponse:
    def __init__(self, body: bytes, chunk_size: int = 7):
        self._body = io.BytesIO(body)
        self._chunk_size = chunk_size
        self.read_sizes: list[int] = []
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._body.read(min(size, self._chunk_size) if size >= 0 else self._chunk_size)


def _object_info() -> dict[str, Any]:
    return {
        "KSampler": {
            "input": {"required": {"steps": ["INT", {"default": 20}]}},
            "output": ["LATENT"],
            "category": "sampling",
        }
    }


def test_stream_snapshot_writes_valid_object_info_atomically(monkeypatch, tmp_path):
    body = json.dumps(_object_info()).encode()
    response = _ChunkedResponse(body)
    monkeypatch.setattr(nodes_cmd, "authed_urlopen", lambda *_a, **_kw: response)
    target = Target(kind="local", base_url="http://127.0.0.1:8188")
    output = tmp_path / "object_info.json"

    result = nodes_cmd._stream_object_info_snapshot(target, output)

    assert json.loads(output.read_text()) == _object_info()
    assert result == {"bytes": len(body), "classes": 1}
    assert len(response.read_sizes) > 1, "the response must be consumed incrementally"
    assert not list(tmp_path.glob("*.tmp"))


def test_stream_snapshot_preserves_existing_file_on_invalid_json(monkeypatch, tmp_path):
    response = _ChunkedResponse(b'{"KSampler":')
    monkeypatch.setattr(nodes_cmd, "authed_urlopen", lambda *_a, **_kw: response)
    target = Target(kind="local", base_url="http://127.0.0.1:8188")
    output = tmp_path / "object_info.json"
    output.write_text('{"existing": true}')

    with pytest.raises(ValueError, match="valid JSON"):
        nodes_cmd._stream_object_info_snapshot(target, output)

    assert output.read_text() == '{"existing": true}'
    assert not list(tmp_path.glob("*.tmp"))


def test_stream_snapshot_accepts_exact_size_limit(monkeypatch, tmp_path):
    body = json.dumps(_object_info()).encode()
    monkeypatch.setattr(
        nodes_cmd, "_OBJECT_INFO_SNAPSHOT_MAX_BYTES", len(body)
    )
    monkeypatch.setattr(
        nodes_cmd,
        "authed_urlopen",
        lambda *_a, **_kw: _ChunkedResponse(body),
    )
    output = tmp_path / "object_info.json"

    result = nodes_cmd._stream_object_info_snapshot(
        Target(kind="local", base_url="http://127.0.0.1:8188"), output
    )

    assert result["bytes"] == len(body)
    assert output.is_file()


def test_stream_snapshot_rejects_over_limit_without_replacing(monkeypatch, tmp_path):
    body = json.dumps(_object_info()).encode()
    monkeypatch.setattr(
        nodes_cmd, "_OBJECT_INFO_SNAPSHOT_MAX_BYTES", len(body) - 1
    )
    monkeypatch.setattr(
        nodes_cmd,
        "authed_urlopen",
        lambda *_a, **_kw: _ChunkedResponse(body),
    )
    output = tmp_path / "object_info.json"
    output.write_text('{"existing": true}')

    with pytest.raises(ValueError, match="snapshot limit"):
        nodes_cmd._stream_object_info_snapshot(
            Target(kind="local", base_url="http://127.0.0.1:8188"), output
        )

    assert output.read_text() == '{"existing": true}'
    assert not list(tmp_path.glob("*.tmp"))


def test_stream_snapshot_rejects_non_catalog_json(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nodes_cmd,
        "authed_urlopen",
        lambda *_a, **_kw: _ChunkedResponse(b'{"status": "ok"}'),
    )
    output = tmp_path / "object_info.json"

    with pytest.raises(ValueError, match="not a node catalog"):
        nodes_cmd._stream_object_info_snapshot(
            Target(kind="local", base_url="http://127.0.0.1:8188"), output
        )

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_snapshot_command_emits_written_catalog(monkeypatch, tmp_path, capsys):
    target = Target(kind="local", base_url="http://127.0.0.1:8188")
    monkeypatch.setattr(nodes_cmd, "_resolve_snapshot_target", lambda *_a, **_kw: target)
    calls: list[tuple[Target, Path]] = []

    def fake_snapshot(resolved_target: Target, output: Path):
        calls.append((resolved_target, output))
        output.write_text(json.dumps(_object_info()))
        return {"bytes": output.stat().st_size, "classes": 1}

    monkeypatch.setattr(nodes_cmd, "_stream_object_info_snapshot", fake_snapshot)
    output = tmp_path / "object_info.json"
    _force_json_renderer()

    result = CliRunner().invoke(nodes_cmd.app, ["snapshot", "--output", str(output)], standalone_mode=False)
    captured = capsys.readouterr().out or result.stdout
    envelope = json.loads(captured.strip().splitlines()[-1])

    assert result.exit_code == 0
    assert envelope["ok"] is True
    assert envelope["data"]["output"] == str(output)
    assert envelope["data"]["classes"] == 1
    assert calls == [(target, output)]
    schema = json.loads(
        (
            Path(nodes_cmd.__file__).parent.parent / "schemas" / "nodes.json"
        ).read_text()
    )
    jsonschema.validate(instance=envelope["data"], schema=schema)


def test_snapshot_command_reports_invalid_where(tmp_path, capsys):
    _force_json_renderer()

    result = CliRunner().invoke(
        nodes_cmd.app,
        [
            "snapshot",
            "--output",
            str(tmp_path / "object_info.json"),
            "--where",
            "somewhere",
        ],
        standalone_mode=False,
    )
    captured = capsys.readouterr().out or result.stdout
    envelope = json.loads(captured.strip().splitlines()[-1])

    assert result.exit_code == 1
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "where_invalid"


def test_snapshot_command_maps_unresolvable_home_to_envelope(
    monkeypatch, capsys
):
    target = Target(kind="local", base_url="http://127.0.0.1:8188")
    monkeypatch.setattr(
        nodes_cmd, "_resolve_snapshot_target", lambda *_a, **_kw: target
    )
    _force_json_renderer()

    result = CliRunner().invoke(
        nodes_cmd.app,
        ["snapshot", "--output", "~comfy-cli-user-that-does-not-exist/catalog.json"],
        standalone_mode=False,
    )
    captured = capsys.readouterr().out or result.stdout
    envelope = json.loads(captured.strip().splitlines()[-1])

    assert result.exit_code == 1
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "nodes_snapshot_failed"
