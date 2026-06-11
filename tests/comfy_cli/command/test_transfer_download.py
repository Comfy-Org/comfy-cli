"""Tests for `comfy download` item-aware naming and output provenance.

When the job state file carries BOTH the final history record and a compose
item_map (stashed by `comfy run` from the workflow's `_meta`), downloads are
named `<item>_<nnn>.<ext>` with a per-item counter and `files[]` entries gain
`node_id`/`item`. Without the map, naming stays `<prompt8>_<idx>` and the new
keys are omitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from comfy_cli import comfy_client, jobs_state
from comfy_cli.comfy_client import extract_output_entries
from comfy_cli.command import transfer
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode, reset_renderer_for_testing
from comfy_cli.target import Target


@pytest.fixture(autouse=True)
def reset_renderer():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def fake_target():
    return Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-api-key",
    )


PROMPT_ID = "prompt-dl-12345"
SHORT_ID = PROMPT_ID[:8]

RECORD = {
    "status": {"completed": True, "status_str": "success"},
    "outputs": {
        "5": {
            "images": [
                {"filename": "ComfyUI_a.png", "subfolder": "", "type": "output"},
                {"filename": "ComfyUI_b.png", "subfolder": "", "type": "output"},
            ]
        },
        "9": {"images": [{"filename": "ComfyUI_c.png", "subfolder": "", "type": "output"}]},
        # Node 12 produced an output but belongs to no foreach item.
        "12": {"images": [{"filename": "stray.png", "subfolder": "", "type": "output"}]},
    },
}

ITEM_MAP = {
    "s1": {"nodes": ["3", "5"], "save_node": "5", "prefix": "o/s1"},
    # save_node listed outside `nodes` — membership must include it anyway.
    "s2": {"nodes": ["8"], "save_node": "9", "prefix": "o/s2"},
}


def _urls(target: Target) -> list[str]:
    return comfy_client.Client(target).extract_output_urls(RECORD)


def _write_state(target: Target, *, record=None, item_map=None) -> list[str]:
    urls = _urls(target)
    state = jobs_state.JobState(
        prompt_id=PROMPT_ID,
        client_id=None,
        workflow="/abs/composed.json",
        where="cloud",
        base_url=target.base_url,
        status="completed",
        outputs=urls,
        record=record,
        item_map=item_map,
    )
    assert jobs_state.write(state) is not None
    return urls


class _FakeResp:
    """Context-manager response yielding one chunk then EOF."""

    def __init__(self, data: bytes = b"\x89PNG-fake"):
        self._chunks = [data]

    def read(self, n: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_download(fake_target, tmp_path, capsys) -> tuple[list[str], dict]:
    """Run execute_download with mocked target + HTTP; return (paths, envelope data)."""
    set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
    with (
        patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
        patch.object(transfer._DOWNLOAD_OPENER, "open", side_effect=lambda req: _FakeResp()),
    ):
        paths = transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    envelope = json.loads(lines[-1])
    assert envelope["type"] == "envelope"
    return paths, envelope["data"]


class TestItemNamedDownloads:
    def test_files_named_by_item_with_per_item_counter(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=RECORD, item_map=ITEM_MAP)
        paths, data = _run_download(fake_target, tmp_path, capsys)

        names = [Path(p).name for p in paths]
        # Per-item counters restart at 000; the unmapped node 12 output keeps
        # the legacy <prompt8>_<global idx> name.
        assert names == ["s1_000.png", "s1_001.png", "s2_000.png", f"{SHORT_ID}_003.png"]
        for p in paths:
            assert Path(p).is_file()
        assert data["out_dir"] == str((tmp_path / "out").resolve())

    def test_files_entries_carry_node_id_and_item(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=RECORD, item_map=ITEM_MAP)
        _, data = _run_download(fake_target, tmp_path, capsys)

        files = data["files"]
        assert [(f.get("node_id"), f.get("item")) for f in files] == [
            ("5", "s1"),
            ("5", "s1"),
            ("9", "s2"),
            ("12", None),
        ]
        # The unmapped entry has a node_id but NO item key (omitted, not null).
        assert "item" not in files[3]

    def test_legacy_names_without_record_or_map(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=None, item_map=None)
        paths, data = _run_download(fake_target, tmp_path, capsys)

        names = [Path(p).name for p in paths]
        assert names == [f"{SHORT_ID}_{i:03d}.png" for i in range(4)]
        for f in data["files"]:
            assert "node_id" not in f
            assert "item" not in f

    def test_record_without_item_map_still_carries_node_id(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=RECORD, item_map=None)
        paths, data = _run_download(fake_target, tmp_path, capsys)

        # No map → legacy names throughout, but node provenance is known.
        names = [Path(p).name for p in paths]
        assert names == [f"{SHORT_ID}_{i:03d}.png" for i in range(4)]
        assert [f["node_id"] for f in data["files"]] == ["5", "5", "9", "12"]
        assert all("item" not in f for f in data["files"])

    def test_download_data_validates_against_transfer_schema(self, fake_target, tmp_path, capsys):
        import jsonschema

        _write_state(fake_target, record=RECORD, item_map=ITEM_MAP)
        _, data = _run_download(fake_target, tmp_path, capsys)

        schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "transfer.json"
        schema = json.loads(schema_path.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(data)
        # The new keys are part of the documented contract, not just tolerated.
        download_files = schema["oneOf"][1]["properties"]["files"]["items"]["properties"]
        assert "node_id" in download_files
        assert "item" in download_files


class TestExtractOutputEntries:
    """Module-level pure flatten — the record half of Client.extract_outputs,
    usable without a Target (download joins URLs to it by query params)."""

    def test_flattens_record_with_node_ids(self):
        entries = extract_output_entries(RECORD)
        assert [(e["node_id"], e["filename"]) for e in entries] == [
            ("5", "ComfyUI_a.png"),
            ("5", "ComfyUI_b.png"),
            ("9", "ComfyUI_c.png"),
            ("12", "stray.png"),
        ]
        for e in entries:
            assert e["subfolder"] == ""
            assert e["type"] == "output"

    def test_tolerates_malformed_records(self):
        assert extract_output_entries({}) == []
        assert extract_output_entries({"outputs": "garbage"}) == []
        assert extract_output_entries({"outputs": {"1": "garbage", "2": {"images": "nope"}}}) == []
        # Items without a filename are skipped.
        assert extract_output_entries({"outputs": {"1": {"images": [{"type": "output"}]}}}) == []

    def test_client_extract_outputs_delegates(self, fake_target):
        client = comfy_client.Client(fake_target)
        outputs = client.extract_outputs(RECORD)
        entries = extract_output_entries(RECORD)
        assert [o["node_id"] for o in outputs] == [e["node_id"] for e in entries]
        assert [o["filename"] for o in outputs] == [e["filename"] for e in entries]
        # URLs are the view_url of each entry — same triple, same encoding.
        assert outputs[0]["url"] == "https://cloud.example.com/api/view?filename=ComfyUI_a.png&subfolder=&type=output"


class TestCollisionSafeNaming:
    """A retry fan-out reusing the same item ids re-downloads into the same
    out-dir; attempt 1 must never be silently clobbered (fennec friction #3,
    P1 — lost the rejected frames). Existing destinations get a deterministic
    numeric suffix: s1_000.png → s1_000.1.png → s1_000.2.png …"""

    def test_item_named_existing_file_gets_numeric_suffix(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=RECORD, item_map=ITEM_MAP)
        out = tmp_path / "out"
        out.mkdir(parents=True)
        (out / "s1_000.png").write_bytes(b"attempt-1-keep-me")

        paths, data = _run_download(fake_target, tmp_path, capsys)

        names = [Path(p).name for p in paths]
        assert names == ["s1_000.1.png", "s1_001.png", "s2_000.png", f"{SHORT_ID}_003.png"]
        # The prior attempt is untouched; the new download lives beside it.
        assert (out / "s1_000.png").read_bytes() == b"attempt-1-keep-me"
        assert (out / "s1_000.1.png").is_file()
        # files[] reports the path that was ACTUALLY written.
        assert data["files"][0]["path"] == str((out / "s1_000.1.png").resolve())

    def test_suffix_increments_past_prior_retries(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=RECORD, item_map=ITEM_MAP)
        out = tmp_path / "out"
        out.mkdir(parents=True)
        (out / "s1_000.png").write_bytes(b"attempt-1")
        (out / "s1_000.1.png").write_bytes(b"attempt-2")

        paths, _ = _run_download(fake_target, tmp_path, capsys)

        assert Path(paths[0]).name == "s1_000.2.png"
        assert (out / "s1_000.png").read_bytes() == b"attempt-1"
        assert (out / "s1_000.1.png").read_bytes() == b"attempt-2"

    def test_legacy_prompt8_naming_also_never_overwrites(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=None, item_map=None)
        out = tmp_path / "out"
        out.mkdir(parents=True)
        (out / f"{SHORT_ID}_000.png").write_bytes(b"attempt-1")

        paths, _ = _run_download(fake_target, tmp_path, capsys)

        names = [Path(p).name for p in paths]
        assert names == [f"{SHORT_ID}_000.1.png"] + [f"{SHORT_ID}_{i:03d}.png" for i in range(1, 4)]
        assert (out / f"{SHORT_ID}_000.png").read_bytes() == b"attempt-1"


class TestPipedErrorEnvelope:
    """`comfy --json run --wait | comfy download` is the SKILL.md-recommended
    pattern, so download must survive a failed upstream: an error envelope
    (`"data": null`) used to crash with a raw AttributeError (fennec friction
    #1). Bad stdin of any shape → structured error envelope + exit 1."""

    def _download_with_stdin(self, stdin_text: str, monkeypatch, capsys) -> dict:
        import io

        import typer

        set_renderer(Renderer(mode=OutputMode.JSON, command="download"))
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
        with pytest.raises(typer.Exit) as excinfo:
            transfer.execute_download(None)
        assert excinfo.value.exit_code == 1
        out_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(out_lines[-1])
        assert envelope["type"] == "envelope"
        assert envelope["ok"] is False
        return envelope

    def test_error_envelope_with_null_data_no_traceback(self, monkeypatch, capsys):
        upstream = {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": False,
            "command": "run",
            "version": "1.0.0",
            "where": "cloud",
            "data": None,
            "error": {
                "code": "cloud_http_error",
                "message": "Cloud server error while polling (HTTP 429): Too Many Requests",
                "hint": None,
                "details": {"status": 429, "prompt_id": "d4f68191-de90-49e1-ba0f-2668b9c5b30f"},
            },
        }
        envelope = self._download_with_stdin(json.dumps(upstream), monkeypatch, capsys)

        err = envelope["error"]
        assert err["code"] == "download_no_prompt"  # registered code, not a crash
        assert "ok=false" in err["message"]
        # The upstream failure and its prompt_id surface for recovery.
        assert err["details"]["upstream_error"]["code"] == "cloud_http_error"
        assert err["details"]["prompt_id"] == "d4f68191-de90-49e1-ba0f-2668b9c5b30f"

    def test_ok_envelope_without_prompt_id_errors_cleanly(self, monkeypatch, capsys):
        upstream = {"schema": "envelope/1", "type": "envelope", "ok": True, "data": {}, "error": None}
        envelope = self._download_with_stdin(json.dumps(upstream), monkeypatch, capsys)
        assert envelope["error"]["code"] == "download_no_prompt"

    def test_non_envelope_stdin_errors_cleanly(self, monkeypatch, capsys):
        envelope = self._download_with_stdin("this is not json {", monkeypatch, capsys)
        assert envelope["error"]["code"] == "download_no_prompt"

    def test_json_scalar_stdin_errors_cleanly(self, monkeypatch, capsys):
        # Valid JSON but not an envelope object (e.g. a bare list of URLs).
        envelope = self._download_with_stdin('["https://x/view?filename=a.png"]', monkeypatch, capsys)
        assert envelope["error"]["code"] == "download_no_prompt"


class TestMachineModeStdoutPurity:
    """`comfy --json download` consumers pipe stdout into jq/json.load: stdout
    must carry NOTHING but JSON (envelope last), and the human "✓ downloaded"
    progress line is pretty-mode-only — it must not appear at all in machine
    modes (fennec friction #4/#5)."""

    def _download(self, fake_target, tmp_path):
        with (
            patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
            patch.object(transfer._DOWNLOAD_OPENER, "open", side_effect=lambda req: _FakeResp()),
        ):
            transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))

    def test_json_mode_stdout_is_pure_json_envelope_last(self, fake_target, tmp_path, capsys):
        from comfy_cli.caller import Caller

        _write_state(fake_target, record=RECORD, item_map=ITEM_MAP)
        # Resolve the mode the way the entrypoint does, from COMFY_OUTPUT=json.
        renderer = Renderer.resolve(
            env={"COMFY_OUTPUT": "json"},
            is_stdout_tty=False,
            caller=Caller(kind="user", agentic=False, source_env=None),
            command="download",
        )
        assert renderer.mode is OutputMode.JSON
        set_renderer(renderer)

        self._download(fake_target, tmp_path)

        captured = capsys.readouterr()
        out_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert out_lines, "the envelope must land on stdout"
        parsed = [json.loads(ln) for ln in out_lines]  # every line is JSON
        assert parsed[-1]["type"] == "envelope"
        assert parsed[-1]["ok"] is True
        # The human progress line is pretty-mode-only: not on stdout, and not
        # duplicated to stderr either — the envelope already says everything.
        assert "downloaded" not in captured.out
        assert "downloaded" not in captured.err

    def test_ndjson_mode_emits_no_human_progress_line(self, fake_target, tmp_path, capsys):
        _write_state(fake_target, record=RECORD, item_map=ITEM_MAP)
        set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))

        self._download(fake_target, tmp_path)

        captured = capsys.readouterr()
        for ln in captured.out.splitlines():
            if ln.strip():
                json.loads(ln)
        assert "downloaded" not in captured.out
        assert "downloaded" not in captured.err


# ---------------------------------------------------------------------------
# _default_out_dir — project/1 root wins over the legacy config key
# ---------------------------------------------------------------------------


class TestDefaultOutDir:
    def test_prefers_governing_project_outputs(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "comfy.yaml").write_text("schema: project/1\ndefaults:\n  where: cloud\n")
        monkeypatch.chdir(proj)

        assert transfer._default_out_dir() == str(proj.resolve() / "outputs")

    def test_falls_back_to_config_key_outside_project(self, tmp_path, monkeypatch):
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        legacy = tmp_path / "legacy"
        (legacy / "outputs").mkdir(parents=True)

        class _FakeCM:
            def get(self, key):
                return str(legacy)

        import comfy_cli.config_manager as config_manager

        monkeypatch.setattr(config_manager, "ConfigManager", _FakeCM)
        assert transfer._default_out_dir() == str(legacy / "outputs")

    def test_defaults_to_relative_outputs(self, tmp_path, monkeypatch):
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)

        class _FakeCM:
            def get(self, key):
                return None

        import comfy_cli.config_manager as config_manager

        monkeypatch.setattr(config_manager, "ConfigManager", _FakeCM)
        assert transfer._default_out_dir() == "./outputs"
