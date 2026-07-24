"""Tests for `comfy download` item-aware naming and output provenance.

When the job state file carries BOTH the final history record and a compose
item_map (stashed by `comfy run` from the workflow's `_meta`), downloads are
named `<item>_<nnn>.<ext>` with a per-item counter and `files[]` entries gain
`node_id`/`item`. Without the map, naming stays `<prompt8>_<idx>` and the new
keys are omitted.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    """Context-manager stand-in for http.client.HTTPResponse. ``read(n)``
    honours ``n`` and advances through the body like the real object, so the
    streaming loop and its running byte total are faithfully exercised.
    Declares a matching Content-Length by default; pass an int to lie, None to
    omit. ``chunk_size`` caps each read below ``n`` to force multi-read
    streaming (and to model a chunked body that over-delivers past a small
    declared Content-Length, which a real clipped Content-Length body cannot)."""

    def __init__(self, data: bytes = b"\x89PNG-fake", *, content_length="auto", chunk_size=None):
        self._buf = data
        self._pos = 0
        self._chunk_size = chunk_size
        self.reads = 0
        if content_length == "auto":
            content_length = len(data)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def read(self, n: int) -> bytes:
        self.reads += 1
        take = n if self._chunk_size is None else min(n, self._chunk_size)
        chunk = self._buf[self._pos : self._pos + take]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_download(fake_target, tmp_path, capsys) -> tuple[list[str], dict]:
    """Run execute_download with mocked target + HTTP; return (paths, envelope data)."""
    set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
    with (
        patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
        patch.object(transfer._DOWNLOAD_OPENER, "open", side_effect=lambda req, timeout=None: _FakeResp()),
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
            patch.object(transfer._DOWNLOAD_OPENER, "open", side_effect=lambda req, timeout=None: _FakeResp()),
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


class TestDownloadIntegrity:
    """A transfer that dies mid-body or oversteps the size cap must fail with a
    structured envelope and leave neither the final file nor a `.part` partial
    behind; completed downloads land atomically."""

    def _failing_download(self, fake_target, tmp_path, capsys, resp=None, raises=None) -> dict:
        import typer

        def _open(req, **kw):
            if raises is not None:
                raise raises
            return resp

        set_renderer(Renderer(mode=OutputMode.JSON, command="download"))
        _write_state(fake_target)
        with (
            patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
            patch.object(transfer._DOWNLOAD_OPENER, "open", side_effect=_open),
        ):
            with pytest.raises(typer.Exit) as excinfo:
                transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        assert excinfo.value.exit_code == 1
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["ok"] is False
        return envelope["error"]

    def test_truncated_body_errors_and_leaves_nothing(self, fake_target, tmp_path, capsys):
        err = self._failing_download(fake_target, tmp_path, capsys, _FakeResp(b"x" * 400, content_length=1000))
        assert err["code"] == "download_failed"
        # Every size failure reports the server-declared length under the same
        # key so a machine consumer never has to guess which spelling is present.
        assert err["details"]["declared_bytes"] == 1000
        assert err["details"]["received_bytes"] == 400
        assert list((tmp_path / "out").iterdir()) == []

    def test_declared_size_over_cap_refused_before_body_read(self, fake_target, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(transfer, "_MAX_DOWNLOAD_BYTES", 100)
        resp = _FakeResp(b"x" * 40, content_length=500)
        err = self._failing_download(fake_target, tmp_path, capsys, resp)
        assert err["code"] == "download_failed"
        assert err["details"]["declared_bytes"] == 500
        assert resp.reads == 0
        assert list((tmp_path / "out").iterdir()) == []

    def test_body_exceeding_declared_length_aborts(self, fake_target, tmp_path, capsys):
        # Chunked over-delivery is the real trigger: a body streamed in small
        # chunks past a declared Content-Length of 100 (a plain Content-Length
        # body would be clipped and never over-deliver). The abort must fire
        # mid-stream, not only after the loop ends.
        resp = _FakeResp(b"x" * 400, content_length=100, chunk_size=32)
        err = self._failing_download(fake_target, tmp_path, capsys, resp)
        assert err["code"] == "download_failed"
        assert err["details"]["declared_bytes"] == 100
        assert err["details"]["received_bytes"] > 100
        assert resp.reads > 1  # streamed across multiple reads before aborting
        assert list((tmp_path / "out").iterdir()) == []

    def test_lengthless_body_over_cap_errors_with_envelope(self, fake_target, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(transfer, "_MAX_DOWNLOAD_BYTES", 100)
        resp = _FakeResp(b"x" * 400, content_length=None, chunk_size=32)
        err = self._failing_download(fake_target, tmp_path, capsys, resp)
        assert err["code"] == "download_failed"
        assert err["details"]["received_bytes"] > 100
        assert resp.reads > 1  # cap tripped while accumulating, not on one read
        assert list((tmp_path / "out").iterdir()) == []

    def test_multichunk_body_reassembled_correctly(self, fake_target, tmp_path, capsys):
        # A large body delivered across many reads must land byte-exact, so the
        # running total and the temp-file writes stay in step with the stream.
        body = bytes(range(256)) * 40  # 10240 bytes
        set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
        _write_state(fake_target)
        with (
            patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
            patch.object(
                transfer._DOWNLOAD_OPENER,
                "open",
                side_effect=lambda req, timeout=None: _FakeResp(body, chunk_size=100),
            ),
        ):
            paths = transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        assert all(Path(p).read_bytes() == body for p in paths)

    def test_lengthless_body_downloads_without_verification(self, fake_target, tmp_path, capsys):
        set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
        _write_state(fake_target)
        with (
            patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
            patch.object(
                transfer._DOWNLOAD_OPENER, "open", side_effect=lambda req, timeout=None: _FakeResp(content_length=None)
            ),
        ):
            paths = transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        assert len(paths) == 4
        assert all(Path(p).read_bytes() == b"\x89PNG-fake" for p in paths)

    def test_completed_download_leaves_no_part_file(self, fake_target, tmp_path, capsys):
        _write_state(fake_target)
        paths, _ = _run_download(fake_target, tmp_path, capsys)
        assert paths
        assert list((tmp_path / "out").glob("*.part")) == []

    def test_preexisting_part_sibling_survives_success(self, fake_target, tmp_path, capsys):
        out = tmp_path / "out"
        out.mkdir(parents=True)
        bystander = out / f"{SHORT_ID}_000.png.part"
        bystander.write_bytes(b"user data, not ours")
        _write_state(fake_target)
        paths, _ = _run_download(fake_target, tmp_path, capsys)
        assert len(paths) == 4
        assert bystander.read_bytes() == b"user data, not ours"

    def test_preexisting_part_sibling_survives_failure(self, fake_target, tmp_path, capsys):
        out = tmp_path / "out"
        out.mkdir(parents=True)
        bystander = out / f"{SHORT_ID}_000.png.part"
        bystander.write_bytes(b"user data, not ours")
        err = self._failing_download(fake_target, tmp_path, capsys, _FakeResp(b"x" * 40, content_length=100))
        assert err["code"] == "download_failed"
        assert bystander.read_bytes() == b"user data, not ours"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_downloaded_file_keeps_umask_permissions(self, fake_target, tmp_path, capsys):
        _write_state(fake_target)
        paths, _ = _run_download(fake_target, tmp_path, capsys)
        umask = os.umask(0)
        os.umask(umask)
        assert (Path(paths[0]).stat().st_mode & 0o777) == 0o666 & ~umask

    def test_connection_failure_emits_envelope_not_traceback(self, fake_target, tmp_path, capsys):
        import urllib.error

        err = self._failing_download(
            fake_target, tmp_path, capsys, raises=urllib.error.URLError("[Errno 111] Connection refused")
        )
        assert err["code"] == "download_failed"
        assert "refused" in err["details"]["reason"]
        assert list((tmp_path / "out").iterdir()) == []

    def test_midstream_reset_emits_envelope_and_cleans_temp(self, fake_target, tmp_path, capsys):
        class _ResettingResp(_FakeResp):
            def read(self, n: int) -> bytes:
                chunk = super().read(n)
                if not chunk:
                    raise ConnectionResetError("Connection reset by peer")
                return chunk

        err = self._failing_download(fake_target, tmp_path, capsys, _ResettingResp(b"x" * 100, content_length=None))
        assert err["code"] == "download_failed"
        assert "reset" in err["details"]["reason"]
        assert list((tmp_path / "out").iterdir()) == []

    def test_chunked_truncation_incompleteread_emits_envelope(self, fake_target, tmp_path, capsys):
        # A truncated chunked body raises http.client.IncompleteRead — an
        # HTTPException, NOT an OSError — which used to escape both except arms
        # as a raw traceback and break the machine-mode envelope contract.
        import http.client

        class _IncompleteResp(_FakeResp):
            def read(self, n: int) -> bytes:
                chunk = super().read(n)
                if not chunk:
                    raise http.client.IncompleteRead(b"", 500)
                return chunk

        err = self._failing_download(fake_target, tmp_path, capsys, _IncompleteResp(b"y" * 100, content_length=None))
        assert err["code"] == "download_failed"
        assert list((tmp_path / "out").iterdir()) == []

    def test_rename_failure_emits_envelope_and_cleans_temp(self, fake_target, tmp_path, capsys):
        with patch("pathlib.Path.replace", side_effect=OSError("Permission denied")):
            err = self._failing_download(fake_target, tmp_path, capsys, _FakeResp())
        assert err["code"] == "download_failed"
        assert "Permission denied" in err["details"]["reason"]
        assert list((tmp_path / "out").iterdir()) == []

    def test_download_passes_socket_timeout(self, fake_target, tmp_path, capsys):
        set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
        _write_state(fake_target)
        opener = MagicMock(side_effect=lambda req, timeout=None: _FakeResp())
        with (
            patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
            patch.object(transfer._DOWNLOAD_OPENER, "open", opener),
        ):
            transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        assert opener.call_args.kwargs["timeout"] == transfer._DOWNLOAD_TIMEOUT_S


class TestDeclaredContentLength:
    """Content-Length parsing must never raise (a ValueError would silently
    disable size verification): unparseable or ambiguous headers degrade to
    None, a folded duplicate that agrees is accepted."""

    class _Resp:
        def __init__(self, value):
            self.headers = {} if value is None else {"Content-Length": value}

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1000", 1000),
            ("0", 0),
            (None, None),
            ("1000, 1000", 1000),  # folded duplicate that agrees
            ("1000, 2000", None),  # folded duplicate that disagrees → ambiguous
            ("not-a-number", None),
            ("-5", None),
            ("", None),
        ],
    )
    def test_parses_or_degrades_to_none(self, value, expected):
        assert transfer._declared_content_length(self._Resp(value)) == expected


class TestPartFileHelper:
    """`_open_part_file` owns the raw fd end-to-end: any failure closes it and
    removes the temp file, and a copy whose source vanishes must not leak the
    descriptor either (the fd is wrapped before the source is opened)."""

    def _open_fd_count(self):
        return len(os.listdir("/proc/self/fd"))

    @pytest.mark.skipif(not hasattr(os, "fchmod"), reason="os.fchmod is POSIX-only")
    def test_fchmod_failure_closes_fd_and_removes_temp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os, "fchmod", lambda *a: (_ for _ in ()).throw(OSError("ENOTSUP")))
        with pytest.raises(OSError, match="ENOTSUP"):
            transfer._open_part_file(tmp_path / "out.png")
        assert list(tmp_path.glob("*.part")) == []

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/fd fd accounting")
    def test_missing_source_copy_leaks_no_fd_and_no_temp(self, tmp_path):
        dst = tmp_path / "dst.bin"
        missing = tmp_path / "gone.bin"
        before = self._open_fd_count()
        for _ in range(5):
            with pytest.raises(FileNotFoundError):
                transfer._copy_local_output_capped(missing, dst)
        assert self._open_fd_count() == before
        assert list(tmp_path.glob("*.part")) == []
        assert not dst.exists()


class TestLocalOutputCopy:
    """A `comfy run --where local --wait` job records bare on-disk output
    paths (execution.format_image_path returns an absolute path for loopback
    hosts), not `/view` URLs. `comfy download` must copy those files into
    --out-dir instead of tripping the anti-SSRF guard, which rejects any
    non-http(s) URL — while STILL rejecting real non-http URLs on the fetch
    path so the SSRF protection stays intact (issue #480)."""

    def _write_local_state(self, outputs: list[str]) -> None:
        state = jobs_state.JobState(
            prompt_id=PROMPT_ID,
            client_id=None,
            workflow="/abs/composed.json",
            where="local",
            base_url="http://127.0.0.1:8188",
            status="completed",
            outputs=outputs,
            record=None,
            item_map=None,
        )
        assert jobs_state.write(state) is not None

    def _run(self, tmp_path, capsys, fake_target) -> tuple[list[str], dict]:
        set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
        with patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target):
            paths = transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        return paths, json.loads(lines[-1])["data"]

    def test_bare_path_output_is_copied_into_out_dir(self, fake_target, tmp_path, capsys):
        src_dir = tmp_path / "output"
        src_dir.mkdir()
        src = src_dir / "ComfyUI_00001_.png"
        src.write_bytes(b"\x89PNG-local-bytes")
        self._write_local_state([str(src)])

        paths, data = self._run(tmp_path, capsys, fake_target)

        assert len(paths) == 1
        copied = Path(paths[0])
        assert copied.is_file()
        assert copied.parent == (tmp_path / "out")
        # The bytes are the on-disk source, copied verbatim.
        assert copied.read_bytes() == b"\x89PNG-local-bytes"
        # Source is left untouched (a copy, not a move).
        assert src.read_bytes() == b"\x89PNG-local-bytes"
        assert data["files"][0]["path"] == str(copied.resolve())
        assert data["files"][0]["size"] == len(b"\x89PNG-local-bytes")

    def test_copied_output_keeps_real_extension_not_png(self, fake_target, tmp_path, capsys):
        # A bare path carries no `?filename=` param; the extension must come
        # from the source file, not the hardcoded `.png` default.
        src = tmp_path / "output" / "clip_out.webp"
        src.parent.mkdir()
        src.write_bytes(b"RIFFfake-webp")
        self._write_local_state([str(src)])

        paths, _ = self._run(tmp_path, capsys, fake_target)

        assert Path(paths[0]).suffix == ".webp"

    def test_file_uri_output_is_copied(self, fake_target, tmp_path, capsys):
        src = tmp_path / "output" / "vid.mp4"
        src.parent.mkdir()
        src.write_bytes(b"fake-mp4")
        self._write_local_state([src.as_uri()])  # file:///…/vid.mp4

        paths, _ = self._run(tmp_path, capsys, fake_target)

        assert Path(paths[0]).suffix == ".mp4"
        assert Path(paths[0]).read_bytes() == b"fake-mp4"

    def test_copy_cap_enforced_during_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transfer, "_MAX_DOWNLOAD_BYTES", 100)
        src = tmp_path / "big.bin"
        src.write_bytes(b"x" * 300)
        dst = tmp_path / "out" / "big.bin"
        dst.parent.mkdir()
        with pytest.raises(ValueError, match="safety limit"):
            transfer._copy_local_output_capped(src, dst)
        assert list(dst.parent.iterdir()) == []

    def test_copy_cap_breach_surfaces_as_envelope(self, fake_target, tmp_path, capsys, monkeypatch):
        import typer

        src = tmp_path / "output" / "grew.png"
        src.parent.mkdir()
        src.write_bytes(b"\x89PNG-local")
        self._write_local_state([str(src)])

        def _cap_breach(src_, dst_):
            raise ValueError("local output exceeds 100 byte safety limit")

        monkeypatch.setattr(transfer, "_copy_local_output_capped", _cap_breach)
        set_renderer(Renderer(mode=OutputMode.JSON, command="download"))
        with patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target):
            with pytest.raises(typer.Exit) as excinfo:
                transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        assert excinfo.value.exit_code == 1
        envelope = json.loads([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1])
        assert envelope["error"]["code"] == "download_failed"
        assert "safety limit" in envelope["error"]["message"]

    def test_missing_local_source_errors_cleanly(self, fake_target, tmp_path, capsys):
        self._write_local_state([str(tmp_path / "output" / "gone.png")])
        set_renderer(Renderer(mode=OutputMode.JSON, command="download"))
        import typer

        with patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target):
            with pytest.raises(typer.Exit) as excinfo:
                transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        assert excinfo.value.exit_code == 1
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "download_failed"
        assert "not found on disk" in envelope["error"]["message"]


class TestSSRFGuardIntact:
    """The local-copy branch is additive: `_assert_download_url` must still
    reject any non-http(s) URL that isn't a bare local path, and the download
    loop must surface that rejection (SSRF protection stays intact)."""

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://evil.example.com/etc/passwd",
            "file://evil.example.com/etc/passwd",  # file:// with a REMOTE host
            "gopher://internal:70/x",
            "data:text/plain,hi",
        ],
    )
    def test_assert_download_url_rejects_non_http(self, url):
        with pytest.raises(ValueError, match="non-HTTP URL"):
            transfer._assert_download_url(url)

    @pytest.mark.parametrize(
        "url",
        ["http://x/view?filename=a.png", "https://x/view?filename=a.png"],
    )
    def test_assert_download_url_allows_http(self, url):
        transfer._assert_download_url(url)  # does not raise

    def test_remote_host_file_uri_is_not_treated_as_local(self):
        # A file:// URL pointing at a non-local host must NOT be copied — it
        # falls through to the SSRF guard, which rejects it.
        assert transfer._local_source_path("file://evil.example.com/etc/passwd") is None
        assert transfer._local_source_path("ftp://evil.example.com/x") is None
        assert transfer._local_source_path("http://x/view?filename=a.png") is None

    def test_non_http_output_still_rejected_by_download_loop(self, fake_target, tmp_path, capsys):
        # A non-http, non-local URL flowing through the download loop must
        # error via the guard rather than being fetched.
        state = jobs_state.JobState(
            prompt_id=PROMPT_ID,
            client_id=None,
            workflow="/abs/composed.json",
            where="cloud",
            base_url="https://cloud.example.com",
            status="completed",
            outputs=["ftp://evil.example.com/etc/passwd"],
            record=None,
            item_map=None,
        )
        assert jobs_state.write(state) is not None
        set_renderer(Renderer(mode=OutputMode.JSON, command="download"))
        import typer

        with patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target):
            with pytest.raises(typer.Exit) as excinfo:
                transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        assert excinfo.value.exit_code == 1
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "download_failed"
        assert "non-HTTP URL" in envelope["error"]["message"]

    def test_bare_path_from_non_local_job_is_not_copied(self, fake_target, tmp_path, capsys):
        # The copy-from-disk branch is gated on `where == "local"`. A cloud/
        # remote job whose (untrusted) metadata sneaks in a bare absolute path
        # must NOT be copied off disk — it falls through to the SSRF guard,
        # which rejects it as a non-HTTP URL. Guards against exfiltration via a
        # tampered piped envelope or a malicious API record.
        secret = tmp_path / "secret.txt"
        secret.write_bytes(b"top-secret")
        state = jobs_state.JobState(
            prompt_id=PROMPT_ID,
            client_id=None,
            workflow="/abs/composed.json",
            where="cloud",
            base_url="https://cloud.example.com",
            status="completed",
            outputs=[str(secret)],
            record=None,
            item_map=None,
        )
        assert jobs_state.write(state) is not None
        set_renderer(Renderer(mode=OutputMode.JSON, command="download"))
        import typer

        out = tmp_path / "out"
        with patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target):
            with pytest.raises(typer.Exit) as excinfo:
                transfer.execute_download(PROMPT_ID, out_dir=str(out))
        assert excinfo.value.exit_code == 1
        envelope = json.loads([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1])
        assert envelope["error"]["code"] == "download_failed"
        assert "non-HTTP URL" in envelope["error"]["message"]
        # The secret was never copied out.
        assert not any(out.glob("*")) if out.exists() else True

    @pytest.mark.parametrize(
        "url",
        [
            "//attacker.com/share/x.png",  # POSIX-style UNC
            "\\\\attacker.com\\share\\x.png",  # Windows UNC
        ],
    )
    def test_unc_paths_are_not_local(self, url):
        # UNC/network paths are "absolute" but resolve over SMB (NTLM leak);
        # they must not be treated as local files.
        assert transfer._local_source_path(url) is None

    @pytest.mark.parametrize(
        "url",
        [
            "file:///abs/path.png",  # no host
            "file://localhost/abs/path.png",  # bare loopback host
            "file://localhost:8080/abs/path.png",  # loopback host + port (port ignored)
            "file://[::1]/abs/path.png",  # IPv6 loopback
        ],
    )
    def test_loopback_file_uris_are_local(self, url):
        # The tightened `hostname`-based check accepts genuine loopback file://
        # URIs — a port or IPv6 brackets on a loopback host don't defeat it
        # (that was the whole point of switching off `netloc`).
        assert transfer._local_source_path(url) == Path("/abs/path.png")


class TestDownloadHelperExtraction:
    """Direct, guard-pinning unit tests for the per-URL helpers extracted from
    ``execute_download`` (BE-3273): ``_download_one_url`` orchestrates naming +
    the symlink-dest refusal + branch dispatch; ``_copy_local_one`` and
    ``_stream_http_one`` carry the two download branches. The behavior-preserving
    extraction must keep every guard, envelope code, and exit intact — and the
    ``_local_source_path`` gate must stay pinned to ``is_local_job``.
    """

    def _json_renderer(self):
        # Pass the renderer explicitly (the helpers take it as an argument), so
        # no process-wide singleton state leaks between direct-call tests.
        return Renderer(mode=OutputMode.JSON, command="download")

    def _error(self, capsys) -> dict:
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["type"] == "envelope"
        assert envelope["ok"] is False
        return envelope["error"]

    # -- _download_one_url ---------------------------------------------------

    def test_non_local_job_never_calls_local_source_path_and_ssrf_rejects(self, tmp_path, capsys, monkeypatch):
        # With is_local_job=False the local-copy path is never consulted:
        # _local_source_path must NOT be called (monkeypatched to explode), and a
        # bare filesystem path flows to the HTTP branch where the SSRF assert
        # rejects it as a non-HTTP URL.
        import typer

        def _boom(url):
            raise AssertionError("_local_source_path must not be called for a non-local job")

        monkeypatch.setattr(transfer, "_local_source_path", _boom)
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(typer.Exit) as excinfo:
            transfer._download_one_url(
                "/etc/passwd",
                0,
                dest=dest,
                annotations=[(None, None)],
                item_counters={},
                short_id="prompt-d",
                is_local_job=False,
                auth_hdrs={},
                renderer=self._json_renderer(),
            )
        assert excinfo.value.exit_code == 1
        err = self._error(capsys)
        assert err["code"] == "download_failed"
        assert "non-HTTP URL" in err["message"]
        assert err["details"] == {"url": "/etc/passwd", "index": 0}
        # Nothing was written.
        assert list(dest.iterdir()) == []

    def test_local_job_copies_and_shares_per_item_counter(self, tmp_path, capsys):
        # A real local source file is copied in, the entry reports the written
        # path + true size, and a shared item_counters dict advances the per-item
        # index across two calls.
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src_a = src_dir / "a.png"
        src_a.write_bytes(b"\x89PNG-a-bytes")
        src_b = src_dir / "b.png"
        src_b.write_bytes(b"\x89PNG-longer-b-bytes")
        dest = tmp_path / "out"
        dest.mkdir()

        annotations = [("5", "s1"), ("5", "s1")]
        item_counters: dict[str, int] = {}
        renderer = self._json_renderer()

        entry0 = transfer._download_one_url(
            str(src_a),
            0,
            dest=dest,
            annotations=annotations,
            item_counters=item_counters,
            short_id="prompt-d",
            is_local_job=True,
            auth_hdrs={},
            renderer=renderer,
        )
        entry1 = transfer._download_one_url(
            str(src_b),
            1,
            dest=dest,
            annotations=annotations,
            item_counters=item_counters,
            short_id="prompt-d",
            is_local_job=True,
            auth_hdrs={},
            renderer=renderer,
        )

        assert Path(entry0["path"]).name == "s1_000.png"
        assert Path(entry1["path"]).name == "s1_001.png"
        assert item_counters == {"s1": 2}
        assert Path(entry0["path"]).read_bytes() == b"\x89PNG-a-bytes"
        assert entry0["size"] == len(b"\x89PNG-a-bytes")
        assert entry1["size"] == len(b"\x89PNG-longer-b-bytes")
        assert entry0["node_id"] == "5" and entry0["item"] == "s1"
        # Source files are copied, not moved.
        assert src_a.is_file() and src_b.is_file()

    def test_symlink_destination_refused_before_copy_or_fetch(self, tmp_path, capsys, monkeypatch):
        # The dest-symlink guard fires before either branch runs. _collision_safe_path
        # normally skips symlinks, so pin it to the planted symlink to reach the guard,
        # and make both branch sinks explode to prove neither is entered.
        import typer

        dest = tmp_path / "out"
        dest.mkdir()
        target = tmp_path / "elsewhere.png"
        target.write_bytes(b"attacker-target")
        link = dest / "s1_000.png"
        link.symlink_to(target)

        monkeypatch.setattr(transfer, "_collision_safe_path", lambda p: link)
        monkeypatch.setattr(
            transfer,
            "_copy_local_output_capped",
            lambda *a: (_ for _ in ()).throw(AssertionError("copy must not run")),
        )
        monkeypatch.setattr(
            transfer._DOWNLOAD_OPENER,
            "open",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetch must not run")),
        )

        src = tmp_path / "src.png"
        src.write_bytes(b"local")
        with pytest.raises(typer.Exit) as excinfo:
            transfer._download_one_url(
                str(src),
                0,
                dest=dest,
                annotations=[("5", "s1")],
                item_counters={},
                short_id="prompt-d",
                is_local_job=True,
                auth_hdrs={},
                renderer=self._json_renderer(),
            )
        assert excinfo.value.exit_code == 1
        err = self._error(capsys)
        assert err["code"] == "download_failed"
        assert "Refusing to write to symlink" in err["message"]
        # The symlink and its target are untouched.
        assert link.is_symlink()
        assert target.read_bytes() == b"attacker-target"

    # -- _stream_http_one ----------------------------------------------------

    def _stream(self, tmp_path, capsys, resp, *, expect_exit=True):
        import typer

        dest = tmp_path / "out"
        dest.mkdir(exist_ok=True)
        local_path = dest / "img.png"
        renderer = self._json_renderer()
        with patch.object(transfer._DOWNLOAD_OPENER, "open", side_effect=lambda req, timeout=None: resp):
            if expect_exit:
                with pytest.raises(typer.Exit) as excinfo:
                    transfer._stream_http_one("https://x/view?filename=a.png", 0, local_path, {}, renderer)
                assert excinfo.value.exit_code == 1
                return self._error(capsys), dest, local_path
            transfer._stream_http_one("https://x/view?filename=a.png", 0, local_path, {}, renderer)
            return None, dest, local_path

    def test_stream_declared_over_cap_refused_before_body_read(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(transfer, "_MAX_DOWNLOAD_BYTES", 100)
        resp = _FakeResp(b"x" * 40, content_length=500)
        err, dest, _ = self._stream(tmp_path, capsys, resp)
        assert err["code"] == "download_failed"
        assert err["details"]["declared_bytes"] == 500
        assert resp.reads == 0  # refused before the first body read
        assert list(dest.glob("*")) == []

    def test_stream_over_declared_midstream_aborts(self, tmp_path, capsys):
        resp = _FakeResp(b"x" * 400, content_length=100, chunk_size=32)
        err, dest, _ = self._stream(tmp_path, capsys, resp)
        assert err["code"] == "download_failed"
        assert err["details"]["declared_bytes"] == 100
        assert err["details"]["received_bytes"] > 100
        assert resp.reads > 1  # aborted mid-stream, not only post-loop
        assert list(dest.glob("*")) == []

    def test_stream_running_cap_aborts(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(transfer, "_MAX_DOWNLOAD_BYTES", 100)
        resp = _FakeResp(b"x" * 400, content_length=None, chunk_size=32)
        err, dest, _ = self._stream(tmp_path, capsys, resp)
        assert err["code"] == "download_failed"
        assert err["details"]["received_bytes"] > 100
        assert list(dest.glob("*")) == []

    def test_stream_truncation_aborts(self, tmp_path, capsys):
        resp = _FakeResp(b"x" * 400, content_length=1000)
        err, dest, _ = self._stream(tmp_path, capsys, resp)
        assert err["code"] == "download_failed"
        assert err["details"]["declared_bytes"] == 1000
        assert err["details"]["received_bytes"] == 400
        assert list(dest.glob("*")) == []

    def test_stream_success_renames_part_to_final_no_sibling(self, tmp_path, capsys):
        _, dest, local_path = self._stream(tmp_path, capsys, _FakeResp(), expect_exit=False)
        assert local_path.read_bytes() == b"\x89PNG-fake"
        assert list(dest.glob("*.part")) == []
        assert [p.name for p in dest.iterdir()] == ["img.png"]

    # -- _copy_local_one -----------------------------------------------------

    def test_copy_missing_source_envelope(self, tmp_path, capsys):
        import typer

        dest = tmp_path / "out"
        dest.mkdir()
        missing = tmp_path / "gone.png"
        with pytest.raises(typer.Exit) as excinfo:
            transfer._copy_local_one(str(missing), 0, missing, dest / "img.png", self._json_renderer())
        assert excinfo.value.exit_code == 1
        err = self._error(capsys)
        assert err["code"] == "download_failed"
        assert "not found on disk" in err["message"]
        assert list(dest.iterdir()) == []

    def test_copy_stat_cap_breach_envelope(self, tmp_path, capsys, monkeypatch):
        import typer

        monkeypatch.setattr(transfer, "_MAX_DOWNLOAD_BYTES", 100)
        src = tmp_path / "big.png"
        src.write_bytes(b"x" * 300)
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(typer.Exit) as excinfo:
            transfer._copy_local_one(str(src), 0, src, dest / "img.png", self._json_renderer())
        assert excinfo.value.exit_code == 1
        err = self._error(capsys)
        assert err["code"] == "download_failed"
        assert "safety limit" in err["message"]
        assert err["details"]["size"] == 300
        assert list(dest.iterdir()) == []

    @pytest.mark.parametrize("exc", [OSError("disk full"), ValueError("mid-copy cap")])
    def test_copy_capped_failure_surfaces_as_envelope(self, tmp_path, capsys, monkeypatch, exc):
        import typer

        src = tmp_path / "src.png"
        src.write_bytes(b"\x89PNG-local")
        dest = tmp_path / "out"
        dest.mkdir()

        def _raise(src_, dst_):
            raise exc

        monkeypatch.setattr(transfer, "_copy_local_output_capped", _raise)
        with pytest.raises(typer.Exit) as excinfo:
            transfer._copy_local_one(str(src), 0, src, dest / "img.png", self._json_renderer())
        assert excinfo.value.exit_code == 1
        err = self._error(capsys)
        assert err["code"] == "download_failed"
        assert "Failed to copy local output" in err["message"]


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


class TestDownloadExtensionSanitized:
    """The download extension can be derived from an untrusted `?filename=`
    query param (a compromised/malicious server). Unlike the `item` token, it
    was never sanitized, so control/ANSI bytes reached the on-disk name and
    the echoed path — a terminal-injection vector in human mode. `_sanitize_ext`
    whitelists it to a safe charset while preserving the no-traversal guarantee
    (BE-3326)."""

    # A real ESC control byte, exactly as a hostile server could return it.
    _ATTACK_NAME = "out.png\x1b[31mHACK"

    @staticmethod
    def _has_control_bytes(s: str) -> bool:
        return any(ord(c) < 0x20 or ord(c) == 0x7F for c in s)

    def _write_cloud_state(self, target: Target, outputs: list[str]) -> None:
        state = jobs_state.JobState(
            prompt_id=PROMPT_ID,
            client_id=None,
            workflow="/abs/composed.json",
            where="cloud",
            base_url=target.base_url,
            status="completed",
            outputs=outputs,
            record=None,
            item_map=None,
        )
        assert jobs_state.write(state) is not None

    def _run(self, fake_target, tmp_path, capsys) -> tuple[list[str], dict]:
        set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
        with (
            patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
            patch.object(transfer._DOWNLOAD_OPENER, "open", side_effect=lambda req, timeout=None: _FakeResp()),
        ):
            paths = transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        return paths, json.loads(lines[-1])["data"]

    def test_query_param_control_bytes_stripped_from_ext(self, fake_target, tmp_path, capsys):
        # A hostile server names the output with a control-byte-laden extension.
        url = f"https://cloud.example.com/api/view?filename={self._ATTACK_NAME}&subfolder=&type=output"
        self._write_cloud_state(fake_target, [url])

        paths, data = self._run(fake_target, tmp_path, capsys)

        assert len(paths) == 1
        name = Path(paths[0]).name
        echoed = data["files"][0]["path"]
        # No control/ANSI bytes survive into the on-disk name or the echoed path.
        assert not self._has_control_bytes(name), repr(name)
        assert not self._has_control_bytes(echoed), repr(echoed)
        # Exact name: control/ANSI bytes are gone (`\x1b`, `[` stripped) while the
        # benign alphanumeric payload remnant survives the whitelist. Asserting the
        # full name (not just a `.png` prefix) catches any regression that leaks
        # control bytes after `.png`.
        assert name == f"{SHORT_ID}_000.png31mHACK", name
        assert Path(paths[0]).is_file()

    def test_query_param_no_directory_traversal(self, fake_target, tmp_path, capsys):
        # `Path(...).suffix` already drops directory components; confirm a
        # traversal-shaped filename never escapes the out-dir and degrades to
        # the `.png` default (no dotted suffix to keep).
        out_dir = tmp_path / "out"
        url = "https://cloud.example.com/api/view?filename=../../../etc/passwd&subfolder=&type=output"
        self._write_cloud_state(fake_target, [url])

        paths, _ = self._run(fake_target, tmp_path, capsys)

        assert len(paths) == 1
        written = Path(paths[0])
        assert written.name == f"{SHORT_ID}_000.png"
        # Nothing was written outside the requested out-dir.
        assert written.resolve().parent == out_dir.resolve()

    def test_local_source_control_bytes_stripped_from_ext(self, fake_target, tmp_path, capsys):
        # The local-copy branch derives `ext` from the on-disk source name and
        # must be sanitized too (BE-3326 applies to both branches).
        src_dir = tmp_path / "output"
        src_dir.mkdir()
        src = src_dir / self._ATTACK_NAME
        src.write_bytes(b"\x89PNG-local")
        state = jobs_state.JobState(
            prompt_id=PROMPT_ID,
            client_id=None,
            workflow="/abs/composed.json",
            where="local",
            base_url="http://127.0.0.1:8188",
            status="completed",
            outputs=[str(src)],
            record=None,
            item_map=None,
        )
        assert jobs_state.write(state) is not None

        set_renderer(Renderer(mode=OutputMode.NDJSON, command="download"))
        with patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target):
            paths = transfer.execute_download(PROMPT_ID, out_dir=str(tmp_path / "out"))

        assert len(paths) == 1
        name = Path(paths[0]).name
        assert not self._has_control_bytes(name), repr(name)
        # Exact name (see query-param twin above): control bytes stripped, benign
        # payload remnant kept — asserting the full name catches control-byte leaks.
        assert name == f"{SHORT_ID}_000.png31mHACK", name
        assert Path(paths[0]).is_file()

    @pytest.mark.parametrize(
        "suffix",
        [
            ".💥",  # emoji-only
            ".日本語",  # unicode-only
            ".\x1b",  # lone control byte
            ".",  # already just a dot
            "..",  # multiple dots
            ".-_",  # dots/dashes/underscores, no alnum
            "",  # empty
        ],
    )
    def test_sanitize_ext_collapses_extensionless_suffix_to_empty(self, suffix):
        # A suffix with no surviving alphanumeric char carries no real extension:
        # it must return "" so the caller's `or ".png"` fallback applies rather than
        # a truthy bare-dot result that bypasses the fallback and writes `<id>_000.`.
        assert transfer._sanitize_ext(suffix) == ""

    def test_sanitize_ext_keeps_real_extension(self):
        assert transfer._sanitize_ext(".png") == ".png"
        assert transfer._sanitize_ext(".7z") == ".7z"
        assert transfer._sanitize_ext(".tar.gz") == ".tar.gz"

    def test_sanitize_ext_caps_length(self):
        # A hostile `?filename=out.<thousands of safe chars>` must not yield an
        # over-long extension that pushes local_name past NAME_MAX.
        result = transfer._sanitize_ext("." + "a" * 5000)
        assert len(result) <= transfer._MAX_EXT_LEN
        assert not self._has_control_bytes(result)
