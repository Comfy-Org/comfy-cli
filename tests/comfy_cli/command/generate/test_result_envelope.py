"""``comfy generate`` result paths emit ``envelope/1``.

The discovery verbs (``list`` / ``schema``) were migrated onto the renderer
envelope earlier. The submit/sync/resume *result* paths were left emitting a
bare partner JSON blob via ``output.print_json`` (``_emit_result``,
``as_json=True``), so an agent parsing stdout by the documented contract got a
document with no ``schema``/``type``/``ok``/``error`` discriminator —
indistinguishable from an envelope except by reading it.

What is pinned here (unit-level; no network, no API key):

  - JSON and NDJSON modes: the terminal result is one ``envelope/1`` whose
    ``data`` validates against the registered ``generate_result.json`` schema;
  - ``--download`` surfaces the saved local paths inside ``data.saved``;
  - the schema is registered in ``COMMAND_SCHEMAS["comfy generate"]`` so
    ``comfy discover`` advertises it;
  - pretty mode with a tail ``--json`` keeps the legacy raw blob — this fix is
    additive for machine consumers only.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import jsonschema
import pytest

from comfy_cli.command.generate import app as generate_app
from comfy_cli.command.generate import poll
from comfy_cli.discovery import COMMAND_SCHEMAS
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

SCHEMAS_DIR = Path(__file__).resolve().parents[4] / "comfy_cli" / "schemas"


def _validator_for(schema_name: str) -> jsonschema.protocols.Validator:
    schema = json.loads((SCHEMAS_DIR / schema_name).read_text())
    store: dict[str, dict] = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        s = json.loads(path.read_text())
        if s.get("$id"):
            store[s["$id"]] = s
        store[path.name] = s
    base = SCHEMAS_DIR.absolute().as_uri() + "/"
    resolver = jsonschema.RefResolver(base_uri=base, referrer=schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


def _succeeded(**overrides) -> poll.PollResult:
    fields = {
        "status": "succeeded",
        "error": None,
        "image_urls": ["https://cdn.example/img.png"],
        "raw": {"status": "succeeded", "urls": ["https://cdn.example/img.png"]},
    }
    fields.update(overrides)
    return poll.PollResult(**fields)


@pytest.fixture()
def pinned_renderer():
    def make(mode: OutputMode) -> tuple[Renderer, io.StringIO, io.StringIO]:
        machine, pretty = io.StringIO(), io.StringIO()
        renderer = Renderer(
            mode=mode,
            command="generate",
            version="test",
            _machine_stream_override=machine,
            _pretty_stream_override=pretty,
        )
        set_renderer(renderer)
        return renderer, machine, pretty

    yield make
    reset_renderer_for_testing()


# --------------------------------------------------------------------------- #
# These four failed before the fix — stdout carried a bare partner blob.
# --------------------------------------------------------------------------- #


def test_success_emits_envelope_in_json_mode(pinned_renderer):
    _, machine, _ = pinned_renderer(OutputMode.JSON)
    generate_app._emit_result(_succeeded(), request_id="req1", download=None, as_json=True)
    envelope = json.loads(machine.getvalue().splitlines()[-1])
    assert envelope["schema"] == "envelope/1"
    assert envelope["type"] == "envelope"
    assert envelope["ok"] is True
    assert envelope["command"] == "generate"
    assert envelope["error"] is None


def test_download_variant_lists_saved_paths_in_data(pinned_renderer, tmp_path):
    _, machine, _ = pinned_renderer(OutputMode.JSON)
    saved = tmp_path / "out.png"
    from comfy_cli.command.generate import output as gen_output

    original = gen_output.save_urls
    gen_output.save_urls = lambda urls, d, rid: [saved]
    try:
        generate_app._emit_result(_succeeded(), request_id="req1", download=str(tmp_path), as_json=True)
    finally:
        gen_output.save_urls = original
    data = json.loads(machine.getvalue().splitlines()[-1])["data"]
    assert data["saved"] == [str(saved)]


def test_payload_validates_against_registered_schema(pinned_renderer):
    assert COMMAND_SCHEMAS.get("comfy generate") == "generate_result", (
        "`comfy generate` must advertise its result schema via `comfy discover`"
    )
    _, machine, _ = pinned_renderer(OutputMode.JSON)
    generate_app._emit_result(_succeeded(), request_id="req1", download=None, as_json=True)
    envelope = json.loads(machine.getvalue().splitlines()[-1])
    _validator_for("generate_result.json").validate(envelope["data"])
    _validator_for("envelope.json").validate(envelope)


def test_ndjson_final_line_is_the_envelope(pinned_renderer):
    _, machine, _ = pinned_renderer(OutputMode.NDJSON)
    generate_app._emit_result(_succeeded(), request_id="req1", download=None, as_json=True)
    lines = [json.loads(x) for x in machine.getvalue().splitlines()]
    assert lines[-1]["schema"] == "envelope/1"
    assert all(line.get("type") != "envelope" for line in lines[:-1])


# --------------------------------------------------------------------------- #
# Guard: pretty mode with a tail --json keeps the legacy raw blob.
# --------------------------------------------------------------------------- #


def test_pretty_mode_tail_json_keeps_legacy_raw_blob(pinned_renderer, capsys):
    # print_json bypasses the renderer and writes builtin sys.stdout, so
    # capture via capsys rather than the renderer's stream overrides.
    pinned_renderer(OutputMode.PRETTY)
    generate_app._emit_result(_succeeded(), request_id="req1", download=None, as_json=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "succeeded"


# --------------------------------------------------------------------------- #
# Review (annehe9) on #809
# --------------------------------------------------------------------------- #


def _failed(**overrides) -> poll.PollResult:
    fields = {
        "status": "failed",
        "error": "partner blew up",
        "image_urls": [],
        "raw": {"status": "failed", "error": "partner blew up"},
    }
    fields.update(overrides)
    return poll.PollResult(**fields)


def test_failed_job_is_an_ok_false_envelope_and_exit_1_in_json_mode(pinned_renderer):
    """A terminally failed job must never be reported as ``ok: true`` — the
    registered ``generate_result`` schema promises failures arrive as an
    ok=false envelope with ``error.code``, and machine consumers trust ``ok``."""
    import typer

    _, machine, _ = pinned_renderer(OutputMode.JSON)
    with pytest.raises(typer.Exit) as e:
        generate_app._emit_result(_failed(), request_id="req1", download=None, as_json=True)
    assert e.value.exit_code == 1
    envelope = json.loads(machine.getvalue().splitlines()[-1])
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "generate_job_failed"
    assert envelope["data"] is None


def test_output_json_alone_gets_the_envelope_not_colored_text(pinned_renderer, capsys):
    """``comfy --output json generate …`` (no tail ``--json``) must emit the
    envelope on the machine stream, never ANSI-colored URLs on stdout."""
    _, machine, _ = pinned_renderer(OutputMode.JSON)
    generate_app._emit_result(_succeeded(), request_id="req1", download=None, as_json=False)
    envelope = json.loads(machine.getvalue().splitlines()[-1])
    assert envelope["schema"] == "envelope/1" and envelope["ok"] is True
    assert envelope["data"]["result"] == {"status": "succeeded", "urls": ["https://cdn.example/img.png"]}
    assert "\x1b[" not in capsys.readouterr().out
