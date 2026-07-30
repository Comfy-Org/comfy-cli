"""``comfy generate list`` / ``comfy generate schema`` emit ``envelope/1`` (BE-4933).

These are the two discovery verbs an agent needs before it can run a partner
model: one enumerates the aliases, the other gives a model's parameters. Both
used to ignore the *global* ``--json`` outright (Rich table / plain prose on
stdout, exit 0) and, when the tail ``--json`` was picked up by
``_separate_meta_flags``, emitted a bare JSON blob rather than the versioned
envelope every other machine-readable command in the family returns. Agents had
to scrape box-drawing characters to get the model list — the catalog is not
reachable any other way (``comfy discover`` does not carry it).

What is pinned here:
  - both verbs, under both ``--json`` spellings, emit one ``envelope/1`` whose
    ``data`` validates against the command's registered schema;
  - the summary/description text is the FULL value, not the ``…``-truncated
    form the human table cuts to fit its column;
  - error paths carry a registered ``error.code`` instead of an ad-hoc blob;
  - pretty (human) output is untouched — this is an additive change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

SCHEMAS_DIR = Path(__file__).resolve().parents[4] / "comfy_cli" / "schemas"


def _validator_for(schema_name: str) -> jsonschema.Validator:
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


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    proc_env = os.environ.copy()
    proc_env.setdefault("NO_COLOR", "1")
    # Keep the CLI out of the user's real config/telemetry during the test run.
    proc_env.setdefault("COMFY_CLI_DISABLE_TELEMETRY", "1")
    if env:
        proc_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "comfy_cli", *args],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )


def _envelope(args: list[str], env: dict | None = None) -> dict:
    result = _run(args, env=env)
    assert result.stdout.strip(), f"empty stdout. stderr={result.stderr!r}"
    last = [line for line in result.stdout.splitlines() if line.strip()][-1]
    return json.loads(last)


# Both spellings must work: the global flag the Typer callback resolves, and the
# tail flag `generate`'s own `_separate_meta_flags` picks out of ctx.args.
LIST_INVOCATIONS = [
    ["--json", "generate", "list"],
    ["generate", "list", "--json"],
]
SCHEMA_INVOCATIONS = [
    ["--json", "generate", "schema", "flux-pro"],
    ["generate", "schema", "flux-pro", "--json"],
]


# --------------------------------------------------------------------------- #
# generate list
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("args", LIST_INVOCATIONS)
def test_list_emits_valid_envelope(args):
    envelope = _envelope(args)
    _validator_for("envelope.json").validate(envelope)
    assert envelope["schema"] == "envelope/1"
    assert envelope["type"] == "envelope"
    assert envelope["ok"] is True
    assert envelope["command"] == "generate list"
    assert envelope["error"] is None


@pytest.mark.parametrize("args", LIST_INVOCATIONS)
def test_list_payload_validates(args):
    envelope = _envelope(args)
    _validator_for("generate_list.json").validate(envelope["data"])


def test_list_carries_one_record_per_model_with_the_table_columns():
    from comfy_cli.command.generate import spec

    data = _envelope(["--json", "generate", "list"])["data"]
    assert data["count"] == len(data["models"]) == len(spec.list_endpoints())
    assert data["count"] > 1, "catalog should not be a one-row table"
    for row in data["models"]:
        # alias/name, partner, style, mode, summary — the columns the table renders.
        assert row["alias"] and row["id"] and row["partner"] and row["category"]
        assert row["mode"] in {"sync", "async"}
        assert isinstance(row["summary"], str)


def test_list_summaries_are_full_not_table_truncated():
    """The table cuts summaries at 60 chars + '…'; the JSON must not."""
    from comfy_cli.command.generate import spec

    data = _envelope(["--json", "generate", "list"])["data"]
    by_id = {e.id: e.summary for e in spec.list_endpoints()}
    for row in data["models"]:
        assert row["summary"] == by_id[row["id"]]
        assert not row["summary"].endswith("…")
    # Guard the assertion above is meaningful: at least one summary is long
    # enough that the table would have truncated it.
    assert any(len(row["summary"]) > 61 for row in data["models"])


def test_list_honors_filters_and_echoes_them():
    envelope = _envelope(["--json", "generate", "list", "--partner", "bfl"])
    data = envelope["data"]
    assert data["filters"] == {"partner": "bfl", "category": None, "query": None}
    assert data["models"], "expected at least one bfl model"
    assert {row["partner"] for row in data["models"]} == {"bfl"}


def test_list_with_no_matches_is_an_empty_success_not_an_error():
    result = _run(["--json", "generate", "list", "--partner", "no-such-partner"])
    envelope = json.loads(result.stdout.splitlines()[-1])
    assert result.returncode == 0
    assert envelope["ok"] is True
    assert envelope["data"]["models"] == []
    assert envelope["data"]["count"] == 0


# --------------------------------------------------------------------------- #
# generate schema <model>
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("args", SCHEMA_INVOCATIONS)
def test_schema_emits_valid_envelope(args):
    envelope = _envelope(args)
    _validator_for("envelope.json").validate(envelope)
    assert envelope["schema"] == "envelope/1"
    assert envelope["ok"] is True
    assert envelope["command"] == "generate schema"
    assert envelope["error"] is None


@pytest.mark.parametrize("args", SCHEMA_INVOCATIONS)
def test_schema_payload_validates(args):
    envelope = _envelope(args)
    _validator_for("generate_schema.json").validate(envelope["data"])


def test_schema_params_carry_name_type_required_description():
    data = _envelope(["--json", "generate", "schema", "flux-pro"])["data"]
    assert data["model"] == "flux-pro"
    assert data["id"] == "bfl/flux-pro-1.1/generate"
    by_name = {p["name"]: p for p in data["params"]}

    prompt = by_name["prompt"]
    assert prompt["type"] == "string"
    assert prompt["kind"] == prompt["type"], "`kind` is a retained alias of `type`"
    assert prompt["required"] is True
    assert prompt["description"], "descriptions must survive into JSON"


def test_schema_surfaces_enum_values():
    data = _envelope(["--json", "generate", "schema", "flux-pro"])["data"]
    by_name = {p["name"]: p for p in data["params"]}
    output_format = by_name["output_format"]
    assert output_format["type"] == "enum"
    assert output_format["enum"] == ["jpeg", "png"]
    assert output_format["required"] is False


def test_schema_descriptions_match_the_spec_verbatim():
    from comfy_cli.command.generate import schema as gen_schema
    from comfy_cli.command.generate import spec

    endpoint = spec.get_endpoint("flux-pro")
    expected = {f.name: f.description for f in gen_schema.flags_for(endpoint)}
    data = _envelope(["--json", "generate", "schema", "flux-pro"])["data"]
    assert {p["name"]: p["description"] for p in data["params"]} == expected


def test_schema_unknown_model_emits_registered_error_code():
    from comfy_cli import error_codes

    result = _run(["--json", "generate", "schema", "definitely-not-a-model"])
    envelope = json.loads(result.stdout.splitlines()[-1])
    assert result.returncode == 1
    assert envelope["ok"] is False
    assert envelope["data"] is None
    assert envelope["error"]["code"] == "generate_unknown_model"
    assert error_codes.is_registered(envelope["error"]["code"])
    assert envelope["error"]["hint"]
    assert envelope["error"]["details"]["requested"] == "definitely-not-a-model"


def test_schema_without_a_model_emits_registered_error_code():
    from comfy_cli import error_codes

    result = _run(["--json", "generate", "schema"])
    envelope = json.loads(result.stdout.splitlines()[-1])
    assert result.returncode == 1
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "generate_bad_args"
    assert error_codes.is_registered(envelope["error"]["code"])


# --------------------------------------------------------------------------- #
# mode resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "args",
    [
        ["generate", "list", "--json"],
        ["generate", "schema", "flux-pro", "--json"],
        # An explicit tail --json also outranks a global --no-json: it is the
        # more specific flag.
        ["--no-json", "generate", "list", "--json"],
    ],
)
def test_tail_json_upgrades_an_otherwise_pretty_renderer(args):
    """`Renderer.force_json` — the tail `--json` is parsed by `generate`'s own
    `_separate_meta_flags`, so it never reaches the global callback that
    resolves output mode. Without the upgrade, `renderer.emit` would no-op in
    pretty mode and the command would print nothing at all."""
    envelope = _envelope(args, env={"COMFY_OUTPUT": "pretty"})
    _validator_for("envelope.json").validate(envelope)
    assert envelope["ok"] is True


@pytest.mark.parametrize(
    "args", [["--json-stream", "generate", "list"], ["--json-stream", "generate", "schema", "flux-pro"]]
)
def test_stream_mode_emits_the_envelope_as_its_final_line(args):
    """These verbs have no intermediate events, so NDJSON degenerates to the
    single terminating envelope — and must not silently downgrade to JSON."""
    result = _run(args)
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    envelope = json.loads(lines[-1])
    assert envelope["type"] == "envelope"
    assert envelope["ok"] is True


def test_json_mode_leaves_stderr_clean():
    """No Rich table leaks to stderr alongside the envelope."""
    result = _run(["--json", "generate", "list"])
    assert result.stderr == ""


# --------------------------------------------------------------------------- #
# additive: pretty output is untouched
# --------------------------------------------------------------------------- #


def test_pretty_list_still_renders_the_table():
    result = _run(["generate", "list"], env={"COMFY_OUTPUT": "pretty", "COLUMNS": "100"})
    assert result.returncode == 0
    assert "Comfy Generate — Models" in result.stdout
    assert "Run `comfy generate schema <model>` to see parameters" in result.stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_pretty_schema_still_renders_prose():
    result = _run(["generate", "schema", "flux-pro"], env={"COMFY_OUTPUT": "pretty", "COLUMNS": "100"})
    assert result.returncode == 0
    assert result.stdout.startswith("Model: flux-pro")
    assert "Parameters (use as `--name value`):" in result.stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_pretty_schema_errors_stay_plain_red_lines():
    """Error paths keep their one-line human form — no envelope, no error panel."""
    env = {"COMFY_OUTPUT": "pretty", "COLUMNS": "100"}
    usage = _run(["generate", "schema"], env=env)
    assert usage.returncode == 1
    assert usage.stdout.strip() == "Usage: comfy generate schema <model>"

    unknown = _run(["generate", "schema", "definitely-not-a-model"], env=env)
    assert unknown.returncode == 1
    assert unknown.stdout.startswith("Unknown model: 'definitely-not-a-model'.")


# --------------------------------------------------------------------------- #
# discovery registration
# --------------------------------------------------------------------------- #


def test_both_verbs_register_a_schema_for_discover():
    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert COMMAND_SCHEMAS["comfy generate list"] == "generate_list"
    assert COMMAND_SCHEMAS["comfy generate schema"] == "generate_schema"


def test_discover_ships_the_new_schemas():
    data = _envelope(["--json", "discover", "--schemas-only"])["data"]
    assert "generate_list" in data["schemas"]
    assert "generate_schema" in data["schemas"]
