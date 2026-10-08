from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest
from build_validate_support import ResolveRecorder, local_model, local_node, remote_model, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.build_validation import (
    HIDDEN_LINK_LABEL,
    _https_url,
    _lacks_extension,
    _valid_filename,
    _valid_model_dir,
)


@pytest.fixture(autouse=True)
def stable_command_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        build,
        "capture_pip_provenance",
        lambda python: {
            "pipDependencies": "example==1.0.0\n",
            "environment": {"os": "Linux", "arch": "x86_64", "pythonVersion": "3.12.0", "torch": None},
        },
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "install"
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "models" / "checkpoints" / "base.safetensors").write_bytes(b"MODEL")
    (root / "custom_nodes" / "local-node").mkdir(parents=True)
    (root / "custom_nodes" / "local-node" / "nodes.py").write_bytes(b"NODE")
    return root


def _invoke(root: Path, *args: str, token: str | None = None, output: str = "json"):
    return CliRunner().invoke(
        cli_app,
        ["build", "validate", *args, str(root)],
        env={
            "AI_AGENT": "1",
            "COMFY_OUTPUT": output,
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": token,
            "COMFY_BUILDER_URL": "https://builder.test",
        },
    )


def _envelope(result) -> dict:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.output
    return json.loads(lines[-1])


def _entry(definition: JsonObject, collection: str) -> JsonObject:
    entries = definition[collection]
    assert isinstance(entries, list)
    entry = entries[0]
    assert isinstance(entry, dict)
    return entry


def test_offline_validate_accepts_local_entries_without_constructing_a_client(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, models=[local_model()], nodes=[local_node()])
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("Builder client constructed"))

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    envelope = _envelope(result)
    assert envelope["command"] == "build validate"
    assert envelope["data"]["remote"] is False
    assert envelope["data"]["wire_definition"]["models"][0].get("blobId") is None


def test_init_then_offline_validate_passes_signed_out(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    init = CliRunner().invoke(
        cli_app,
        [
            "build",
            "init",
            "--name",
            "Fixture",
            "--python",
            sys.executable,
            "--comfy-version",
            "0.3.0",
            str(workspace),
        ],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1", "COMFY_BUILDER_TOKEN": None},
    )
    assert init.exit_code == 0, init.stderr
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("Builder client constructed"))

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 0, result.stderr


def test_wire_projection_collapses_cached_sources_and_preserves_unknown_keys(workspace: Path) -> None:
    # Given
    model = local_model(sourceUri="https://models.example/base", blobId="blob-model", future={"nested": True})
    node = local_node(
        repository="https://github.com/example/node",
        gitRef="main",
        commit="a" * 40,
        registryVersion="1.2.3",
        blobId="blob-node",
        future={"nested": True},
    )
    write_spec(workspace, models=[model], nodes=[node])

    # When
    result = _invoke(workspace)
    projected = build.project_wire_definition({"models": [model], "customNodes": [node], "future": {"top": True}})

    # Then
    assert result.exit_code == 0, result.stderr
    assert projected["future"] == {"top": True}
    assert _entry(projected, "models") == {
        "type": "checkpoints",
        "filename": "base.safetensors",
        "blobId": "blob-model",
        "future": {"nested": True},
    }
    assert _entry(projected, "customNodes") == {
        "name": "local-node",
        "blobId": "blob-node",
        "future": {"nested": True},
    }


@pytest.mark.parametrize(
    ("models", "nodes", "offending"),
    [
        ([{"type": "checkpoints", "filename": "public.safetensors"}], [], "definition.models[0]"),
        ([], [{"name": "public-node"}], "definition.customNodes[0]"),
    ],
)
def test_non_local_entry_without_an_effective_source_fails(
    workspace: Path, models: list[JsonObject], nodes: list[JsonObject], offending: str
) -> None:
    # Given
    write_spec(workspace, models=models, nodes=nodes)

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert offending in error["message"]


def test_whitespace_source_is_unset_and_non_string_is_rejected_before_precedence(workspace: Path) -> None:
    # Given
    write_spec(
        workspace,
        models=[
            {"type": "checkpoints", "filename": "blank.safetensors", "sourceUri": "   "},
            {"type": "loras", "filename": "malformed.safetensors", "sourceUri": ["bad"], "blobId": "winner"},
        ],
        nodes=[],
    )

    # When
    result = _invoke(workspace)

    # Then: the malformed second entry is rejected even though blobId would win precedence.
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert "definition.models[1].sourceUri" in error["message"]
    assert "must be a string" in error["message"]


@pytest.mark.parametrize(
    ("models", "nodes", "offending"),
    [
        ([local_model(localPath="checkpoints")], [local_node()], "definition.models[0].localPath"),
        ([local_model()], [local_node(localPath="../models/checkpoints/base.safetensors")], "customNodes[0]"),
    ],
)
def test_local_path_must_be_lexical_and_match_the_entry_kind(
    workspace: Path, models: list[JsonObject], nodes: list[JsonObject], offending: str
) -> None:
    # Given
    write_spec(workspace, models=models, nodes=nodes)

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert offending in error["message"]


def test_remote_reports_all_four_states_without_failing_on_no_candidate(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    recorder = ResolveRecorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", recorder)
    models = [
        remote_model("public.safetensors"),
        remote_model("private.safetensors", 1),
        remote_model("outage.safetensors", 2),
        remote_model(None, 3),
    ]
    write_spec(workspace, models=models, nodes=[])

    # When
    result = _invoke(workspace, "--remote", token="tok_test")

    # Then
    assert result.exit_code == 0, result.stderr
    lookups = _envelope(result)["data"]["model_lookups"]
    assert [lookup["state"] for lookup in lookups] == [
        "candidate_found",
        "none_found",
        "lookup_error",
        "not_lookupable",
    ]
    assert lookups[1].get("error") is None
    assert lookups[2]["error"] == "providers unavailable"
    assert recorder.calls == [{"method": "POST", "body": {"filenames": [model["filename"] for model in models[:3]]}}]


def test_remote_batches_seventy_local_filenames_in_spec_order(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    recorder = ResolveRecorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", recorder)
    models = [remote_model(f"model-{index:03}.safetensors", index) for index in range(70)]
    write_spec(workspace, models=models, nodes=[])

    # When
    result = _invoke(workspace, "--remote", token="tok_test")

    # Then
    assert result.exit_code == 0, result.stderr
    assert [len(call["body"]["filenames"]) for call in recorder.calls] == [32, 32, 6]
    sent = [filename for call in recorder.calls for filename in call["body"]["filenames"]]
    expected = [model["filename"] for model in models]
    assert sent == expected
    assert [lookup["filename"] for lookup in _envelope(result)["data"]["model_lookups"]] == expected


@pytest.mark.parametrize("models", [[], [remote_model(None)]])
def test_remote_with_no_lookupable_models_makes_zero_requests(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, models: list[JsonObject]
) -> None:
    # Given
    recorder = ResolveRecorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", recorder)
    write_spec(workspace, models=models, nodes=[])

    # When
    result = _invoke(workspace, "--remote", token="tok_test")

    # Then
    assert result.exit_code == 0, result.stderr
    assert recorder.calls == []
    states = [lookup["state"] for lookup in _envelope(result)["data"]["model_lookups"]]
    assert states == (["not_lookupable"] if models else [])


def test_remote_auth_precedes_workspace_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    recorder = ResolveRecorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", recorder)

    # When
    result = _invoke(tmp_path / "missing-workspace", "--remote")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "build_not_signed_in"
    assert recorder.calls == []


def test_pretty_remote_output_keeps_none_and_lookup_errors_distinct(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    recorder = ResolveRecorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", recorder)
    write_spec(
        workspace,
        models=[remote_model("private.safetensors"), remote_model("outage.safetensors", 1)],
        nodes=[],
    )

    # When
    result = _invoke(workspace, "--remote", token="tok_test", output="pretty")

    # Then
    assert result.exit_code == 0, result.output
    assert "none_found" in result.stdout
    assert "lookup_error" in result.stdout


#: Model entries as people paste them from Civitai or Hugging Face, each breaking
#: one rule of the builder's release cut.
PASTED_MODELS: list[JsonObject] = [
    {"type": "Loras", "sourceUri": "https://h.example/a.safetensors"},
    {"type": "loras", "sourceUri": "https://civitai.com/api/download/models/128713"},
    {"type": "loras", "filename": "add_detail (v1.1).safetensors", "sourceUri": "https://h.example/b.safetensors"},
    {"type": "loras", "sha256": "8A0B5F4C2E", "sourceUri": "https://h.example/c.safetensors"},
]


def _refused(result) -> dict[str, str]:
    """The refused entries, as the model each names to the rule it breaks."""
    error = _envelope(result)["error"]
    assert error["code"] == "build_spec_invalid", error
    return {issue["model"]: issue["field"].rsplit(".", 1)[1] for issue in error["details"]["invalid"]}


def test_validate_names_every_model_entry_the_builder_would_refuse(workspace: Path) -> None:
    """Offline it cannot know the builder's folders, so a case variant waits for the
    save; the rest are named together, before anything is uploaded."""
    # Given
    write_spec(workspace, models=PASTED_MODELS, nodes=[])

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 1
    assert _refused(result) == {
        "https://civitai.com/api/download/models/128713": "filename",
        "add_detail (v1.1).safetensors": "filename",
        "https://h.example/c.safetensors": "sha256",
    }
    assert _envelope(result)["error"]["message"].startswith("3 problems the builder would refuse this build for:")


def test_remote_validate_also_names_a_case_variant_of_a_folder(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    recorder = ResolveRecorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", recorder)
    write_spec(workspace, models=PASTED_MODELS, nodes=[])

    # When
    result = _invoke(workspace, "--remote", token="tok_test")

    # Then
    assert result.exit_code == 1
    assert _refused(result)["https://h.example/a.safetensors"] == "type"
    assert len(_refused(result)) == 4
    assert recorder.calls == [], "a refused spec looks nothing up"


def test_pretty_validate_prints_each_problem_on_its_own_line(workspace: Path) -> None:
    # Given
    write_spec(workspace, models=PASTED_MODELS, nodes=[])

    # When
    result = _invoke(workspace, output="pretty")

    # Then
    assert result.exit_code == 1
    lines = [line.strip("│ ") for line in result.output.splitlines()]
    assert sum(line.startswith("definition.models[") for line in lines) == 3, result.output


@pytest.mark.parametrize(
    "model",
    [
        pytest.param({"type": "ipadapter/custom", "blobId": "blob-1"}, id="a-new-folder"),
        pytest.param(
            {"type": "loras", "filename": "ok.safetensors", "sourceUri": "https://civitai.com/api/download/models/1"},
            id="a-link-with-a-filename",
        ),
        pytest.param(
            {"type": "loras", "sourceUri": "https://h.example/x.safetensors", "sha256": "A" * 64}, id="upper-hex"
        ),
        pytest.param(
            {"type": "loras", "sourceUri": "https://h.example/x.safetensors", "sha256": f" {'a' * 64}\n"},
            id="padded-sha256",
        ),
        pytest.param(
            {"type": "loras", "filename": "  ", "sourceUri": "https://h.example/x.safetensors"}, id="blank-filename"
        ),
        pytest.param(
            {"type": "loras", "sourceUri": "https://h.example/x.safetensors", "sha256": ""}, id="empty-sha256"
        ),
    ],
)
def test_validate_passes_what_the_builder_accepts(workspace: Path, model: JsonObject) -> None:
    # Given
    write_spec(workspace, models=[model], nodes=[])

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 0, result.stdout


def test_a_local_models_sha256_is_left_to_the_push(workspace: Path) -> None:
    """Push hashes a local model and replaces its sha256, so a stale one refuses nothing."""
    # Given
    write_spec(workspace, models=[local_model(sha256="a")], nodes=[])

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 0, result.stdout


#: Links with the builder's answer for each: (``validHTTPSURL``, ``derivedFilenameLacksExt``).
#: Every row was checked against those two Go functions, copied verbatim from
#: comfy-builder's definition.go and run under go1.26.4. The builder reads a link
#: with Go's ``url.Parse``, which decodes the path and refuses some links Python's
#: ``urlsplit`` takes.
BUILDER_LINK_ANSWERS = [
    pytest.param("https://h.example/x.safetensors", True, False, id="plain"),
    pytest.param("https://civitai.com/api/download/models/128713", True, True, id="civitai-id"),
    pytest.param("https://h.example/m%2Esafetensors", True, False, id="escaped-dot-is-an-extension"),
    pytest.param("https://h.example/model.safetensors%2F123", True, True, id="escaped-slash-ends-the-name"),
    pytest.param("https://h.example/a%20b.safetensors", True, False, id="escaped-space"),
    pytest.param("https://h.example/100%.safetensors", False, False, id="bad-escape"),
    pytest.param("https://h.example/x.safetensors#100%", False, False, id="bad-escape-in-fragment"),
    pytest.param("https://h.example:abc/x.safetensors", False, False, id="port-not-a-number"),
    pytest.param("https://h.example:8443/x.safetensors", True, False, id="port"),
    pytest.param("https://h.example:1:2/x.safetensors", False, False, id="two-ports"),
    pytest.param("https://h.example/a\tb.safetensors", False, False, id="tab"),
    pytest.param("\x1fhttps://h.example/x.safetensors", False, False, id="control-char-str-strip-drops"),
    pytest.param("https://h.example/x.safetensors#frag\x01", True, False, id="control-char-in-fragment"),
    pytest.param("  https://h.example/x.safetensors\n", True, False, id="surrounding-space"),
    pytest.param("HTTPS://h.example/x.safetensors", True, False, id="scheme-case"),
    pytest.param("http://h.example/x.safetensors", False, False, id="http"),
    pytest.param("https:h.example/x.safetensors", False, False, id="opaque"),
    pytest.param("https:models/128713", False, False, id="opaque-has-no-path"),
    pytest.param("https:///x.safetensors", False, False, id="no-host"),
    pytest.param("https://h ex.example/x.safetensors", False, False, id="space-in-host"),
    pytest.param("https://%41.example/x.safetensors", False, False, id="ascii-escape-in-host"),
    pytest.param("https://us er@h.example/x.safetensors", False, False, id="space-in-userinfo"),
    pytest.param("https://[::1]/x.safetensors", True, False, id="ipv6"),
    pytest.param("https://[1.2.3.4]/x.safetensors", False, False, id="ipv4-in-brackets"),
    pytest.param("https://h.example", True, False, id="no-path"),
    pytest.param("https://h.example/models/", True, True, id="trailing-slash"),
    pytest.param("https://h.example/x.safetensors/", True, False, id="trailing-slash-after-a-name"),
    pytest.param("https://h.example/dl?file=x.safetensors", True, True, id="name-in-query"),
    pytest.param("//h.example/model", False, True, id="no-scheme-with-a-host"),
    pytest.param("models/128713", False, True, id="no-scheme-relative-path"),
    pytest.param("/model", False, True, id="no-scheme-absolute-path"),
    pytest.param("///models/128713", False, True, id="no-scheme-three-slashes-is-all-path"),
    pytest.param("https:x", False, False, id="opaque-short"),
    pytest.param("1h:80/model", False, False, id="no-scheme-colon-in-first-segment"),
    pytest.param("https://localhost", True, False, id="host-with-no-dot-and-no-path"),
    pytest.param("https://h.example:abc/model", False, False, id="bad-port-before-a-bare-name"),
    pytest.param("https://u:p@h.example/model", True, True, id="userinfo"),
    pytest.param("https://us er@h.example/model", False, False, id="bad-userinfo-before-a-bare-name"),
    pytest.param("https://u%zz@h.example/model", False, False, id="bad-escape-in-userinfo"),
    pytest.param("https://h[x.example/model", False, False, id="bracket-inside-a-host"),
    pytest.param("https://[::1]:8443/model", True, True, id="ipv6-with-port"),
    pytest.param("https://[::1]:x/model", False, False, id="ipv6-bad-port"),
    pytest.param("https://[::1]x/model", False, False, id="ipv6-junk-after-the-bracket"),
    pytest.param("https://[::1/model", False, False, id="ipv6-unclosed"),
    pytest.param("https://[fe80::1%25en0]/model", True, True, id="ipv6-zone"),
    pytest.param("https://[fe80::1%25en0]:8443/model", True, True, id="ipv6-zone-with-port"),
    pytest.param("https://[fe80::1%25]/model", False, False, id="ipv6-empty-zone"),
    pytest.param("https://[fe80::1%25en%200]/model", True, True, id="ipv6-zone-escaped-space"),
    pytest.param("https://[fe80::1%25en%2F0]/model", False, False, id="ipv6-zone-escaped-slash"),
    pytest.param("https://[fe80::1%en0]/model", False, False, id="ipv6-zone-unescaped-percent"),
    pytest.param("https://[fe%3A80::1]/model", False, False, id="ipv6-escape-before-the-zone"),
    pytest.param("https://h%zz.example/model", False, False, id="bad-escape-in-host"),
    pytest.param("https://h%4/model", False, False, id="short-escape-in-host"),
    pytest.param("https://h%C3%A9.example/model", True, True, id="non-ascii-escape-in-host"),
    pytest.param("https://h%25.example/model", True, True, id="escaped-percent-in-host"),
    pytest.param("https://h.example:/model", True, True, id="empty-port"),
    pytest.param("https://h.example:1:2/model", False, False, id="https-port-ends-at-the-first-colon"),
    pytest.param("http://h.example:1:2/model", False, False, id="http-port-ends-at-the-first-colon"),
    pytest.param("ftp://h.example:1:2/model", False, True, id="other-scheme-port-ends-at-the-last-colon"),
    pytest.param("ftp://h.example:abc/model", False, False, id="other-scheme-bad-port"),
]


@pytest.mark.parametrize(("uri", "https", "lacks_extension"), BUILDER_LINK_ANSWERS)
def test_links_read_as_the_builder_reads_them(uri: str, https: bool, lacks_extension: bool) -> None:
    assert (_https_url(uri), _lacks_extension(uri)) == (https, lacks_extension)


@pytest.mark.parametrize(
    ("uri", "field"),
    [
        pytest.param("https://h.example/m%2Esafetensors", None, id="escaped-dot"),
        pytest.param("https://h.example/model.safetensors%2F123", "filename", id="escaped-slash"),
        pytest.param("https://h.example/100%.safetensors", "sourceUri", id="bad-escape"),
    ],
)
def test_validate_answers_a_link_as_the_builder_would(workspace: Path, uri: str, field: str | None) -> None:
    # Given
    write_spec(workspace, models=[{"type": "loras", "sourceUri": uri}], nodes=[])

    # When
    result = _invoke(workspace)

    # Then
    if field is None:
        assert result.exit_code == 0, result.stdout
    else:
        assert result.exit_code == 1
        assert _refused(result) == {uri: field}


#: Values Python's ``str.strip`` reads as blank and Go's ``strings.TrimSpace`` does
#: not (\x1c-\x1f), each with the field the builder refuses, as ``definition.go``
#: ``validateModels`` answers it under go1.26.4.
@pytest.mark.parametrize(
    ("model", "field"),
    [
        pytest.param(
            {"type": "loras", "sourceUri": "https://h.example/x.safetensors", "sha256": "a" * 64 + "\x1c"},
            "sha256",
            id="sha256-with-a-separator",
        ),
        pytest.param(
            {"type": "loras", "filename": "\x1f", "sourceUri": "https://h.example/x.safetensors"},
            "filename",
            id="filename-of-a-separator",
        ),
        pytest.param(
            {"type": "loras", "filename": "x.safetensors", "sourceUri": "\x1c"},
            "sourceUri",
            id="link-of-a-separator",
        ),
    ],
)
def test_the_rules_trim_as_the_builder_does(workspace: Path, model: JsonObject, field: str) -> None:
    # Given
    write_spec(workspace, models=[model], nodes=[])

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 1
    assert list(_refused(result).values()) == [field]


@pytest.mark.parametrize("output", ["json", "pretty"])
def test_a_refused_link_is_named_without_its_credentials(workspace: Path, output: str) -> None:
    """A link names an entry that has no filename, and a signed link or a Civitai
    ``?token=`` carries a credential that must not reach a terminal or a CI log."""
    # Given
    link = "https://me:hunter2@civitai.com/api/download/models/1?token=s3cr3t&type=Model#part"
    write_spec(workspace, models=[{"type": "loras", "sourceUri": link}], nodes=[])

    # When
    result = _invoke(workspace, output=output)

    # Then
    assert result.exit_code == 1
    assert "definition.models[0].filename" in result.output
    for secret in ("hunter2", "s3cr3t", "type=Model", "#part"):
        assert secret not in result.output
    if output == "json":
        assert _refused(result) == {"https://civitai.com/api/download/models/1": "filename"}


@pytest.mark.parametrize("output", ["json", "pretty"])
@pytest.mark.parametrize(
    ("link", "secrets", "label"),
    [
        pytest.param(
            "https://alice:s3cr/et+tok@host.example/m.safetensors",
            ("alice", "s3cr", "et+tok"),
            "…/m.safetensors",
            id="slash-in-the-password",
        ),
        pytest.param(
            "https://alice:s3cr?et@host.example/m.safetensors",
            ("alice", "s3cr", "et@"),
            HIDDEN_LINK_LABEL,
            id="question-mark-in-the-password",
        ),
        pytest.param(
            "https://alice:s3cr#et@host.example/m.safetensors",
            ("alice", "s3cr", "et@"),
            HIDDEN_LINK_LABEL,
            id="hash-in-the-password",
        ),
        pytest.param(
            "https://bob:Pq7?Zr8#Wk9/Xy6@host.example/m.safetensors",
            ("bob", "Pq7", "Zr8", "Wk9", "Xy6"),
            HIDDEN_LINK_LABEL,
            id="question-mark-and-hash-in-the-password",
        ),
        pytest.param(
            # Go parses this one: host alice, port 8841, path /Zk9x@host.example/...
            "https://alice:8841/Zk9x@host.example/m.safetensors",
            ("alice", ":8841", "Zk9x"),
            "…/m.safetensors",
            id="password-go-reads-as-a-port",
        ),
        pytest.param(
            "https://host.example/m.safetensors?sig=Ab3@Zk9q",
            ("sig=", "Ab3", "Zk9q"),
            HIDDEN_LINK_LABEL,
            id="at-in-the-query",
        ),
        pytest.param(
            "https://alice:s3cret@host.example/dir/m.safetensors?token=T0kn#frag",
            ("alice", "s3cret", "T0kn", "frag"),
            "https://host.example/dir/m.safetensors",
            id="a-link-that-parses",
        ),
    ],
)
def test_a_model_label_never_shows_a_links_credentials(
    workspace: Path, link: str, secrets: tuple[str, ...], label: str, output: str
) -> None:
    """A password pasted as is (a base64 token holds '/', and may hold '?' or '#')
    makes a link Go cannot parse, or parses as something else; the label names no
    part of it, and still shows the scheme, host and path of a link that parses. A
    hidden link is named by the file after its last '@', unless a '?' or '#' before
    that '@' may have started a query it sits in."""
    # Given
    write_spec(workspace, models=[{"type": "loras", "sourceUri": link, "sha256": "nope"}], nodes=[])

    # When
    result = _invoke(workspace, output=output)

    # Then
    assert result.exit_code == 1, result.output
    said = "".join(result.output.split())
    for secret in secrets:
        assert secret not in said, result.output
    assert "".join(label.split()) in said
    if output == "json":
        issues = _envelope(result)["error"]["details"]["invalid"]
        assert {issue["model"] for issue in issues} == {label}


@pytest.mark.parametrize("output", ["json", "pretty"])
def test_hidden_links_still_name_their_models_apart(workspace: Path, output: str) -> None:
    """A jsDelivr ``@version`` path puts an '@' after the first '/', so the link is
    not shown; each model is still named by the file its link points at, as
    ``models[n]`` counts the sorted spec and a JSON consumer keys by ``model``."""
    # Given
    write_spec(
        workspace,
        models=[
            {"type": "loras", "sourceUri": "https://cdn.jsdelivr.net/gh/team/loras@v1/a.safetensors", "sha256": "no"},
            {"type": "loras", "sourceUri": "https://cdn.jsdelivr.net/gh/team/loras@v1/b", "sha256": "a" * 64},
        ],
        nodes=[],
    )

    # When
    result = _invoke(workspace, output=output)

    # Then
    assert result.exit_code == 1, result.output
    said = "".join(result.output.split())
    assert "".join(HIDDEN_LINK_LABEL.split()) not in said
    assert "…/a.safetensors" in said
    assert "(…/b)" in said or '"…/b"' in said
    if output == "json":
        issues = _envelope(result)["error"]["details"]["invalid"]
        assert {issue["model"]: issue["field"].rsplit(".", 1)[-1] for issue in issues} == {
            "…/a.safetensors": "sha256",
            "…/b": "filename",
        }


def test_a_refused_link_is_named_without_its_fragment(workspace: Path) -> None:
    # Given
    write_spec(
        workspace, models=[{"type": "loras", "sourceUri": "https://civitai.com/api/download/models/1#part"}], nodes=[]
    )

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 1
    assert _refused(result) == {"https://civitai.com/api/download/models/1": "filename"}


def test_a_blank_filename_does_not_name_a_link_with_no_extension(workspace: Path) -> None:
    """The builder trims the filename first, so blank is no filename at all."""
    # Given
    link = "https://civitai.com/api/download/models/1"
    write_spec(workspace, models=[{"type": "loras", "filename": " \t", "sourceUri": link}], nodes=[])

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 1
    (issue,) = _envelope(result)["error"]["details"]["invalid"]
    assert (issue["field"], issue["reason"]) == (
        "definition.models[0].filename",
        "sourceUri has no file extension; set an explicit filename",
    )


#: The builder's vetted folders as the tests' list carries them.
DIRECTORIES = frozenset({"checkpoints", "loras", "vae"})


#: Each row checked against Go ``common.ValidModelDir`` (whose own list holds these
#: three), imported from the cloud repo and run under go1.26.4.
@pytest.mark.parametrize(
    ("value", "valid"),
    [
        pytest.param("loras", True, id="vetted"),
        pytest.param("Loras", False, id="case-variant"),
        pytest.param("LORAS", False, id="case-variant-upper"),
        pytest.param("ipadapter/custom", True, id="new-folder"),
        pytest.param("insightface/models/antelopev2", True, id="nested"),
        pytest.param("configs", False, id="reserved-configs"),
        pytest.param("CONFIGS/x", False, id="reserved-configs-any-case"),
        pytest.param("custom_nodes", False, id="reserved-custom-nodes"),
        pytest.param("Custom_Nodes/foo", False, id="reserved-custom-nodes-any-case"),
        pytest.param("foo./bar", False, id="trailing-dot-segment"),
        pytest.param("con", False, id="windows-device"),
        pytest.param("aux.x", False, id="windows-device-with-extension"),
        pytest.param("lpt9.bin", False, id="windows-lpt"),
        pytest.param("com0", True, id="not-a-windows-device"),
        pytest.param("a b", False, id="space"),
        pytest.param("-lead", False, id="leading-dash"),
        pytest.param("a//b", False, id="empty-segment"),
        pytest.param("/abs", False, id="absolute"),
        pytest.param("../x", False, id="parent"),
        pytest.param("a" * 127 + "/" + "a" * 127, True, id="255-chars"),
        pytest.param("a" * 127 + "/" + "a" * 128, False, id="too-long"),
        pytest.param("", False, id="empty"),
    ],
)
def test_model_directories_as_the_builder_reads_them(value: str, valid: bool) -> None:
    assert _valid_model_dir(value, DIRECTORIES) is valid


def test_a_case_variant_passes_without_the_builders_list() -> None:
    """Offline there is no list to tell "Loras" from a new folder, so it is left to
    the save, which warns about it; the builder itself refuses it."""
    assert _valid_model_dir("Loras", None) is True
    assert _valid_model_dir("Loras", DIRECTORIES) is False


#: Each row checked against Go ``definition.ValidFilename``, under go1.26.4.
@pytest.mark.parametrize(
    ("value", "valid"),
    [
        pytest.param("x.safetensors", True, id="plain"),
        pytest.param("a.b.c", True, id="dots"),
        pytest.param("..", False, id="dot-dot"),
        pytest.param("a..b", False, id="dot-dot-inside"),
        pytest.param(".hidden", False, id="leading-dot"),
        pytest.param("a/b", False, id="separator"),
        pytest.param("a b", False, id="space"),
        pytest.param("a" * 256, False, id="too-long"),
    ],
)
def test_filenames_as_the_builder_reads_them(value: str, valid: bool) -> None:
    assert _valid_filename(value) is valid


def test_one_problem_is_counted_singular_and_names_its_model(workspace: Path) -> None:
    # Given
    write_spec(
        workspace,
        models=[{"type": "loras", "sha256": "8A0B5F4C2E", "sourceUri": "https://h.example/c.safetensors"}],
        nodes=[],
    )

    # When
    result = _invoke(workspace)

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["message"] == (
        "1 problem the builder would refuse this build for:\n"
        "  definition.models[0].sha256 (https://h.example/c.safetensors): must be a 64-character sha256"
    )


def test_an_entry_with_nothing_to_name_it_by_has_no_parentheses(workspace: Path) -> None:
    # Given
    write_spec(workspace, models=[{"type": "Bad Folder", "blobId": "blob-1"}], nodes=[])

    # When
    result = _invoke(workspace)

    # Then
    message = _envelope(result)["error"]["message"]
    assert message.splitlines()[1] == "  definition.models[0].type: " + (
        "must be a model directory under models/ (e.g. checkpoints, insightface/models/antelopev2)"
    )


def test_pretty_validate_leaves_the_problem_list_out_of_the_details(workspace: Path) -> None:
    """Pretty mode already lists every problem in the message; the list is for JSON."""
    # Given
    write_spec(workspace, models=PASTED_MODELS, nodes=[])

    # When
    json_result = _invoke(workspace)
    pretty_result = _invoke(workspace, output="pretty")

    # Then
    assert len(_envelope(json_result)["error"]["details"]["invalid"]) == 3
    lines = [line.strip("│ ") for line in pretty_result.output.splitlines()]
    assert any(line.startswith("path") for line in lines), pretty_result.output
    assert not any(line.startswith("invalid") for line in lines), pretty_result.output


#: What validate says when it did not check a folder's case.
_FOLDER_CASE_UNCHECKED = "a folder's case (`Loras` for `loras`) was not checked"

#: A case variant only the builder's list tells from a new folder.
CASE_VARIANT: list[JsonObject] = [{"type": "Loras", "sourceUri": "https://h.example/a.safetensors"}]


def _schema(name: str) -> JsonObject:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_offline_validate_says_it_did_not_check_a_folders_case(workspace: Path) -> None:
    """The builder refuses "Loras" at the release, so a pass that could not tell it
    from a new folder says so, to a person and to an agent."""
    # Given
    write_spec(workspace, models=CASE_VARIANT, nodes=[])

    # When
    pretty = _invoke(workspace, output="pretty")
    agent = _invoke(workspace)

    # Then
    assert pretty.exit_code == 0, pretty.output
    assert _FOLDER_CASE_UNCHECKED in " ".join(pretty.output.split())
    assert agent.exit_code == 0, agent.output
    data = _envelope(agent)["data"]
    assert data["folder_case_checked"] is False
    jsonschema.Draft202012Validator(_schema("build_validate.json")).validate(data)


def test_remote_validate_says_it_checked_a_folders_case(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.builder_api.request_json", ResolveRecorder())
    write_spec(workspace, models=[{"type": "loras", "sourceUri": "https://h.example/a.safetensors"}], nodes=[])

    # When
    pretty = _invoke(workspace, "--remote", token="tok_test", output="pretty")
    agent = _invoke(workspace, "--remote", token="tok_test")

    # Then
    assert pretty.exit_code == 0, pretty.output
    assert _FOLDER_CASE_UNCHECKED not in " ".join(pretty.output.split())
    assert _envelope(agent)["data"]["folder_case_checked"] is True


def test_validate_without_models_says_nothing_of_a_folders_case(workspace: Path) -> None:
    # Given
    write_spec(workspace, models=[], nodes=[])

    # When
    pretty = _invoke(workspace, output="pretty")
    agent = _invoke(workspace)

    # Then
    assert _FOLDER_CASE_UNCHECKED not in " ".join(pretty.output.split())
    assert "folder_case_checked" not in _envelope(agent)["data"]


@pytest.mark.parametrize(
    "listed",
    [
        pytest.param({"directories": []}, id="empty-list"),
        pytest.param({}, id="no-list"),
        pytest.param({"directories": [1, 2]}, id="no-names"),
        pytest.param({"directories": [""]}, id="an-empty-name"),
        pytest.param({"directories": ["  "]}, id="a-blank-name"),
    ],
)
def test_remote_validate_treats_a_list_naming_no_folder_as_unread(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, listed: JsonObject
) -> None:
    """A 200 that names no folder checks no case, so "Loras" passes as it does when
    the read fails, and the command says so."""
    # Given
    recorder = ResolveRecorder()

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        if url == "https://builder.test/v1/model-directories":
            return 200, listed
        return recorder(url, target, method=method, body=body, timeout=timeout, max_bytes=max_bytes)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    write_spec(workspace, models=CASE_VARIANT, nodes=[])

    # When
    pretty = _invoke(workspace, "--remote", token="tok_test", output="pretty")
    agent = _invoke(workspace, "--remote", token="tok_test")

    # Then
    assert pretty.exit_code == 0, pretty.output
    said = " ".join(pretty.output.split())
    assert f"could not read the builder's model folders, so {_FOLDER_CASE_UNCHECKED}" in said
    assert said.count(_FOLDER_CASE_UNCHECKED) == 1, pretty.output
    assert agent.exit_code == 0, agent.output
    assert _envelope(agent)["data"]["folder_case_checked"] is False
