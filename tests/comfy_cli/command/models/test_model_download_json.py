"""`comfy --json model download` failure paths must emit `envelope/1` errors.

The invariant under test: **every** failure of `model download` exits non-zero
AND puts exactly one `envelope/1` object with `ok: false` on stdout. Two of these
paths used to exit 0 with no envelope (file-exists, HF-unauthorized), which the
local MCP's `plain_ok` synthesizer turned into a synthesized *success* for a
download that never happened.

The success path is deliberately unchanged (prints + exit 0, no envelope) — that
same synthesizer depends on exit-0-no-envelope meaning success.
"""

from __future__ import annotations

import io
import json
import sys
from typing import Any
from unittest.mock import Mock, patch

import pytest
import typer.testing

from comfy_cli import constants
from comfy_cli.command.models.models import app
from comfy_cli.file_utils import DownloadException
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)

runner = typer.testing.CliRunner()

# The renderer's machine stream (stdout in real runs) is pinned to this buffer so
# the "exactly one envelope on stdout, nothing else" assertion holds regardless of
# whether the installed Click version mixes stderr into CliRunner's captured
# output. Human log lines go to `_HUMAN` (stderr in real runs) and must never
# appear in `_MACHINE`.
_MACHINE = io.StringIO()
_HUMAN = io.StringIO()


@pytest.fixture(autouse=True)
def json_renderer():
    """Install a JSON-mode renderer with pinned streams, and tear it down.

    The renderer is a process-wide singleton, so the teardown matters: without it
    the pretty-mode expectations of the neighbouring `test_models.py` would break
    depending on test ordering.
    """
    global _MACHINE, _HUMAN
    reset_renderer_for_testing()
    _MACHINE = io.StringIO()
    _HUMAN = io.StringIO()
    r = Renderer()
    r.mode = OutputMode.JSON
    r.machine_stream = _MACHINE
    r.pretty_stream = _HUMAN
    set_renderer(r)
    yield r
    reset_renderer_for_testing()


def _stdout() -> str:
    return _MACHINE.getvalue()


def _envelope() -> dict[str, Any]:
    """Parse the single envelope the command is allowed to put on stdout."""
    output = _stdout()
    objects = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            pytest.fail(f"non-JSON line on stdout in --json mode: {line!r}\nfull stdout:\n{output}")
    assert len(objects) == 1, f"expected exactly one envelope on stdout, got {len(objects)}:\n{output}"
    return objects[0]


def _assert_error_envelope(result, code: str) -> dict[str, Any]:
    assert result.exit_code == 1, f"expected exit 1, got {result.exit_code}\n{result.output}"
    env = _envelope()
    assert env["schema"] == "envelope/1"
    assert env["ok"] is False
    assert env["error"]["code"] == code
    assert env["error"]["hint"], "every error must carry a navigation hint"
    return env


def _invoke(args: list[str], **patches):
    """Run `model download` with the network/prompt surface stubbed out.

    Defaults: not a CivitAI URL, not a Hugging Face URL, no config values. Each
    test overrides only the piece it exercises.

    Note: there is deliberately no ``patch("comfy_cli.tracking.track_command")``
    here. ``download`` is decorated at import time, so patching the factory
    afterwards never rebinds the wrapper — the mock would be pure decoration. The
    real wrapper runs and no-ops because telemetry consent is off under pytest.
    """
    cfg = patches.pop("config_manager", None)
    if cfg is None:
        cfg = Mock()
        cfg.get_or_override.return_value = None
        cfg.get.return_value = None

    # The two source probes are defaults rather than fixtures so a CivitAI/HF test
    # can override them through the same `**patches` door as everything else,
    # instead of rebuilding the whole stack by hand.
    merged = {
        "check_civitai_url": {"return_value": (False, False, None, None)},
        "check_huggingface_url": {"return_value": (False, None, None, None, None)},
        **patches,
    }

    with patch("comfy_cli.command.models.models.config_manager", cfg):
        stack = []
        try:
            for target, kwargs in merged.items():
                p = patch(f"comfy_cli.command.models.models.{target}", **kwargs)
                stack.append(p)
                p.start()
            return runner.invoke(app, ["download", *args])
        finally:
            for p in reversed(stack):
                p.stop()


# --------------------------------------------------------------------------- #
# transfer failure — the headline acceptance case
# --------------------------------------------------------------------------- #


def test_download_exception_emits_download_failed(tmp_path):
    result = _invoke(
        ["--url", "https://example.com/missing.safetensors", "--filename", "x.safetensors"],
        get_workspace={"return_value": tmp_path},
        download_file={"side_effect": DownloadException("404 Not Found")},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert "404 Not Found" in env["error"]["message"]
    assert env["error"]["details"]["url"] == "https://example.com/missing.safetensors"


def test_local_write_failure_emits_download_failed(tmp_path):
    """`download_file` converts *network* failures to DownloadException, but a local
    filesystem failure still surfaces as OSError — which used to end the command
    with a traceback and no envelope."""
    result = _invoke(
        ["--url", "https://example.com/m.bin", "--filename", "x.bin"],
        get_workspace={"return_value": tmp_path},
        download_file={"side_effect": PermissionError(13, "Permission denied")},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert "Permission denied" in env["error"]["message"]


def test_download_exception_message_with_markup_survives(tmp_path):
    """A server message containing rich-markup metacharacters must reach the
    envelope verbatim — the JSON path never runs it through Rich markup."""
    result = _invoke(
        ["--url", "https://example.com/m.bin", "--filename", "x.bin"],
        get_workspace={"return_value": tmp_path},
        download_file={"side_effect": DownloadException("server said [/] at /path/[id]/resource")},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert env["error"]["message"] == "server said [/] at /path/[id]/resource"


# --------------------------------------------------------------------------- #
# file already exists — used to exit 0 with no envelope
# --------------------------------------------------------------------------- #


def test_file_exists_emits_error_and_exits_nonzero(tmp_path):
    target = tmp_path / "models" / "checkpoints" / "x.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already here")

    result = _invoke(
        [
            "--url",
            "https://example.com/x.safetensors",
            "--relative-path",
            "models/checkpoints",
            "--filename",
            "x.safetensors",
        ],
        get_workspace={"return_value": tmp_path},
        download_file={},
    )

    env = _assert_error_envelope(result, "model_file_exists")
    assert env["error"]["details"]["path"] == str(target)


def test_file_exists_does_not_download(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"already here")

    with patch("comfy_cli.command.models.models.download_file") as mock_dl:
        result = _invoke(
            ["--url", "https://example.com/x.bin", "--relative-path", ".", "--filename", "x.bin"],
            get_workspace={"return_value": tmp_path},
        )

    assert result.exit_code == 1
    assert not mock_dl.called


# --------------------------------------------------------------------------- #
# Hugging Face unauthorized with no token — used to exit 0 with no envelope
# --------------------------------------------------------------------------- #


def test_hf_unauthorized_without_token_emits_error(tmp_path):
    # `_invoke`'s default config_manager already reports no CivitAI and no HF token.
    with patch("comfy_cli.command.models.models.download_file") as mock_dl:
        result = _invoke(
            [
                "--url",
                "https://huggingface.co/org/repo/resolve/main/m.safetensors",
                "--relative-path",
                "models/checkpoints",
                "--filename",
                "m.safetensors",
            ],
            check_huggingface_url={"return_value": (True, "org/repo", "m.safetensors", None, "main")},
            check_unauthorized={"return_value": True},
            get_workspace={"return_value": tmp_path},
        )

    env = _assert_error_envelope(result, "hf_unauthorized")
    assert constants.HF_API_TOKEN_ENV_KEY in env["error"]["hint"]
    assert env["error"]["details"]["repo_id"] == "org/repo"
    assert not mock_dl.called


# --------------------------------------------------------------------------- #
# no resolvable filename — used to be a bare `typer.Exit(1)` / a raw traceback
# --------------------------------------------------------------------------- #


def test_unprompted_empty_filename_emits_missing_argument(tmp_path):
    """`ui.prompt_input` returns its (empty) default when prompting is skipped —
    e.g. for an agentic caller. That used to raise an unhandled
    DownloadException("Filename cannot be empty") and print a traceback."""
    result = _invoke(
        ["--url", "https://example.com/"],
        get_workspace={"return_value": tmp_path},
        download_file={},
        ui={"new": Mock(**{"prompt_input.return_value": ""})},
    )

    _assert_error_envelope(result, "missing_argument")


def test_cancelled_filename_prompt_emits_missing_argument(tmp_path):
    """`questionary` returns None when the prompt is cancelled / gets EOF. That
    used to be a bare `typer.Exit(1)` with no message and no envelope."""
    result = _invoke(
        ["--url", "https://example.com/"],
        get_workspace={"return_value": tmp_path},
        download_file={},
        ui={"new": Mock(**{"prompt_input.return_value": None})},
    )

    _assert_error_envelope(result, "missing_argument")


# --------------------------------------------------------------------------- #
# CivitAI metadata resolution — used to escape as an unhandled traceback
# --------------------------------------------------------------------------- #


def test_civitai_lookup_failure_emits_download_failed(tmp_path):
    result = _invoke(
        ["--url", "https://civitai.com/models/4242", "--filename", "x.safetensors"],
        check_civitai_url={"return_value": (True, False, 4242, None)},
        request_civitai_model_api={"side_effect": RuntimeError("404 Client Error")},
        get_workspace={"return_value": tmp_path},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert env["error"]["details"]["stage"] == "resolve"


def test_civitai_version_without_primary_file_emits_download_failed(tmp_path):
    result = _invoke(
        ["--url", "https://civitai.com/api/download/models/777", "--filename", "x.safetensors"],
        check_civitai_url={"return_value": (False, True, None, 777)},
        request_civitai_model_version_api={"return_value": None},
        get_workspace={"return_value": tmp_path},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert env["error"]["details"]["stage"] == "resolve"


# --------------------------------------------------------------------------- #
# the success path is unchanged: no envelope, exit 0
# --------------------------------------------------------------------------- #


def test_direct_url_success_still_emits_no_envelope(tmp_path):
    """A plain (non-CivitAI, non-Hugging-Face) file URL is a SUPPORTED source —
    it must still download, and must still leave stdout empty on success so the
    MCP's exit-0-no-envelope success synthesizer keeps working."""
    with patch("comfy_cli.command.models.models.download_file") as mock_dl:
        result = _invoke(
            ["--url", "https://example.com/model.bin", "--relative-path", ".", "--filename", "model.bin"],
            get_workspace={"return_value": tmp_path},
        )

    assert result.exit_code == 0, result.output
    assert mock_dl.called, "a direct file URL must still be downloaded"
    assert _stdout().strip() == "", f"success path must keep stdout clean, got: {_stdout()!r}"
    assert "Done in" in _HUMAN.getvalue(), "the human-facing log still goes to stderr"


# --------------------------------------------------------------------------- #
# the URL is never echoed back with its credentials attached
# --------------------------------------------------------------------------- #


def test_error_details_url_is_scrubbed_of_token(tmp_path):
    """CivitAI download links carry the API token as `?token=`. `tracking._scrub_value`
    strips it before telemetry; an error envelope is a louder channel than telemetry
    (agent/MCP wrappers log stdout verbatim), so it must be stripped there too."""
    result = _invoke(
        ["--url", "https://example.com/m.bin?token=SUPERSECRET#frag", "--filename", "x.bin"],
        get_workspace={"return_value": tmp_path},
        download_file={"side_effect": DownloadException("boom")},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert env["error"]["details"]["url"] == "https://example.com/m.bin"
    assert "SUPERSECRET" not in json.dumps(env)
    assert "SUPERSECRET" not in _HUMAN.getvalue(), "the human log leaks the token too"


def test_userinfo_is_stripped_from_details_url(tmp_path):
    result = _invoke(
        ["--url", "https://user:pw@example.com/m.bin", "--filename", "x.bin"],
        get_workspace={"return_value": tmp_path},
        download_file={"side_effect": DownloadException("boom")},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert env["error"]["details"]["url"] == "https://example.com/m.bin"


# --------------------------------------------------------------------------- #
# a filename must not escape the workspace
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["../../.bashrc", "..", "sub/dir.bin", "C:evil.bin", "a\\b.bin"])
def test_traversing_filename_is_rejected_before_any_download(tmp_path, bad):
    """`--filename` — and, in a non-interactive run, the CivitAI-supplied `file["name"]`
    that becomes its default — is joined into the destination with no sanitisation by
    `pathlib`, so `..` or an absolute value writes outside the workspace."""
    with patch("comfy_cli.command.models.models.download_file") as mock_dl:
        result = _invoke(
            ["--url", "https://example.com/m.bin", "--relative-path", ".", "--filename", bad],
            get_workspace={"return_value": tmp_path},
        )

    env = _assert_error_envelope(result, "invalid_argument")
    assert "--relative-path" in env["error"]["hint"], "the hint must point at the supported way to pick a directory"
    assert not mock_dl.called


def test_plain_filename_is_still_accepted(tmp_path):
    """The traversal guard must not deny the ordinary case."""
    with patch("comfy_cli.command.models.models.download_file") as mock_dl:
        result = _invoke(
            ["--url", "https://example.com/m.bin", "--relative-path", "models/checkpoints", "--filename", "v1.5.bin"],
            get_workspace={"return_value": tmp_path},
        )

    assert result.exit_code == 0, result.output
    assert mock_dl.called


def test_traversing_civitai_basemodel_is_rejected(tmp_path):
    """`version["baseModel"]` is remote input joined straight into the destination path."""
    with patch("comfy_cli.command.models.models.download_file") as mock_dl:
        result = _invoke(
            ["--url", "https://civitai.com/api/download/models/777"],
            check_civitai_url={"return_value": (False, True, None, 777)},
            request_civitai_model_version_api={
                "return_value": ("m.safetensors", "https://civitai.com/x", "checkpoint", "../../../..")
            },
            get_workspace={"return_value": tmp_path},
        )

    _assert_error_envelope(result, "invalid_argument")
    assert not mock_dl.called


# --------------------------------------------------------------------------- #
# rich markup in dynamic values must not become the crash it was meant to catch
# --------------------------------------------------------------------------- #


def test_markup_hostile_url_does_not_crash_the_log_line(tmp_path):
    """`print` here is Rich's markup-parsing print. A URL holding `[/]` (or an IPv6
    literal) raised MarkupError out of the *logging*, ending the command with a
    traceback and no envelope — the exact failure this command promises not to have."""
    with patch("comfy_cli.command.models.models.download_file") as mock_dl:
        result = _invoke(
            ["--url", "http://[::1]/a/[/]/m.bin", "--relative-path", ".", "--filename", "m.bin"],
            get_workspace={"return_value": tmp_path},
        )

    assert result.exit_code == 0, result.output
    assert mock_dl.called


# --------------------------------------------------------------------------- #
# nothing escapes without an envelope, including what we didn't name
# --------------------------------------------------------------------------- #


def test_unexpected_downloader_exception_still_emits_envelope(tmp_path):
    """The downloader can raise something neither `DownloadException` nor `OSError` —
    a malformed `Content-Length` used to surface as ValueError out of `int()`. The
    `--json` contract is an envelope on *every* failure, so there's a backstop."""
    result = _invoke(
        ["--url", "https://example.com/m.bin", "--filename", "x.bin"],
        get_workspace={"return_value": tmp_path},
        download_file={"side_effect": ValueError("invalid literal for int()")},
    )

    env = _assert_error_envelope(result, "download_failed")
    assert "ValueError" in env["error"]["message"]


# --------------------------------------------------------------------------- #
# --filename is honoured on the gated Hugging Face path too
# --------------------------------------------------------------------------- #


def test_hf_download_honours_explicit_filename(tmp_path):
    """`hf_hub_download` names the file after the repo path, so `--filename` was
    silently ignored on the *authenticated* HF branch only. That made the
    `local_filepath.exists()` guard inspect a path this branch never writes, and made
    the `model_file_exists` hint ("pass `--filename`") impossible to act on."""
    hf_written = tmp_path / "models" / "checkpoints" / "repo-name.safetensors"
    hf_written.parent.mkdir(parents=True)
    hf_written.write_bytes(b"weights")

    fake_hub = Mock()
    fake_hub.hf_hub_download.return_value = str(hf_written)

    cfg = Mock()
    cfg.get_or_override.return_value = "hf_token"
    cfg.get.return_value = None

    with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
        result = _invoke(
            [
                "--url",
                "https://huggingface.co/org/repo/resolve/main/repo-name.safetensors",
                "--relative-path",
                "models/checkpoints",
                "--filename",
                "mine.safetensors",
            ],
            config_manager=cfg,
            check_huggingface_url={"return_value": (True, "org/repo", "repo-name.safetensors", None, "main")},
            check_unauthorized={"return_value": True},
            get_workspace={"return_value": tmp_path},
        )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "models" / "checkpoints" / "mine.safetensors").read_bytes() == b"weights"
    assert not hf_written.exists()


def test_hf_download_honours_prompted_filename(tmp_path):
    """The same split, via the prompt door: with no `--filename` the name comes from
    `ui.prompt_input`, whose default is only a *suggestion* — a user can type
    something else. Gating the move on `filename is not None` left that case landing
    at the repo path while the exists-guard and its hint pointed at the typed name."""
    hf_written = tmp_path / "models" / "checkpoints" / "repo-name.safetensors"
    hf_written.parent.mkdir(parents=True)
    hf_written.write_bytes(b"weights")

    fake_hub = Mock()
    fake_hub.hf_hub_download.return_value = str(hf_written)

    cfg = Mock()
    cfg.get_or_override.return_value = "hf_token"
    cfg.get.return_value = None

    with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
        result = _invoke(
            [
                "--url",
                "https://huggingface.co/org/repo/resolve/main/repo-name.safetensors",
                "--relative-path",
                "models/checkpoints",
            ],
            config_manager=cfg,
            check_huggingface_url={"return_value": (True, "org/repo", "repo-name.safetensors", None, "main")},
            check_unauthorized={"return_value": True},
            get_workspace={"return_value": tmp_path},
            ui={"new": Mock(**{"prompt_input.return_value": "typed.safetensors"})},
        )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "models" / "checkpoints" / "typed.safetensors").read_bytes() == b"weights"
    assert not hf_written.exists()
