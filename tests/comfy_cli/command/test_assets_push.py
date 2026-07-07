"""``comfy assets push`` — sync assets/ to the run target over /upload/image.

CliRunner-based (not subprocess) so ``_upload_file`` can be monkeypatched:
no test ever performs a real upload, and the hard rule — the CLI never
touches a ComfyUI install's folders directly — is enforced by construction
(the command has no other ingestion path to fall back to).
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import project as project_cmd
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)
from comfy_cli.target import Target

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"

MARKER = "schema: project/1\ndefaults:\n  where: cloud\n"


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _invoke(args: list[str], capsys) -> tuple[int, dict]:
    """Invoke `comfy assets ...` and parse the trailing JSON envelope."""
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(project_cmd.assets_app, args, standalone_mode=False)
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    # In non-standalone mode click returns typer.Exit's code as the command's
    # return value (result.exit_code stays 0).
    exit_code = result.return_value if isinstance(result.return_value, int) else result.exit_code
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return exit_code, json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, out={captured[:500]!r})")


def _validator_for(name: str) -> jsonschema.Validator:
    schema = json.loads((SCHEMAS_DIR / name).read_text())
    store: dict[str, dict] = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        s = json.loads(path.read_text())
        if s.get("$id"):
            store[s["$id"]] = s
        store[path.name] = s
    base = SCHEMAS_DIR.absolute().as_uri() + "/"
    resolver = jsonschema.RefResolver(base_uri=base, referrer=schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def proj(tmp_path, monkeypatch) -> Path:
    """A minimal project/1 tree, with cwd inside it."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "comfy.yaml").write_text(MARKER)
    for d in ("assets", "fragments", "blueprints", "outputs", ".comfy"):
        (root / d).mkdir()
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def uploads(monkeypatch) -> list:
    """Monkeypatch _upload_file + resolve_target; record every upload call."""
    calls: list[tuple[Path, Target, bool]] = []

    def fake_upload(path, target, *, overwrite):
        calls.append((Path(path), target, overwrite))
        return {"name": f"srv-{Path(path).name}", "subfolder": "", "type": "input"}

    def fake_resolve_target(*, where=None, **kw):
        kind = where or "local"
        return Target(kind=kind, base_url="http://127.0.0.1:8188")

    monkeypatch.setattr(project_cmd, "_upload_file", fake_upload)
    monkeypatch.setattr(project_cmd, "resolve_target", fake_resolve_target)
    return calls


def _lock(root: Path) -> dict:
    return json.loads((root / ".comfy" / "assets.lock.json").read_text())


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


def test_push_uploads_and_writes_lock(proj, uploads, capsys):
    (proj / "assets" / "keyframes").mkdir()
    (proj / "assets" / "keyframes" / "s1.png").write_bytes(b"png-one")
    (proj / "assets" / "flat.png").write_bytes(b"png-two")

    rc, env = _invoke(["push", "--where", "local"], capsys)
    assert rc == 0
    _validator_for("envelope.json").validate(env)
    _validator_for("assets.json").validate(env["data"])
    assert env["ok"] is True
    assert env["changed"] is True

    data = env["data"]
    assert data["where"] == "local"
    assert data["skipped"] == 0
    assert data["current"] == []
    assert data["lock"] == str(proj / ".comfy" / "assets.lock.json")
    by_name = {p["name"]: p for p in data["pushed"]}
    assert set(by_name) == {"keyframes/s1.png", "flat.png"}
    assert by_name["flat.png"]["sha256"] == hashlib.sha256(b"png-two").hexdigest()
    assert by_name["flat.png"]["cloud_name"] == "srv-flat.png"
    assert by_name["flat.png"]["size"] == len(b"png-two")

    lock = _lock(proj)
    assert lock["schema"] == "assets-lock/1"
    entry = lock["assets"]["keyframes/s1.png"]
    assert entry["sha256"] == hashlib.sha256(b"png-one").hexdigest()
    assert entry["cloud_name"] == "srv-s1.png"
    assert entry["size"] == len(b"png-one")
    assert entry["where"] == "local"
    assert "T" in entry["pushed_at"]  # UTC ISO timestamp
    assert len(uploads) == 2


def test_push_skips_dotfiles_and_dot_dirs(proj, uploads, capsys):
    (proj / "assets" / ".DS_Store").write_bytes(b"junk")
    (proj / "assets" / ".hidden").mkdir()
    (proj / "assets" / ".hidden" / "x.png").write_bytes(b"hidden")
    (proj / "assets" / "real.png").write_bytes(b"real")

    rc, env = _invoke(["push", "--where", "local"], capsys)
    assert rc == 0
    assert [p["name"] for p in env["data"]["pushed"]] == ["real.png"]
    assert len(uploads) == 1


def test_second_push_skips_unchanged(proj, uploads, capsys):
    (proj / "assets" / "a.png").write_bytes(b"same")
    assert _invoke(["push", "--where", "local"], capsys)[0] == 0
    uploads.clear()

    rc, env = _invoke(["push", "--where", "local"], capsys)
    assert rc == 0
    assert env["changed"] is False
    assert env["data"]["pushed"] == []
    assert env["data"]["skipped"] == 1
    # `current` lists every asset verified up-to-date on this target, so a
    # script can assert "my files are on the server" from the push envelope
    # alone: pushed ∪ current ⊇ my files. (A truthful pushed:[] previously
    # read as failure to agents whose peer had already pushed.)
    assert env["data"]["current"] == ["a.png"]
    assert uploads == []


def test_force_repushes_unchanged(proj, uploads, capsys):
    (proj / "assets" / "a.png").write_bytes(b"same")
    assert _invoke(["push", "--where", "local"], capsys)[0] == 0
    uploads.clear()

    rc, env = _invoke(["push", "--where", "local", "--force"], capsys)
    assert rc == 0
    assert env["changed"] is True
    assert [p["name"] for p in env["data"]["pushed"]] == ["a.png"]
    assert len(uploads) == 1


def test_changed_sha_repushes(proj, uploads, capsys):
    (proj / "assets" / "a.png").write_bytes(b"v1")
    assert _invoke(["push", "--where", "local"], capsys)[0] == 0
    (proj / "assets" / "a.png").write_bytes(b"v2-different")
    uploads.clear()

    rc, env = _invoke(["push", "--where", "local"], capsys)
    assert rc == 0
    assert [p["name"] for p in env["data"]["pushed"]] == ["a.png"]
    assert _lock(proj)["assets"]["a.png"]["sha256"] == hashlib.sha256(b"v2-different").hexdigest()


def test_different_where_repushes(proj, uploads, capsys):
    """The lock records WHERE an asset was pushed — a lock entry for local
    does not satisfy a cloud push."""
    (proj / "assets" / "a.png").write_bytes(b"same")
    assert _invoke(["push", "--where", "local"], capsys)[0] == 0
    uploads.clear()

    rc, env = _invoke(["push", "--where", "cloud"], capsys)
    assert rc == 0
    assert [p["name"] for p in env["data"]["pushed"]] == ["a.png"]
    assert env["data"]["where"] == "cloud"
    assert _lock(proj)["assets"]["a.png"]["where"] == "cloud"


def test_push_outside_project_errors(tmp_path, monkeypatch, uploads, capsys):
    monkeypatch.chdir(tmp_path)
    rc, env = _invoke(["push", "--where", "local"], capsys)
    assert rc == 1
    assert env["ok"] is False
    assert env["error"]["code"] == "project_not_found"
    assert "comfy project init" in env["error"]["hint"]
    assert uploads == []


def test_push_upload_http_error_maps_to_upload_failed(proj, monkeypatch, capsys):
    (proj / "assets" / "a.png").write_bytes(b"x")

    def boom(path, target, *, overwrite):
        raise urllib.error.HTTPError("http://x/upload/image", 503, "down", {}, io.BytesIO())

    monkeypatch.setattr(project_cmd, "_upload_file", boom)
    monkeypatch.setattr(
        project_cmd,
        "resolve_target",
        lambda *, where=None, **kw: Target(kind="local", base_url="http://127.0.0.1:8188"),
    )
    rc, env = _invoke(["push", "--where", "local"], capsys)
    assert rc == 1
    assert env["error"]["code"] == "upload_failed"
    # A failed push must not record the file as pushed.
    assert not (proj / ".comfy" / "assets.lock.json").exists()


def test_push_status_join_agrees(proj, uploads, capsys):
    """`project status` must read the pushed lock as pushed=True, stale=False
    — and flip stale when the file changes afterwards."""
    (proj / "assets" / "a.png").write_bytes(b"v1")
    assert _invoke(["push", "--where", "local"], capsys)[0] == 0

    from comfy_cli.project import find_project

    entries = {e["name"]: e for e in project_cmd._asset_entries(find_project(proj))}
    assert entries["a.png"]["pushed"] is True
    assert entries["a.png"]["stale"] is False

    (proj / "assets" / "a.png").write_bytes(b"v2")
    entries = {e["name"]: e for e in project_cmd._asset_entries(find_project(proj))}
    assert entries["a.png"]["stale"] is True


def test_push_pretty_mode_lists_files(proj, uploads, monkeypatch, capsys):
    (proj / "assets" / "a.png").write_bytes(b"x")
    runner = CliRunner()
    result = runner.invoke(project_cmd.assets_app, ["push", "--where", "local"], standalone_mode=False)
    out = capsys.readouterr().out + (result.stdout or "")
    assert "a.png" in out
    assert "srv-a.png" in out


# ---------------------------------------------------------------------------
# ratchets
# ---------------------------------------------------------------------------


def test_assets_push_registered_in_discovery():
    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert COMMAND_SCHEMAS["comfy assets push"] == "assets"


def test_assets_schema_shipped():
    schema = json.loads((SCHEMAS_DIR / "assets.json").read_text())
    assert schema["$id"] == "https://comfy.org/schemas/assets.json"
