"""``comfy project`` integration: init / status via subprocess (the same
pattern as tests/comfy_cli/auth/test_auth_command.py)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"

MARKER = "schema: project/1\ndefaults:\n  where: cloud\n"
CONVENTIONAL = ("assets", "fragments", "blueprints", "outputs", ".comfy")


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


@pytest.fixture
def proj_dir(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    return d


def _run(args, cwd):
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "comfy_cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        check=False,
    )


def _last_json(stdout: str) -> dict:
    last = [line for line in stdout.splitlines() if line.strip()][-1]
    return json.loads(last)


# ---------------------------------------------------------------------------
# comfy project init
# ---------------------------------------------------------------------------


def test_init_creates_marker_and_conventional_dirs(proj_dir):
    # Pin the backend via the flag: auto-detect depends on whether the host
    # machine has cloud credentials, and tests must not.
    res = _run(["--json", "project", "init", "--where", "cloud"], proj_dir)
    assert res.returncode == 0, res.stderr
    env = _last_json(res.stdout)
    _validator_for("envelope.json").validate(env)
    _validator_for("project.json").validate(env["data"])
    assert env["ok"] is True
    assert env["changed"] is True
    assert env["data"]["action"] == "init"
    assert env["data"]["root"] == str(proj_dir.resolve())
    assert env["data"]["where_default"] == "cloud"

    marker = proj_dir / "comfy.yaml"
    assert marker.is_file()
    assert "schema: project/1" in marker.read_text()
    assert "where: cloud" in marker.read_text()
    for d in CONVENTIONAL:
        assert (proj_dir / d).is_dir(), d


def test_init_where_local_writes_local_default(proj_dir):
    res = _run(["--json", "project", "init", "--where", "local"], proj_dir)
    assert res.returncode == 0, res.stderr
    env = _last_json(res.stdout)
    assert env["data"]["where_default"] == "local"
    assert "where: local" in (proj_dir / "comfy.yaml").read_text()


def test_init_invalid_where_errors(proj_dir):
    res = _run(["--json", "project", "init", "--where", "marsbase"], proj_dir)
    assert res.returncode == 1
    env = _last_json(res.stdout)
    assert env["ok"] is False
    assert env["error"]["code"] == "invalid_argument"
    assert not (proj_dir / "comfy.yaml").exists()


def test_reinit_errors_with_project_already_exists(proj_dir):
    assert _run(["--json", "project", "init"], proj_dir).returncode == 0
    res = _run(["--json", "project", "init"], proj_dir)
    assert res.returncode == 1
    env = _last_json(res.stdout)
    assert env["ok"] is False
    assert env["error"]["code"] == "project_already_exists"
    assert str(proj_dir.resolve()) in env["error"]["message"]


def test_init_inside_parent_project_errors(proj_dir):
    # A nested dir under an existing project is already governed — init refuses.
    assert _run(["--json", "project", "init"], proj_dir).returncode == 0
    nested = proj_dir / "blueprints"
    res = _run(["--json", "project", "init"], nested)
    assert res.returncode == 1
    env = _last_json(res.stdout)
    assert env["error"]["code"] == "project_already_exists"
    assert str(proj_dir.resolve()) in env["error"]["message"]


# ---------------------------------------------------------------------------
# comfy project status
# ---------------------------------------------------------------------------


def test_status_outside_project_errors(proj_dir):
    res = _run(["--json", "project", "status"], proj_dir)
    assert res.returncode == 1
    env = _last_json(res.stdout)
    assert env["ok"] is False
    assert env["error"]["code"] == "project_not_found"
    assert "comfy project init" in env["error"]["hint"]


def test_status_inside_project_envelope_validates(proj_dir):
    # Pin --where so the marker's default doesn't fall through to credential
    # auto-detect, which would resolve to "cloud" on a machine with cloud
    # creds but "local" on a clean CI runner.
    assert _run(["--json", "project", "init", "--where", "cloud"], proj_dir).returncode == 0
    (proj_dir / "blueprints" / "story.yaml").write_text("pipeline: []\n")
    asset = proj_dir / "assets" / "keyframes" / "s1.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"fake-png-bytes")
    (proj_dir / "assets" / ".DS_Store").write_bytes(b"junk")  # dotfiles skipped
    (proj_dir / "scratch").mkdir()  # unconventional → warning

    res = _run(["--json", "project", "status"], proj_dir)
    assert res.returncode == 0, res.stderr
    env = _last_json(res.stdout)
    _validator_for("envelope.json").validate(env)
    _validator_for("project.json").validate(env["data"])

    data = env["data"]
    assert data["root"] == str(proj_dir.resolve())
    assert data["schema"] == "project/1"
    assert data["defaults"] == {"where": "cloud"}
    assert data["blueprints"] == ["story.yaml"]

    assert len(data["assets"]) == 1
    entry = data["assets"][0]
    assert entry["name"] == "keyframes/s1.png"
    assert entry["sha256"] == hashlib.sha256(b"fake-png-bytes").hexdigest()
    assert entry["size"] == len(b"fake-png-bytes")
    assert entry["pushed"] is False
    assert entry["stale"] is False

    assert data["recent_runs"] == []
    assert data["warnings"] == ["unknown top-level directory: scratch"]


def test_status_joins_assets_against_lock(proj_dir):
    assert _run(["--json", "project", "init"], proj_dir).returncode == 0
    fresh = proj_dir / "assets" / "fresh.png"
    fresh.write_bytes(b"fresh")
    stale = proj_dir / "assets" / "stale.png"
    stale.write_bytes(b"new-content")
    (proj_dir / "assets" / "unpushed.png").write_bytes(b"unpushed")

    lock = {
        "schema": "assets-lock/1",
        "assets": {
            "fresh.png": {"sha256": hashlib.sha256(b"fresh").hexdigest(), "cloud_name": "ab12.png"},
            "stale.png": {"sha256": hashlib.sha256(b"old-content").hexdigest(), "cloud_name": "cd34.png"},
        },
    }
    (proj_dir / ".comfy" / "assets.lock.json").write_text(json.dumps(lock))

    res = _run(["--json", "project", "status"], proj_dir)
    assert res.returncode == 0, res.stderr
    by_name = {a["name"]: a for a in _last_json(res.stdout)["data"]["assets"]}
    assert by_name["fresh.png"]["pushed"] is True
    assert by_name["fresh.png"]["stale"] is False
    assert by_name["stale.png"]["pushed"] is True
    assert by_name["stale.png"]["stale"] is True
    assert by_name["unpushed.png"]["pushed"] is False
    assert by_name["unpushed.png"]["stale"] is False


def test_status_surfaces_recent_runs_from_journal(proj_dir):
    assert _run(["--json", "project", "init"], proj_dir).returncode == 0
    runs = proj_dir / ".comfy" / "runs.jsonl"
    lines = [json.dumps({"ts": f"2026-06-10T00:00:{i:02d}+00:00", "cmd": "run", "seq": i}) for i in range(3)]
    runs.write_text("\n".join(lines) + "\n")

    res = _run(["--json", "project", "status"], proj_dir)
    data = _last_json(res.stdout)["data"]
    assert [r["seq"] for r in data["recent_runs"]] == [0, 1, 2]  # newest last


def test_status_pretty_mode_mentions_root(proj_dir):
    assert _run(["project", "init"], proj_dir).returncode == 0
    res = _run(["project", "status"], proj_dir)
    assert res.returncode == 0, res.stderr
    assert str(proj_dir.resolve()) in res.stdout


# ---------------------------------------------------------------------------
# ratchets
# ---------------------------------------------------------------------------


def test_project_commands_registered_in_discovery():
    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert COMMAND_SCHEMAS["comfy project init"] == "project"
    assert COMMAND_SCHEMAS["comfy project status"] == "project"


def test_project_error_codes_registered():
    from comfy_cli import error_codes

    assert error_codes.is_registered("project_already_exists")
    assert error_codes.is_registered("project_not_found")
