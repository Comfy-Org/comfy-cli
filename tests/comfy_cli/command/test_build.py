from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
import typer
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_push import SkippedSymlink
from comfy_cli.command.build_spec import read_build_spec


@pytest.fixture
def models_tree(tmp_path):
    """A models/ tree covering nesting, a symlinked-out dir, and skip cases."""
    root = tmp_path / "models"
    (root / "checkpoints").mkdir(parents=True)
    (root / "ultralytics" / "bbox").mkdir(parents=True)
    (root / "checkpoints" / "base.safetensors").write_bytes(b"CKPT")
    (root / "ultralytics" / "bbox" / "face.pt").write_bytes(b"FACE")
    # non-model extension and dotfile are both skipped
    (root / "checkpoints" / "notes.txt").write_bytes(b"nope")
    (root / "checkpoints" / ".hidden.pt").write_bytes(b"nope")
    # a file sitting directly in models/ has no placement folder -> skipped
    (root / "loose.safetensors").write_bytes(b"nope")
    # a model dir symlinked to external storage must still be followed
    ext = tmp_path / "external" / "vae_store"
    ext.mkdir(parents=True)
    (ext / "ae.safetensors").write_bytes(b"VAE")
    os.symlink(ext, root / "vae")
    return root


def test_scan_classifies_by_folder_and_hashes(models_tree):
    models = build.scan_models(models_tree)
    by_name = {m["filename"]: m for m in models}

    assert set(by_name) == {"base.safetensors", "face.pt", "ae.safetensors"}
    assert by_name["base.safetensors"]["type"] == "checkpoints"
    assert by_name["face.pt"]["type"] == "ultralytics/bbox"  # nested folder preserved
    assert by_name["ae.safetensors"]["type"] == "vae"  # symlinked-out dir followed

    assert by_name["base.safetensors"]["sha256"] == hashlib.sha256(b"CKPT").hexdigest()
    assert by_name["base.safetensors"]["sizeBytes"] == 4


def test_scan_models_records_scan_root_relative_posix_paths(models_tree):
    # Given / When
    models = {model["filename"]: model for model in build.scan_models(models_tree)}

    # Then
    assert models["base.safetensors"]["localPath"] == "checkpoints/base.safetensors"
    assert models["face.pt"]["localPath"] == "ultralytics/bbox/face.pt"
    assert models["ae.safetensors"]["localPath"] == "vae/ae.safetensors"
    for model in models.values():
        local_path = Path(model["localPath"])
        assert not local_path.is_absolute()
        assert ".." not in local_path.parts
        assert (models_tree / local_path).read_bytes()


def test_scan_skips_non_models_and_dotfiles_and_root_files(models_tree):
    names = {m["filename"] for m in build.scan_models(models_tree)}
    assert "notes.txt" not in names  # wrong extension
    assert ".hidden.pt" not in names  # dotfile
    assert "loose.safetensors" not in names  # no placement folder


def test_scan_is_sorted_and_stable(models_tree):
    once = build.scan_models(models_tree)
    twice = build.scan_models(models_tree)
    assert once == twice  # deterministic
    keys = [(m["type"], m["filename"]) for m in once]
    assert keys == sorted(keys)


def test_build_definition_shape():
    definition = build.build_definition(
        [{"type": "loras", "filename": "x.safetensors"}],
        [{"name": "comfyui_essentials", "source": "git"}],
    )
    assert definition["schema"] == build.DEFINITION_SCHEMA
    assert definition["models"][0]["type"] == "loras"
    assert definition["customNodes"][0]["name"] == "comfyui_essentials"
    assert "baseComfyVersion" not in definition  # omitted when not detected


def test_build_definition_includes_base_version_when_given():
    definition = build.build_definition([], [], base_comfy_version="0.3.40")
    assert definition["baseComfyVersion"] == "0.3.40"


def test_detect_comfy_version_from_marker(tmp_path):
    (tmp_path / "comfyui_version.py").write_text('# comment\n__version__ = "0.3.41"\n')
    assert build.detect_comfy_version(tmp_path) == "0.3.41"


def test_detect_comfy_version_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "ComfyUI"\nversion = "0.3.9"\n')
    assert build.detect_comfy_version(tmp_path) == "0.3.9"


def test_detect_comfy_version_none_when_absent(tmp_path):
    assert build.detect_comfy_version(tmp_path) is None


def test_detect_comfy_version_from_server(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"system": {"comfyui_version": "0.3.55"}}

    monkeypatch.setattr("comfy_cli.command.build.requests.get", lambda url, timeout: _Resp())
    assert build.detect_comfy_version_from_server("http://127.0.0.1:8188") == "0.3.55"


def test_detect_comfy_version_from_server_none_when_down(monkeypatch):
    def boom(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("comfy_cli.command.build.requests.get", boom)
    assert build.detect_comfy_version_from_server("http://127.0.0.1:8188") is None


def test_find_comfy_python_prefers_colocated_venv(tmp_path):
    bindir = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True)
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    py.write_text("")  # just needs to exist as a file
    assert build.find_comfy_python(tmp_path, None) == py


def test_find_comfy_python_finds_foreign_layout_venv(tmp_path):
    # Under WSL a Windows install's venv is Scripts/python.exe on a POSIX host, and
    # WSL runs it through interop — probing only the host's layout misses it entirely.
    foreign = ("bin", "python") if os.name == "nt" else ("Scripts", "python.exe")
    bindir = tmp_path / ".venv" / foreign[0]
    bindir.mkdir(parents=True)
    py = bindir / foreign[1]
    py.write_text("")
    assert build.find_comfy_python(tmp_path, None) == py


def test_find_comfy_python_prefers_host_layout_when_both_exist(tmp_path):
    native = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    foreign = ("bin", "python") if os.name == "nt" else ("Scripts", "python.exe")
    for sub, name in (native, foreign):
        (tmp_path / ".venv" / sub).mkdir(parents=True)
        (tmp_path / ".venv" / sub / name).write_text("")
    assert build.find_comfy_python(tmp_path, None) == tmp_path / ".venv" / native[0] / native[1]


def test_find_comfy_python_none_for_data_only_dir(tmp_path):
    # a data-only ComfyUI dir (no venv) must NOT silently fall back to some python
    assert build.find_comfy_python(tmp_path, None) is None


def test_find_comfy_python_explicit(tmp_path):
    exe = tmp_path / "py"
    exe.write_text("")
    assert build.find_comfy_python(None, str(exe)) == exe
    assert build.find_comfy_python(None, str(tmp_path / "nope")) is None


def test_capture_pip_provenance_freezes_and_labels():
    # freeze this test venv's own interpreter — a real freeze + platform probe
    prov = build.capture_pip_provenance(sys.executable)
    assert prov is not None
    assert set(prov["environment"]) >= {"os", "arch", "pythonVersion", "torch"}
    assert prov["environment"]["os"]  # non-empty platform.system()
    # the freeze carries a self-describing source-platform header for the resolver
    assert prov["pipDependencies"].startswith("# Captured by comfy-cli")
    assert "retarget" in prov["pipDependencies"]


def test_capture_pip_provenance_none_on_bad_python(tmp_path):
    assert build.capture_pip_provenance(str(tmp_path / "nonexistent-python")) is None


def test_build_definition_includes_pip_and_environment():
    d = build.build_definition(
        [], [], pip_dependencies="numpy==1.26.0\n", environment={"os": "Darwin", "arch": "arm64"}
    )
    assert d["pipDependencies"] == "numpy==1.26.0\n"
    assert d["environment"]["os"] == "Darwin"


def _git_init_node(path, *, remote=None):
    """Create a committed git repo at ``path`` (optional origin remote)."""
    path.mkdir(parents=True)
    (path / "__init__.py").write_text("# node")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def run(*a):
        subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, env=env)

    run("init", "-q")
    if remote:
        run("remote", "add", "origin", remote)
    run("add", "-A")
    run("commit", "-q", "-m", "init")


@pytest.fixture
def custom_nodes_tree(tmp_path):
    """A custom_nodes/ tree: a git node with a remote, a git node without one,
    a plain (non-git) dir, and directories that must be skipped."""
    root = tmp_path / "custom_nodes"
    root.mkdir()
    _git_init_node(root / "comfyui_essentials", remote="https://github.com/cubiq/ComfyUI_essentials")
    _git_init_node(root / "local_only_node")  # git, but no origin remote
    (root / "hand_dropped").mkdir()  # not a git repo at all
    (root / "hand_dropped" / "nodes.py").write_text("# node")
    (root / "__pycache__").mkdir()  # skipped
    (root / ".hidden").mkdir()  # skipped
    return root


def test_scan_custom_nodes_records_repo_and_ref(custom_nodes_tree):
    nodes = {n["name"]: n for n in build.scan_custom_nodes(custom_nodes_tree)}
    assert set(nodes) == {"comfyui_essentials", "local_only_node", "hand_dropped"}

    essentials = nodes["comfyui_essentials"]
    assert essentials["source"] == "git"
    assert essentials["repository"] == "https://github.com/cubiq/ComfyUI_essentials"
    assert len(essentials["gitRef"]) == 40  # full commit sha


def test_scan_custom_nodes_records_scan_root_relative_posix_paths(custom_nodes_tree):
    # Given / When
    nodes = {node["name"]: node for node in build.scan_custom_nodes(custom_nodes_tree)}

    # Then
    assert {name: node["localPath"] for name, node in nodes.items()} == {
        "comfyui_essentials": "comfyui_essentials",
        "hand_dropped": "hand_dropped",
        "local_only_node": "local_only_node",
    }
    for node in nodes.values():
        local_path = Path(node["localPath"])
        assert not local_path.is_absolute()
        assert ".." not in local_path.parts
        assert (custom_nodes_tree / local_path).is_dir()


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("git@github.com:org/repo.git", "https://github.com/org/repo"),
        ("ssh://git@github.com/org/repo/", "https://github.com/org/repo"),
        ("https://github.com/org/repo.git/", "https://github.com/org/repo"),
    ],
)
def test_scan_canonicalizes_valid_github_origins(tmp_path, monkeypatch, origin, expected):
    # Given
    node_dir = _write_pack(tmp_path, "pack", git=True)
    monkeypatch.setattr(
        build,
        "_git_output",
        lambda path, *args: origin if args[0] == "remote" else "cafebabe",
    )

    # When
    (node,) = build.scan_custom_nodes(node_dir.parent)

    # Then
    assert node["source"] == "git"
    assert node["repository"] == expected


@pytest.mark.parametrize(
    "origin",
    [
        "https://gitlab.com/o/r",
        "http://github.com/o/r",
        "https://github.com:8443/o/r",
        "https://user@github.com/o/r",
        "https://github.com/o/r/sub",
        "https://github.com/o",
        "https://github.com/o/r?x=1",
        "https://github.com/o/r#frag",
        "https://github.com/-bad/r",
        "https://github.com/o/../r",
        f"https://github.com/{'o' * 101}/r",
    ],
)
def test_scan_keeps_origins_outside_builder_contract_local(tmp_path, monkeypatch, origin):
    # Given
    node_dir = _write_pack(tmp_path, "pack", git=True)
    monkeypatch.setattr(
        build,
        "_git_output",
        lambda path, *args: origin if args[0] == "remote" else "cafebabe",
    )

    # When
    (node,) = build.scan_custom_nodes(node_dir.parent)

    # Then
    assert node["source"] == "local"
    assert not node.get("repository")


def test_scan_custom_nodes_marks_unfetchable_as_local(custom_nodes_tree):
    nodes = {n["name"]: n for n in build.scan_custom_nodes(custom_nodes_tree)}
    # git repo but no origin -> can't fetch repo@ref -> must be uploaded
    assert nodes["local_only_node"]["source"] == "local"
    assert nodes["local_only_node"]["repository"] is None
    # plain directory, not a git checkout
    assert nodes["hand_dropped"]["source"] == "local"
    assert nodes["hand_dropped"]["gitRef"] is None


def test_scan_custom_nodes_missing_dir_is_empty(tmp_path):
    assert build.scan_custom_nodes(tmp_path / "nope") == []


def test_init_command_json_envelope(models_tree, custom_nodes_tree, tmp_path):
    """End-to-end: the command emits an ok envelope carrying the definition.

    models_tree (tmp_path/models) and custom_nodes_tree (tmp_path/custom_nodes)
    are siblings, so the command auto-discovers the nodes from --models-dir."""
    out = tmp_path / "comfy-build.yaml"
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "comfy_cli",
            "build",
            "init",
            "--name",
            "Fixture",
            "--models-dir",
            str(models_tree),
            "--python",
            sys.executable,
            "--comfy-version",
            "master",
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    stdout_lines = proc.stdout.strip().splitlines()
    assert len(stdout_lines) == 1
    envelope = json.loads(stdout_lines[0])
    assert envelope["ok"] is True
    assert envelope["command"] == "build init"
    assert envelope["data"]["count"] == 3
    assert envelope["data"]["custom_node_count"] == 3  # auto-found the sibling custom_nodes/
    # the written file round-trips to the same definition
    written = read_build_spec(out)["definition"]
    assert isinstance(written, dict)
    assert written["models"] == envelope["data"]["definition"]["models"]
    assert written["customNodes"] == envelope["data"]["definition"]["customNodes"]


def test_init_command_missing_dir_errors(tmp_path):
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "comfy_cli",
            "build",
            "init",
            "--name",
            "Fixture",
            "--models-dir",
            str(tmp_path / "nope"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "build_models_dir_missing"


class _FakeResolver:
    def __init__(self, results):
        self._results = results
        self.batches = []

    def resolve_models(self, filenames):
        self.batches.append(list(filenames))
        return [self._results[f] for f in filenames if f in self._results]


def test_resolve_models_via_builder_only_trusts_hash_match():
    models = [
        {"type": "vae", "filename": "ae.safetensors", "sha256": "AAA"},  # exact hash match (case-insensitive)
        {"type": "unet", "filename": "mine.safetensors", "sha256": "BBB"},  # public candidate, different hash
        {"type": "clip", "filename": "nohash.safetensors", "sha256": "CCC"},  # candidate has no sha256
    ]
    resolver = _FakeResolver(
        {
            "ae.safetensors": {
                "filename": "ae.safetensors",
                "candidates": [{"sourceUri": "https://hf/ae", "sha256": "aaa"}],
            },
            "mine.safetensors": {
                "filename": "mine.safetensors",
                "candidates": [{"sourceUri": "https://hf/x", "sha256": "zzz"}],
            },
            "nohash.safetensors": {"filename": "nohash.safetensors", "candidates": [{"sourceUri": "https://hf/n"}]},
        }
    )
    n = build.resolve_models_via_builder(models, resolver)
    assert n == 1
    assert models[0]["sourceUri"] == "https://hf/ae"  # hash match → referenced by URL
    assert "sourceUri" not in models[1]  # hash mismatch (renamed local tune) → upload
    assert "sourceUri" not in models[2]  # provider gave no sha256 → safe default: upload


def test_resolve_models_via_builder_batches_over_32():
    models = [{"type": "x", "filename": f"m{i}.safetensors", "sha256": f"h{i}"} for i in range(70)]
    resolver = _FakeResolver({})  # nothing resolves; we only care about batching
    build.resolve_models_via_builder(models, resolver)
    assert [len(b) for b in resolver.batches] == [32, 32, 6]
    assert all(len(b) <= 32 for b in resolver.batches)


def test_resolve_models_via_builder_skips_a_model_with_no_filename():
    """`filename` is optional in the spec, so a model without one must be passed
    over rather than raising a KeyError the caller would blame on the builder."""
    models = [
        {"type": "unet", "sha256": "AAA"},
        {"type": "vae", "filename": "ae.safetensors", "sha256": "BBB"},
    ]
    resolver = _FakeResolver(
        {
            "ae.safetensors": {
                "filename": "ae.safetensors",
                "candidates": [{"sourceUri": "https://hf/ae", "sha256": "bbb"}],
            }
        }
    )
    assert build.resolve_models_via_builder(models, resolver) == 1
    assert resolver.batches == [["ae.safetensors"]]
    assert "sourceUri" not in models[0]
    assert models[1]["sourceUri"] == "https://hf/ae"


def test_delete_declined_at_the_prompt_aborts_without_touching_the_builder(monkeypatch):
    """A decline is only reachable in pretty mode — `--json` refuses upstream in
    `interaction.confirm` rather than prompting."""
    deleted = []

    class _Client:
        def delete_build(self, build_id):
            deleted.append(build_id)

    monkeypatch.setattr(build, "_builder_client", lambda renderer, url: _Client())
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", agentic=False, source_env=None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda _question: False)

    result = CliRunner().invoke(cli_app, ["--no-json", "build", "delete", "--id", "build-1"], env={"COLUMNS": "400"})

    assert result.exit_code == 0
    assert "Aborted." in result.stdout
    assert deleted == []


def test_json_delete_on_a_tty_refuses_instead_of_opening_a_prompt(monkeypatch):
    """`comfy --json build delete` used to draw questionary on stdout and hang
    forever; it must emit one refusal envelope and exit 1 instead."""
    deleted = []

    class _Client:
        def delete_build(self, build_id):
            deleted.append(build_id)

    monkeypatch.setattr(build, "_builder_client", lambda renderer, url: _Client())
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", agentic=False, source_env=None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda _q: pytest.fail("prompted a --json caller"))

    result = CliRunner().invoke(cli_app, ["--json", "build", "delete", "--id", "build-1"])

    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert result.exit_code == 1
    assert envelope["error"]["code"] == "build_delete_needs_confirm"
    assert envelope["error"]["details"]["buildId"] == "build-1"
    assert deleted == []


def test_json_delete_with_skip_prompt_runs_to_completion(monkeypatch):
    """`--skip-prompt` means "run to completion without asking". The hand-rolled
    gate this replaced never consulted the flag on the `--json` path — it saw a
    non-pretty renderer and refused — so this combination used to exit 1."""
    deleted: list[str] = []

    class _Client:
        def delete_build(self, build_id):
            deleted.append(build_id)

    monkeypatch.setattr(build, "_builder_client", lambda renderer, url: _Client())
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", agentic=False, source_env=None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: True)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda _question: pytest.fail("prompted"))

    result = CliRunner().invoke(cli_app, ["--json", "build", "delete", "--id", "build-1"])

    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert result.exit_code == 0
    assert envelope["changed"] is True
    assert deleted == ["build-1"]


class _WarnRecorder:
    """Minimal renderer stand-in that records the warnings emitted."""

    def __init__(self):
        self.messages = []

    def warn(self, message, *, hint=None):
        self.messages.append(message)


def _skipped(count):
    return [SkippedSymlink(f"definition.customNodes[{i}]", f"node-{i}", "vendor") for i in range(count)]


def test_the_skipped_symlink_warning_truncates_its_preview_but_not_its_rows():
    """The human line is capped so a pathological node cannot flood the terminal;
    the machine payload is never truncated."""
    renderer = _WarnRecorder()

    returned = build._warn_skipped_symlinks(renderer, _skipped(7))

    (message,) = renderer.messages
    assert "excluded 7 symlinks" in message
    assert message.count(": vendor") == 5
    assert message.endswith("and 2 more")
    assert len(returned) == 7


def test_a_single_skipped_symlink_reads_in_the_singular():
    renderer = _WarnRecorder()

    build._warn_skipped_symlinks(renderer, _skipped(1))

    (message,) = renderer.messages
    assert "excluded 1 symlink from" in message
    assert "more" not in message


def test_no_skipped_symlinks_says_nothing_at_all():
    renderer = _WarnRecorder()

    assert build._warn_skipped_symlinks(renderer, []) == []
    assert renderer.messages == []


def test_builder_client_endpoints_and_parsing(monkeypatch):
    calls = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        calls.append((method, url))
        if url.endswith("/v1/blobs"):
            return 200, {"blobId": "b1", "uploadUrl": "https://put", "expiresAt": "t"}
        if url.endswith("/v1/builds"):
            return 201, {"id": "d1", "name": "n"}
        if url.endswith("/releases"):
            return 202, {"releaseId": "v1", "statusUrl": "https://s"}
        if url.endswith("/v1/models/resolve"):
            return 200, {
                "results": [{"filename": "a.safetensors", "candidates": [{"sourceUri": "https://u", "sha256": "h"}]}]
            }
        return 200, {}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    from comfy_cli.builder_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt-token")
    assert c.create_blob("model", "f.safetensors", "hash", 5) == ("b1", "https://put")
    assert c.create_build("n", {"models": [], "customNodes": []}) == "d1"
    assert c.create_release("d1", [{"os": "linux", "gpu": "nvidia"}]) == ("v1", "https://s")
    results = c.resolve_models(["a.safetensors"])
    assert results[0]["candidates"][0]["sourceUri"] == "https://u"
    # URLs carry the /v1 prefix, and the JWT rides on the cloud target
    assert ("POST", "https://builder.test/v1/blobs") in calls
    assert ("POST", "https://builder.test/v1/models/resolve") in calls
    assert c.target.auth_token == "jwt-token" and c.target.is_cloud


def test_builder_client_read_endpoints(monkeypatch):
    calls = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        calls.append((method, url))
        # The paged reads carry a `?limit=`, so route on the path, not the URL.
        path = url.split("?", 1)[0]
        if path.endswith("/v1/builds"):
            return 200, {"builds": [{"id": "d1", "name": "n"}]}
        if path.endswith("/v1/builds/d1"):
            return 200, {"id": "d1", "name": "n", "definition": {"models": []}}
        if path.endswith("/v1/builds/d1/releases"):
            return 200, {"releases": [{"id": "v1", "status": "complete"}]}
        if "/v1/releases/v1/logs" in url:
            return 200, {
                "versionId": "v1",
                "releaseId": "v1",
                "os": "linux",
                "gpu": "nvidia",
                "log": "hello",
                "truncated": False,
            }
        return 200, {}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    from comfy_cli.builder_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt")
    assert c.list_builds() == [{"id": "d1", "name": "n"}]
    assert c.get_build("d1")["definition"] == {"models": []}
    assert c.list_releases("d1") == [{"id": "v1", "status": "complete"}]
    logs = c.get_release_logs("v1", os="linux", gpu="nvidia")
    assert logs["log"] == "hello" and logs["truncated"] is False
    # reads are GETs under /v1, and the log target selector rides as query params
    assert ("GET", "https://builder.test/v1/builds?limit=100") in calls
    assert any("/logs?" in u and "os=linux" in u and "gpu=nvidia" in u for _, u in calls)


_CURSOR_LISTINGS = [
    pytest.param("builds", lambda client: client.list_builds(), id="builds"),
    pytest.param("releases", lambda client: client.list_releases("d1"), id="releases"),
]


def _paging_client(monkeypatch, listing, next_cursor):
    """A builder whose every page answers with ``next_cursor``, counting requests."""
    from comfy_cli.builder_api import BuilderClient

    requested = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        requested.append(url)
        return 200, {listing: [{"id": f"row-{len(requested)}"}], "nextCursor": next_cursor(len(requested))}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    return BuilderClient("https://builder.test/", "jwt"), requested


@pytest.mark.parametrize(("listing", "call"), _CURSOR_LISTINGS)
def test_a_non_string_next_cursor_stops_the_walk(monkeypatch, listing, call):
    """`nextCursor: 5` is truthy and survives urlencode, so an untyped walk never ends."""
    # Given
    client, requested = _paging_client(monkeypatch, listing, lambda _page: 5)

    # When
    rows = call(client)

    # Then
    assert rows == [{"id": "row-1"}]
    assert len(requested) == 1


@pytest.mark.parametrize(("listing", "call"), _CURSOR_LISTINGS)
def test_a_repeated_cursor_is_refused_rather_than_paged_in_a_circle(monkeypatch, listing, call):
    # Given
    from comfy_cli.builder_pagination import BuilderPaginationError

    client, requested = _paging_client(monkeypatch, listing, lambda _page: "same")

    # When / Then
    with pytest.raises(BuilderPaginationError, match="repeated"):
        call(client)
    assert len(requested) == 2


@pytest.mark.parametrize(("listing", "call"), _CURSOR_LISTINGS)
def test_an_endless_cursor_chain_stops_at_the_page_cap(monkeypatch, listing, call):
    # Given
    from comfy_cli.builder_pagination import _MAX_LIST_PAGES, BuilderPaginationError

    client, requested = _paging_client(monkeypatch, listing, lambda page: f"cursor-{page}")

    # When / Then
    with pytest.raises(BuilderPaginationError, match="did not end"):
        call(client)
    assert len(requested) == _MAX_LIST_PAGES


def test_a_pagination_refusal_reaches_the_user_as_a_builder_error(monkeypatch):
    """Not a traceback: `_builder_call`'s ValueError clause would have relabeled
    this a caller input mistake, so the error type has to stay outside it."""
    # Given
    from comfy_cli.builder_pagination import BuilderPaginationError

    reported = {}

    class Recorder:
        def error(self, *, code, message, **extra):
            reported.update(code=code, message=message)

    def raise_pagination():
        raise BuilderPaginationError("builder repeated a builds page cursor")

    # When
    with pytest.raises(typer.Exit):
        build._builder_call(Recorder(), raise_pagination)

    # Then
    assert reported["code"] == "build_builder_error"
    assert "repeated a builds page cursor" in reported["message"]


def test_a_full_blob_page_is_reported_as_possibly_incomplete(monkeypatch):
    """The endpoint mints no cursor, so length is the only truncation signal."""
    # Given
    from comfy_cli.builder_pagination import _BLOB_PAGE_LIMIT, blob_listing_is_clamped

    # When / Then
    assert blob_listing_is_clamped([{"blobId": str(n)} for n in range(_BLOB_PAGE_LIMIT)])
    assert not blob_listing_is_clamped([{"blobId": str(n)} for n in range(_BLOB_PAGE_LIMIT - 1)])


def test_builder_client_delete_and_validate(monkeypatch):
    calls = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        calls.append((method, url))
        if url.endswith("/validate"):
            return 200, {"resolvable": True}
        return 204, None

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    from comfy_cli.builder_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt")
    c.delete_build("d1")
    assert ("DELETE", "https://builder.test/v1/builds/d1") in calls
    assert c.validate_build("d1") == {"resolvable": True}
    assert ("POST", "https://builder.test/v1/builds/d1/validate") in calls


def test_builder_client_reference_and_update_endpoints(monkeypatch):
    calls = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        calls.append((method, url, body))
        if url.endswith("/v1/base-images"):
            return 200, {"baseImages": [{"id": "cuda"}]}
        if url.endswith("/v1/build-targets"):
            return 200, {"targets": [{"os": "linux", "gpu": "nvidia"}]}
        if url.endswith("/v1/model-directories"):
            return 200, {"directories": ["checkpoints", "vae"]}
        if url.endswith("/v1/blobs"):
            return 200, {"blobs": [{"blobId": "b1", "filename": "m.safetensors"}]}
        if url.endswith("/v1/builds/d1"):
            return 200, {"id": "d1", "definition": {"models": []}}
        if url.endswith("/manifest"):
            return 200, {"models": [{"filename": "ae"}]}
        if url.endswith("/download"):
            return 200, {"downloadUrl": "https://dl", "expiresAt": "t"}
        return 200, {}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    from comfy_cli.builder_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt")
    assert c.list_base_images() == [{"id": "cuda"}]
    assert c.list_build_targets() == [{"os": "linux", "gpu": "nvidia"}]
    assert c.list_model_directories() == ["checkpoints", "vae"]
    assert c.list_blobs() == [{"blobId": "b1", "filename": "m.safetensors"}]
    assert c.update_build("d1", {"models": []}, "2026-08-01T00:00:00Z")["id"] == "d1"
    assert c.get_release_manifest("v1")["models"][0]["filename"] == "ae"
    assert c.get_artifact_download("a1")["downloadUrl"] == "https://dl"
    # update is a PATCH carrying the definition AND the expectedUpdatedAt guard the
    # builder requires (a missing one 409s STALE); the rest are GETs under /v1
    assert (
        "PATCH",
        "https://builder.test/v1/builds/d1",
        {"definition": {"models": []}, "expectedUpdatedAt": "2026-08-01T00:00:00Z"},
    ) in calls
    assert ("GET", "https://builder.test/v1/base-images", None) in calls
    assert ("GET", "https://builder.test/v1/build-targets", None) in calls
    assert ("GET", "https://builder.test/v1/model-directories", None) in calls
    assert ("GET", "https://builder.test/v1/blobs", None) in calls
    assert ("GET", "https://builder.test/v1/releases/v1/manifest", None) in calls
    assert ("GET", "https://builder.test/v1/build-artifacts/a1/download", None) in calls


def test_delete_command_needs_confirm_non_interactive():
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json", "COMFY_BUILDER_TOKEN": "test-token"}
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "build", "delete", "--id", "some-id"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["error"]["code"] == "build_delete_needs_confirm"


class _RecordingRenderer:
    """Minimal renderer stand-in that records the error code emitted."""

    def __init__(self):
        self.codes = []

    def error(self, code, message, details=None):
        self.codes.append(code)


def test_builder_client_uses_injected_token(monkeypatch):
    # The agent service / CI inject a JWT via COMFY_BUILDER_TOKEN; from_session (the
    # interactive OAuth path) must not be consulted.
    monkeypatch.setenv("COMFY_BUILDER_TOKEN", "injected-jwt")
    called = {"from_session": False}

    def fake_from_session(cls, base):
        called["from_session"] = True
        raise AssertionError("from_session should not be called when a token is injected")

    monkeypatch.setattr("comfy_cli.builder_api.BuilderClient.from_session", classmethod(fake_from_session))
    from comfy_cli.command.build import _builder_client

    client = _builder_client(_RecordingRenderer(), "https://builder.test/")
    assert client.target.auth_token == "injected-jwt"
    assert called["from_session"] is False


def test_builder_client_falls_back_to_session(monkeypatch):
    monkeypatch.delenv("COMFY_BUILDER_TOKEN", raising=False)
    sentinel = object()
    monkeypatch.setattr("comfy_cli.builder_api.BuilderClient.from_session", classmethod(lambda cls, base: sentinel))
    from comfy_cli.command.build import _builder_client

    assert _builder_client(_RecordingRenderer(), None) is sentinel


def test_builder_call_maps_the_limited_beta_403_to_build_not_enabled():
    import io
    import urllib.error

    from comfy_cli.command.build import _builder_call

    def raise_403():
        raise urllib.error.HTTPError(
            "https://x", 403, "Forbidden", None, io.BytesIO(b'{"error":"FEATURE_NOT_ENABLED"}')
        )

    r = _RecordingRenderer()
    with pytest.raises(typer.Exit):
        _builder_call(r, raise_403)
    assert r.codes == ["build_not_enabled"]


def test_builder_call_maps_other_errors_to_builder_error():
    import io
    import urllib.error

    from comfy_cli.command.build import _builder_call

    def raise_500():
        raise urllib.error.HTTPError("https://x", 500, "Server Error", None, io.BytesIO(b"boom"))

    r = _RecordingRenderer()
    with pytest.raises(typer.Exit):
        _builder_call(r, raise_500)
    assert r.codes == ["build_builder_error"]


def test_upload_blob_sends_generation_match_header(monkeypatch, tmp_path):
    captured = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_put(url, data=None, headers=None, timeout=None, allow_redirects=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["allow_redirects"] = allow_redirects
        return _Resp()

    monkeypatch.setattr("comfy_cli.builder_api.requests.put", fake_put)
    from comfy_cli.builder_api import BuilderClient

    f = tmp_path / "m.safetensors"
    f.write_bytes(b"x")
    BuilderClient("https://builder.test/", "jwt").upload_blob("https://storage.example/put?sig=1", f)
    # the builder signs the URL requiring this header; without it GCS 400s
    assert captured["headers"].get("x-goog-if-generation-match") == "0"
    # presigned PUTs must not follow redirects (a 3xx could divert the file stream)
    assert captured["allow_redirects"] is False


def test_get_release_logs_uses_large_cap(monkeypatch):
    seen = {}

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        seen["max_bytes"] = max_bytes
        return 200, {"versionId": "v1", "log": "x", "truncated": False}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    from comfy_cli.builder_api import BuilderClient

    BuilderClient("https://builder.test/", "jwt").get_release_logs("v1")
    # the builder caps a served log at 8 MiB; the client cap must sit above that
    assert seen["max_bytes"] > 8 * 1024 * 1024


def test_builder_call_catches_response_too_large():
    from comfy_cli.command.build import _builder_call
    from comfy_cli.http import ResponseTooLarge

    def raise_too_large():
        raise ResponseTooLarge("response exceeds cap")

    r = _RecordingRenderer()
    with pytest.raises(typer.Exit):
        _builder_call(r, raise_too_large)
    assert r.codes == ["build_builder_error"]


# --- create: the definition survives the trip ---------------------------------

# --- scan: a registry-installed pack has an upstream --------------------------


def _write_pack(root, name, *, project=None, git=False):
    # Under `custom_nodes/`, not tmp_path itself: autouse fixtures put their own
    # directories there, and scan reads every directory it is handed as a pack.
    d = root / "custom_nodes" / name
    d.mkdir(parents=True)
    if project is not None:
        d.joinpath("pyproject.toml").write_text(project, encoding="utf-8")
    if git:
        d.joinpath(".git").mkdir()
    return d


def test_scan_drops_the_git_keys_when_it_records_a_registry_pin(tmp_path, monkeypatch):
    """Exactly one source per node, which the builder enforces. A half-git checkout
    (an origin but no resolvable HEAD) would otherwise carry both, which `create`
    hides by picking one but `update --from` sends verbatim and the builder 400s."""
    _write_pack(tmp_path, "half", project='[project]\nname = "half"\nversion = "1.0.0"\n', git=True)
    monkeypatch.setattr(
        build, "_git_output", lambda path, *args: "https://github.com/x/half" if args[0] == "remote" else None
    )
    (node,) = build.scan_custom_nodes(tmp_path / "custom_nodes")
    assert node["source"] == "registry"
    assert node.get("repository") is None and node.get("gitRef") is None


def test_scan_reads_the_registry_pin_off_an_archive_install(tmp_path):
    """`comfy node install` unpacks archives, so a pack has no git history at all.
    Its pyproject still names the published version the builder can fetch."""
    _write_pack(tmp_path, "comfyui-kjnodes", project='[project]\nname = "comfyui-kjnodes"\nversion = "1.4.9"\n')
    (node,) = build.scan_custom_nodes(tmp_path / "custom_nodes")
    assert node["source"] == "registry"
    assert node["id"] == "comfyui-kjnodes"
    assert node["registryVersion"] == "1.4.9"


def test_scan_keeps_a_pack_local_when_nothing_names_an_upstream(tmp_path):
    """No git, no usable pyproject: it really must be uploaded."""
    _write_pack(tmp_path, "handmade")
    _write_pack(tmp_path, "half", project='[project]\nname = "half"\n')
    assert {n["name"]: n["source"] for n in build.scan_custom_nodes(tmp_path / "custom_nodes")} == {
        "handmade": "local",
        "half": "local",
    }


def test_scan_prefers_git_over_the_registry_pin(tmp_path, monkeypatch):
    """A commit pins bytes exactly; a package version is resolved later. When a pack
    has both, the more precise one wins."""
    _write_pack(tmp_path, "dual", project='[project]\nname = "dual"\nversion = "2.0.0"\n', git=True)
    monkeypatch.setattr(
        build,
        "_git_output",
        lambda path, *args: "https://github.com/x/dual" if args[0] == "remote" else "cafebabe",
    )
    (node,) = build.scan_custom_nodes(tmp_path / "custom_nodes")
    assert node["source"] == "git"
    assert "registryVersion" not in node


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("this is not toml [[[", "malformed"),
        ('[project]\nname = "ok"\nversion = 1.0\n', "version is a toml float, not a string"),
        ('[project]\nname = "ok"\nversion = 2024-01-01\n', "version is a toml date"),
        ('[project]\nname = ""\nversion = "1.0.0"\n', "empty id"),
        ('[project]\nname = "ok"\nversion = "1.0"\n', "not a package version; the builder rejects it"),
        ('[project]\nname = "ok"\nversion = "0.1.0b1"\n', "prerelease is not a package version"),
        ('[project]\nname = "../../publishers/x"\nversion = "1.0.0"\n', "escapes the registry path"),
        ('[project]\nname = "ok#frag"\nversion = "1.0.0"\n', "truncates server-side to a different id"),
        ('project = "hello"\n', "project is not a table"),
    ],
)
def test_read_registry_pin_is_silent_on_anything_it_cannot_trust(tmp_path, body, why):
    """A pack that cannot answer has no pin. It must never crash the scan, and never
    hand on a value the registry or the builder would reject."""
    d = _write_pack(tmp_path, "pack", project=body)
    assert build._read_registry_pin(d) is None, why


# --- scan: the ComfyUI pin has to be a ref that resolves ----------------------


@pytest.mark.parametrize(
    ("detected", "expected"),
    [
        ("0.30.2", "v0.30.2"),  # the packaged marker, the common case
        ("0.30", "v0.30"),
        ("v0.30.2", "v0.30.2"),  # already a tag: untouched
        ("master", "master"),  # a branch
        ("0.30.2-5-gdeadbee", "0.30.2-5-gdeadbee"),  # git describe output
        ("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
    ],
)
def test_as_comfy_git_ref(detected, expected):
    """Upstream tags releases `vX.Y.Z` while every version source reports the bare
    number, so only a bare release number is rewritten."""
    assert build.as_comfy_git_ref(detected) == expected


def test_init_command_writes_a_resolvable_comfy_ref(models_tree, tmp_path):
    """End-to-end: a bare `--comfy-version` reaches the definition as a tag. The
    builder resolves this field with git ls-remote, so the bare number it used to
    record could only ever be discovered by a failed build."""
    out = tmp_path / "comfy-build.yaml"
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "comfy_cli",
            "build",
            "init",
            "--name",
            "Fixture",
            "--models-dir",
            str(models_tree),
            "--comfy-version",
            "0.30.2",
            "--python",
            sys.executable,
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    definition = read_build_spec(out)["definition"]
    assert isinstance(definition, dict)
    assert definition["baseComfyVersion"] == "v0.30.2"


# --- create: the builder reads the pins, not us -------------------------------


def test_snapshot_from_definition_maps_each_source_to_its_snapshot_kind():
    """The importer reads a Desktop export; a scan holds the same facts under other
    names. Translating is what lets one implementation of registry truth serve both."""
    snap = build.snapshot_from_definition(
        {
            "baseComfyVersion": "v0.30.2",
            "environment": {"pythonVersion": "3.12.7"},
            "customNodes": [
                {"name": "was-node-suite-comfyui", "id": "pr-was-47064894", "registryVersion": "1.0.1"},
                {"name": "gitpack", "repository": "https://github.com/x/gitpack", "gitRef": "deadbeef"},
                {"name": "handmade"},
            ],
        }
    )
    assert snap["type"] == "comfyui-desktop-2-snapshot"
    (entry,) = snap["snapshots"]
    assert entry["comfyui"]["baseTag"] == "v0.30.2"
    assert entry["pythonVersion"] == "3.12.7"
    assert entry["customNodes"] == [
        {
            "type": "cnr",
            "id": "pr-was-47064894",
            "dirName": "was-node-suite-comfyui",
            "version": "1.0.1",
            "enabled": True,
        },
        {
            "type": "git",
            "id": "gitpack",
            "dirName": "gitpack",
            "url": "https://github.com/x/gitpack",
            "commit": "deadbeef",
            "enabled": True,
        },
    ]


def test_snapshot_from_definition_keeps_only_plain_pins():
    """A freeze is one pin per line; a snapshot is a map. A comment, a git direct
    reference and a bare name are not pins and have no key to occupy."""
    snap = build.snapshot_from_definition(
        {
            "pipDependencies": (
                "# captured on Darwin/arm64\n"
                "numpy==1.26.4\n"
                "cstr @ git+https://github.com/WASasquatch/cstr@0520c29\n"
                "torch\n"
                "\n"
                "timm==1.0.28\n"
            )
        }
    )
    assert snap["snapshots"][0]["pipPackages"] == {"numpy": "1.26.4", "timm": "1.0.28"}


def test_snapshot_from_definition_carries_no_models():
    """A snapshot describes an environment. Models are the caller's half and stay
    with the definition, which is why the importer's answer is merged, not adopted."""
    snap = build.snapshot_from_definition({"models": [{"type": "vae", "filename": "ae.safetensors"}]})
    assert "models" not in snap["snapshots"][0]


def test_report_advisories_names_every_thing_the_import_could_not_carry():
    lines = build.report_advisories(
        {"notInRegistry": ["was-node-suite-comfyui"], "unpinnablePins": ["torch"], "pythonSatisfied": True}
    )
    assert any("was-node-suite-comfyui" in line and "does not publish" in line for line in lines)
    assert any("torch" in line and "not a public PyPI release" in line for line in lines)
    assert len(lines) == 2  # pythonSatisfied True says nothing


def test_report_advisories_says_when_no_base_image_matches_the_python():
    """The scan ran on one Python and the build runs on another, which is how a
    freeze that resolved locally fails in the build."""
    (line,) = build.report_advisories({"pythonSatisfied": False})
    assert "closest one" in line


# --- the freeze describes the target env, and carries no credential -----------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "cstr @ git+https://x-access-token:ghp_SECRET@github.com/org/p.git@abc",
            "cstr @ git+https://***@github.com/org/p.git@abc",
        ),
        ("pkg @ https://user:pw@example.com/x.whl", "pkg @ https://***@example.com/x.whl"),
        # no userinfo: untouched, including the @ that pins a git ref
        (
            "ffmpy @ git+https://github.com/WASasquatch/ffmpy.git@f000",
            "ffmpy @ git+https://github.com/WASasquatch/ffmpy.git@f000",
        ),
        ("numpy==1.26.4", "numpy==1.26.4"),
    ],
)
def test_redact_freeze_credentials(line, expected):
    """A definition is written to disk, POSTed to the builder and often committed,
    so a token pip recorded in a direct reference travels far from where it was
    minted."""
    assert build._redact_freeze_credentials(line) == expected


def test_freeze_ignores_the_callers_pythonpath(tmp_path, monkeypatch):
    """PYTHONPATH puts whatever it points at into the freeze as an installed
    package, and the builder then tries to resolve a pin for a phantom."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["env"] = kwargs.get("env") or {}

        class R:
            returncode = 0
            stdout = "numpy==1.26.4\n"

        return R()

    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.setattr(build.subprocess, "run", fake_run)
    assert build._freeze_env("/x/bin/python") == "numpy==1.26.4\n"
    assert "PYTHONPATH" not in seen["env"]


# --- blobs: a private file is uploaded once and referenced by id ---------------


def test_report_advisories_reads_the_refused_release_as_one_value():
    """`droppedComfyVersion` is a string where its neighbours are lists. Iterated as
    a list it renders one entry per character."""
    (line,) = build.report_advisories({"droppedComfyVersion": "v9.9.9"})
    assert "'v9.9.9'" in line and "6 " not in line


def test_report_advisories_names_a_folder_collision():
    (line,) = build.report_advisories({"collidingNodes": ["ComfyUI-Easy-Use"]})
    assert "already claimed the folder" in line and "ComfyUI-Easy-Use" in line


def test_as_snapshot_envelope_wraps_the_file_desktop_actually_writes():
    """Desktop stores one bare snapshot per file under `.launcher/snapshots/` and
    only its export action wraps them, so the file a user has is refused."""
    bare = {"comfyui": {"baseTag": "v0.30.2"}, "customNodes": [], "pipPackages": {}, "pythonVersion": "3.12.7"}
    wrapped = build.as_snapshot_envelope(bare)
    assert wrapped["type"] == "comfyui-desktop-2-snapshot"
    assert wrapped["snapshots"] == [bare]


def test_as_snapshot_envelope_leaves_a_real_export_alone():
    export = {"type": "comfyui-desktop-2-snapshot", "version": 2, "snapshots": [{"customNodes": []}]}
    assert build.as_snapshot_envelope(export) is export


# --- the builder's import report, rendered ------------------------------------
#
# Ported with #779's workflow importer. The renderer is shared by
# --from-snapshot and --from-workflow, so a snapshot report must not move.

WORKFLOW_REPORT = {
    "comfyVersionRequired": True,
    "pinnedToLatest": True,
    "unresolvedClasses": ["ReActorFaceSwap", "TotallyMadeUpNodeXYZ", "WAS_Image_Blend"],
    "uncheckedClasses": ["ImpactSimpleDetectorSEGS"],
    "packsWithoutVersion": ["comfyui-kjnodes"],
    "collidingPacks": ["comfyui-manager"],
    "unknownClasses": [
        {"classType": "ReActorFaceSwap", "status": "missing"},
        {"classType": "TotallyMadeUpNodeXYZ", "status": "missing"},
        {
            "classType": "WAS_Image_Blend",
            "status": "suggested",
            "suggestions": [
                {
                    "nearestClass": "Image Blend",
                    "packId": "https://github.com/ltdrdata/was-node-suite-comfyui",
                    "score": 0.8888888888888888,
                },
                {"nearestClass": "ImageTile+", "packId": "comfyui_essentials", "score": 0.5555555555555556},
            ],
        },
    ],
    "models": [
        {
            "filename": "v1-5-pruned-emaonly-fp16.safetensors",
            "status": "matched",
            "directories": ["checkpoints"],
            "usedBy": ["CheckpointLoaderSimple"],
        },
        {"filename": "definitely-not-a-real-lora-v3.safetensors", "status": "missing", "usedBy": ["LoraLoader"]},
    ],
    "partnerClasses": {"LumaImageNode": "Luma", "OpenAIGPTImage1": "OpenAI (inc. Sora)"},
}


def test_report_advisories_renders_a_workflow_report_line_for_line():
    """A workflow report shares no key with a snapshot report, so the reader sees
    exactly these lines and no others."""
    assert build.report_advisories(WORKFLOW_REPORT) == [
        "the importer pinned every pack the workflow named without a version to the registry's newest published "
        "one, so importing the same file later can build something different",
        "3 node classes nothing installable provides; the graph will not run without them: ReActorFaceSwap, "
        "TotallyMadeUpNodeXYZ, WAS_Image_Blend",
        "1 node classes the registry never answered for, so the build may not carry them: ImpactSimpleDetectorSEGS",
        "1 packs the build fetches from their repository, because the registry publishes no version of them: "
        "comfyui-kjnodes",
        "1 packs the build leaves out, because another pack already claimed their install folder: comfyui-manager",
        "1 node classes the registry could not attribute, with the closest pack it named: WAS_Image_Blend "
        "(maybe https://github.com/ltdrdata/was-node-suite-comfyui)",
        "1 models the shared catalog holds that this build does not carry, each needing a sourceUri in the "
        "definition before you cut: v1-5-pruned-emaonly-fp16.safetensors",
        "1 models the graph loads that nothing has a source for; `comfy build refs resolve` finds candidates: "
        "definitely-not-a-real-lora-v3.safetensors",
        "2 node classes call a partner API rather than run from an installed pack: LumaImageNode (Luma), "
        "OpenAIGPTImage1 (OpenAI (inc. Sora))",
    ]


def test_report_advisories_still_owes_the_models_the_catalog_matched():
    """A workflow import builds custom nodes and no models, so a matched model is
    one the catalog holds and the build does not. Dropping the line lets a user
    cut a paid release whose graph dies at CheckpointLoaderSimple."""
    lines = build.report_advisories(
        {"models": [{"filename": "v1-5-pruned-emaonly-fp16.safetensors", "status": "matched"}]}
    )
    assert lines == [
        "1 models the shared catalog holds that this build does not carry, each needing a sourceUri in the "
        "definition before you cut: v1-5-pruned-emaonly-fp16.safetensors"
    ]


def test_report_advisories_names_the_catalog_lead_for_a_suggested_model():
    """The catalog ranks what it thinks the filename meant, and a name Cloud
    already has beats sending the reader to HuggingFace for it."""
    lines = build.report_advisories(
        {
            "models": [
                {
                    "filename": "sd15.safetensors",
                    "status": "suggested",
                    "suggestions": [
                        {"filename": "v1-5-pruned-emaonly.safetensors", "score": 0.62},
                        {"filename": "v1-5-pruned-emaonly-fp16.safetensors", "score": 0.91},
                    ],
                }
            ]
        }
    )
    assert lines == [
        "1 models the graph loads that nothing has a source for; `comfy build refs resolve` finds candidates: "
        "sd15.safetensors (maybe v1-5-pruned-emaonly-fp16.safetensors)"
    ]


def test_report_advisories_counts_every_name_and_says_how_many_it_held_back():
    classes = [f"WAS_Image_Filter_{i}" for i in range(30)]
    (line,) = build.report_advisories({"unresolvedClasses": classes})
    assert line.startswith("30 node classes nothing installable provides")
    assert line.endswith(f"{', '.join(classes[:8])} (+22 more)")


def test_report_advisories_says_when_a_key_arrives_in_a_shape_it_cannot_render():
    """Dropping a key the server did send is how a partial import comes to look
    clean, which is the failure this command exists to prevent."""
    lines = build.report_advisories({"partnerClasses": ["LumaImageNode"], "unresolvedClasses": "ReActorFaceSwap"})
    assert lines == [
        "the builder sent `unresolvedClasses` as str, which this CLI cannot render; read it with --json",
        "the builder sent `partnerClasses` as list, which this CLI cannot render; read it with --json",
    ]


def test_report_advisories_scrubs_the_newlines_a_crafted_workflow_could_carry():
    """Class names travel to the builder from the workflow file and come back in
    the report, so an attacker-authored file must not forge CLI lines."""
    (line,) = build.report_advisories(
        {"unresolvedClasses": ["KSampler\n\u2714 Created build dist-9\nAll classes resolved"]}
    )
    assert "\n" not in line
    assert "KSampler\\n" in line


def test_report_advisories_leaves_a_snapshot_report_unchanged():
    """`comfy build from-snapshot` shares the renderer, so its output must not move."""
    assert build.report_advisories({"notInRegistry": ["was-node-suite-comfyui"]}) == [
        "1 pinned to something the Comfy Registry does not publish: was-node-suite-comfyui"
    ]
