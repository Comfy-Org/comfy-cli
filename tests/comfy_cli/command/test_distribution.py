"""``comfy distribution scan`` — unit tests for the pure scan/hash logic and a
subprocess check of the JSON envelope (same pattern as test_project_command)."""

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

from comfy_cli.command import distribution
from comfy_cli.output import get_renderer


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
    models = distribution.scan_models(models_tree)
    by_name = {m["filename"]: m for m in models}

    assert set(by_name) == {"base.safetensors", "face.pt", "ae.safetensors"}
    assert by_name["base.safetensors"]["type"] == "checkpoints"
    assert by_name["face.pt"]["type"] == "ultralytics/bbox"  # nested folder preserved
    assert by_name["ae.safetensors"]["type"] == "vae"  # symlinked-out dir followed

    assert by_name["base.safetensors"]["sha256"] == hashlib.sha256(b"CKPT").hexdigest()
    assert by_name["base.safetensors"]["sizeBytes"] == 4


def test_scan_skips_non_models_and_dotfiles_and_root_files(models_tree):
    names = {m["filename"] for m in distribution.scan_models(models_tree)}
    assert "notes.txt" not in names  # wrong extension
    assert ".hidden.pt" not in names  # dotfile
    assert "loose.safetensors" not in names  # no placement folder


def test_scan_is_sorted_and_stable(models_tree):
    once = distribution.scan_models(models_tree)
    twice = distribution.scan_models(models_tree)
    assert once == twice  # deterministic
    keys = [(m["type"], m["filename"]) for m in once]
    assert keys == sorted(keys)


def test_build_definition_shape():
    definition = distribution.build_definition(
        [{"type": "loras", "filename": "x.safetensors"}],
        [{"name": "comfyui_essentials", "source": "git"}],
    )
    assert definition["schema"] == distribution.DEFINITION_SCHEMA
    assert definition["models"][0]["type"] == "loras"
    assert definition["customNodes"][0]["name"] == "comfyui_essentials"
    assert "baseComfyVersion" not in definition  # omitted when not detected


def test_build_definition_includes_base_version_when_given():
    definition = distribution.build_definition([], [], base_comfy_version="0.3.40")
    assert definition["baseComfyVersion"] == "0.3.40"


def test_detect_comfy_version_from_marker(tmp_path):
    (tmp_path / "comfyui_version.py").write_text('# comment\n__version__ = "0.3.41"\n')
    assert distribution.detect_comfy_version(tmp_path) == "0.3.41"


def test_detect_comfy_version_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "ComfyUI"\nversion = "0.3.9"\n')
    assert distribution.detect_comfy_version(tmp_path) == "0.3.9"


def test_detect_comfy_version_none_when_absent(tmp_path):
    assert distribution.detect_comfy_version(tmp_path) is None


def test_detect_comfy_version_from_server(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"system": {"comfyui_version": "0.3.55"}}

    monkeypatch.setattr("comfy_cli.command.distribution.requests.get", lambda url, timeout: _Resp())
    assert distribution.detect_comfy_version_from_server("http://127.0.0.1:8188") == "0.3.55"


def test_detect_comfy_version_from_server_none_when_down(monkeypatch):
    import requests

    def boom(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("comfy_cli.command.distribution.requests.get", boom)
    assert distribution.detect_comfy_version_from_server("http://127.0.0.1:8188") is None


def test_find_comfy_python_prefers_colocated_venv(tmp_path):
    bindir = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True)
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    py.write_text("")  # just needs to exist as a file
    assert distribution.find_comfy_python(tmp_path, None) == py


def test_find_comfy_python_none_for_data_only_dir(tmp_path):
    # a data-only ComfyUI dir (no venv) must NOT silently fall back to some python
    assert distribution.find_comfy_python(tmp_path, None) is None


def test_find_comfy_python_explicit(tmp_path):
    exe = tmp_path / "py"
    exe.write_text("")
    assert distribution.find_comfy_python(None, str(exe)) == exe
    assert distribution.find_comfy_python(None, str(tmp_path / "nope")) is None


def test_capture_pip_provenance_freezes_and_labels():
    # freeze this test venv's own interpreter — a real freeze + platform probe
    prov = distribution.capture_pip_provenance(sys.executable)
    assert prov is not None
    assert set(prov["environment"]) >= {"os", "arch", "pythonVersion", "torch"}
    assert prov["environment"]["os"]  # non-empty platform.system()
    # the freeze carries a self-describing source-platform header for the resolver
    assert prov["pipDependencies"].startswith("# Captured by comfy-cli")
    assert "retarget" in prov["pipDependencies"]


def test_capture_pip_provenance_none_on_bad_python(tmp_path):
    assert distribution.capture_pip_provenance(str(tmp_path / "nonexistent-python")) is None


def test_build_definition_includes_pip_and_environment():
    d = distribution.build_definition(
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
    nodes = {n["name"]: n for n in distribution.scan_custom_nodes(custom_nodes_tree)}
    assert set(nodes) == {"comfyui_essentials", "local_only_node", "hand_dropped"}

    essentials = nodes["comfyui_essentials"]
    assert essentials["source"] == "git"
    assert essentials["repository"] == "https://github.com/cubiq/ComfyUI_essentials"
    assert len(essentials["gitRef"]) == 40  # full commit sha


def test_scan_custom_nodes_marks_unfetchable_as_local(custom_nodes_tree):
    nodes = {n["name"]: n for n in distribution.scan_custom_nodes(custom_nodes_tree)}
    # git repo but no origin -> can't fetch repo@ref -> must be uploaded
    assert nodes["local_only_node"]["source"] == "local"
    assert nodes["local_only_node"]["repository"] is None
    # plain directory, not a git checkout
    assert nodes["hand_dropped"]["source"] == "local"
    assert nodes["hand_dropped"]["gitRef"] is None


def test_scan_custom_nodes_missing_dir_is_empty(tmp_path):
    assert distribution.scan_custom_nodes(tmp_path / "nope") == []


def test_scan_command_json_envelope(models_tree, custom_nodes_tree, tmp_path):
    """End-to-end: the command emits an ok envelope carrying the definition.

    models_tree (tmp_path/models) and custom_nodes_tree (tmp_path/custom_nodes)
    are siblings, so the command auto-discovers the nodes from --models-dir."""
    out = tmp_path / "definition.json"
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "distribution", "scan", "--models-dir", str(models_tree), "-o", str(out)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is True
    assert envelope["command"] == "distribution scan"
    assert envelope["data"]["count"] == 3
    assert envelope["data"]["custom_node_count"] == 3  # auto-found the sibling custom_nodes/
    # the written file round-trips to the same definition
    written = json.loads(out.read_text())
    assert written["models"] == envelope["data"]["definition"]["models"]
    assert written["customNodes"] == envelope["data"]["definition"]["customNodes"]


def test_scan_command_missing_dir_errors(tmp_path):
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "distribution", "scan", "--models-dir", str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "distribution_models_dir_missing"


# --- create: plan_create mapping + command ------------------------------------

SCAN_DEF = {
    "schema": "distribution-definition/0",
    "baseComfyVersion": "master",
    "models": [
        {"type": "unet", "filename": "z.safetensors", "sha256": "abc", "sizeBytes": 100, "source": "local"},
        {"type": "vae", "filename": "ae.safetensors", "sha256": "def", "sizeBytes": 50, "source": "local"},
    ],
    "customNodes": [
        {"name": "essentials", "repository": "https://github.com/x/essentials", "gitRef": "deadbeef", "source": "git"},
        {"name": "handmade", "repository": None, "gitRef": None, "source": "local"},
    ],
}


def test_plan_create_maps_to_builder_schema_all_upload():
    """With the default (no-op) resolver every model is an upload; git nodes are
    referenced by repo@ref, local nodes are uploads."""
    plan = distribution.plan_create(SCAN_DEF)
    d = plan["definition"]

    # models: builder shape, no scan-only fields, no source yet (pending upload)
    assert d["models"][0] == {"type": "unet", "filename": "z.safetensors", "sha256": "abc"}
    assert "sizeBytes" not in d["models"][0] and "source" not in d["models"][0]

    # git node -> repository+gitRef; local node -> name only (upload)
    ess = next(n for n in d["customNodes"] if n["name"] == "essentials")
    assert ess == {"name": "essentials", "repository": "https://github.com/x/essentials", "gitRef": "deadbeef"}
    hand = next(n for n in d["customNodes"] if n["name"] == "handmade")
    assert hand == {"name": "handmade"}

    # upload plan: 2 models + 1 local node = 3 items, 150 model bytes
    assert plan["upload_count"] == 3
    assert plan["upload_bytes"] == 150
    kinds = sorted(u["kind"] for u in plan["uploads"])
    assert kinds == ["model", "model", "node_zip"]


def test_plan_create_carries_environment_fields():
    """pipDependencies passes straight through; baseComfyVersion is normalized to a
    ref the builder can resolve, here rather than only in `scan`, so a definition
    written before that fix does not spend its cut discovering the difference."""
    definition = {**SCAN_DEF, "baseComfyVersion": "0.3.40", "pipDependencies": "numpy==1.26.0\n"}
    d = distribution.plan_create(definition)["definition"]
    assert d["baseComfyVersion"] == "v0.3.40"
    assert d["pipDependencies"] == "numpy==1.26.0\n"


def test_plan_create_resolver_turns_model_into_sourceuri():
    """A resolver that returns a URL makes the model a public sourceUri (no upload)."""

    def resolver(m):
        return "https://hf.co/z" if m["filename"] == "z.safetensors" else None

    plan = distribution.plan_create(SCAN_DEF, resolver=resolver)
    z = next(m for m in plan["definition"]["models"] if m["filename"] == "z.safetensors")
    assert z["sourceUri"] == "https://hf.co/z"
    assert "blobId" not in z
    # only the unresolved vae model (+ local node) remain as uploads
    assert plan["upload_count"] == 2
    assert {u.get("filename") for u in plan["uploads"] if u["kind"] == "model"} == {"ae.safetensors"}


def test_create_command_preview(tmp_path):
    def_path = tmp_path / "def.json"
    def_path.write_text(json.dumps(SCAN_DEF))
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "distribution", "create", "--from", str(def_path), "--name", "demo"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is True
    assert envelope["data"]["executed"] is False
    assert envelope["data"]["name"] == "demo"
    assert envelope["data"]["plan"]["upload_count"] == 3


def test_create_command_execute_requires_login(tmp_path):
    """--execute with no OAuth session (isolated secrets) → clean not-signed-in error."""
    def_path = tmp_path / "def.json"
    def_path.write_text(json.dumps(SCAN_DEF))
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "COMFY_OUTPUT": "json",
        "COMFY_SECRETS_PATH": str(tmp_path / "secrets.json"),  # empty → no session
        "VIRTUAL_ENV": "",
        "CONDA_PREFIX": "",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "distribution", "create", "--from", str(def_path), "--execute"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["error"]["code"] == "distribution_not_signed_in"


class _FakeBuilder:
    """Stand-in BuilderClient recording calls, for execute_create orchestration tests."""

    def __init__(self):
        self.uploaded = []
        self.blobs_created = []
        self.created = None
        self.cut = None

    def create_blob(self, kind, filename, sha256, size_bytes):
        self.blobs_created.append(filename)
        return f"blob-{filename}", f"https://put.example/{filename}"

    def upload_blob(self, upload_url, path):
        self.uploaded.append((upload_url, str(path)))

    def create_distribution(self, name, definition, description=None):
        self.created = (name, definition)
        return "dist-123"

    def cut_version(self, distribution_id, targets=None):
        self.cut = (distribution_id, targets)
        return "ver-456", "https://status.example/ver-456"


def test_execute_create_uploads_stitches_and_cuts():
    scan_def = {
        "models": [{"type": "vae", "filename": "ae.safetensors", "sha256": "h", "sizeBytes": 5, "source": "local"}],
        "customNodes": [{"name": "ess", "repository": "https://github.com/x/ess", "gitRef": "abc", "source": "git"}],
    }
    plan = distribution.plan_create(scan_def)
    fake = _FakeBuilder()
    result = distribution.execute_create(
        plan, client=fake, name="demo", locate_bytes=lambda u: Path("/tmp/ae.safetensors")
    )

    assert result == {
        "distributionId": "dist-123",
        "versionId": "ver-456",
        "statusUrl": "https://status.example/ver-456",
        "uploaded": 1,
    }
    # the uploaded model's blobId is stitched into the definition sent to create
    assert plan["definition"]["models"][0]["blobId"] == "blob-ae.safetensors"
    # the git node was referenced by repo@ref, not uploaded
    assert fake.created[1]["customNodes"][0]["repository"] == "https://github.com/x/ess"
    assert fake.cut[0] == "dist-123"


def test_execute_create_rejects_local_node_upload():
    scan_def = {"models": [], "customNodes": [{"name": "hand", "source": "local"}]}
    plan = distribution.plan_create(scan_def)
    with pytest.raises(NotImplementedError):
        distribution.execute_create(plan, client=_FakeBuilder(), name="d", locate_bytes=lambda u: Path("/x"))


def test_execute_create_preflights_uploads_before_moving_bytes():
    # One model upload, then a local (non-git) node that makes the whole create
    # unsupported. The failure must be detected before any model bytes upload,
    # otherwise a real install uploads gigabytes only to raise and create nothing.
    scan_def = {
        "models": [{"type": "vae", "filename": "ae.safetensors", "sha256": "h", "sizeBytes": 5, "source": "local"}],
        "customNodes": [{"name": "handmade", "repository": None, "gitRef": None, "source": "local"}],
    }
    plan = distribution.plan_create(scan_def)
    fake = _FakeBuilder()
    with pytest.raises(NotImplementedError):
        distribution.execute_create(plan, client=fake, name="demo", locate_bytes=lambda u: Path("/tmp/x"))
    assert fake.blobs_created == [], "no model should upload when a later node makes the create unsupported"
    assert fake.created is None


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
    n = distribution.resolve_models_via_builder(models, resolver)
    assert n == 1
    assert models[0]["sourceUri"] == "https://hf/ae"  # hash match → referenced by URL
    assert "sourceUri" not in models[1]  # hash mismatch (renamed local tune) → upload
    assert "sourceUri" not in models[2]  # provider gave no sha256 → safe default: upload


def test_resolve_models_via_builder_batches_over_32():
    models = [{"type": "x", "filename": f"m{i}.safetensors", "sha256": f"h{i}"} for i in range(70)]
    resolver = _FakeResolver({})  # nothing resolves; we only care about batching
    distribution.resolve_models_via_builder(models, resolver)
    assert [len(b) for b in resolver.batches] == [32, 32, 6]
    assert all(len(b) <= 32 for b in resolver.batches)


def test_resolve_model_source_reads_annotation():
    assert distribution.resolve_model_source({"filename": "x", "sourceUri": "https://u"}) == "https://u"
    assert distribution.resolve_model_source({"filename": "x"}) is None


def test_plan_create_uses_resolved_sourceuri():
    # a model the resolve step annotated with sourceUri is referenced by URL, not
    # uploaded, under the DEFAULT resolver.
    defn = {
        "models": [
            {
                "type": "vae",
                "filename": "ae.safetensors",
                "sha256": "h",
                "sizeBytes": 5,
                "source": "local",
                "sourceUri": "https://hf/ae",
            }
        ],
        "customNodes": [],
    }
    plan = distribution.plan_create(defn)
    assert plan["definition"]["models"][0]["sourceUri"] == "https://hf/ae"
    assert plan["upload_count"] == 0


def test_builder_client_endpoints_and_parsing(monkeypatch):
    calls = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        calls.append((method, url))
        if url.endswith("/v1/blobs"):
            return 200, {"blobId": "b1", "uploadUrl": "https://put", "expiresAt": "t"}
        if url.endswith("/v1/distributions"):
            return 201, {"id": "d1", "name": "n"}
        if url.endswith("/versions"):
            return 202, {"distributionVersionId": "v1", "statusUrl": "https://s"}
        if url.endswith("/v1/models/resolve"):
            return 200, {
                "results": [{"filename": "a.safetensors", "candidates": [{"sourceUri": "https://u", "sha256": "h"}]}]
            }
        return 200, {}

    monkeypatch.setattr("comfy_cli.distribution_api.request_json", fake_request_json)
    from comfy_cli.distribution_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt-token")
    assert c.create_blob("model", "f.safetensors", "hash", 5) == ("b1", "https://put")
    assert c.create_distribution("n", {"models": [], "customNodes": []}) == "d1"
    assert c.cut_version("d1") == ("v1", "https://s")
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
        if url.endswith("/v1/distributions"):
            return 200, {"distributions": [{"id": "d1", "name": "n"}]}
        if url.endswith("/v1/distributions/d1"):
            return 200, {"id": "d1", "name": "n", "definition": {"models": []}}
        if url.endswith("/v1/distributions/d1/versions"):
            return 200, {"versions": [{"id": "v1", "status": "complete"}]}
        if "/v1/distribution-versions/v1/logs" in url:
            return 200, {"versionId": "v1", "os": "linux", "gpu": "nvidia", "log": "hello", "truncated": False}
        return 200, {}

    monkeypatch.setattr("comfy_cli.distribution_api.request_json", fake_request_json)
    from comfy_cli.distribution_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt")
    assert c.list_distributions() == [{"id": "d1", "name": "n"}]
    assert c.get_distribution("d1")["definition"] == {"models": []}
    assert c.list_distribution_versions("d1") == [{"id": "v1", "status": "complete"}]
    logs = c.get_version_logs("v1", os="linux", gpu="nvidia")
    assert logs["log"] == "hello" and logs["truncated"] is False
    # reads are GETs under /v1, and the log target selector rides as query params
    assert ("GET", "https://builder.test/v1/distributions") in calls
    assert any("/logs?" in u and "os=linux" in u and "gpu=nvidia" in u for _, u in calls)


def test_builder_client_delete_and_validate(monkeypatch):
    calls = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        calls.append((method, url))
        if url.endswith("/validate"):
            return 200, {"resolvable": True}
        return 204, None

    monkeypatch.setattr("comfy_cli.distribution_api.request_json", fake_request_json)
    from comfy_cli.distribution_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt")
    c.delete_distribution("d1")
    assert ("DELETE", "https://builder.test/v1/distributions/d1") in calls
    assert c.validate_distribution("d1") == {"resolvable": True}
    assert ("POST", "https://builder.test/v1/distributions/d1/validate") in calls


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
        if url.endswith("/v1/distributions/d1"):
            return 200, {"id": "d1", "definition": {"models": []}}
        if url.endswith("/manifest"):
            return 200, {"models": [{"filename": "ae"}]}
        if url.endswith("/download"):
            return 200, {"downloadUrl": "https://dl", "expiresAt": "t"}
        return 200, {}

    monkeypatch.setattr("comfy_cli.distribution_api.request_json", fake_request_json)
    from comfy_cli.distribution_api import BuilderClient

    c = BuilderClient("https://builder.test/", "jwt")
    assert c.list_base_images() == [{"id": "cuda"}]
    assert c.list_build_targets() == [{"os": "linux", "gpu": "nvidia"}]
    assert c.list_model_directories() == ["checkpoints", "vae"]
    assert c.list_blobs() == [{"blobId": "b1", "filename": "m.safetensors"}]
    assert c.update_distribution("d1", {"models": []}, "2026-08-01T00:00:00Z")["id"] == "d1"
    assert c.get_version_manifest("v1")["models"][0]["filename"] == "ae"
    assert c.get_artifact_download("a1")["downloadUrl"] == "https://dl"
    # update is a PATCH carrying the definition AND the expectedUpdatedAt guard the
    # builder requires (a missing one 409s STALE); the rest are GETs under /v1
    assert (
        "PATCH",
        "https://builder.test/v1/distributions/d1",
        {"definition": {"models": []}, "expectedUpdatedAt": "2026-08-01T00:00:00Z"},
    ) in calls
    assert ("GET", "https://builder.test/v1/base-images", None) in calls
    assert ("GET", "https://builder.test/v1/build-targets", None) in calls
    assert ("GET", "https://builder.test/v1/model-directories", None) in calls
    assert ("GET", "https://builder.test/v1/blobs", None) in calls
    assert ("GET", "https://builder.test/v1/distribution-versions/v1/manifest", None) in calls
    assert ("GET", "https://builder.test/v1/build-artifacts/a1/download", None) in calls


def test_delete_command_needs_confirm_non_interactive():
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "distribution", "delete", "some-id"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["error"]["code"] == "distribution_delete_needs_confirm"


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

    monkeypatch.setattr("comfy_cli.distribution_api.BuilderClient.from_session", classmethod(fake_from_session))
    from comfy_cli.command.distribution import _builder_client

    client = _builder_client(_RecordingRenderer(), "https://builder.test/")
    assert client.target.auth_token == "injected-jwt"
    assert called["from_session"] is False


def test_builder_client_falls_back_to_session(monkeypatch):
    monkeypatch.delenv("COMFY_BUILDER_TOKEN", raising=False)
    sentinel = object()
    monkeypatch.setattr(
        "comfy_cli.distribution_api.BuilderClient.from_session", classmethod(lambda cls, base: sentinel)
    )
    from comfy_cli.command.distribution import _builder_client

    assert _builder_client(_RecordingRenderer(), None) is sentinel


def test_update_command_reads_updated_at_before_patch(monkeypatch, tmp_path):
    # Regression guard for "update always 409s": the command must GET the current
    # updatedAt and echo it back as expectedUpdatedAt in the PATCH. Also exercises a
    # nodes-only definition (no `models` key), which `update` must accept.
    monkeypatch.setenv("COMFY_BUILDER_TOKEN", "jwt")
    calls = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        calls.append((method, url, body))
        if method == "GET" and url.endswith("/v1/distributions/d1"):
            return 200, {"id": "d1", "updatedAt": "2026-08-01T12:00:00Z"}
        if method == "PATCH":
            return 200, {"id": "d1"}
        return 200, {}

    monkeypatch.setattr("comfy_cli.distribution_api.request_json", fake_request_json)
    from comfy_cli.command import distribution

    d = tmp_path / "def.json"  # nodes-only, no `models` key
    d.write_text(json.dumps({"customNodes": [{"name": "x", "repository": "https://g/x", "gitRef": "a"}]}))
    distribution.update_cmd(distribution_id="d1", from_=str(d), builder_url="https://builder.test/")

    assert [c[0] for c in calls] == ["GET", "PATCH"]  # read-modify-write order
    patch = next(c for c in calls if c[0] == "PATCH")
    assert patch[2]["expectedUpdatedAt"] == "2026-08-01T12:00:00Z"
    import io
    import urllib.error

    import typer

    from comfy_cli.command.distribution import _builder_call

    def raise_403():
        raise urllib.error.HTTPError(
            "https://x", 403, "Forbidden", None, io.BytesIO(b'{"error":"FEATURE_NOT_ENABLED"}')
        )

    r = _RecordingRenderer()
    with pytest.raises(typer.Exit):
        _builder_call(r, raise_403)
    assert r.codes == ["distribution_not_enabled"]


def test_builder_call_maps_other_errors_to_builder_error():
    import io
    import urllib.error

    import typer

    from comfy_cli.command.distribution import _builder_call

    def raise_500():
        raise urllib.error.HTTPError("https://x", 500, "Server Error", None, io.BytesIO(b"boom"))

    r = _RecordingRenderer()
    with pytest.raises(typer.Exit):
        _builder_call(r, raise_500)
    assert r.codes == ["distribution_builder_error"]


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

    monkeypatch.setattr("comfy_cli.distribution_api.requests.put", fake_put)
    from comfy_cli.distribution_api import BuilderClient

    f = tmp_path / "m.safetensors"
    f.write_bytes(b"x")
    BuilderClient("https://builder.test/", "jwt").upload_blob("https://storage.example/put?sig=1", f)
    # the builder signs the URL requiring this header; without it GCS 400s
    assert captured["headers"].get("x-goog-if-generation-match") == "0"
    # presigned PUTs must not follow redirects (a 3xx could divert the file stream)
    assert captured["allow_redirects"] is False


def test_get_version_logs_uses_large_cap(monkeypatch):
    seen = {}

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        seen["max_bytes"] = max_bytes
        return 200, {"versionId": "v1", "log": "x", "truncated": False}

    monkeypatch.setattr("comfy_cli.distribution_api.request_json", fake_request_json)
    from comfy_cli.distribution_api import BuilderClient

    BuilderClient("https://builder.test/", "jwt").get_version_logs("v1")
    # the builder caps a served log at 8 MiB; the client cap must sit above that
    assert seen["max_bytes"] > 8 * 1024 * 1024


def test_builder_call_catches_response_too_large():
    import typer

    from comfy_cli.command.distribution import _builder_call
    from comfy_cli.http import ResponseTooLarge

    def raise_too_large():
        raise ResponseTooLarge("response exceeds cap")

    r = _RecordingRenderer()
    with pytest.raises(typer.Exit):
        _builder_call(r, raise_too_large)
    assert r.codes == ["distribution_builder_error"]


def test_create_command_missing_comfy_version(tmp_path):
    d = tmp_path / "def.json"
    d.write_text(json.dumps({"schema": "distribution-definition/0", "models": [], "customNodes": []}))
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "distribution", "create", "--from", str(d)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["error"]["code"] == "distribution_missing_comfy_version"


def test_create_command_bad_definition(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "distribution", "create", "--from", str(bad)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    envelope = json.loads(proc.stdout.strip().splitlines()[-1])
    assert envelope["error"]["code"] == "distribution_definition_invalid"


# --- create: the definition survives the trip ---------------------------------

REGISTRY_DEF = {
    "schema": "distribution-definition/0",
    "baseComfyVersion": "v0.3.40",
    "baseImage": "cuda130-py312",
    "modelPolicy": {"mode": "allowlist", "list": ["ae.safetensors"]},
    "partnerNodePolicy": {"mode": "blocklist", "list": []},
    "models": [],
    "customNodes": [
        {"name": "comfyui-kjnodes", "id": "comfyui-kjnodes", "registryVersion": "1.4.9", "source": "registry"},
        {"name": "priv", "blobId": "blob-1"},
        {"name": "gitpack", "repository": "https://github.com/x/gitpack", "gitRef": "deadbeef"},
    ],
}


def test_plan_create_keeps_registry_pins_out_of_the_upload_plan():
    """A registry-pinned node is a source the builder can resolve, so it travels as
    (id, registryVersion) rather than becoming an upload `--execute` would refuse."""
    plan = distribution.plan_create(REGISTRY_DEF)
    kj = next(n for n in plan["definition"]["customNodes"] if n["name"] == "comfyui-kjnodes")
    assert kj == {"name": "comfyui-kjnodes", "id": "comfyui-kjnodes", "registryVersion": "1.4.9"}
    assert plan["upload_count"] == 0


def test_plan_create_keeps_a_blob_reference():
    """An already-uploaded node names its blob; re-uploading it would orphan bytes."""
    priv = next(n for n in distribution.plan_create(REGISTRY_DEF)["definition"]["customNodes"] if n["name"] == "priv")
    assert priv == {"name": "priv", "blobId": "blob-1"}


def test_plan_create_honours_a_hand_written_git_node_with_no_source_tag():
    """`source` is the scan's own annotation. A definition written by hand names its
    fields and nothing else, and must not be read as an upload."""
    g = next(n for n in distribution.plan_create(REGISTRY_DEF)["definition"]["customNodes"] if n["name"] == "gitpack")
    assert g == {"name": "gitpack", "repository": "https://github.com/x/gitpack", "gitRef": "deadbeef"}


def test_plan_create_carries_base_image_and_policies():
    """These reach the builder or they do not exist: an absent baseImage silently
    becomes the catalog default, and an absent policy seals as allow-all."""
    d = distribution.plan_create(REGISTRY_DEF)["definition"]
    assert d["baseImage"] == "cuda130-py312"
    assert d["modelPolicy"] == {"mode": "allowlist", "list": ["ae.safetensors"]}
    assert d["partnerNodePolicy"] == {"mode": "blocklist", "list": []}
    assert d["baseComfyVersion"] == "v0.3.40"


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
        distribution, "_git_output", lambda path, *args: "https://github.com/x/half" if args[0] == "remote" else None
    )
    (node,) = distribution.scan_custom_nodes(tmp_path / "custom_nodes")
    assert node["source"] == "registry"
    assert node.get("repository") is None and node.get("gitRef") is None


def test_scan_reads_the_registry_pin_off_an_archive_install(tmp_path):
    """`comfy node install` unpacks archives, so a pack has no git history at all.
    Its pyproject still names the published version the builder can fetch."""
    _write_pack(tmp_path, "comfyui-kjnodes", project='[project]\nname = "comfyui-kjnodes"\nversion = "1.4.9"\n')
    (node,) = distribution.scan_custom_nodes(tmp_path / "custom_nodes")
    assert node["source"] == "registry"
    assert node["id"] == "comfyui-kjnodes"
    assert node["registryVersion"] == "1.4.9"


def test_scan_keeps_a_pack_local_when_nothing_names_an_upstream(tmp_path):
    """No git, no usable pyproject: it really must be uploaded."""
    _write_pack(tmp_path, "handmade")
    _write_pack(tmp_path, "half", project='[project]\nname = "half"\n')
    assert {n["name"]: n["source"] for n in distribution.scan_custom_nodes(tmp_path / "custom_nodes")} == {
        "handmade": "local",
        "half": "local",
    }


def test_scan_prefers_git_over_the_registry_pin(tmp_path, monkeypatch):
    """A commit pins bytes exactly; a package version is resolved later. When a pack
    has both, the more precise one wins."""
    _write_pack(tmp_path, "dual", project='[project]\nname = "dual"\nversion = "2.0.0"\n', git=True)
    monkeypatch.setattr(
        distribution,
        "_git_output",
        lambda path, *args: "https://github.com/x/dual" if args[0] == "remote" else "cafebabe",
    )
    (node,) = distribution.scan_custom_nodes(tmp_path / "custom_nodes")
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
    assert distribution._read_registry_pin(d) is None, why


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
    assert distribution.as_comfy_git_ref(detected) == expected


def test_scan_command_writes_a_resolvable_comfy_ref(models_tree, tmp_path):
    """End-to-end: a bare `--comfy-version` reaches the definition as a tag. The
    builder resolves this field with git ls-remote, so the bare number it used to
    record could only ever be discovered by a failed build."""
    out = tmp_path / "definition.json"
    env = {**os.environ, "NO_COLOR": "1", "COMFY_OUTPUT": "json"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "comfy_cli",
            "distribution",
            "scan",
            "--models-dir",
            str(models_tree),
            "--comfy-version",
            "0.30.2",
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["baseComfyVersion"] == "v0.30.2"


# --- validate: say what was checked, and surface the warnings -----------------


def _run_validate(monkeypatch, capsys, result):
    """Drive validate_cmd against a stand-in builder, in pretty mode."""

    class FakeClient:
        def validate_distribution(self, distribution_id):
            return result

    monkeypatch.setattr(distribution, "_builder_client", lambda renderer, url: FakeClient())
    distribution.validate_cmd("d1")
    return capsys.readouterr().out


def test_validate_prints_the_warnings_beside_the_verdict(monkeypatch, capsys):
    """A ref the builder cannot find rides alongside `ok: true`. It was reachable
    only by reading the JSON dump, under a line that said the definition resolved,
    so the one thing that will fail the cut was the easiest thing to miss."""
    out = _run_validate(
        monkeypatch,
        capsys,
        {
            "ok": True,
            "warnings": [{"field": "baseComfyVersion", "reason": "ref not found in remote advertisement"}],
        },
    )
    # Assert on the text BEFORE the JSON dump: the dump repeats every one of these
    # strings, so a whole-output assertion passes even with the warn lines deleted.
    head = out.split("{", 1)[0]
    assert "1 reference(s)" in head
    assert "the build will fail on these" in head
    assert "baseComfyVersion: ref not found in remote advertisement" in head


def test_validate_does_not_claim_the_definition_resolves(monkeypatch, capsys):
    """The endpoint checks shape and pin existence; whether the set installs
    together is only answered by a build."""
    out = _run_validate(monkeypatch, capsys, {"ok": True})
    assert "not a full resolve" in out
    assert "Definition resolves." not in out


# --- create: the builder reads the pins, not us -------------------------------


def test_snapshot_from_definition_maps_each_source_to_its_snapshot_kind():
    """The importer reads a Desktop export; a scan holds the same facts under other
    names. Translating is what lets one implementation of registry truth serve both."""
    snap = distribution.snapshot_from_definition(
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
    snap = distribution.snapshot_from_definition(
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
    snap = distribution.snapshot_from_definition({"models": [{"type": "vae", "filename": "ae.safetensors"}]})
    assert "models" not in snap["snapshots"][0]


def test_report_advisories_names_every_thing_the_import_could_not_carry():
    lines = distribution.report_advisories(
        {"notInRegistry": ["was-node-suite-comfyui"], "unpinnablePins": ["torch"], "pythonSatisfied": True}
    )
    assert any("was-node-suite-comfyui" in line and "does not publish" in line for line in lines)
    assert any("torch" in line and "not a public PyPI release" in line for line in lines)
    assert len(lines) == 2  # pythonSatisfied True says nothing


def test_report_advisories_says_when_no_base_image_matches_the_python():
    """The scan ran on one Python and the build runs on another, which is how a
    freeze that resolved locally fails in the build."""
    (line,) = distribution.report_advisories({"pythonSatisfied": False})
    assert "closest one" in line


class _ImportingBuilder:
    """Stands in for the builder across the whole execute path."""

    def __init__(self, resolved=None, fail=None):
        self.resolved = resolved
        self.fail = fail
        self.created_with = None
        self.snapshot_seen = None

    def resolve_snapshot(self, snapshot):
        self.snapshot_seen = snapshot
        if self.fail:
            raise self.fail
        return self.resolved

    def resolve_models(self, filenames):
        return []

    def create_distribution(self, name, definition, description=None):
        self.created_with = definition
        return "dist-1"

    def cut_version(self, distribution_id, targets=None):
        return ("ver-1", "status-url")


def _execute(monkeypatch, builder, definition):
    monkeypatch.setattr(distribution, "_builder_client", lambda renderer, url: builder)
    distribution._create_execute(get_renderer(), definition, name="demo", builder_url=None, models_dir=None)


def test_create_execute_sends_the_definition_for_reading_and_takes_back_what_it_says(monkeypatch):
    """The whole point of the call: the id the builder vouched for is the id that
    gets created, without the CLI holding its own copy of registry truth."""
    builder = _ImportingBuilder(
        resolved={
            "definition": {
                "customNodes": [{"name": "was", "id": "was-node-suite-comfyui", "registryVersion": "1.0.1"}]
            },
            "report": {},
        }
    )
    _execute(
        monkeypatch,
        builder,
        {
            "baseComfyVersion": "v0.30.2",
            "models": [],
            "customNodes": [{"name": "was", "id": "pr-was-47064894", "registryVersion": "1.0.1"}],
        },
    )
    assert builder.snapshot_seen["snapshots"][0]["customNodes"][0]["id"] == "pr-was-47064894"
    (node,) = builder.created_with["customNodes"]
    assert node["id"] == "was-node-suite-comfyui"


def test_create_execute_refuses_when_the_builder_drops_a_pack(monkeypatch):
    """A pack the importer could not vouch for is absent from what it returns.
    Building an image quietly missing what the user asked for is worse than failing."""
    builder = _ImportingBuilder(resolved={"definition": {"customNodes": []}, "report": {"notInRegistry": ["was"]}})
    with pytest.raises(typer.Exit):
        _execute(
            monkeypatch,
            builder,
            {
                "baseComfyVersion": "v0.30.2",
                "models": [],
                "customNodes": [{"name": "was", "id": "nope", "registryVersion": "1.0.1"}],
            },
        )
    assert builder.created_with is None


def test_create_execute_still_creates_when_the_builder_cannot_read_pins(monkeypatch):
    """An older builder has no importer. The pins go to the cut unchecked, which is
    where they were checked before this existed; refusing to create would be worse."""
    builder = _ImportingBuilder(fail=requests.RequestException("404 no such endpoint"))
    _execute(
        monkeypatch,
        builder,
        {
            "baseComfyVersion": "v0.30.2",
            "models": [],
            "customNodes": [{"name": "was", "id": "pr-was-47064894", "registryVersion": "1.0.1"}],
        },
    )
    assert builder.created_with["customNodes"][0]["id"] == "pr-was-47064894"


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
    assert distribution._redact_freeze_credentials(line) == expected


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
    monkeypatch.setattr(distribution.subprocess, "run", fake_run)
    assert distribution._freeze_env("/x/bin/python") == "numpy==1.26.4\n"
    assert "PYTHONPATH" not in seen["env"]


# --- update: the same file works through either command -----------------------


def test_update_maps_a_scan_definition_instead_of_sending_it_raw(monkeypatch):
    """`update --from` sent the file verbatim, so a scan definition's models
    arrived with neither sourceUri nor blobId and the builder rejected every one.
    The same file must work through create and update alike."""
    sent = {}

    class FakeClient:
        def get_distribution(self, did):
            return {"updatedAt": "t0"}

        def update_distribution(self, did, definition, updated_at):
            sent.update(definition)
            return {"id": did}

    monkeypatch.setattr(distribution, "_builder_client", lambda renderer, url: FakeClient())
    monkeypatch.setattr(
        distribution,
        "resolve_models_via_builder",
        lambda models, client: [m.__setitem__("sourceUri", "https://hf.co/x") for m in models] and len(models),
    )
    defn = {
        "models": [{"type": "vae", "filename": "a.safetensors", "sha256": "d", "sizeBytes": 1, "source": "local"}],
        "customNodes": [{"name": "kj", "id": "kj", "registryVersion": "1.4.9", "source": "registry"}],
        "baseComfyVersion": "0.30.2",
    }
    monkeypatch.setattr(distribution, "_load_definition", lambda p, require_models=False: defn)
    distribution.update_cmd("d1", from_="ignored.json")

    assert sent["models"][0]["sourceUri"] == "https://hf.co/x"
    assert "sizeBytes" not in sent["models"][0] and "source" not in sent["models"][0]
    assert sent["customNodes"][0] == {"name": "kj", "id": "kj", "registryVersion": "1.4.9"}
    assert sent["baseComfyVersion"] == "v0.30.2"


def test_update_refuses_what_it_cannot_upload(monkeypatch):
    """update has no upload step, so a model with no public source is a clear
    refusal naming the command that can, not a builder 400."""

    class FakeClient:
        def get_distribution(self, did):
            return {"updatedAt": "t0"}

    monkeypatch.setattr(distribution, "_builder_client", lambda renderer, url: FakeClient())
    monkeypatch.setattr(distribution, "resolve_models_via_builder", lambda models, client: 0)
    defn = {
        "models": [{"type": "vae", "filename": "private.safetensors", "sha256": "d", "source": "local"}],
        "customNodes": [],
    }
    monkeypatch.setattr(distribution, "_load_definition", lambda p, require_models=False: defn)
    with pytest.raises(typer.Exit) as e:
        distribution.update_cmd("d1", from_="ignored.json")
    assert e.value.exit_code == 1


# --- blobs: a private file is uploaded once and referenced by id ---------------


def test_plan_create_keeps_a_model_blob_reference():
    """A model the caller already uploaded names its blob. Re-uploading it would
    orphan the bytes it replaced and spend the transfer twice."""
    plan = distribution.plan_create(
        {"models": [{"type": "vae", "filename": "ae.safetensors", "sha256": "def", "blobId": "blob-9"}]}
    )
    assert plan["definition"]["models"][0]["blobId"] == "blob-9"
    assert plan["upload_count"] == 0


def test_a_definition_naming_only_blobs_goes_through_update_untouched():
    """Every member names a builder source, so update has nothing to map and sends
    the file as written."""
    assert not distribution._is_scan_shaped(
        {
            "models": [{"type": "vae", "filename": "ae.safetensors", "blobId": "blob-9"}],
            "customNodes": [{"name": "priv", "blobId": "blob-1"}],
        }
    )
