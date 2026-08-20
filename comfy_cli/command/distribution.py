"""``comfy distribution`` — package a local ComfyUI environment into a
serverless distribution build.

POC scope: ``scan`` inspects a local ComfyUI install and emits a builder-ready
*definition* covering the two halves a build needs:

- **models** — walk the ``models/`` tree and hash every model file, producing
  ``{type, filename, sha256, sizeBytes}``. This closes the model-path resolution
  gap: a workflow only carries a bare model name (path stripped, no hash), but a
  build needs the placement folder, filename, and a content hash to fetch and
  verify the right file.
- **custom nodes** — walk ``custom_nodes/`` and record each node's git
  ``repository`` + commit ``gitRef`` (the builder fetches ``repo@ref``). Nodes
  with no fetchable upstream are marked ``source: local`` — they must be uploaded
  as a blob.

The scan runs entirely on the user's machine — the one place the real files and
node checkouts exist — and is the agreed compatibility shim for users who won't
upgrade desktop. The emitted definition maps 1:1 onto the builder's
``definition.models[]`` / ``definition.customNodes[]``; a later ``create``
command uploads private blobs and cuts a version from it.

Surface module (Typer + renderer) in the ``command/project.py`` style; the
scan/hash logic is kept in pure helpers so it stays unit-testable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import requests
import typer

from comfy_cli import tracking
from comfy_cli._safe_exec import BinaryNotFoundError, resolve_required_binary
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH, SUPPORTED_PT_EXTENSIONS
from comfy_cli.file_utils import atomic_write_text
from comfy_cli.output import get_renderer
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Package a local ComfyUI environment into a serverless distribution. "
        "[Limited beta] — builder access is granted per account; commands that reach the "
        "builder return a 'not enabled yet' message until your account is enabled."
    ),
)

# `comfy distribution version <verb>` — the resource-verb surface for versions
# (a version is an immutable build cut of a distribution).
version_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and cut distribution versions (builds).",
)
app.add_typer(version_app, name="version")

# `comfy distribution artifact <verb>` / `blob <verb>` — the per-target build
# outputs and the workspace's private uploaded content.
artifact_app = typer.Typer(no_args_is_help=True, help="Build artifacts (per-target outputs of a cut version).")
app.add_typer(artifact_app, name="artifact")
blob_app = typer.Typer(no_args_is_help=True, help="Workspace private blobs (uploaded models / nodes).")
app.add_typer(blob_app, name="blob")

# @singleton — the same instance the top-level callback set up, so
# `workspace_path` is already resolved by the time a command runs.
workspace_manager = WorkspaceManager()

# ComfyUI's custom-node directory name, sibling of models/ under the install
# root. No constant for it in constants.py (only DEFAULT_COMFY_MODEL_PATH), so
# it's named here.
DEFAULT_COMFY_CUSTOM_NODES_PATH = "custom_nodes"

# Bumped only on a breaking change to the emitted definition shape (see the
# machine-output contract note in comfy_cli/output/renderer.py).
DEFINITION_SCHEMA = "distribution-definition/0"

# Bound each git call so a wedged repo (network remote, lock contention) can't
# stall the whole scan.
_GIT_TIMEOUT = 10


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file through SHA-256 in fixed-size chunks so multi-GB models
    aren't loaded fully into memory. SHA-256 (not blake3) because the builder
    is SHA-256 end-to-end: it re-verifies this exact digest at build time."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_model_files(models_root: Path) -> Iterator[tuple[Path, str, str]]:
    """Yield ``(abs_path, model_type, filename)`` for model files under
    ``models_root``.

    - **Follows symlinks.** ComfyUI model dirs are routinely symlinked to
      external storage (``models/loras`` → a shared drive), so a
      symlink-skipping walk would miss most real installs. Directory cycles are
      broken with a resolved-path visited set.
    - ``model_type`` is the POSIX directory path relative to ``models/`` (e.g.
      ``checkpoints``, ``ultralytics/bbox``) — exactly the builder's
      ``Model.type`` placement target.
    - Files sitting directly in ``models/`` (no folder) and dotfiles are
      skipped; a build needs a placement folder.
    """
    visited: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(models_root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []  # cycle via symlink — don't recurse back in
            continue
        visited.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))

        model_type = Path(dirpath).relative_to(models_root).as_posix()
        if model_type == ".":
            continue  # bare models/ root: no placement folder to build into

        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if not name.lower().endswith(SUPPORTED_PT_EXTENSIONS):
                continue
            yield Path(dirpath) / name, model_type, name


def scan_models(models_root: Path) -> list[dict]:
    """Walk ``models_root`` and return one definition entry per model file,
    sorted by ``type`` then ``filename`` for stable, diffable output."""
    models: list[dict] = []
    for path, model_type, filename in _iter_model_files(models_root):
        models.append(
            {
                "type": model_type,
                "filename": filename,
                "sha256": _sha256_file(path),
                "sizeBytes": path.stat().st_size,
                # POC: every scanned file is a local blob candidate. The create
                # step decides sourceUri (public catalog match) vs blobId
                # (private upload) per entry.
                "source": "local",
            }
        )
    return models


def _git_output(repo_path: Path, *args: str) -> str | None:
    """Return stripped stdout of ``git -C <repo_path> <args>``, or None on any
    failure (git absent, non-zero exit, timeout, empty output). Never raises —
    a node that can't answer git just falls back to ``source: local``."""
    try:
        git_bin = resolve_required_binary("git")
    except BinaryNotFoundError:
        return None
    try:
        result = subprocess.run(
            [git_bin, "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def scan_custom_nodes(custom_nodes_root: Path) -> list[dict]:
    """Return one definition entry per custom-node directory, sorted by name.

    Each top-level directory under ``custom_nodes/`` is one node. For a git
    checkout we record its ``repository`` (origin remote) and ``gitRef`` (HEAD
    commit) so the builder can fetch ``repo@ref`` reproducibly. A directory that
    is not a git checkout, or has no fetchable origin, is marked ``source:
    local`` — it has no upstream and must be uploaded as a blob. Returns an empty
    list when the folder is absent (a workflow may use no custom nodes)."""
    nodes: list[dict] = []
    if not custom_nodes_root.is_dir():
        return nodes
    for entry in sorted(custom_nodes_root.iterdir()):
        if not entry.is_dir():
            continue  # single-file .py nodes aren't buildable units here
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        is_git = (entry / ".git").exists()
        repository = _git_output(entry, "remote", "get-url", "origin") if is_git else None
        git_ref = _git_output(entry, "rev-parse", "HEAD") if is_git else None
        nodes.append(
            {
                "name": entry.name,
                "repository": repository,
                "gitRef": git_ref,
                # Fetchable only when we have BOTH a remote and a pinned commit;
                # a local-only or detached checkout can't be reconstructed from
                # a repo@ref, so it must be uploaded.
                "source": "git" if (repository and git_ref) else "local",
            }
        )
    return nodes


# Matches ComfyUI's `__version__ = "0.3.x"` in comfyui_version.py — the same
# marker comfy_cli/command/outdated.py reads.
_COMFY_VERSION_RE = re.compile(r"""^__version__\s*=\s*['"]([^'"\n]+)['"]""", re.MULTILINE)


def detect_comfy_version(root: Path) -> str | None:
    """Best-effort ComfyUI version for the definition's ``baseComfyVersion``.

    Prefers the packaged marker ``comfyui_version.py`` (the clean released
    version), then ``git describe``, then ``pyproject.toml``. Returns None when
    the root isn't a recognizable ComfyUI install — the builder can still fall
    back to its own default base in that case."""
    marker = root / "comfyui_version.py"
    try:
        m = _COMFY_VERSION_RE.search(marker.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    except (OSError, UnicodeDecodeError):
        pass
    described = _git_output(root, "describe", "--tags", "--always")
    if described:
        return described.lstrip("vV")
    try:
        for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("version") and "=" in stripped:
                return stripped.split("=", 1)[1].strip().strip("\"'")
    except (OSError, UnicodeDecodeError):
        pass
    return None


# ComfyUI's default local address; the running server reports its version at
# /system_stats even when the code dir has no version marker (split layouts).
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


def detect_comfy_version_from_server(base_url: str) -> str | None:
    """Ask a running ComfyUI for its version via GET /system_stats. This is the
    fallback that covers split/Desktop layouts where the version isn't on disk
    at the data root. Returns None if no server answers (fast-fail) or the field
    is absent."""
    try:
        resp = requests.get(base_url.rstrip("/") + "/system_stats", timeout=2)
        resp.raise_for_status()
        version = resp.json().get("system", {}).get("comfyui_version")
    except (requests.RequestException, ValueError):
        return None
    return version or None


def build_definition(
    models: list[dict],
    custom_nodes: list[dict],
    base_comfy_version: str | None = None,
    pip_dependencies: str | None = None,
    environment: dict | None = None,
) -> dict:
    """Assemble the builder-ready definition from scanned models + nodes.
    Environment fields (baseComfyVersion / pipDependencies / environment) are
    included only when captured (omitted → builder falls back to its default)."""
    definition = {"schema": DEFINITION_SCHEMA, "models": models, "customNodes": custom_nodes}
    if base_comfy_version:
        definition["baseComfyVersion"] = base_comfy_version
    if pip_dependencies:
        definition["pipDependencies"] = pip_dependencies
    if environment:
        definition["environment"] = environment
    return definition


def _venv_python(venv_dir: Path) -> Path:
    """The Python executable inside a venv dir (Windows vs POSIX layout)."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def find_comfy_python(comfy_root: Path | None, explicit: str | None) -> Path | None:
    """Locate the ComfyUI environment's Python for a `pip freeze`.

    ``--python`` wins. Otherwise only a venv co-located with the ComfyUI *code*
    root (``.venv``/``venv``) counts. We deliberately do NOT fall back to the
    CLI's own interpreter or an ambient ``VIRTUAL_ENV`` (as
    ``resolve_workspace_python`` does): freezing the wrong environment produces
    confidently-wrong provenance — the CLI's packages, not ComfyUI's — which is
    worse than none. Returns None when it can't be located (e.g. a data-only
    ComfyUI directory whose code lives elsewhere)."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    if comfy_root:
        for name in (".venv", "venv"):
            py = _venv_python(comfy_root / name)
            if py.is_file():
                return py
    return None


_TORCH_RE = re.compile(r"^torch==(\S+)", re.MULTILINE)


def _freeze_env(python_exe: str) -> str | None:
    """`pip freeze` for a Python env, falling back to `uv pip freeze` when the
    env has no pip module (uv-created venvs, common for ComfyUI). Returns the
    requirements text, or None if both fail."""
    attempts = [[python_exe, "-m", "pip", "freeze"]]
    # Resolve uv the same way as git (absolute path + typed miss) rather than a
    # bare PATH name; skip the fallback only when uv is genuinely absent.
    try:
        attempts.append([resolve_required_binary("uv"), "pip", "freeze", "--python", python_exe])
    except BinaryNotFoundError:
        pass
    for cmd in attempts:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (subprocess.SubprocessError, OSError):
            continue
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    return None


def capture_pip_provenance(python_exe: str) -> dict | None:
    """Freeze ``python_exe``'s env and capture source-platform provenance.

    Returns ``{environment: {os, arch, pythonVersion, torch}, pipDependencies}``.
    ``pipDependencies`` is the freeze prefixed with a comment header naming the
    source platform, so the pins carry their own "captured on macOS/arm64,
    retarget for the build platform" context downstream — this is *evidence* for
    a resolver, not authoritative target pins (a Mac freeze can't run on
    Linux+CUDA as-is). Returns None if the freeze or platform probe fails."""
    reqs = _freeze_env(python_exe)
    if reqs is None:
        return None
    try:
        probe = subprocess.run(
            [
                python_exe,
                "-c",
                "import platform,json;print(json.dumps({'os':platform.system(),"
                "'arch':platform.machine(),'pythonVersion':platform.python_version()}))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if probe.returncode != 0:
        return None
    try:
        env = json.loads(probe.stdout.strip())
    except json.JSONDecodeError:
        return None
    m = _TORCH_RE.search(reqs)
    env["torch"] = m.group(1) if m else None
    header = (
        f"# Captured by comfy-cli from the source environment: {env['os']}/{env['arch']}, "
        f"Python {env['pythonVersion']}" + (f", torch {env['torch']}" if env["torch"] else "") + ".\n"
        "# Resolved pins from where the workflow ran locally — a build for a different "
        "OS/GPU must retarget them.\n"
    )
    return {"environment": env, "pipDependencies": header + reqs}


# --- create: scan definition -> builder definition + upload plan --------------
#
# The builder's definition schema (services/comfy-builder/definition/definition.go):
#   Model = { type, filename?, sourceUri? | blobId?, sha256? }  (exactly one source)
#   Node  = { name, repository? + gitRef? | registryVersion? | blobId? }
# The scan already emits type/filename/sha256/name/repository/gitRef with the same
# keys, so mapping is: decide each item's source (public URL vs upload) and drop
# the scan-only fields (sizeBytes, source, top-level schema).


def resolve_model_source(model: dict) -> str | None:
    """Return the public ``sourceUri`` chosen for a model, or None if it must be
    uploaded. This reads the annotation left by ``resolve_models_via_builder``
    (the network resolve + hash-verify step) — on its own it makes no network
    call, so ``plan_create`` stays offline and testable. Injectable, so tests can
    still substitute a fake resolver."""
    return model.get("sourceUri")


# The builder's POST /v1/models/resolve accepts at most this many filenames/call.
_RESOLVE_BATCH = 32


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def resolve_models_via_builder(models: list[dict], client) -> int:
    """Resolve each model's filename against the builder and, when a returned
    candidate's sha256 matches the local file's hash, annotate the model with
    that candidate's ``sourceUri`` (so ``plan_create`` references it by URL
    instead of uploading it). Returns how many were resolved.

    Only a **hash-confirmed** public match is trusted: a filename alone is a weak
    key, and a renamed local fine-tune must never be mistaken for a public model.
    Candidates the provider didn't hash are left to upload (the safe default).
    The builder re-verifies the sha256 on fetch regardless, so this is an
    optimization to avoid needless uploads — not the integrity boundary."""
    names = sorted({m["filename"] for m in models if m.get("filename")})
    if not names:
        return 0
    by_name: dict[str, list[dict]] = {}
    for batch in _chunks(names, _RESOLVE_BATCH):
        for res in client.resolve_models(batch):
            by_name[res.get("filename", "")] = res.get("candidates") or []

    resolved = 0
    for m in models:
        local = (m.get("sha256") or "").lower()
        if not local or m.get("sourceUri"):
            continue
        for cand in by_name.get(m["filename"], []):
            uri = cand.get("sourceUri")
            if uri and (cand.get("sha256") or "").lower() == local:
                m["sourceUri"] = uri
                resolved += 1
                break
    return resolved


def plan_create(definition: dict, resolver=resolve_model_source) -> dict:
    """Turn a scan definition into a builder definition + an upload plan.

    For each model: if the resolver finds a public URL → reference it as
    ``sourceUri`` (no upload); otherwise it's a private blob → goes in the upload
    plan and will be referenced by ``blobId`` once uploaded. For each custom node:
    ``repository``+``gitRef``, ``id``+``registryVersion`` or ``blobId`` is carried
    through as-is; only a node naming none of them has no upstream and is uploaded
    as a ``node_zip`` blob. Every other definition key travels untouched, because
    a dropped one is not an error the builder reports, just a different build.

    Pure and backend-free: it decides *what* would be sent/uploaded without making
    any network call. The ``blobId`` values are filled in later by the upload step;
    here upload-bound items carry no source yet and are listed in ``uploads``."""
    models_out: list[dict] = []
    nodes_out: list[dict] = []
    uploads: list[dict] = []

    for i, m in enumerate(definition.get("models", [])):
        entry: dict = {"type": m["type"], "filename": m["filename"]}
        if m.get("sha256"):
            entry["sha256"] = m["sha256"]
        uri = resolver(m)
        if uri:
            entry["sourceUri"] = uri
        else:
            # `slot` points at the builder-definition entry this upload fills, so
            # execute can stitch the returned blobId back after uploading.
            uploads.append(
                {
                    "kind": "model",
                    "type": m["type"],
                    "filename": m["filename"],
                    "sha256": m.get("sha256"),
                    "sizeBytes": m.get("sizeBytes", 0),
                    "slot": ["models", i],
                }
            )
        models_out.append(entry)

    for j, n in enumerate(definition.get("customNodes", [])):
        entry = {"name": n["name"]}
        # The builder takes exactly one source per node. Try them most-precise
        # first: a pinned git checkout reconstructs an exact commit, a registry
        # version an exact published artifact, a blob is bytes we already hold.
        # Keyed on the fields themselves rather than the scan's `source` tag, so a
        # hand-edited definition that names a source without one is honoured too.
        if n.get("repository") and n.get("gitRef"):
            entry["repository"] = n["repository"]
            entry["gitRef"] = n["gitRef"]
        elif n.get("registryVersion") and n.get("id"):
            entry["id"] = n["id"]
            entry["registryVersion"] = n["registryVersion"]
        elif n.get("blobId"):
            entry["blobId"] = n["blobId"]
        else:
            uploads.append({"kind": "node_zip", "name": n["name"], "sizeBytes": 0, "slot": ["customNodes", j]})
        nodes_out.append(entry)

    builder_definition: dict = {"models": models_out, "customNodes": nodes_out}
    # Everything else the definition declares goes through untouched. Dropping a
    # key here does not fail loudly: the builder just applies its own default, so a
    # base image silently becomes the catalog's current one and an absent policy
    # seals as allow-all.
    for key in ("baseComfyVersion", "baseImage", "pipDependencies", "modelPolicy", "partnerNodePolicy"):
        if definition.get(key):
            builder_definition[key] = definition[key]
    return {
        "definition": builder_definition,
        "uploads": uploads,
        "upload_count": len(uploads),
        "upload_bytes": sum(u.get("sizeBytes", 0) for u in uploads),
    }


def execute_create(plan: dict, *, client, name: str, locate_bytes, targets=None) -> dict:
    """Run the live create: upload each private blob, stitch its blobId into the
    definition, create the distribution, and cut a build.

    ``client`` is a BuilderClient (or a stand-in with the same methods).
    ``locate_bytes(upload) -> Path`` finds the local file for a model upload.
    node_zip uploads (hand-dropped local nodes) aren't packaged yet — raises
    NotImplementedError so the caller can surface a clear message. Returns
    ``{distributionId, versionId, statusUrl, uploaded}``."""
    definition = plan["definition"]
    # Preflight the whole upload list before any bytes move: an unsupported local
    # node, or a model whose file can't be found (missing --models-dir), must fail
    # here — not after uploading gigabytes of other models that would be orphaned
    # when the create never happens.
    prepared = []
    for u in plan["uploads"]:
        if u["kind"] != "model":
            raise NotImplementedError(f"uploading a local custom node ({u.get('name')}) isn't supported yet")
        prepared.append((u, locate_bytes(u)))
    uploaded = 0
    for u, path in prepared:
        blob_id, upload_url = client.create_blob("model", u["filename"], u["sha256"], u["sizeBytes"])
        client.upload_blob(upload_url, path)
        section, idx = u["slot"]
        definition[section][idx]["blobId"] = blob_id
        uploaded += 1
    dist_id = client.create_distribution(name, definition)
    try:
        version_id, status_url = client.cut_version(dist_id, targets)
    except Exception as e:
        # The distribution now exists server-side; carry its id on the error so the
        # command can surface it (the user can then delete or retry it, not orphan it).
        e.distribution_id = dist_id
        raise
    return {"distributionId": dist_id, "versionId": version_id, "statusUrl": status_url, "uploaded": uploaded}


def _resolve_models_dir(models_dir: str | None) -> Path | None:
    """``--models-dir`` wins; otherwise the selected workspace's ``models/``.
    Returns None when neither yields a path (no workspace selected)."""
    if models_dir:
        return Path(models_dir).expanduser()
    ws = workspace_manager.workspace_path
    if not ws:
        return None
    return Path(ws) / DEFAULT_COMFY_MODEL_PATH


def _resolve_custom_nodes_dir(custom_nodes_dir: str | None, models_dir: str | None) -> Path | None:
    """``--custom-nodes-dir`` wins; else the ``custom_nodes/`` sibling of the
    chosen models dir (they share a ComfyUI root); else the workspace's
    ``custom_nodes/``. Returns None when nothing resolves."""
    if custom_nodes_dir:
        return Path(custom_nodes_dir).expanduser()
    if models_dir:
        return Path(models_dir).expanduser().parent / DEFAULT_COMFY_CUSTOM_NODES_PATH
    ws = workspace_manager.workspace_path
    if not ws:
        return None
    return Path(ws) / DEFAULT_COMFY_CUSTOM_NODES_PATH


def _resolve_comfy_root(models_dir: str | None) -> Path | None:
    """The ComfyUI install root — parent of the chosen models dir, else the
    selected workspace. Used to read the ComfyUI version for baseComfyVersion."""
    if models_dir:
        return Path(models_dir).expanduser().parent
    ws = workspace_manager.workspace_path
    return Path(ws) if ws else None


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _render_models_table(renderer, models: list[dict], total_bytes: int) -> None:
    from rich.table import Table

    table = Table(title=f"{len(models)} models · {_human_size(total_bytes)}")
    table.add_column("type", style="cyan", no_wrap=True)
    table.add_column("filename", style="white")
    table.add_column("size", justify="right", style="green")
    table.add_column("sha256", style="dim")
    for m in models:
        table.add_row(m["type"], m["filename"], _human_size(m["sizeBytes"]), m["sha256"][:12] + "…")
    renderer.console().print(table)


def _render_nodes_table(renderer, nodes: list[dict]) -> None:
    from rich.table import Table

    local = sum(1 for n in nodes if n["source"] == "local")
    title = f"{len(nodes)} custom nodes"
    if local:
        title += f" · {local} need upload"
    table = Table(title=title)
    table.add_column("node", style="cyan", no_wrap=True)
    table.add_column("source", style="green")
    table.add_column("repository", style="white")
    table.add_column("ref", style="dim")
    for n in nodes:
        ref = (n["gitRef"][:12] + "…") if n["gitRef"] else "—"
        table.add_row(n["name"], n["source"], n["repository"] or "—", ref)
    renderer.console().print(table)


@app.command(
    "scan",
    help="Scan a local ComfyUI install (models + custom nodes) and emit a builder-ready definition.",
)
@tracking.track_command("distribution")
def scan_cmd(
    models_dir: Annotated[
        str | None,
        typer.Option(
            "--models-dir",
            help="Models folder to scan. Default: the selected workspace's models/ directory.",
        ),
    ] = None,
    custom_nodes_dir: Annotated[
        str | None,
        typer.Option(
            "--custom-nodes-dir",
            help="Custom-nodes folder to scan. Default: the custom_nodes/ sibling of the models dir.",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the definition JSON to this path. Default: emit it in the command result.",
        ),
    ] = None,
    python: Annotated[
        str | None,
        typer.Option(
            "--python",
            help="ComfyUI's Python executable, to capture pip deps. Default: a venv beside the install; "
            "required for split/Desktop layouts where the code lives apart from the data dir.",
        ),
    ] = None,
    comfy_version: Annotated[
        str | None,
        typer.Option("--comfy-version", help="Override the detected ComfyUI version (baseComfyVersion)."),
    ] = None,
    comfy_url: Annotated[
        str | None,
        typer.Option(
            "--comfy-url", help=f"Running ComfyUI URL to read the version from. Default: {DEFAULT_COMFY_URL}."
        ),
    ] = None,
):
    renderer = get_renderer()

    root = _resolve_models_dir(models_dir)
    if root is None or not root.is_dir():
        renderer.error(
            code="distribution_models_dir_missing",
            message=f"No models/ directory to scan at {root!s}." if root else "No workspace selected.",
            details={"path": str(root) if root else None},
        )
        raise typer.Exit(code=1)

    renderer.info(f"Scanning models in {root} …")
    models = scan_models(root)
    total_bytes = sum(m["sizeBytes"] for m in models)

    nodes_root = _resolve_custom_nodes_dir(custom_nodes_dir, models_dir)
    nodes: list[dict] = []
    if nodes_root and nodes_root.is_dir():
        renderer.info(f"Scanning custom nodes in {nodes_root} …")
        nodes = scan_custom_nodes(nodes_root)

    # baseComfyVersion: explicit flag → version marker at the code root → a
    # running ComfyUI's /system_stats (covers split layouts with no on-disk marker).
    comfy_root = _resolve_comfy_root(models_dir)
    base_version = comfy_version or (detect_comfy_version(comfy_root) if comfy_root else None)
    if not base_version:
        base_version = detect_comfy_version_from_server(comfy_url or DEFAULT_COMFY_URL)

    # pip deps: freeze the ComfyUI env (evidence for the resolver, tagged with the
    # source platform). Never guess the interpreter — omit if we can't find it.
    python_exe = find_comfy_python(comfy_root, python)
    provenance = capture_pip_provenance(str(python_exe)) if python_exe else None
    if python is None and python_exe is None:
        renderer.warn("pip deps not captured — couldn't locate ComfyUI's Python. Pass --python <path> to include them.")
    elif python_exe is None:
        renderer.warn(f"pip deps not captured — no Python at {python}.")
    elif provenance is None:
        renderer.warn(f"pip deps not captured — `pip freeze` failed for {python_exe}.")

    definition = build_definition(
        models,
        nodes,
        base_comfy_version=base_version,
        pip_dependencies=provenance["pipDependencies"] if provenance else None,
        environment=provenance["environment"] if provenance else None,
    )
    renderer.event(
        "scan_complete",
        models=len(models),
        custom_nodes=len(nodes),
        total_bytes=total_bytes,
        base_comfy_version=base_version,
        pip_captured=provenance is not None,
    )

    if output:
        out_path = Path(output).expanduser()
        try:
            atomic_write_text(out_path, json.dumps(definition, indent=2, sort_keys=True), fsync=True)
        except OSError as e:
            renderer.error(
                code="distribution_output_write_error",
                message=f"Could not write definition to {out_path}: {e}",
                details={"path": str(out_path), "error": str(e)},
            )
            raise typer.Exit(code=1) from e

    if renderer.is_pretty():
        if base_version:
            renderer.print(f"ComfyUI version: {base_version}")
        else:
            renderer.warn(
                "ComfyUI version not detected — a build requires it. "
                "Re-run with --comfy-version <ref>, or set baseComfyVersion before `create`."
            )
        if provenance:
            e = provenance["environment"]
            torch = f", torch {e['torch']}" if e["torch"] else ""
            renderer.print(f"pip deps: captured from {e['os']}/{e['arch']} (Python {e['pythonVersion']}{torch})")
        if models:
            _render_models_table(renderer, models, total_bytes)
        else:
            renderer.warn(f"No model files found under {root}.")
        if nodes:
            _render_nodes_table(renderer, nodes)
        if output:
            renderer.success(f"Wrote definition → {output}")

    renderer.emit(
        {
            "models_dir": str(root),
            "custom_nodes_dir": str(nodes_root) if nodes_root and nodes_root.is_dir() else None,
            "base_comfy_version": base_version,
            "pip_captured": provenance is not None,
            "environment": provenance["environment"] if provenance else None,
            "count": len(models),
            "custom_node_count": len(nodes),
            "total_bytes": total_bytes,
            "definition": definition,
        },
        command="distribution scan",
        changed=bool(output),
    )


def _load_definition(path: Path, *, require_models: bool = True) -> dict:
    """Read a scan definition JSON file. Raises ValueError on any problem so the
    command can map it to one error envelope. ``require_models`` is relaxed for
    ``update``, where a nodes-only or pins-only edit is legitimate."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"cannot read {path}: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or (require_models and "models" not in data):
        raise ValueError(f"{path} is not a distribution definition (no 'models' key)")
    # Guard the fields the create path indexes, so a hand-edited definition returns
    # an error envelope rather than a bare KeyError traceback downstream.
    for i, m in enumerate(data.get("models") or []):
        if not isinstance(m, dict) or not m.get("type") or not m.get("filename"):
            raise ValueError(f"{path}: models[{i}] needs both 'type' and 'filename'")
    for i, n in enumerate(data.get("customNodes") or []):
        if not isinstance(n, dict) or not n.get("name"):
            raise ValueError(f"{path}: customNodes[{i}] needs a 'name'")
    return data


# Builder base URL: the comfy-builder service behind the Developer Platform
# gateway (the gateway strips the /builder prefix and forwards /v1/* to it).
# Prod issuer is cloud.comfy.org, matching the OAuth login token. Override with
# --builder-url or COMFY_BUILDER_URL (e.g. https://stagingplatformapi.comfy.org/builder).
DEFAULT_BUILDER_URL = "https://platformapi.comfy.org/builder"


@app.command(
    "create",
    help="Create a serverless distribution from a scan definition (preview by default; --execute goes live).",
)
@tracking.track_command("distribution")
def create_cmd(
    from_: Annotated[
        str,
        typer.Option("--from", "-f", help="Path to a definition JSON produced by `comfy distribution scan -o`."),
    ],
    name: Annotated[
        str,
        typer.Option("--name", help="Name for the distribution."),
    ] = "untitled-distribution",
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Go live: upload blobs, create the distribution, and cut a build."),
    ] = False,
    builder_url: Annotated[
        str | None,
        typer.Option("--builder-url", help="Builder base URL. Default: $COMFY_BUILDER_URL, else the placeholder."),
    ] = None,
    models_dir: Annotated[
        str | None,
        typer.Option("--models-dir", help="Models folder, to locate bytes for private-model uploads (--execute)."),
    ] = None,
):
    renderer = get_renderer()

    try:
        definition = _load_definition(Path(from_).expanduser())
    except ValueError as e:
        renderer.error(code="distribution_definition_invalid", message=str(e), details={"path": from_})
        raise typer.Exit(code=1) from e

    # Preflight: the builder allows a version-less definition to be *created* but
    # refuses to *cut a build* from one (it needs a pinned ComfyUI version). create
    # cuts a build, so catch the missing version here with a clear fix rather than
    # letting it surface as a raw builder 400 after work has already happened.
    if not definition.get("baseComfyVersion"):
        renderer.error(
            code="distribution_missing_comfy_version",
            message="the definition has no baseComfyVersion — the builder needs a pinned ComfyUI version to build.",
            hint='re-scan with `--comfy-version <ref>` (a tag, branch, or commit), or add "baseComfyVersion" to the definition',
            details={"path": from_},
        )
        raise typer.Exit(code=1)

    if execute:
        _create_execute(renderer, definition, name=name, builder_url=builder_url, models_dir=models_dir)
        return

    # Preview (default): offline plan, no network — so no resolution, everything
    # shows as an upload (the live --execute path resolves first).
    plan = plan_create(definition)
    resolved = sum(1 for m in plan["definition"]["models"] if m.get("sourceUri"))
    git_nodes = sum(1 for n in plan["definition"]["customNodes"] if n.get("repository"))
    if renderer.is_pretty():
        renderer.info(f"Distribution '{name}' — preview (no calls made)")
        renderer.print(
            f"  models: {len(plan['definition']['models'])}  "
            f"({resolved} resolved to URL, {plan['upload_count']} to upload · {_human_size(plan['upload_bytes'])})"
        )
        renderer.print(
            f"  custom nodes: {len(plan['definition']['customNodes'])}  "
            f"({git_nodes} from git, {len(plan['definition']['customNodes']) - git_nodes} to upload)"
        )
        renderer.print("\n  Builder definition that would be sent:")
        renderer.console().print_json(json.dumps(plan["definition"]))

    renderer.emit({"name": name, "plan": plan, "executed": False}, command="distribution create")


def _create_execute(renderer, definition: dict, *, name: str, builder_url: str | None, models_dir: str | None) -> None:
    """The live --execute path: auth, resolve, upload, create, cut. Maps every
    failure class to a single error envelope + exit(1)."""
    import urllib.error

    client = _builder_client(renderer, builder_url)

    # Resolve models to public URLs where a hash-verified match exists. Best
    # effort: if resolution is unavailable, everything just falls through to
    # upload rather than failing the whole create.
    try:
        n = resolve_models_via_builder(definition.get("models", []), client)
        if renderer.is_pretty() and n:
            renderer.info(f"Resolved {n} model(s) to public URLs — skipping their upload")
    except (urllib.error.URLError, requests.RequestException, KeyError) as e:
        renderer.warn(f"model resolution unavailable ({e}); all models will be uploaded")

    plan = plan_create(definition)

    # Locate bytes for private-model uploads: models/<type>/<filename>. Only
    # needed when something actually uploads.
    mroot = Path(models_dir).expanduser() if models_dir else None

    def locate_bytes(upload: dict) -> Path:
        if mroot is None:
            raise FileNotFoundError("pass --models-dir to locate model bytes for upload")
        p = mroot / upload["type"] / upload["filename"]
        if not p.is_file():
            raise FileNotFoundError(f"model file not found: {p}")
        return p

    try:
        result = execute_create(plan, client=client, name=name, locate_bytes=locate_bytes)
    except NotImplementedError as e:
        renderer.error(code="distribution_upload_unavailable", message=str(e))
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        renderer.error(code="distribution_upload_unavailable", message=str(e))
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, requests.RequestException, KeyError) as e:
        # Same handler as the read verbs: surface the builder's own message (and the
        # limited-beta 403), plus the created distribution's id when a cut failed
        # after create, so the caller can delete or retry it rather than orphan it.
        _report_builder_error(renderer, e, dist_id=getattr(e, "distribution_id", None))
        raise typer.Exit(code=1) from e

    if renderer.is_pretty():
        renderer.success(f"Created distribution '{name}'")
        renderer.print(f"  distribution: {result['distributionId']}")
        renderer.print(f"  version:      {result['versionId']}  (uploaded {result['uploaded']} blob(s))")
        renderer.print(f"  status:       {result['statusUrl']}")
    renderer.emit({"name": name, "executed": True, **result}, command="distribution create", changed=True)


def _builder_client(renderer, builder_url: str | None):
    """Build an authed BuilderClient, or emit a not-signed-in envelope + exit(1).

    A caller that already holds a Cloud JWT — the Developer Platform agent service
    forwarding the request's token, or CI — injects it via ``COMFY_BUILDER_TOKEN``
    and skips the interactive OAuth session ``from_session`` uses. The env var wins
    over a stored session so an explicit token always takes precedence.
    """
    from comfy_cli.distribution_api import BuilderAuthError, BuilderClient

    base_url = builder_url or os.environ.get("COMFY_BUILDER_URL") or DEFAULT_BUILDER_URL
    token = os.environ.get("COMFY_BUILDER_TOKEN")
    if token:
        return BuilderClient(base_url, token)
    try:
        return BuilderClient.from_session(base_url)
    except BuilderAuthError as e:
        renderer.error(code="distribution_not_signed_in", message=str(e))
        raise typer.Exit(code=1) from e


def _report_builder_error(renderer, e, *, dist_id: str | None = None) -> None:
    """Emit one error envelope for a builder failure. Prefers the limited-beta 403,
    then the builder's own error body (e.g. `INVALID_DEFINITION: …` or
    `SUBSCRIPTION_REQUIRED: …`) over urllib's opaque "HTTP Error 400", then the
    generic transport error. Includes a created distribution's id when supplied,
    so a cut that fails after create doesn't orphan an unnamed distribution."""
    import urllib.error

    base = {"distributionId": dist_id} if dist_id else {}
    if isinstance(e, urllib.error.HTTPError):
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        if e.code == 403 and "FEATURE_NOT_ENABLED" in body:
            renderer.error(
                code="distribution_not_enabled",
                message="The developer platform is in limited beta and your account isn't enabled yet.",
                details=base or None,
            )
            return
        detail = _builder_msg(body) or getattr(e, "reason", None) or str(e)
        renderer.error(
            code="distribution_builder_error",
            message=f"builder call failed ({e.code}): {detail}",
            details={**base, "status": e.code, "body": body[:1000]},
        )
        return
    renderer.error(code="distribution_builder_error", message=f"builder call failed: {e}", details=base or None)


def _builder_call(renderer, fn):
    """Run a builder API call, mapping every failure class to one error envelope
    + exit(1) via _report_builder_error."""
    import urllib.error

    from comfy_cli.http import ResponseTooLarge

    try:
        return fn()
    except ResponseTooLarge as e:
        # Mostly the build log outgrowing even the raised logs cap; a clear message
        # beats an unhandled traceback.
        renderer.error(
            code="distribution_builder_error", message=f"builder response exceeded the client size cap ({e})"
        )
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, requests.RequestException, KeyError) as e:
        _report_builder_error(renderer, e)
        raise typer.Exit(code=1) from e


def _builder_msg(body: str) -> str:
    """Pull ``"<error>: <message>"`` out of a builder JSON error body ({error, message}),
    or ``""`` when the body isn't the expected shape."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    err = str(parsed.get("error") or "").strip()
    msg = str(parsed.get("message") or "").strip()
    return f"{err}: {msg}".strip(": ").strip() if (err or msg) else ""


_BUILDER_URL_OPT = typer.Option("--builder-url", help="Builder base URL. Default: $COMFY_BUILDER_URL, else built-in.")


@app.command("list", help="List the workspace's distributions.")
@tracking.track_command("distribution")
def list_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    dists = _builder_call(renderer, client.list_distributions)
    if renderer.is_pretty():
        if not dists:
            renderer.info("No distributions yet.")
        for d in dists:
            renderer.print(f"  {d.get('id', '?')}  {d.get('name', '')}")
    renderer.emit({"distributions": dists}, command="distribution list")


@app.command("get", help="Show a distribution and its full definition.")
@tracking.track_command("distribution")
def get_cmd(
    distribution_id: Annotated[str, typer.Argument(help="Distribution id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    dist = _builder_call(renderer, lambda: client.get_distribution(distribution_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(dist))
    renderer.emit(dist, command="distribution get")


@version_app.command("create", help="Cut a new version (build) of a distribution.")
@tracking.track_command("distribution")
def version_create(
    distribution_id: Annotated[str, typer.Argument(help="Distribution id to build.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    version_id, status_url = _builder_call(renderer, lambda: client.cut_version(distribution_id))
    if renderer.is_pretty():
        renderer.success(f"Cut version {version_id}")
        renderer.print(f"  status: {status_url}")
    renderer.emit(
        {"distributionId": distribution_id, "versionId": version_id, "statusUrl": status_url},
        command="distribution version create",
        changed=True,
    )


@version_app.command("list", help="List a distribution's versions.")
@tracking.track_command("distribution")
def version_list(
    distribution_id: Annotated[str, typer.Argument(help="Distribution id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    versions = _builder_call(renderer, lambda: client.list_distribution_versions(distribution_id))
    if renderer.is_pretty():
        if not versions:
            renderer.info("No versions yet.")
        for v in versions:
            renderer.print(f"  {v.get('id', '?')}  {v.get('status', '')}")
    renderer.emit({"versions": versions}, command="distribution version list")


@version_app.command("get", help="Show a version's build status.")
@tracking.track_command("distribution")
def version_get(
    version_id: Annotated[str, typer.Argument(help="Distribution version id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    version = _builder_call(renderer, lambda: client.get_version(version_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(version))
    renderer.emit(version, command="distribution version get")


@version_app.command("logs", help="Read a version's build log for one target.")
@tracking.track_command("distribution")
def version_logs(
    version_id: Annotated[str, typer.Argument(help="Distribution version id.")],
    os_target: Annotated[
        str | None, typer.Option("--os", help="Target OS to read (linux/windows/mac); builder picks one if omitted.")
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="Target GPU to read (nvidia/amd/cpu/mps).")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    content = _builder_call(renderer, lambda: client.get_version_logs(version_id, os=os_target, gpu=gpu))
    if renderer.is_pretty():
        log = content.get("log", "")
        renderer.print(log if log else "(no build log captured yet)")
        if content.get("truncated"):
            renderer.warn("log truncated (head and tail kept; middle dropped)")
    renderer.emit(content, command="distribution version logs")


@app.command("validate", help="Dry-run resolve a distribution's definition (no build).")
@tracking.track_command("distribution")
def validate_cmd(
    distribution_id: Annotated[str, typer.Argument(help="Distribution id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    # A 400 here means "the definition has problems" — a normal validate outcome,
    # not a transport error; _builder_call surfaces the builder's issue list in the
    # message + details.body so the caller sees exactly what failed to resolve.
    result = _builder_call(renderer, lambda: client.validate_distribution(distribution_id))
    if renderer.is_pretty():
        renderer.success("Definition resolves.")
        renderer.console().print_json(json.dumps(result))
    renderer.emit(result, command="distribution validate")


@app.command("delete", help="Delete a distribution (soft-delete).")
@tracking.track_command("distribution")
def delete_cmd(
    distribution_id: Annotated[str, typer.Argument(help="Distribution id.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    if not yes:
        if renderer.is_pretty():
            if not typer.confirm(f"Delete distribution {distribution_id}?"):
                renderer.info("Aborted.")
                raise typer.Exit(code=0)
        else:
            # No TTY to prompt on (JSON / agent / pipe): refuse rather than block.
            renderer.error(
                code="distribution_delete_needs_confirm",
                message="refusing to delete without confirmation in non-interactive mode.",
                hint="pass --yes to confirm the delete",
                details={"distributionId": distribution_id},
            )
            raise typer.Exit(code=1)
    client = _builder_client(renderer, builder_url)
    _builder_call(renderer, lambda: client.delete_distribution(distribution_id))
    if renderer.is_pretty():
        renderer.success(f"Deleted distribution {distribution_id}")
    renderer.emit({"distributionId": distribution_id, "deleted": True}, command="distribution delete", changed=True)


@app.command("update", help="Replace a distribution's definition from a scan/edited JSON.")
@tracking.track_command("distribution")
def update_cmd(
    distribution_id: Annotated[str, typer.Argument(help="Distribution id.")],
    from_: Annotated[str, typer.Option("--from", "-f", help="Path to a definition JSON.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    try:
        definition = _load_definition(Path(from_).expanduser(), require_models=False)
    except ValueError as e:
        renderer.error(code="distribution_definition_invalid", message=str(e), details={"path": from_})
        raise typer.Exit(code=1) from e
    client = _builder_client(renderer, builder_url)
    # The builder guards the save with the updatedAt the caller last saw (optimistic
    # concurrency, meant for the website's read-edit-save). For a CLI that's just
    # last-writer-wins: fetch the current updatedAt and echo it back, else the save
    # 409s STALE on a missing/zero timestamp.
    current = _builder_call(renderer, lambda: client.get_distribution(distribution_id))
    dist = _builder_call(
        renderer,
        lambda: client.update_distribution(distribution_id, definition, current.get("updatedAt")),
    )
    if renderer.is_pretty():
        renderer.success(f"Updated distribution {distribution_id}")
    renderer.emit(dist, command="distribution update", changed=True)


@app.command("resolve", help="Resolve model filenames to public download candidates (HF/CivitAI).")
@tracking.track_command("distribution")
def resolve_cmd(
    filenames: Annotated[list[str], typer.Argument(help="Model filenames to resolve.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    # The builder caps /v1/models/resolve at 32 filenames per call; batch so a large
    # argument list doesn't 400.
    results = _builder_call(
        renderer,
        lambda: [r for batch in _chunks(list(filenames), _RESOLVE_BATCH) for r in client.resolve_models(batch)],
    )
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(results))
    renderer.emit({"results": results}, command="distribution resolve")


@app.command("base-images", help="List the curated base images a distribution can build on.")
@tracking.track_command("distribution")
def base_images_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    images = _builder_call(renderer, client.list_base_images)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(images))
    renderer.emit({"baseImages": images}, command="distribution base-images")


@app.command("build-targets", help="List the build targets (os/gpu) a version can be cut for.")
@tracking.track_command("distribution")
def build_targets_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    targets = _builder_call(renderer, client.list_build_targets)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(targets))
    renderer.emit({"targets": targets}, command="distribution build-targets")


@app.command("model-dirs", help="List the model directories a model may be placed in.")
@tracking.track_command("distribution")
def model_dirs_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    dirs = _builder_call(renderer, client.list_model_directories)
    if renderer.is_pretty():
        for d in dirs:
            renderer.print(f"  {d}")
    renderer.emit({"directories": dirs}, command="distribution model-dirs")


@version_app.command("manifest", help="Show a version's models and runtime policies.")
@tracking.track_command("distribution")
def version_manifest(
    version_id: Annotated[str, typer.Argument(help="Distribution version id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    manifest = _builder_call(renderer, lambda: client.get_version_manifest(version_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(manifest))
    renderer.emit(manifest, command="distribution version manifest")


@artifact_app.command("download", help="Get a signed download link for a build artifact.")
@tracking.track_command("distribution")
def artifact_download(
    artifact_id: Annotated[str, typer.Argument(help="Build artifact id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    signed = _builder_call(renderer, lambda: client.get_artifact_download(artifact_id))
    if renderer.is_pretty():
        renderer.print(signed.get("downloadUrl", "(no url)"))
    renderer.emit(signed, command="distribution artifact download")


@blob_app.command("list", help="List the workspace's private blobs.")
@tracking.track_command("distribution")
def blob_list(
    kind: Annotated[str | None, typer.Option("--kind", help="Filter by blob kind: model or node_zip.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    blobs = _builder_call(renderer, lambda: client.list_blobs(kind))
    if renderer.is_pretty():
        if not blobs:
            renderer.info("No blobs.")
        for b in blobs:
            renderer.print(f"  {b.get('blobId', b.get('id', '?'))}  {b.get('filename', '')}")
    renderer.emit({"blobs": blobs}, command="distribution blob list")
