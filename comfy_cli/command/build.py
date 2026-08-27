"""``comfy build`` — package a local ComfyUI environment into a
serverless build.

POC scope: ``scan`` inspects a local ComfyUI install and emits a builder-ready
*definition* covering the two halves a build needs:

- **models** — walk the ``models/`` tree and hash every model file, producing
  ``{type, filename, sha256, sizeBytes}``. This closes the model-path resolution
  gap: a workflow only carries a bare model name (path stripped, no hash), but a
  build needs the placement folder, filename, and a content hash to fetch and
  verify the right file.
- **custom nodes** — walk ``custom_nodes/`` and record each node's most precise
  source: a git ``repository`` + commit ``gitRef`` (the builder fetches
  ``repo@ref``), else the ``id`` + ``registryVersion`` its ``pyproject.toml``
  declares (the builder fetches the published artifact). Nodes with neither are
  marked ``source: local`` — they must be uploaded as a blob.

The scan runs entirely on the user's machine — the one place the real files and
node checkouts exist — and is the agreed compatibility shim for users who won't
upgrade desktop. The emitted definition maps 1:1 onto the builder's
``definition.models[]`` / ``definition.customNodes[]``; a later ``create``
command uploads private blobs and cuts a version from it.

Surface module (Typer + renderer) in the ``command/project.py`` style; the
scan/hash logic is kept in pure helpers so it stays unit-testable.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
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
from comfy_cli.command.pack_scan import read_pyproject
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH, SUPPORTED_PT_EXTENSIONS
from comfy_cli.file_utils import atomic_write_text
from comfy_cli.output import get_renderer
from comfy_cli.registry.api import sanitize_error_body
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Package a local ComfyUI environment into a serverless build. "
        "[Limited beta] — builder access is granted per account; commands that reach the "
        "builder return a 'not enabled yet' message until your account is enabled."
    ),
)

# `comfy build release <verb>`: the resource-verb surface for releases
# (a release is an immutable cut of a build; the builder's public
# API renamed build versions to releases). The module keeps the version_app
# name so internal references stay put.
version_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and cut build releases.",
)
app.add_typer(version_app, name="release")


def _warn_version_deprecated() -> None:
    """One stderr warning per invocation of the retired `comfy build version`
    spelling, same wiring as cmdline.py's _add_deprecated_alias. No envelope
    re-stamp is needed here: the root callback stamps only the top-level group
    name ("build", identical for both spellings), and every command in this
    group passes its canonical `build release ...` label to emit explicitly."""
    renderer = get_renderer()
    renderer.stderr_console().print(
        "[yellow]`comfy build version` is deprecated; use `comfy build release` instead.[/yellow]"
    )


# `comfy build version` was the subgroup's name before the builder's public API
# renamed build versions to releases; kept as a warning alias for old scripts.
app.add_typer(version_app, name="version", hidden=True, callback=_warn_version_deprecated)

# `comfy build artifact <verb>` / `blob <verb>` — the per-target build
# outputs and the workspace's private uploaded content.
artifact_app = typer.Typer(no_args_is_help=True, help="Build artifacts (per-target outputs of a cut release).")
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


# What the two consumers of a pin accept. The builder rejects a registryVersion
# that is not a package version (definition.go), and a node id is a slug: reading
# either straight out of a pyproject and sending it on lets a pack put arbitrary
# text into a registry URL and into the definition.
_REGISTRY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_REGISTRY_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _read_registry_pin(node_dir: Path) -> tuple[str, str] | None:
    """Return the ``(id, version)`` a pack's ``pyproject.toml`` CLAIMS, or None.

    A pack installed by ``comfy node install`` comes from the Comfy Registry as an
    archive, so it has no git history at all — but its ``pyproject.toml`` still
    carries the two fields that name a published version: ``project.name`` is the
    registry node id and ``project.version`` its package version.

    This is the pack's own claim about itself, not evidence: the file is written by
    whoever wrote the pack, so a pin from here is unverified until the registry is
    asked (see :func:`verify_registry_pins`). Silent on every failure — absent,
    unreadable, malformed, missing either field, or a value neither the registry
    nor the builder would accept — because a pack that cannot answer simply has no
    registry pin, and the scan must not die on one bad neighbour."""
    path = node_dir / "pyproject.toml"
    # Guarded rather than caught: the shared parser reports an absent file to the
    # user, and most packs in a scan do not have one.
    if not path.is_file():
        return None
    try:
        # The repo's one pack-pyproject parser, which also resolves a PEP 621
        # `dynamic = ["version"]`. Muted: it writes advice for someone publishing
        # a pack, which is noise when reading somebody else's install.
        with contextlib.redirect_stderr(io.StringIO()):
            config = read_pyproject(str(path))
    except Exception:
        return None
    if config is None:
        return None
    node_id, version = config.project.name, config.project.version
    if not isinstance(node_id, str) or not isinstance(version, str):
        return None
    node_id, version = node_id.strip(), version.strip()
    if not _REGISTRY_ID_RE.match(node_id) or not _REGISTRY_VERSION_RE.match(version):
        return None
    return node_id, version


def _identify_node(node_dir: Path) -> dict:
    """One definition entry for one pack, naming the single source it has."""
    repository = git_ref = None
    if (node_dir / ".git").exists():
        repository = _git_output(node_dir, "remote", "get-url", "origin")
        git_ref = _git_output(node_dir, "rev-parse", "HEAD")
    # A repo@ref reconstructs an exact commit, so it wins where both exist.
    if repository and git_ref:
        return {"name": node_dir.name, "repository": repository, "gitRef": git_ref, "source": "git"}
    pin = _read_registry_pin(node_dir)
    if pin:
        # The builder takes exactly one source, so a half-git checkout keeps none
        # of its git fields: an origin without a resolvable HEAD is not fetchable.
        return {"name": node_dir.name, "id": pin[0], "registryVersion": pin[1], "source": "registry"}
    return {"name": node_dir.name, "repository": repository, "gitRef": git_ref, "source": "local"}


def scan_custom_nodes(custom_nodes_root: Path) -> list[dict]:
    """Return one definition entry per custom-node directory, sorted by name.

    Each top-level directory under ``custom_nodes/`` is one node, recorded as the
    most precise source the builder can rebuild it from:

    - a git checkout with a fetchable origin → ``repository`` + ``gitRef`` (HEAD),
      which pins an exact commit;
    - otherwise a published registry version → ``id`` + ``registryVersion`` read
      from its ``pyproject.toml``, which pins an exact artifact. This is the case
      for everything ``comfy node install`` writes, since it unpacks archives
      rather than cloning;
    - otherwise ``source: local`` — no upstream, so it must be uploaded as a blob.

    Returns an empty list when the folder is absent (a workflow may use no custom
    nodes)."""
    nodes: list[dict] = []
    if not custom_nodes_root.is_dir():
        return nodes
    for entry in sorted(custom_nodes_root.iterdir()):
        if not entry.is_dir():
            continue  # single-file .py nodes aren't buildable units here
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        nodes.append(_identify_node(entry))
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


# A bare release number, as `comfyui_version.py` and `/system_stats` report it.
_BARE_RELEASE_RE = re.compile(r"^\d+(?:\.\d+)+$")


def as_comfy_git_ref(version: str) -> str:
    """Return ``version`` as a ref the builder can actually resolve.

    ``baseComfyVersion`` is resolved with ``git ls-remote`` against upstream
    ComfyUI, which tags its releases ``vX.Y.Z`` — but every source we detect the
    version from reports the bare number, so the value we recorded was guaranteed
    to miss, and a build was the first thing to say so. Only a bare release number
    is rewritten; a branch, a commit sha, a describe string or an
    already-prefixed tag is left exactly as given."""
    v = version.strip()
    return "v" + v if _BARE_RELEASE_RE.match(v) else v


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


# userinfo in a direct-reference URL, e.g.
# `pkg @ git+https://x-access-token:ghp_xxx@github.com/org/private.git@sha`.
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<userinfo>[^/@\s]+)@")


def _redact_freeze_credentials(freeze: str) -> str:
    """Strip userinfo from any URL in a freeze.

    A freeze can carry direct references to private repositories, and pip writes
    those with whatever credential was used to install them. The definition is
    written to disk, POSTed to the builder, and often committed, so a token in it
    travels a long way from the machine that minted it."""
    return _URL_USERINFO_RE.sub(lambda m: m.group("scheme") + "***@", freeze)


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
    # PYTHONPATH would put whatever it points at into the freeze as an installed
    # package, so a caller with one set gets phantom pins the builder then tries
    # to resolve. The freeze must describe the target env and nothing else.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for cmd in attempts:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        except (subprocess.SubprocessError, OSError):
            continue
        if r.returncode == 0 and r.stdout.strip():
            return _redact_freeze_credentials(r.stdout)
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


def snapshot_from_definition(definition: dict) -> dict:
    """Wrap a scanned definition as the snapshot envelope the builder reads.

    The builder's importer is the one place that knows what the Comfy Registry
    actually publishes, which curated base image a Python fits, and how a pin
    normalizes. It takes a Desktop export, and a scan collects the same facts
    under different names, so the scan is translated rather than that knowledge
    being reimplemented here and drifting from it.

    Models have no place in a snapshot and stay with the caller."""
    nodes = []
    for n in definition.get("customNodes", []):
        name = n.get("name") or ""
        if n.get("registryVersion"):
            nodes.append(
                {
                    "type": "cnr",
                    "id": n.get("id") or name,
                    "dirName": name,
                    "version": n["registryVersion"],
                    "enabled": True,
                }
            )
        elif n.get("repository"):
            node = {"type": "git", "id": name, "dirName": name, "url": n["repository"], "enabled": True}
            if n.get("gitRef"):
                node["commit"] = n["gitRef"]
            nodes.append(node)

    # A freeze is one pin per line; a snapshot is a name -> version map. Anything
    # that is not a plain `==` pin (a comment, a direct git reference, an extras
    # marker) has no place in that map, and the importer reports what it drops.
    pips: dict[str, str] = {}
    for line in (definition.get("pipDependencies") or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        pip_name, _, pip_version = line.partition("==")
        if pip_name.strip() and pip_version.strip():
            pips[pip_name.strip()] = pip_version.strip()

    environment = definition.get("environment") or {}
    return {
        "type": "comfyui-desktop-2-snapshot",
        "version": 2,
        "snapshots": [
            {
                "comfyui": {"baseTag": definition.get("baseComfyVersion") or ""},
                "customNodes": nodes,
                "pipPackages": pips,
                "pythonVersion": environment.get("pythonVersion") or "",
            }
        ],
    }


# Each entry is (report key, what it means to the reader).
_REPORT_ADVISORIES = (
    ("notInRegistry", "pinned to something the Comfy Registry does not publish"),
    ("collidingNodes", "left out: another pack already claimed the folder they install into"),
    ("registryPending", "published but not yet servable; the build will wait or fail on it"),
    ("unresolvedNodes", "named nothing the builder can install from"),
    ("unverifiedPins", "left unchecked because the registry did not answer"),
    ("skippedPins", "dropped: the build owns these packages"),
    ("unpinnablePins", "dropped: not a public PyPI release"),
    ("unresolvedClasses", "node classes nothing installable provides; the graph will not run without them"),
    ("uncheckedClasses", "node classes the registry never answered for, so the build may not carry them"),
    (
        "packsWithoutVersion",
        "packs the build fetches from their repository, because the registry publishes no version of them",
    ),
    ("collidingPacks", "packs the build leaves out, because another pack already claimed their install folder"),
)

# How many names an advisory line prints before it says how many it held back.
_ADVISORY_NAMES = 8


def _from_server(value) -> str:
    """Scrub one builder-supplied fragment. Class names and filenames travel to
    the builder from the workflow file and come back in the report, so a crafted
    file could otherwise forge terminal lines that read as the CLI's own."""
    return sanitize_error_body(str(value))


def _advisory_line(count: int, meaning: str, names: list[str]) -> str:
    """Count first, then the names, then how many names the line held back."""
    shown = names[:_ADVISORY_NAMES]
    held_back = f" (+{count - len(shown)} more)" if count > len(shown) else ""
    return f"{count} {meaning}: {', '.join(shown)}{held_back}"


def _unrenderable(key: str, value) -> str:
    """The builder sent a key this renderer knows, in a shape it cannot read.
    Say so: dropping it silently is how a partial import comes to look clean."""
    return f"the builder sent `{key}` as {type(value).__name__}, which this CLI cannot render; read it with --json"


def _best_suggestion(entry: dict, field: str) -> str:
    """The highest-scored suggestion's ``field``, or "" when none carries one.
    The catalog ranks what it thinks the workflow meant, so the reader gets the
    lead rather than the whole list."""
    ranked = [s for s in (entry.get("suggestions") or []) if isinstance(s, dict) and s.get(field)]
    if not ranked:
        return ""
    best = max(ranked, key=lambda s: s["score"] if isinstance(s.get("score"), int | float) else 0.0)
    return _from_server(best[field])


def _suggested_pack_lines(entries: list) -> list[str]:
    """`unknownClasses` is the detailed form of `unresolvedClasses`, which already
    prints the names. What it adds is the pack the registry came closest to, so
    that is all this renders, and only for the classes that carry one."""
    suggested = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pack = _best_suggestion(entry, "packId")
        if pack:
            suggested.append(f"{_from_server(entry.get('classType'))} (maybe {pack})")
    if not suggested:
        return []
    meaning = "node classes the registry could not attribute, with the closest pack it named"
    return [_advisory_line(len(suggested), meaning, suggested)]


def _model_lines(entries: list) -> list[str]:
    """A workflow import builds custom nodes and no models, because a workflow
    names a model without saying where it comes from. So every model the graph
    loads is still owed, and ``status`` shapes the wording rather than deciding
    whether a line exists: the shared catalog already holds a matched one, which
    needs only a source pointer, while the rest have to be found first."""
    held, owed = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            owed.append(_from_server(entry))
            continue
        name = _from_server(entry.get("filename"))
        if entry.get("status") == "matched":
            held.append(name)
            continue
        lead = _best_suggestion(entry, "filename")
        owed.append(f"{name} (maybe {lead})" if lead else name)
    lines = []
    if held:
        meaning = (
            "models the shared catalog holds that this build does not carry, each needing a sourceUri in the "
            "definition before you cut"
        )
        lines.append(_advisory_line(len(held), meaning, held))
    if owed:
        meaning = "models the graph loads that nothing has a source for; `comfy build resolve` finds candidates"
        lines.append(_advisory_line(len(owed), meaning, owed))
    return lines


# Each entry is (report key, what turns its entries into lines).
_REPORT_ENTRY_ADVISORIES = (
    ("unknownClasses", _suggested_pack_lines),
    ("models", _model_lines),
)


def report_advisories(report: dict) -> list[str]:
    """One line per thing the importer could not carry, in reader's terms."""
    lines = []
    # Present and false is the finding; absent means the importer did not say.
    if report.get("pythonSatisfied") is False:
        lines.append(
            "no curated base image matches the scanned Python exactly; the build runs on the closest one, "
            "so a pin resolved against your Python may not resolve against the build's"
        )
    # The release the importer refused, so the empty version field has a reason.
    dropped = report.get("droppedComfyVersion")
    if dropped:
        lines.append(f"the ComfyUI release {str(dropped)!r} is not a ref the build can use, so none was set")
    if report.get("pinnedToLatest") is True:
        lines.append(
            "the importer pinned every pack the workflow named without a version to the registry's newest "
            "published one, so importing the same file later can build something different"
        )
    for key, meaning in _REPORT_ADVISORIES:
        entries = report.get(key)
        if not entries:
            continue
        # A scalar here would otherwise render one line per character.
        if not isinstance(entries, list | tuple):
            lines.append(_unrenderable(key, entries))
            continue
        lines.append(_advisory_line(len(entries), meaning, [_from_server(e) for e in entries]))
    for key, render in _REPORT_ENTRY_ADVISORIES:
        entries = report.get(key)
        if not entries:
            continue
        if not isinstance(entries, list | tuple):
            lines.append(_unrenderable(key, entries))
            continue
        lines.extend(render(entries))
    # A mapping of class name -> provider, so each name carries who serves it.
    partners = report.get("partnerClasses")
    if partners and not isinstance(partners, dict):
        lines.append(_unrenderable("partnerClasses", partners))
    elif partners:
        served = [f"{_from_server(cls)} ({_from_server(provider)})" for cls, provider in partners.items()]
        meaning = "node classes call a partner API rather than run from an installed pack"
        lines.append(_advisory_line(len(partners), meaning, served))
    return lines


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


def _is_scan_shaped(definition: dict) -> bool:
    """Whether this definition still carries the scan's own vocabulary.

    A definition round-tripped out of the builder already names sources the way
    the builder does; one straight from `scan` does not. The tell is a member that
    names no builder source at all, which the builder would reject."""
    for m in definition.get("models", []):
        if not m.get("sourceUri") and not m.get("blobId"):
            return True
    for n in definition.get("customNodes", []):
        if not n.get("repository") and not n.get("registryVersion") and not n.get("blobId"):
            return True
    return False


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
        # A blob the caller already holds is a source, so it is neither resolved
        # nor uploaded again: re-uploading would orphan the bytes it replaced.
        if m.get("blobId"):
            entry["blobId"] = m["blobId"]
        elif uri:
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
        # `id` is not a source: the builder uses it to name the install directory,
        # so it rides along whatever the source turns out to be.
        if n.get("id"):
            entry["id"] = n["id"]
        # Most precise first, and keyed on the fields rather than the scan's
        # `source` tag, so a hand-written definition is read the same way.
        if n.get("repository"):
            entry["repository"] = n["repository"]
            # The website emits repository+commit and no gitRef, so requiring
            # gitRef here would send those nodes to be uploaded instead.
            for key in ("gitRef", "commit"):
                if n.get(key):
                    entry[key] = n[key]
        elif n.get("registryVersion"):
            # A hand-written node often names the pack once. scan writes id and
            # name the same for a registry pack, and the builder prefers id then
            # name, so the name is the id when no other is given.
            entry["id"] = n.get("id") or n["name"]
            entry["registryVersion"] = n["registryVersion"]
        elif n.get("blobId"):
            entry["blobId"] = n["blobId"]
        else:
            uploads.append({"kind": "node_zip", "name": n["name"], "sizeBytes": 0, "slot": ["customNodes", j]})
        nodes_out.append(entry)

    builder_definition: dict = {"models": models_out, "customNodes": nodes_out}
    # Every key the builder's Definition accepts, which is an allowlist rather than
    # a passthrough: `schema` and `environment` are the scan's own bookkeeping and
    # stay local. Dropping one of these does not fail loudly — the builder applies
    # its own default, so a base image silently becomes the catalog's current one.
    for key in ("baseComfyVersion", "baseImage", "pipDependencies"):
        if definition.get(key):
            builder_definition[key] = definition[key]
    # A policy is permission-bearing, so absent and empty must not be the same: an
    # absent policy seals as allow-all, while `{}` is something the user wrote.
    for key in ("modelPolicy", "partnerNodePolicy"):
        if definition.get(key) is not None:
            builder_definition[key] = definition[key]
    # The ref the builder resolves, not the number every version source reports.
    # Applied here rather than only in `scan` so a definition written before this
    # existed does not spend its cut discovering the difference.
    if builder_definition.get("baseComfyVersion"):
        builder_definition["baseComfyVersion"] = as_comfy_git_ref(builder_definition["baseComfyVersion"])
    return {
        "definition": builder_definition,
        "uploads": uploads,
        "upload_count": len(uploads),
        "upload_bytes": sum(u.get("sizeBytes", 0) for u in uploads),
    }


def _id_keys(*, build: str | None = None, release: str | None = None) -> dict[str, str]:
    """Both spellings of an id. `buildId` and `releaseId` are canonical;
    `distributionId` and `versionId` go after one release, once pinned scripts have moved."""
    keys: dict[str, str] = {}
    if build is not None:
        keys["buildId"] = keys["distributionId"] = build
    if release is not None:
        keys["releaseId"] = keys["versionId"] = release
    return keys


def execute_create(plan: dict, *, client, name: str, locate_bytes, targets=None) -> dict:
    """Run the live create: upload each private blob, stitch its blobId into the
    definition, create the build, and cut a release.

    ``client`` is a BuilderClient (or a stand-in with the same methods).
    ``locate_bytes(upload) -> Path`` finds the local file for a model upload.
    node_zip uploads (hand-dropped local nodes) aren't packaged yet — raises
    NotImplementedError so the caller can surface a clear message. Returns
    ``{buildId, distributionId, releaseId, versionId, statusUrl, uploaded}``."""
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
    build_id = client.create_build(name, definition)
    try:
        version_id, status_url = client.cut_version(build_id, targets)
    except Exception as e:
        # The build now exists server-side; carry its id on the error so the
        # command can surface it (the user can then delete or retry it, not orphan it).
        e.build_id = build_id
        raise
    return {
        **_id_keys(build=build_id, release=version_id),
        "statusUrl": status_url,
        "uploaded": uploaded,
    }


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
    table.add_column("from", style="white")
    table.add_column("pinned to", style="dim")
    for n in nodes:
        # One row per node whatever its source, so a registry pin is as visible as
        # a git one rather than rendering as two dashes.
        if n.get("registryVersion"):
            origin, pin = n.get("id") or "—", n["registryVersion"]
        else:
            origin = n.get("repository") or "—"
            pin = (n["gitRef"][:12] + "…") if n.get("gitRef") else "—"
        table.add_row(n["name"], n["source"], origin, pin)
    renderer.console().print(table)


@app.command(
    "scan",
    help="Scan a local ComfyUI install (models + custom nodes) and emit a builder-ready definition.",
)
@tracking.track_command("build")
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
            code="build_models_dir_missing",
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
    # Whatever named it, the definition records a ref the builder can resolve.
    if base_version:
        base_version = as_comfy_git_ref(base_version)

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
                code="build_output_write_error",
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
        command="build scan",
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
        raise ValueError(f"{path} is not a build definition (no 'models' key)")
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
    help="Create a serverless build from a scan definition (preview by default; --execute goes live).",
)
@tracking.track_command("build")
def create_cmd(
    from_: Annotated[
        str,
        typer.Option("--from", "-f", help="Path to a definition JSON produced by `comfy build scan -o`."),
    ],
    name: Annotated[
        str,
        typer.Option("--name", help="Name for the build."),
    ] = "untitled-build",
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Go live: upload blobs, create the build, and cut a release."),
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
        renderer.error(code="build_definition_invalid", message=str(e), details={"path": from_})
        raise typer.Exit(code=1) from e

    # Preflight: the builder allows a version-less definition to be *created* but
    # refuses to *cut a build* from one (it needs a pinned ComfyUI version). create
    # cuts a build, so catch the missing version here with a clear fix rather than
    # letting it surface as a raw builder 400 after work has already happened.
    if not definition.get("baseComfyVersion"):
        renderer.error(
            code="build_missing_comfy_version",
            message="the definition has no baseComfyVersion — the builder needs a pinned ComfyUI version to build.",
            hint='re-scan with `--comfy-version <ref>` (a tag, branch, or commit), or add "baseComfyVersion" to the definition',
            details={"path": from_},
        )
        raise typer.Exit(code=1)

    if execute:
        _create_execute(
            renderer,
            definition,
            name=name,
            builder_url=builder_url,
            models_dir=models_dir,
        )
        return

    # Preview (default): offline plan, no network — so no resolution, everything
    # shows as an upload (the live --execute path resolves first).
    plan = plan_create(definition)
    resolved = sum(1 for m in plan["definition"]["models"] if m.get("sourceUri"))
    nodes_out = plan["definition"]["customNodes"]
    git_nodes = sum(1 for n in nodes_out if n.get("repository"))
    registry_nodes = sum(1 for n in nodes_out if n.get("registryVersion"))
    # From the plan, not from the definition: a registry-pinned node has no
    # repository, and counting it as an upload overstates the one number a user
    # reads before deciding to spend.
    node_uploads = sum(1 for u in plan["uploads"] if u["kind"] == "node_zip")
    model_uploads = plan["upload_count"] - node_uploads
    if renderer.is_pretty():
        renderer.info(f"Build '{name}' — preview (no calls made)")
        renderer.print(
            f"  models: {len(plan['definition']['models'])}  "
            f"({resolved} resolved to URL, {model_uploads} to upload · {_human_size(plan['upload_bytes'])})"
        )
        renderer.print(
            f"  custom nodes: {len(nodes_out)}  "
            f"({git_nodes} from git, {registry_nodes} from the registry, {node_uploads} to upload)"
        )
        renderer.print("\n  Builder definition that would be sent:")
        renderer.console().print_json(json.dumps(plan["definition"]))

    renderer.emit({"name": name, "plan": plan, "executed": False}, command="build create")


def _create_execute(
    renderer,
    definition: dict,
    *,
    name: str,
    builder_url: str | None,
    models_dir: str | None,
) -> None:
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

    # The importer is the side of the boundary that knows what the registry
    # publishes. Best effort: a builder without it leaves the pins to the cut.
    declared = [n.get("name") for n in definition.get("customNodes", [])]
    try:
        imported = client.resolve_snapshot(snapshot_from_definition(definition))
    except (urllib.error.URLError, requests.RequestException, KeyError, ValueError) as e:
        renderer.warn(f"the builder could not read the definition's pins ({e}); they go to the cut unchecked")
    else:
        imported_report = imported.get("report") or {}
        for line in report_advisories(imported_report):
            renderer.warn(line)
        checked = (imported.get("definition") or {}).get("customNodes")
        if checked is not None:
            # A pack it could not vouch for is absent from what it returns, and
            # an image quietly missing a pack is worse than a refused create.
            # Matched case-insensitively: the importer derives a pack's folder by
            # lowercasing, so a name it chose to normalise must not read as a pack
            # it dropped, which would refuse a create the builder was happy with.
            def _key(name):
                return (name or "").strip().casefold()

            kept = {_key(n.get("name")) for n in checked}
            # A collision is the importer resolving a conflict the cut would refuse
            # outright, so its definition is the buildable one and stopping would
            # help nobody. Every other absence means the user asked for something
            # that does not exist, which only they can fix.
            colliding = {_key(n) for n in imported_report.get("collidingNodes") or []}
            missing = [n for n in declared if _key(n) not in kept and _key(n) not in colliding]
            if missing:
                renderer.error(
                    code="build_registry_pin_missing",
                    message="the builder could not resolve these custom nodes: " + ", ".join(sorted(missing)),
                    hint="see the advisories above; edit the definition to name a published version, or remove the pack",
                )
                raise typer.Exit(code=1)
            definition["customNodes"] = checked

    plan = plan_create(definition)

    # An absent policy seals as allow-all. Said rather than written on the user's
    # behalf, which would pin a posture nobody chose if the default ever moves.
    missing = [k for k in ("modelPolicy", "partnerNodePolicy") if definition.get(k) is None]
    if missing:
        renderer.warn(
            f"no {' or '.join(missing)} set; this version will be sealed permitting everything in those categories"
        )

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
        renderer.error(code="build_upload_unavailable", message=str(e))
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        renderer.error(code="build_upload_unavailable", message=str(e))
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, requests.RequestException, KeyError) as e:
        # Same handler as the read verbs: surface the builder's own message (and the
        # limited-beta 403), plus the created build's id when a cut failed
        # after create, so the caller can delete or retry it rather than orphan it.
        _report_builder_error(renderer, e, build_id=getattr(e, "build_id", None))
        raise typer.Exit(code=1) from e

    if renderer.is_pretty():
        renderer.success(f"Created build '{name}'")
        renderer.print(f"  build:   {result['buildId']}")
        renderer.print(f"  release: {result['releaseId']}  (uploaded {result['uploaded']} blob(s))")
        renderer.print(f"  status:  {result['statusUrl']}")
    renderer.emit({"name": name, "executed": True, **result}, command="build create", changed=True)


def _builder_client(renderer, builder_url: str | None):
    """Build an authed BuilderClient, or emit a not-signed-in envelope + exit(1).

    A caller that already holds a Cloud JWT — the Developer Platform agent service
    forwarding the request's token, or CI — injects it via ``COMFY_BUILDER_TOKEN``
    and skips the interactive OAuth session ``from_session`` uses. The env var wins
    over a stored session so an explicit token always takes precedence.
    """
    from comfy_cli.builder_api import BuilderAuthError, BuilderClient

    base_url = builder_url or os.environ.get("COMFY_BUILDER_URL") or DEFAULT_BUILDER_URL
    token = os.environ.get("COMFY_BUILDER_TOKEN")
    if token:
        return BuilderClient(base_url, token)
    try:
        return BuilderClient.from_session(base_url)
    except BuilderAuthError as e:
        renderer.error(code="build_not_signed_in", message=str(e))
        raise typer.Exit(code=1) from e


def _report_builder_error(renderer, e, *, build_id: str | None = None) -> None:
    """Emit one error envelope for a builder failure. Prefers the limited-beta 403,
    then the builder's own error body (e.g. `INVALID_DEFINITION: …` or
    `SUBSCRIPTION_REQUIRED: …`) over urllib's opaque "HTTP Error 400", then the
    generic transport error. Includes a created build's id when supplied,
    so a cut that fails after create doesn't orphan an unnamed build."""
    import urllib.error

    base = _id_keys(build=build_id)
    if isinstance(e, urllib.error.HTTPError):
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        if e.code == 403 and "FEATURE_NOT_ENABLED" in body:
            renderer.error(
                code="build_not_enabled",
                message="The developer platform is in limited beta and your account isn't enabled yet.",
                details=base or None,
            )
            return
        detail = _builder_msg(body) or getattr(e, "reason", None) or str(e)
        renderer.error(
            code="build_builder_error",
            message=f"builder call failed ({e.code}): {detail}",
            details={**base, "status": e.code, "body": body[:1000]},
        )
        return
    renderer.error(code="build_builder_error", message=f"builder call failed: {e}", details=base or None)


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
        renderer.error(code="build_builder_error", message=f"builder response exceeded the client size cap ({e})")
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


@app.command("list", help="List the workspace's builds.")
@tracking.track_command("build")
def list_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    dists = _builder_call(renderer, client.list_builds)
    if renderer.is_pretty():
        if not dists:
            renderer.info("No builds yet.")
        for d in dists:
            renderer.print(f"  {d.get('id', '?')}  {d.get('name', '')}")
    # `builds` is canonical; `distributions` carries the same rows and goes after one release.
    renderer.emit({"builds": dists, "distributions": dists}, command="build list")


@app.command("get", help="Show a build and its full definition.")
@tracking.track_command("build")
def get_cmd(
    build_id: Annotated[str, typer.Argument(help="Build id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    dist = _builder_call(renderer, lambda: client.get_build(build_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(dist))
    renderer.emit(dist, command="build get")


@version_app.command("create", help="Cut a new release of a build.")
@tracking.track_command("build")
def version_create(
    build_id: Annotated[str, typer.Argument(help="Build id to cut a release of.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    release_id, status_url = _builder_call(renderer, lambda: client.cut_version(build_id))
    if renderer.is_pretty():
        renderer.success(f"Cut release {release_id}")
        renderer.print(f"  status: {status_url}")
    renderer.emit(
        {**_id_keys(build=build_id, release=release_id), "statusUrl": status_url},
        command="build release create",
        changed=True,
    )


@version_app.command("list", help="List a build's releases.")
@tracking.track_command("build")
def version_list(
    build_id: Annotated[str, typer.Argument(help="Build id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    versions = _builder_call(renderer, lambda: client.list_releases(build_id))
    if renderer.is_pretty():
        if not versions:
            renderer.info("No releases yet.")
        for v in versions:
            renderer.print(f"  {v.get('id', '?')}  {v.get('status', '')}")
    renderer.emit({"releases": versions, "versions": versions}, command="build release list")


@version_app.command("get", help="Show a release's build status.")
@tracking.track_command("build")
def version_get(
    version_id: Annotated[str, typer.Argument(help="Build release id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    version = _builder_call(renderer, lambda: client.get_version(version_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(version))
    renderer.emit(version, command="build release get")


@version_app.command("logs", help="Read a release's build log for one target.")
@tracking.track_command("build")
def version_logs(
    version_id: Annotated[str, typer.Argument(help="Build release id.")],
    os_target: Annotated[
        str | None, typer.Option("--os", help="Target OS to read (linux/windows/mac); builder picks one if omitted.")
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="Target GPU to read (nvidia/amd/cpu/mps).")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    content = _builder_call(renderer, lambda: client.get_version_logs(version_id, os=os_target, gpu=gpu))
    # Mirror whichever spelling the builder returned, so either server generation
    # yields both keys.
    log_id = content.get("releaseId") or content.get("versionId")
    if log_id:
        content = {**content, **_id_keys(release=log_id)}
    if renderer.is_pretty():
        log = content.get("log", "")
        renderer.print(log if log else "(no build log captured yet)")
        if content.get("truncated"):
            renderer.warn("log truncated (head and tail kept; middle dropped)")
    renderer.emit(content, command="build release logs")


@app.command("validate", help="Dry-run resolve a build's definition (nothing is built).")
@tracking.track_command("build")
def validate_cmd(
    build_id: Annotated[str, typer.Argument(help="Build id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    # A 400 here means "the definition has problems" — a normal validate outcome,
    # not a transport error; _builder_call surfaces the builder's issue list in the
    # message + details.body so the caller sees exactly what failed to resolve.
    result = _builder_call(renderer, lambda: client.validate_build(build_id))
    if renderer.is_pretty():
        # The endpoint checks shape and pin existence. Whether the set installs
        # together is answered by a build, so the line claims only what it did.
        renderer.success("Definition is valid: shape and pin existence checked, not a full resolve.")
        # Non-blocking here and fatal at freeze, so they are printed rather than
        # left to be found in the JSON below. The cut itself still passes.
        warnings = result.get("warnings") if isinstance(result, dict) else None
        warnings = warnings if isinstance(warnings, list) else []
        if warnings:
            renderer.warn(f"{len(warnings)} reference(s) the builder could not resolve; the build will fail on these:")
            for w in warnings:
                # Builder-supplied text, so it goes through the same scrub as any
                # other remote error body rather than straight at the terminal.
                if isinstance(w, dict):
                    field, reason = w.get("field", "?"), w.get("reason", "")
                else:
                    field, reason = "?", w
                renderer.warn(f"  {sanitize_error_body(str(field))}: {sanitize_error_body(str(reason))}")
        renderer.console().print_json(json.dumps(result))
    renderer.emit(result, command="build validate")


@app.command("delete", help="Delete a build (soft-delete).")
@tracking.track_command("build")
def delete_cmd(
    build_id: Annotated[str, typer.Argument(help="Build id.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    if not yes:
        if renderer.is_pretty():
            if not typer.confirm(f"Delete build {build_id}?"):
                renderer.info("Aborted.")
                raise typer.Exit(code=0)
        else:
            # No TTY to prompt on (JSON / agent / pipe): refuse rather than block.
            renderer.error(
                code="build_delete_needs_confirm",
                message="refusing to delete without confirmation in non-interactive mode.",
                hint="pass --yes to confirm the delete",
                details=_id_keys(build=build_id),
            )
            raise typer.Exit(code=1)
    client = _builder_client(renderer, builder_url)
    _builder_call(renderer, lambda: client.delete_build(build_id))
    if renderer.is_pretty():
        renderer.success(f"Deleted build {build_id}")
    renderer.emit({**_id_keys(build=build_id), "deleted": True}, command="build delete", changed=True)


@app.command("update", help="Replace a build's definition from a scan/edited JSON.")
@tracking.track_command("build")
def update_cmd(
    build_id: Annotated[str, typer.Argument(help="Build id.")],
    from_: Annotated[str, typer.Option("--from", "-f", help="Path to a definition JSON.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    try:
        definition = _load_definition(Path(from_).expanduser(), require_models=False)
    except ValueError as e:
        renderer.error(code="build_definition_invalid", message=str(e), details={"path": from_})
        raise typer.Exit(code=1) from e
    client = _builder_client(renderer, builder_url)

    # A scan definition names sources the builder does not take verbatim, so it is
    # mapped the way create maps it and one file works through either command.
    if _is_scan_shaped(definition):
        # Batched, one call per 32 filenames, exactly as create does it. The
        # annotation it leaves is what the default resolver in plan_create reads.
        try:
            resolve_models_via_builder(definition.get("models", []), client)
        except (requests.RequestException, KeyError, ValueError) as e:
            renderer.warn(f"model resolution unavailable ({e})")
        plan = plan_create(definition)
        if plan["uploads"]:
            names = ", ".join(sorted(u.get("filename") or u["name"] for u in plan["uploads"]))
            renderer.error(
                code="build_upload_unavailable",
                message=f"these need uploading, which update cannot do: {names}",
                hint="use `comfy build create --from ... --execute`, which uploads private blobs",
            )
            raise typer.Exit(code=1)
        definition = plan["definition"]

    # The builder guards the save with the updatedAt the caller last saw (optimistic
    # concurrency, meant for the website's read-edit-save). For a CLI that's just
    # last-writer-wins: fetch the current updatedAt and echo it back, else the save
    # 409s STALE on a missing/zero timestamp.
    current = _builder_call(renderer, lambda: client.get_build(build_id))
    dist = _builder_call(
        renderer,
        lambda: client.update_build(build_id, definition, current.get("updatedAt")),
    )
    if renderer.is_pretty():
        renderer.success(f"Updated build {build_id}")
    renderer.emit(dist, command="build update", changed=True)


def as_snapshot_envelope(data: dict) -> dict:
    """Wrap a bare Desktop snapshot in the envelope the importer requires.

    Desktop writes one snapshot per file under `.launcher/snapshots/`, and only
    its export action wraps them. The importer takes the wrapped shape, so the
    file a user actually has on disk is refused as not a Desktop export."""
    if data.get("type") or "snapshots" in data:
        return data
    if "customNodes" not in data and "pipPackages" not in data:
        return data
    return {"type": "comfyui-desktop-2-snapshot", "version": 2, "snapshots": [data]}


@app.command("from-snapshot", help="Create a build from a ComfyUI Desktop snapshot export.")
@tracking.track_command("build")
def from_snapshot_cmd(
    from_: Annotated[str, typer.Option("--from", "-f", help="Path to a Desktop snapshot export JSON.")],
    name: Annotated[str, typer.Option("--name", help="Name for the new build.")],
    description: Annotated[str | None, typer.Option("--description", help="Optional description.")] = None,
    base_image: Annotated[
        str | None,
        typer.Option("--base-image", help="Build on this base image rather than the one the snapshot's Python picks."),
    ] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    path = Path(from_).expanduser()
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        # `null`, a list and a bare scalar all parse. Caught here so the caller
        # gets the error envelope rather than an AttributeError from the wrapper.
        if not isinstance(snapshot, dict):
            raise ValueError(f"expected a JSON object, got {type(snapshot).__name__}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
        renderer.error(
            code="build_definition_invalid",
            message=f"could not read {path}: {e}",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from e

    client = _builder_client(renderer, builder_url)
    result = _builder_call(
        renderer,
        lambda: client.create_build_from_snapshot(
            name, as_snapshot_envelope(snapshot), description=description, base_image_id=base_image
        ),
    )
    created = result.get("build") or {}
    build_id = created.get("id")
    if renderer.is_pretty():
        renderer.success(f"Created build {build_id} from {path.name}")
        for line in report_advisories(result.get("report") or {}):
            renderer.warn(line)
        renderer.info(f"cut a build with `comfy build release create {build_id}`")
    renderer.emit(result, command="build from-snapshot", changed=True)


@app.command("from-workflow", help="Create a build from a ComfyUI workflow file.")
@tracking.track_command("build")
def from_workflow_cmd(
    from_: Annotated[
        str, typer.Option("--from", "-f", help="Path to a ComfyUI workflow JSON (editing format or API export).")
    ],
    name: Annotated[str, typer.Option("--name", help="Name for the new build.")],
    description: Annotated[str | None, typer.Option("--description", help="Optional description.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    path = Path(from_).expanduser()
    try:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        # The dialect is the builder's call: it reads the editing format and the
        # API export, so a client-side guess would refuse files it accepts.
        if not isinstance(workflow, dict):
            raise ValueError(f"expected a JSON object, got {type(workflow).__name__}")
    # A deeply nested file makes json.loads recurse until the interpreter stops
    # it, and that lands outside the ValueError family.
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as e:
        renderer.error(
            code="build_workflow_invalid",
            message=f"could not read {path}: {e}",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from e

    client = _builder_client(renderer, builder_url)
    result = _builder_call(renderer, lambda: client.create_build_from_workflow(name, workflow, description=description))
    created = result.get("build") if isinstance(result, dict) else None
    report = result.get("report") if isinstance(result, dict) else None
    distribution_id = created.get("id") if isinstance(created, dict) else None
    # The row is already written by the time the answer arrives, so a shape this
    # command cannot read means a build exists that nothing here can name.
    if not distribution_id or not isinstance(report, dict):
        renderer.error(
            code="build_builder_error",
            message=(
                "the builder answered from-workflow with a shape this CLI cannot read, so a build may exist "
                "that this command cannot name; list your builds with `comfy build list`"
            ),
            details={"received": sorted(result) if isinstance(result, dict) else type(result).__name__},
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        renderer.success(f"Created build {distribution_id} from {path.name}")
        if not report:
            renderer.warn("the builder sent an empty import report, so nothing here says what the workflow mapped to")
        for line in report_advisories(report):
            renderer.warn(line)
        # comfyVersionRequired is the builder's word on whether a version is
        # pinned, so only an explicit false says this build can be cut.
        if report.get("comfyVersionRequired") is False:
            renderer.info(f"cut a build with `comfy build release create {distribution_id}`")
        else:
            renderer.info(
                "no ComfyUI version is pinned, so this build cannot be cut yet: run `comfy build get "
                f"{distribution_id} --json | jq .data.definition > def.json`, add `baseComfyVersion` to def.json, "
                f"then run `comfy build update {distribution_id} --from def.json`"
            )
    renderer.emit(result, command="build from-workflow", changed=True)


@blob_app.command("upload", help="Upload a private file and print the blobId a definition can reference.")
@tracking.track_command("build")
def blob_upload_cmd(
    path: Annotated[str, typer.Argument(help="File to upload.")],
    kind: Annotated[str, typer.Option("--kind", help="What the file is: model or node_zip.")] = "model",
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    if kind not in ("model", "node_zip"):
        renderer.error(
            code="build_definition_invalid",
            message=f"unknown blob kind {kind!r}",
            hint="use --kind model or --kind node_zip",
        )
        raise typer.Exit(code=1)
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        renderer.error(
            code="build_definition_invalid",
            message=f"no file at {file_path}",
            details={"path": str(file_path)},
        )
        raise typer.Exit(code=1)

    client = _builder_client(renderer, builder_url)
    size_bytes = file_path.stat().st_size
    sha256 = _sha256_file(file_path)
    blob_id, upload_url = _builder_call(renderer, lambda: client.create_blob(kind, file_path.name, sha256, size_bytes))
    _builder_call(renderer, lambda: client.upload_blob(upload_url, file_path))
    if renderer.is_pretty():
        renderer.success(f"Uploaded {file_path.name} as {blob_id}")
        # A model entry needs the hash as well as the id, and the cut refuses one
        # that is not a real sha256, so both are printed rather than just the id.
        renderer.info(f'reference it as "blobId": "{blob_id}", "sha256": "{sha256}"')
    renderer.emit(
        {"blobId": blob_id, "kind": kind, "filename": file_path.name, "sha256": sha256, "sizeBytes": size_bytes},
        command="build blob upload",
        changed=True,
    )


@app.command("resolve", help="Resolve model filenames to public download candidates (HF/CivitAI).")
@tracking.track_command("build")
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
    renderer.emit({"results": results}, command="build resolve")


@app.command("base-images", help="List the curated base images a build can build on.")
@tracking.track_command("build")
def base_images_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    images = _builder_call(renderer, client.list_base_images)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(images))
    renderer.emit({"baseImages": images}, command="build base-images")


@app.command("build-targets", help="List the build targets (os/gpu) a release can be cut for.")
@tracking.track_command("build")
def build_targets_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    targets = _builder_call(renderer, client.list_build_targets)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(targets))
    renderer.emit({"targets": targets}, command="build build-targets")


@app.command("model-dirs", help="List the model directories a model may be placed in.")
@tracking.track_command("build")
def model_dirs_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    dirs = _builder_call(renderer, client.list_model_directories)
    if renderer.is_pretty():
        for d in dirs:
            renderer.print(f"  {d}")
    renderer.emit({"directories": dirs}, command="build model-dirs")


@version_app.command("manifest", help="Show a release's models and runtime policies.")
@tracking.track_command("build")
def version_manifest(
    version_id: Annotated[str, typer.Argument(help="Build release id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    manifest = _builder_call(renderer, lambda: client.get_version_manifest(version_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(manifest))
    renderer.emit(manifest, command="build release manifest")


@artifact_app.command("download", help="Get a signed download link for a build artifact.")
@tracking.track_command("build")
def artifact_download(
    artifact_id: Annotated[str, typer.Argument(help="Build artifact id.")],
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    signed = _builder_call(renderer, lambda: client.get_artifact_download(artifact_id))
    if renderer.is_pretty():
        renderer.print(signed.get("downloadUrl", "(no url)"))
    renderer.emit(signed, command="build artifact download")


@blob_app.command("list", help="List the workspace's private blobs.")
@tracking.track_command("build")
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
    renderer.emit({"blobs": blobs}, command="build blob list")
