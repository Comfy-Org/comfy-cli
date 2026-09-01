"""``comfy build`` — package a local ComfyUI environment into a serverless build.

``init`` inspects a local ComfyUI install and writes a canonical authoring spec
covering the two local asset classes a build needs:

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

Initialization runs entirely on the user's machine, including deterministic
content identities for private assets. It does not authenticate or contact the
Builder service.

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
import tempfile
import time
import urllib.error
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, NoReturn
from urllib.parse import urlsplit

import requests
import typer

from comfy_cli import tracking
from comfy_cli._safe_exec import BinaryNotFoundError, resolve_required_binary
from comfy_cli.command.build_diff import (
    DefinitionDiff,
    diff_definitions,
    merge_definition,
    render_definition_diff,
    summarize_definition_diff,
)
from comfy_cli.command.build_digest_cache import ModelDigestCache
from comfy_cli.command.build_package import NodePackageError, package_node
from comfy_cli.command.build_paths import (
    BuildPaths,
    BuildSpecNotFoundError,
    InstallOverrides,
    resolve_build_paths,
    resolve_local_path,
)
from comfy_cli.command.build_pull import UnsyncedDefinitionError, merge_pulled_spec
from comfy_cli.command.build_push import (
    SkippedSymlink,
    pending_uploads,
    prepare_push,
    public_node_identities,
    public_node_projection,
    spec_without_stale_source_uris,
    unresolved_models,
    upload_assets,
)
from comfy_cli.command.build_spec import (
    SPEC_SCHEMA,
    BuildSpecInvalidError,
    BuildSpecWriteError,
    read_build_spec,
    write_build_spec,
)
from comfy_cli.command.build_targets import (
    TARGET_FORM,
    BuildTarget,
    BuildTargetInvalidError,
    catalog_choices,
    parse_build_targets,
)
from comfy_cli.command.build_validation import (
    lookup_public_model_sources,
    project_wire_definition,
    validate_local_build_spec,
)
from comfy_cli.command.pack_scan import read_pyproject
from comfy_cli.constants import SUPPORTED_PT_EXTENSIONS
from comfy_cli.interaction import confirm, require_option
from comfy_cli.output import get_renderer
from comfy_cli.registry.api import sanitize_error_body
from comfy_cli.utils import parse_rfc3339

if TYPE_CHECKING:
    from comfy_cli.builder_api import BuilderClient

__all__ = ("project_wire_definition", "resolve_local_path")

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Package a local ComfyUI environment into a serverless build. "
        "[Limited beta] — builder access is granted per account; commands that reach the "
        "builder return a 'not enabled yet' message until your account is enabled."
    ),
)

release_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and cut Build releases.",
)
app.add_typer(release_app, name="release")

refs_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect the Builder's reference catalogs.",
)
app.add_typer(refs_app, name="refs")

blob_app = typer.Typer(no_args_is_help=True, help="Workspace private blobs (uploaded models / nodes).")
app.add_typer(blob_app, name="blob", hidden=True)

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
    root = models_root.resolve()
    models: list[dict] = []
    for path, model_type, filename in _iter_model_files(root):
        models.append(
            {
                "type": model_type,
                "filename": filename,
                "localPath": path.relative_to(root).as_posix(),
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
_REPOSITORY_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._-]{0,99}$")
_GITHUB_SCP_RE = re.compile(r"^git@github\.com:(?P<path>.+)$", re.IGNORECASE)


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


def _canonical_github_repository(origin: str) -> str | None:
    """Canonicalize supported GitHub origins and enforce the builder contract."""
    raw = origin.strip()
    scp_match = _GITHUB_SCP_RE.fullmatch(raw)
    if scp_match is not None:
        candidate = f"https://github.com/{scp_match.group('path')}"
    else:
        try:
            parsed_origin = urlsplit(raw)
            origin_port = parsed_origin.port
        except ValueError:
            return None
        if parsed_origin.scheme == "ssh":
            if (
                parsed_origin.hostname != "github.com"
                or parsed_origin.username != "git"
                or parsed_origin.password is not None
                or origin_port is not None
                or parsed_origin.query
                or parsed_origin.fragment
            ):
                return None
            candidate = f"https://github.com{parsed_origin.path}"
        else:
            candidate = raw

    try:
        repository = urlsplit(candidate)
        port = repository.port
    except ValueError:
        return None
    if (
        repository.scheme != "https"
        or repository.hostname != "github.com"
        or repository.username is not None
        or repository.password is not None
        or port is not None
        or repository.query
        or repository.fragment
    ):
        return None

    segments = repository.path.strip("/").split("/")
    if len(segments) != 2:
        return None
    segments[1] = segments[1].removesuffix(".git")
    if any(not _REPOSITORY_SEGMENT_RE.fullmatch(segment) or ".." in segment for segment in segments):
        return None
    return f"https://github.com/{segments[0]}/{segments[1]}"


def _identify_node(node_dir: Path) -> dict:
    """One definition entry for one pack, naming the single source it has."""
    origin = git_ref = None
    if (node_dir / ".git").exists():
        origin = _git_output(node_dir, "remote", "get-url", "origin")
        git_ref = _git_output(node_dir, "rev-parse", "HEAD")
    repository = _canonical_github_repository(origin) if origin else None
    # A repo@ref reconstructs an exact commit, so it wins where both exist.
    if repository and git_ref:
        return {"name": node_dir.name, "repository": repository, "gitRef": git_ref, "source": "git"}
    pin = _read_registry_pin(node_dir)
    if pin:
        # The builder takes exactly one source, so a half-git checkout keeps none
        # of its git fields: an origin without a resolvable HEAD is not fetchable.
        return {"name": node_dir.name, "id": pin[0], "registryVersion": pin[1], "source": "registry"}
    return {"name": node_dir.name, "repository": None, "gitRef": None, "source": "local"}


_SKIPPED_SYMLINK_PREVIEW = 5


def _warn_skipped_symlinks(renderer, skipped: Sequence[SkippedSymlink]) -> list[dict]:
    """Announce symlinks packaging left out, and return them as payload rows.

    Node archives deliberately exclude symlinks, so a node that vendors its
    dependencies through one packages to a near-empty archive. That archive's
    digest becomes the node's ``localDigest`` in a spec the user commits, so
    staying silent ships a release whose first symptom is an ImportError inside
    a container, arbitrarily far from the symlink that caused it.

    Every command that recomputes a node's ``localDigest`` renders its skips
    through here — `init`, `update`, `push`, `pull` and `status` — so they
    cannot drift into describing the same omission differently. The first four
    also carry the returned rows in their payload, including on a `--dry-run`
    that writes nothing; `status` recomputes but exposes no definition to point
    into, so for it the stderr warning is the whole report.
    """
    if not skipped:
        return []
    listed = ", ".join(f"{item.local_path}: {item.member}" for item in skipped[:_SKIPPED_SYMLINK_PREVIEW])
    if len(skipped) > _SKIPPED_SYMLINK_PREVIEW:
        listed += f", and {len(skipped) - _SKIPPED_SYMLINK_PREVIEW} more"
    plural = "s" if len(skipped) > 1 else ""
    renderer.warn(
        f"packaging excluded {len(skipped)} symlink{plural} from the custom nodes: {listed}",
        hint="a symlinked dependency is never packaged; replace it with real files if the node needs it at runtime",
    )
    return [item.as_row() for item in skipped]


def _raise_node_package_error(renderer, error: NodePackageError) -> NoReturn:
    """The single rendering of a packaging failure, for every command that packages.

    ``details.path`` names the *node directory* the user has to fix. Each caller
    reporting its own path would make the same failure arrive as a node folder
    from `init` and as an unrelated spec YAML from `push`, under one error code.
    """
    renderer.error(
        code="build_spec_invalid",
        message=str(error),
        hint=(
            "packaging is all-or-nothing, so anything it cannot package fails the command rather than "
            "shipping a short archive; fix the named path, or move it out of the custom-nodes directory"
        ),
        details={"path": str(error.path)},
    )
    raise typer.Exit(code=1) from error


def _relocate_skipped_symlinks(skipped: Sequence[SkippedSymlink], definition: object) -> list[SkippedSymlink]:
    """Re-point rows at *definition*, dropping nodes it no longer carries.

    ``prepare_push`` numbers ``location`` against the **local** spec, which is
    the definition `push` and `init` go on to ship. `pull` ships the *server's*
    node order and omits any local node the server does not carry, so an
    un-relocated index would name a different node — or none at all. Dropping
    the unmatched rows is not merely tidiness: those nodes are being deleted
    from the spec by this pull, so their packaging is no longer the user's
    problem and warning about them would be noise.
    """
    entries = definition.get("customNodes") if isinstance(definition, dict) else None
    if not isinstance(entries, list):
        return []
    # Only `source: local` entries are candidates. Every row came from a local
    # node, and an unmatched server entry is deep-copied verbatim — so matching
    # on localPath alone would let a server-side path collision re-point a row
    # onto an unrelated node, the exact mislabelling this function prevents.
    #
    # Indexes are pooled per path and consumed one per row, not looked up: a
    # spec may legitimately point two nodes at one directory, and a plain dict
    # would keep only the last, collapsing both rows onto it. Rows left without
    # an index are nodes this pull removes, and are dropped.
    available: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and isinstance(entry.get("localPath"), str) and entry.get("source") == "local":
            available.setdefault(entry["localPath"], []).append(index)
    relocated: list[SkippedSymlink] = []
    for item in skipped:
        indexes = available.get(item.local_path)
        if indexes:
            relocated.append(replace(item, location=f"definition.customNodes[{indexes.pop(0)}]"))
    return relocated


def scan_custom_nodes(
    custom_nodes_root: Path,
    *,
    on_skip: Callable[[SkippedSymlink], None] | None = None,
) -> list[dict]:
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
    root = custom_nodes_root.resolve()
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue  # single-file .py nodes aren't buildable units here
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        node = _identify_node(entry)
        local_path = entry.relative_to(root).as_posix()
        node["localPath"] = local_path
        if node["source"] == "local":
            package = package_node(entry)
            node["localDigest"], node["localSizeBytes"] = package.sha256, package.size_bytes
            if on_skip is not None:
                location = f"definition.customNodes[{len(nodes)}]"
                for member in package.skipped_symlinks:
                    on_skip(SkippedSymlink(location, local_path, member))
        nodes.append(node)
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


def _venv_pythons(venv_dir: Path) -> tuple[Path, ...]:
    """Both layouts' Python executable inside a venv dir, host's own first.

    A venv's layout is fixed by the platform that CREATED it, not by the one
    reading it, and the two are not always the same machine: under WSL a
    Windows ComfyUI install's venv is a ``Scripts/python.exe`` sitting on a
    POSIX host, which WSL still executes through interop. Probing only the
    host's layout is what made ``--python`` mandatory for that case. The host's
    own layout is returned first so a native venv wins if a tree somehow holds
    both."""
    posix = venv_dir / "bin" / "python"
    windows = venv_dir / "Scripts" / "python.exe"
    return (windows, posix) if os.name == "nt" else (posix, windows)


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
            for py in _venv_pythons(comfy_root / name):
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


# --- translating a scanned definition for the builder -------------------------
#
# The builder's definition schema (services/comfy-builder/definition/definition.go):
#   Model = { type, filename?, sourceUri? | blobId?, sha256? }  (exactly one source)
#   Node  = { name, repository? + gitRef? | registryVersion? | blobId? }
# The scan already emits type/filename/sha256/name/repository/gitRef under the same
# keys, so translating is a matter of dropping the scan-only fields (sizeBytes,
# source, top-level schema) rather than remapping them.


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
        meaning = "models the graph loads that nothing has a source for; `comfy build refs resolve` finds candidates"
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


def _model_filename(model: dict) -> str | None:
    """``definition.models[].filename`` is optional, so it is never indexed blindly."""
    value = model.get("filename")
    return value if isinstance(value, str) and value else None


def resolve_models_via_builder(models: list[dict], client) -> int:
    """Resolve each model's filename against the builder and, when a returned
    candidate's sha256 matches the local file's hash, annotate the model with
    that candidate's ``sourceUri`` (so ``push`` references it by URL instead of
    uploading it). Returns how many were resolved.

    Only a **hash-confirmed** public match is trusted: a filename alone is a weak
    key, and a renamed local fine-tune must never be mistaken for a public model.
    Candidates the provider didn't hash are left to upload (the safe default).
    The builder re-verifies the sha256 on fetch regardless, so this is an
    optimization to avoid needless uploads — not the integrity boundary."""
    names = sorted({name for name in map(_model_filename, models) if name is not None})
    if not names:
        return 0
    by_name: dict[str, list[dict]] = {}
    for batch in _chunks(names, _RESOLVE_BATCH):
        for res in client.resolve_models(batch):
            by_name[res.get("filename", "")] = res.get("candidates") or []

    resolved = 0
    for m in models:
        local = (m.get("sha256") or "").lower()
        filename = _model_filename(m)
        if not local or m.get("sourceUri") or filename is None:
            continue
        for cand in by_name.get(filename, []):
            uri = cand.get("sourceUri")
            if uri and (cand.get("sha256") or "").lower() == local:
                m["sourceUri"] = uri
                resolved += 1
                break
    return resolved


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


@dataclass(frozen=True, slots=True)
class ScanRequest:
    """What the shared scan needs beyond the renderer and the click context."""

    paths: BuildPaths
    comfy_version: str | None = None
    comfy_url: str | None = None


class ScanUnavailableError(Exception):
    """This directory cannot be scanned as a ComfyUI installation.

    Raised rather than rendered so the caller decides what it means: fatal for
    `init` and `update`, where a scan is the whole command, but only a skipped
    section for `status`, whose spec-vs-remote half needs no install.
    """

    def __init__(self, code: str, message: str, *, hint: str | None = None, details: dict | None = None) -> None:
        self.code = code
        self.hint = hint
        self.details = details or {}
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """One local scan. ``definition`` is the builder-ready document; the rest is
    what the payload reports about how that document was produced."""

    definition: dict
    environment: dict
    total_bytes: int
    skipped_symlinks: tuple[dict, ...] = ()


def _scan_install(renderer, ctx, request: ScanRequest, *, optional: bool = False) -> ScanResult:
    """Scan an installation into a ``definition`` — the path `init` and `update`
    share, so a rescan can never drift from the original scan.

    Every reason the scan cannot run raises :class:`ScanUnavailableError`, so a
    caller for which "no install here" is an ordinary answer can say so instead
    of exiting. ``optional=True`` marks such a caller: nothing prompts, and even
    the packaging failure that otherwise renders itself and exits is raised.
    """
    paths = request.paths
    if not paths.models_dir.is_dir():
        raise ScanUnavailableError(
            code="build_models_dir_missing",
            message=f"No models/ directory to scan at {paths.models_dir}.",
            details={"path": str(paths.models_dir)},
        )

    explicit_python = str(paths.python) if paths.python is not None else None
    python_exe = find_comfy_python(paths.install_root, explicit_python)
    if python_exe is None and not optional:
        selected_python = require_option(
            "--python",
            None,
            prompt_fn=lambda: typer.prompt("ComfyUI Python executable"),
            error_code="build_missing_input",
            ctx=ctx,
        )
        python_exe = find_comfy_python(paths.install_root, selected_python)
    if python_exe is None:
        raise ScanUnavailableError(
            code="build_missing_input",
            message=(
                f"No ComfyUI Python executable under {paths.install_root}."
                if optional
                else "The selected --python executable could not be resolved."
            ),
            hint="pass --python <path> pointing at the ComfyUI environment's Python executable",
            details={"missing": ["--python"], "path": explicit_python},
        )

    provenance = capture_pip_provenance(str(python_exe))
    if provenance is None:
        raise ScanUnavailableError(
            code="build_missing_input",
            message=f"Could not capture dependency provenance from {python_exe}.",
            hint="pass --python <path> pointing at a working ComfyUI environment",
            details={"missing": ["--python"], "path": str(python_exe)},
        )

    renderer.info(f"Scanning models in {paths.models_dir} …")
    models = scan_models(paths.models_dir)

    nodes: list[dict] = []
    reported_symlinks: list[dict] = []
    if paths.custom_nodes_dir.is_dir():
        renderer.info(f"Scanning custom nodes in {paths.custom_nodes_dir} …")
        skipped: list[SkippedSymlink] = []
        try:
            nodes = scan_custom_nodes(paths.custom_nodes_dir, on_skip=skipped.append)
        except NodePackageError as error:
            if optional:
                raise ScanUnavailableError(
                    code="build_spec_invalid", message=str(error), details={"path": str(error.path)}
                ) from error
            _raise_node_package_error(renderer, error)
        reported_symlinks = _warn_skipped_symlinks(renderer, skipped)

    # baseComfyVersion: explicit flag → version marker at the code root → a
    # running ComfyUI's /system_stats (covers split layouts with no on-disk marker).
    base_version = request.comfy_version or detect_comfy_version(paths.install_root)
    if not base_version:
        base_version = detect_comfy_version_from_server(request.comfy_url or DEFAULT_COMFY_URL)
    # Whatever named it, the definition records a ref the builder can resolve.
    if base_version:
        base_version = as_comfy_git_ref(base_version)

    return ScanResult(
        definition=build_definition(
            models,
            nodes,
            base_comfy_version=base_version,
            pip_dependencies=provenance["pipDependencies"],
            environment=provenance["environment"],
        ),
        environment=provenance["environment"],
        total_bytes=sum(m["sizeBytes"] for m in models),
        skipped_symlinks=tuple(reported_symlinks),
    )


def _require_scan(renderer, ctx, request: ScanRequest) -> ScanResult:
    """Scan, or render the reason and exit — for the commands a scan IS."""
    try:
        return _scan_install(renderer, ctx, request)
    except ScanUnavailableError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error


def _reject_scan_options(
    renderer,
    flag: str,
    *,
    models_dir: str | None,
    custom_nodes_dir: str | None,
    python: str | None,
    comfy_url: str | None,
) -> None:
    """Refuse an import ``flag`` alongside any option that only steers a scan.

    The two name different *sources* for the same ``definition``, so combining
    them has no meaning a precedence rule could rescue — silently picking one
    would write a spec nobody asked for. ``--comfy-version`` is deliberately
    absent: it overrides a value in the resulting definition rather than
    directing a scan, so it applies to either source.
    """
    conflicting = [
        option
        for option, value in (
            ("--models-dir", models_dir),
            ("--custom-nodes-dir", custom_nodes_dir),
            ("--python", python),
            ("--comfy-url", comfy_url),
        )
        if value
    ]
    if not conflicting:
        return
    joined = ", ".join(conflicting)
    renderer.error(
        code="build_missing_input",
        message=f"{flag} cannot be combined with {joined}: an imported definition is not a local scan.",
        hint=f"drop {joined} to import, or drop {flag} to scan the install",
        details={"conflict": [flag, *conflicting]},
    )
    raise typer.Exit(code=1)


def _reject_rival_import(renderer, from_snapshot: str | None, from_workflow: str | None) -> None:
    """Refuse ``--from-snapshot`` and ``--from-workflow`` together.

    Two sources for one ``definition``, and they disagree by construction: a
    snapshot records what was installed, a workflow only what a graph refers to.
    """
    if not (from_snapshot and from_workflow):
        return
    renderer.error(
        code="build_missing_input",
        message="--from-snapshot cannot be combined with --from-workflow: a spec takes its definition from one source.",
        hint="pass whichever one describes the environment you want",
        details={"conflict": ["--from-snapshot", "--from-workflow"]},
    )
    raise typer.Exit(code=1)


@dataclass(frozen=True, slots=True)
class ImportSource:
    """What differs between the two importable sources, and nothing else.

    A snapshot records what was *installed*; a workflow records only what a
    graph *refers to*. Everything after the resolve — advisories, the
    ``--comfy-version`` override, the merge, the diff, the write — is one path,
    so the two can never drift.
    """

    name: str
    flag: str
    label: str
    # Named `code` because it is passed straight to `renderer.error(code=...)`.
    code: str
    resolve: Callable[[BuilderClient, dict], dict]


_SNAPSHOT_SOURCE = ImportSource(
    name="snapshot",
    flag="--from-snapshot",
    label="ComfyUI Desktop snapshot",
    code="build_spec_invalid",
    resolve=lambda client, data: client.resolve_snapshot(as_snapshot_envelope(data)),
)
_WORKFLOW_SOURCE = ImportSource(
    name="workflow",
    flag="--from-workflow",
    label="ComfyUI workflow",
    code="build_workflow_invalid",
    resolve=lambda client, data: client.resolve_workflow(data),
)


@dataclass(frozen=True, slots=True)
class DefinitionImport:
    """One importer round-trip — the other way to obtain a ``definition``.

    ``advisories`` have already been rendered as warnings; they are carried here
    as well so a ``--json`` caller reads them out of the envelope rather than
    parsing stderr.
    """

    definition: dict
    advisories: list[str]
    path: Path
    source: ImportSource


def _import_definition(renderer, source: ImportSource, value: str, *, comfy_version: str | None) -> DefinitionImport:
    """Resolve a snapshot or a workflow into a ``definition``, in one call.

    The builder's importer is the one place that knows what the Comfy Registry
    actually publishes, which curated base image a Python fits, and how a pin
    normalizes, so this path **requires** auth — unlike a plain scan, which is
    entirely local.

    Auth is checked **before** the file is opened. A signed-out caller has no
    route to a definition at all, so reading first would only be work thrown
    away, and it would make ``build_not_signed_in`` depend on whether the file
    happens to exist — two different failures for one cause.
    """
    client = _builder_client(renderer, None)
    path = Path(value).expanduser()
    hint = f"pass {source.flag} <path> naming a {source.label} JSON file"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        renderer.error(
            code=source.code,
            message=f"could not read the {source.label} at {path}: {error}",
            hint=hint,
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from error
    if not isinstance(data, dict):
        renderer.error(
            code=source.code,
            message=f"{path} is not a {source.label}: expected a JSON object.",
            hint=hint,
            details={"path": str(path)},
        )
        raise typer.Exit(code=1)

    imported = _builder_call(renderer, lambda: source.resolve(client, data))
    definition = imported.get("definition") if isinstance(imported, dict) else None
    if not isinstance(definition, dict):
        renderer.error(
            code="build_builder_error",
            message=f"the builder's importer returned no definition for this {source.name}.",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1)

    # Advisories are warnings, never failures: the importer names what it could
    # not carry so it is visible now rather than inside a build. Every line goes
    # through `renderer.warn`, which writes to stderr in JSON mode, so stdout
    # still carries exactly one envelope.
    advisories = report_advisories(imported.get("report") or {})
    for line in advisories:
        renderer.warn(line)

    # `--comfy-version` overrides a value in the definition rather than steering
    # a scan, so it is honoured here too; ignoring it on this path would leave a
    # flag that reads as accepted and does nothing.
    if comfy_version:
        definition["baseComfyVersion"] = as_comfy_git_ref(comfy_version)
    return DefinitionImport(definition=definition, advisories=advisories, path=path, source=source)


def _entry_count(definition: dict, collection: str) -> int:
    """How many entries a definition carries, tolerating an absent collection.

    An imported definition is the builder's document, not the scanner's, so it
    may legitimately omit a collection the scan always writes.
    """
    entries = definition.get(collection)
    return len(entries) if isinstance(entries, list) else 0


def _scanned_payload(paths: BuildPaths, scan: ScanResult) -> dict:
    """`init`'s JSON payload for the local-scan source.

    ``skipped_symlinks`` mirrors `push`'s key so the two commands report the same
    omission the same way — this is the payload that mints the ``localDigest``
    the user commits, so the incompleteness has to be machine-visible here too.
    """
    skipped: dict = {"skipped_symlinks": list(scan.skipped_symlinks)} if scan.skipped_symlinks else {}
    return {
        **skipped,
        "source": "scan",
        "models_dir": str(paths.models_dir),
        "custom_nodes_dir": str(paths.custom_nodes_dir) if paths.custom_nodes_dir.is_dir() else None,
        "base_comfy_version": scan.definition.get("baseComfyVersion"),
        "pip_captured": True,
        "environment": scan.environment,
        "count": _entry_count(scan.definition, "models"),
        "custom_node_count": _entry_count(scan.definition, "customNodes"),
        "total_bytes": scan.total_bytes,
    }


def _imported_payload(imported: DefinitionImport) -> dict:
    """`init`'s JSON payload for an imported source.

    The scan-only numbers are absent rather than zeroed: nothing local was
    measured, and a ``0`` would read as "measured, and there was none".
    """
    return {
        "source": imported.source.name,
        imported.source.name: str(imported.path),
        "base_comfy_version": imported.definition.get("baseComfyVersion"),
        "advisories": imported.advisories,
        "count": _entry_count(imported.definition, "models"),
        "custom_node_count": _entry_count(imported.definition, "customNodes"),
    }


def _render_scan_summary(renderer, paths: BuildPaths, scan: ScanResult) -> None:
    base_version = scan.definition.get("baseComfyVersion")
    if base_version:
        renderer.print(f"ComfyUI version: {base_version}")
    else:
        renderer.warn(
            "ComfyUI version not detected — a build requires it. "
            "Re-run with --comfy-version <ref>, or set baseComfyVersion before `create`."
        )
    environment = scan.environment
    torch = f", torch {environment['torch']}" if environment["torch"] else ""
    renderer.print(
        f"pip deps: captured from {environment['os']}/{environment['arch']} "
        f"(Python {environment['pythonVersion']}{torch})"
    )
    models = scan.definition["models"]
    if models:
        _render_models_table(renderer, models, scan.total_bytes)
    else:
        renderer.warn(f"No model files found under {paths.models_dir}.")
    nodes = scan.definition["customNodes"]
    if nodes:
        _render_nodes_table(renderer, nodes)


def _render_import_summary(renderer, imported: DefinitionImport) -> None:
    """The pretty view of what the importer returned.

    Deliberately not the scan tables: an imported entry carries the builder's
    own fields, not the scanner's ``source`` / ``sizeBytes``, so those tables
    would read keys that are not there.
    """
    definition = imported.definition
    base_version = definition.get("baseComfyVersion")
    renderer.print(f"Imported {imported.path} through the builder's {imported.source.name} importer.")
    if base_version:
        renderer.print(f"ComfyUI version: {base_version}")
    else:
        renderer.warn(
            "The importer set no ComfyUI version — a build requires it. "
            "Re-run with --comfy-version <ref>, or set baseComfyVersion in the spec."
        )
    renderer.print(
        f"custom nodes: {_entry_count(definition, 'customNodes')}  ·  models: {_entry_count(definition, 'models')}"
    )


@app.command(
    "init",
    help="Scan a local ComfyUI install and write a comfy-build spec.",
)
@tracking.track_command("build")
def init_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Name for the build."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing local build spec."),
    ] = False,
    from_snapshot: Annotated[
        str | None,
        typer.Option(
            "--from-snapshot",
            help="Import a ComfyUI Desktop snapshot JSON through the builder instead of scanning the install. "
            "Requires sign-in; cannot be combined with --models-dir, --custom-nodes-dir, --python or --comfy-url.",
        ),
    ] = None,
    from_workflow: Annotated[
        str | None,
        typer.Option(
            "--from-workflow",
            help="Import a ComfyUI workflow JSON (editing format or API export) through the builder instead of "
            "scanning the install. A workflow names no model sources, so the spec starts with none — the report "
            "lists what the graph loads. Requires sign-in; same option conflicts as --from-snapshot.",
        ),
    ] = None,
    models_dir: Annotated[
        str | None,
        typer.Option(
            "--models-dir",
            help="Models folder to scan. Default: PATH/models/.",
        ),
    ] = None,
    custom_nodes_dir: Annotated[
        str | None,
        typer.Option(
            "--custom-nodes-dir",
            help="Custom-nodes folder to scan. Default: PATH/custom_nodes/.",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the build spec to this path instead of PATH/comfy-build.yaml.",
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
    base_image: Annotated[
        str | None,
        typer.Option(
            "--base-image",
            help="Build on this curated base image by id, instead of the default the builder picks. "
            "`comfy build refs base-images` lists the ids.",
        ),
    ] = None,
    comfy_url: Annotated[
        str | None,
        typer.Option(
            "--comfy-url", help=f"Running ComfyUI URL to read the version from. Default: {DEFAULT_COMFY_URL}."
        ),
    ] = None,
):
    renderer = get_renderer()
    _reject_rival_import(renderer, from_snapshot, from_workflow)
    import_source = _SNAPSHOT_SOURCE if from_snapshot else _WORKFLOW_SOURCE if from_workflow else None
    if import_source is not None:
        _reject_scan_options(
            renderer,
            import_source.flag,
            models_dir=models_dir,
            custom_nodes_dir=custom_nodes_dir,
            python=python,
            comfy_url=comfy_url,
        )
    build_name = require_option(
        "--name",
        name,
        prompt_fn=lambda: typer.prompt("Build name"),
        error_code="build_missing_input",
        ctx=ctx,
    )
    paths = resolve_build_paths(
        path,
        overrides=InstallOverrides.from_options(models_dir, custom_nodes_dir, python),
        require_spec=False,
    )
    spec_path = Path(output).expanduser() if output else paths.spec_file
    if spec_path.exists() and not force:
        renderer.error(
            code="build_spec_exists",
            message=f"Build spec already exists at {spec_path}.",
            hint="pass --force to overwrite the local spec",
            details={"path": str(spec_path)},
        )
        raise typer.Exit(code=1)

    scan: ScanResult | None = None
    imported: DefinitionImport | None = None
    if import_source is not None:
        imported = _import_definition(
            renderer, import_source, from_snapshot or from_workflow or "", comfy_version=comfy_version
        )
        definition = imported.definition
        payload = _imported_payload(imported)
    else:
        scan = _require_scan(renderer, ctx, ScanRequest(paths, comfy_version=comfy_version, comfy_url=comfy_url))
        definition = scan.definition
        payload = _scanned_payload(paths, scan)
        renderer.event(
            "init_complete",
            models=payload["count"],
            custom_nodes=payload["custom_node_count"],
            total_bytes=scan.total_bytes,
            base_comfy_version=payload["base_comfy_version"],
            pip_captured=True,
        )

    # The builder validates the id on every write and the baker resolves it at
    # build time, so this is the whole of `--base-image`: an absent key means
    # the catalog default.
    if base_image:
        definition["baseImage"] = base_image

    spec = {
        "schema": SPEC_SCHEMA,
        "id": None,
        "name": build_name,
        "description": "",
        "syncedRevision": None,
        "definition": definition,
    }
    try:
        write_build_spec(spec_path, spec)
    except (BuildSpecInvalidError, BuildSpecWriteError) as error:
        renderer.error(
            code=error.code,
            message=str(error),
            details={"path": str(spec_path)},
        )
        raise typer.Exit(code=1) from error

    if renderer.is_pretty():
        if imported is not None:
            _render_import_summary(renderer, imported)
        elif scan is not None:
            _render_scan_summary(renderer, paths, scan)
        renderer.success(f"Wrote build spec → {spec_path}")

    renderer.emit(
        {"spec_file": str(spec_path), "name": build_name, **payload, "definition": definition},
        command="build init",
        changed=True,
    )


def _read_spec(renderer, spec_file: Path) -> dict:
    try:
        return read_build_spec(spec_file)
    except BuildSpecInvalidError as error:
        renderer.error(code=error.code, message=str(error), details={"path": str(spec_file)})
        raise typer.Exit(code=1) from error


@app.command(
    "update",
    help="Rescan the local install and rewrite the build spec's definition.",
)
@tracking.track_command("build")
def update_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Apply the rescan without confirming.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the diff and write nothing.")] = False,
    from_snapshot: Annotated[
        str | None,
        typer.Option(
            "--from-snapshot",
            help="Take the new definition from a ComfyUI Desktop snapshot JSON via the builder instead of "
            "rescanning. Requires sign-in; cannot be combined with --models-dir, --custom-nodes-dir, "
            "--python or --comfy-url. A snapshot describes no models, so the diff will show the spec's "
            "models being removed — review it before confirming.",
        ),
    ] = None,
    from_workflow: Annotated[
        str | None,
        typer.Option(
            "--from-workflow",
            help="Take the new definition from a ComfyUI workflow JSON (editing format or API export) via the "
            "builder instead of rescanning. Requires sign-in; same option conflicts as --from-snapshot. A "
            "workflow names no model sources either, so the diff will show the spec's models being removed — "
            "review it before confirming.",
        ),
    ] = None,
    models_dir: Annotated[
        str | None,
        typer.Option("--models-dir", help="Models folder to scan. Default: PATH/models/."),
    ] = None,
    custom_nodes_dir: Annotated[
        str | None,
        typer.Option("--custom-nodes-dir", help="Custom-nodes folder to scan. Default: PATH/custom_nodes/."),
    ] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="ComfyUI's Python executable, to capture pip deps."),
    ] = None,
    comfy_version: Annotated[
        str | None,
        typer.Option("--comfy-version", help="Override the detected ComfyUI version (baseComfyVersion)."),
    ] = None,
    base_image: Annotated[
        str | None,
        typer.Option(
            "--base-image",
            help="Change the curated base image the build is built on, by id. "
            "`comfy build refs base-images` lists the ids.",
        ),
    ] = None,
    comfy_url: Annotated[
        str | None,
        typer.Option(
            "--comfy-url", help=f"Running ComfyUI URL to read the version from. Default: {DEFAULT_COMFY_URL}."
        ),
    ] = None,
):
    renderer = get_renderer()
    _reject_rival_import(renderer, from_snapshot, from_workflow)
    import_source = _SNAPSHOT_SOURCE if from_snapshot else _WORKFLOW_SOURCE if from_workflow else None
    if import_source is not None:
        _reject_scan_options(
            renderer,
            import_source.flag,
            models_dir=models_dir,
            custom_nodes_dir=custom_nodes_dir,
            python=python,
            comfy_url=comfy_url,
        )
    try:
        paths = resolve_build_paths(
            path,
            overrides=InstallOverrides.from_options(models_dir, custom_nodes_dir, python),
        )
    except BuildSpecNotFoundError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error

    spec = _read_spec(renderer, paths.spec_file)
    # `canonicalize_build_spec` has already refused anything else (build_spec.py).
    stored = spec["definition"]
    # Only the *source* of the new definition branches. Everything after it —
    # merge, diff, confirmation, metadata preservation, the canonical write — is
    # the one path, so an imported update can never drift from a rescanned one.
    imported: DefinitionImport | None = None
    scanned: ScanResult | None = None
    if import_source is not None:
        imported = _import_definition(
            renderer, import_source, from_snapshot or from_workflow or "", comfy_version=comfy_version
        )
        incoming = imported.definition
    else:
        scanned = _require_scan(renderer, ctx, ScanRequest(paths, comfy_version=comfy_version, comfy_url=comfy_url))
        incoming = scanned.definition
    definition = merge_definition(stored, incoming)
    # After the merge, never before: a scan reports no base image, so the merge
    # carries the stored one forward and would overwrite an incoming value.
    if base_image:
        definition["baseImage"] = base_image
    diff = diff_definitions(stored, definition)
    summary = summarize_definition_diff(diff)

    if renderer.is_pretty():
        render_definition_diff(renderer, diff)

    payload = {
        "spec_file": str(paths.spec_file),
        "source": imported.source.name if imported is not None else "scan",
        "models_dir": str(paths.models_dir),
        "custom_nodes_dir": str(paths.custom_nodes_dir) if paths.custom_nodes_dir.is_dir() else None,
        "dry_run": dry_run,
        "summary": summary,
        "diff": diff.as_json(),
        "definition": definition,
    }
    if imported is not None:
        payload[imported.source.name] = str(imported.path)
        payload["advisories"] = imported.advisories
    # `update` rewrites the committed spec's localDigest exactly as `init` mints
    # it, so an archive that packaging left incomplete has to be as visible here.
    if scanned is not None and scanned.skipped_symlinks:
        payload["skipped_symlinks"] = list(scanned.skipped_symlinks)

    # Before the confirmation, not after: --dry-run promises to write nothing,
    # and a prompt whose only outcome is a write it will not perform is noise.
    if dry_run:
        if renderer.is_pretty():
            renderer.info(f"--dry-run: {paths.spec_file} left untouched.")
        renderer.emit({**payload, "written": False}, command="build update", changed=False)
        return

    if not confirm(
        f"Update the build spec at {paths.spec_file} ({summary})?",
        yes=yes,
        error_code="build_update_needs_confirm",
        ctx=ctx,
    ):
        # Declining needs a prompt, and a prompt needs pretty mode, where `emit`
        # is a no-op; a machine caller is refused above with
        # `build_update_needs_confirm` rather than reaching this branch.
        renderer.info("Aborted.")
        return

    # Only `definition` is recomputed: `id`, `name`, `description` and
    # `syncedRevision` describe the Build, not the installation, and a rescan
    # knows nothing about any of them.
    spec["definition"] = definition
    try:
        write_build_spec(paths.spec_file, spec)
    except (BuildSpecInvalidError, BuildSpecWriteError) as error:
        renderer.error(code=error.code, message=str(error), details={"path": str(paths.spec_file)})
        raise typer.Exit(code=1) from error

    if renderer.is_pretty():
        renderer.success(f"Updated build spec → {paths.spec_file}")
    renderer.emit({**payload, "written": True}, command="build update", changed=not diff.is_empty)


# Builder base URL: the comfy-builder service behind the Developer Platform
# gateway (the gateway strips the /builder prefix and forwards /v1/* to it).
# Prod issuer is cloud.comfy.org, matching the OAuth login token. Override with
# --builder-url or COMFY_BUILDER_URL (e.g. https://stagingplatformapi.comfy.org/builder).
DEFAULT_BUILDER_URL = "https://platformapi.comfy.org/builder"
_BUILDER_URL_OPT = typer.Option("--builder-url", help="Builder base URL. Default: $COMFY_BUILDER_URL, else built-in.")


class _StaleBuildError(Exception):
    pass


def _raise_build_spec_stale(renderer, message: str) -> NoReturn:
    renderer.error(
        code="build_spec_stale",
        message=message,
        hint="run `comfy build pull` to review the remote changes, then retry the push; use --force to overwrite them",
    )
    raise typer.Exit(code=1)


def _update_for_push(client, build_id: str, definition: dict, revision: str | None, name: str, description: str):
    try:
        return client.update_build(
            build_id,
            definition,
            revision,
            name=name,
            description=description,
        )
    except urllib.error.HTTPError as error:
        if error.code == 409:
            raise _StaleBuildError from error
        raise


def _force_update(renderer, client, build_id: str, definition: dict, name: str, description: str) -> dict:
    for _attempt in range(3):
        remote = _builder_call(renderer, lambda: client.get_build(build_id))
        try:
            return _builder_call(
                renderer,
                lambda: _update_for_push(client, build_id, definition, remote["updatedAt"], name, description),
            )
        except _StaleBuildError:
            continue
    _raise_build_spec_stale(renderer, f"build {build_id} changed during all 3 overwrite attempts.")


def _parse_release_targets(renderer, values: Sequence[str]) -> list[BuildTarget]:
    """Every ``--target`` parsed, or one envelope naming the expected form."""
    try:
        return parse_build_targets(values)
    except BuildTargetInvalidError as error:
        renderer.error(
            code="build_missing_input",
            message=str(error),
            hint=f"pass --target {TARGET_FORM}, for example --target linux/nvidia",
            details={"invalid": list(error.values)},
        )
        raise typer.Exit(code=1) from error


def _prompt_release_targets(renderer, builder_url: str | None) -> list[str] | None:
    """Tenet 1's picker: the builder's own build-targets catalog, never a default.

    ``None`` back — an aborted prompt, or a catalog with nothing offerable —
    falls through to ``require_option``'s refusal rather than cutting a release
    for targets nobody chose.
    """
    from comfy_cli.ui import prompt_multi_select

    client = _builder_client(renderer, builder_url)
    choices = catalog_choices(_builder_call(renderer, client.list_build_targets))
    if not choices:
        return None
    return prompt_multi_select("Build targets to release", choices) or None


def _write_spec(renderer, path: Path, spec: dict) -> None:
    try:
        write_build_spec(path, spec)
    except (BuildSpecInvalidError, BuildSpecWriteError) as error:
        renderer.error(code=error.code, message=str(error), details={"path": str(path)})
        raise typer.Exit(code=1) from error


@app.command("push", help="Push the local comfy-build spec to the builder.")
@tracking.track_command("build")
def push_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    build_id: Annotated[
        str | None, typer.Option("--id", help="Push to this Build id instead of the spec's id.")
    ] = None,
    release: Annotated[
        bool,
        typer.Option("--release", help="Cut a release for every --target once the push lands."),
    ] = False,
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            help=f"Build target to release, as {TARGET_FORM} (e.g. linux/nvidia). Repeatable; required with --release.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite remote changes, retrying a bounded GET-then-PATCH."),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Compute uploads locally; send no HTTP requests.")] = False,
    models_dir: Annotated[
        str | None,
        typer.Option("--models-dir", help="Models folder used to resolve model localPath values."),
    ] = None,
    custom_nodes_dir: Annotated[
        str | None,
        typer.Option("--custom-nodes-dir", help="Custom-nodes folder used to resolve node localPath values."),
    ] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    if release and dry_run:
        renderer.error(
            code="build_missing_input",
            message="--release cuts a release from a real push, and --dry-run sends nothing.",
            hint="drop --dry-run to cut the release, or drop --release to preview the push",
            details={"conflict": ["--release", "--dry-run"]},
        )
        raise typer.Exit(code=1)
    targets = _parse_release_targets(renderer, target or ())
    if targets and not release:
        renderer.error(
            code="build_missing_input",
            message="--target applies only to the release --release cuts.",
            hint="pass --release to cut a release for these targets",
            details={"missing": ["--release"]},
        )
        raise typer.Exit(code=1)
    if release and not targets:
        targets = _parse_release_targets(
            renderer,
            require_option(
                "--target",
                None,
                prompt_fn=lambda: _prompt_release_targets(renderer, builder_url),
                error_code="build_missing_input",
                ctx=ctx,
            ),
        )
    client = None if dry_run else _builder_client(renderer, builder_url)
    try:
        paths = resolve_build_paths(
            path,
            overrides=InstallOverrides.from_options(models_dir, custom_nodes_dir),
        )
    except BuildSpecNotFoundError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    spec = _read_spec(renderer, paths.spec_file)
    stored_id = spec.get("id")
    if build_id is not None and stored_id is not None and build_id != stored_id and not force:
        _raise_build_spec_stale(
            renderer,
            f"--id {build_id} differs from the spec's Build id {stored_id}; its syncedRevision belongs to another Build.",
        )
    # The scratch directory holds one archive per local node, and it is released
    # the moment the last upload lands: nothing after this block reads an
    # archive, and the create/update round-trip that follows can take a while.
    with tempfile.TemporaryDirectory(prefix="comfy-build-push-") as package_dir:
        try:
            validate_local_build_spec(spec, paths)
            preparation = prepare_push(spec, paths, ModelDigestCache(_sha256_file), package_dir=Path(package_dir))
        except NodePackageError as error:
            _raise_node_package_error(renderer, error)
        except BuildSpecInvalidError as error:
            renderer.error(code=error.code, message=str(error), details={"path": str(paths.spec_file)})
            raise typer.Exit(code=1) from error

        uploads = pending_uploads(preparation)
        payload = {
            "spec_file": str(paths.spec_file),
            "id": build_id or stored_id,
            "syncedRevision": spec.get("syncedRevision"),
            "created": False,
            "dry_run": dry_run,
            "upload_count": len(uploads),
            "upload_bytes": sum(upload.size_bytes for upload in uploads),
            "uploaded": 0,
            "deduped": 0,
        }
        reported = _warn_skipped_symlinks(renderer, preparation.skipped_symlinks)
        if reported:
            payload["skipped_symlinks"] = reported
        if dry_run:
            renderer.emit(payload, command="build push", changed=False)
            return
        assert client is not None

        if preparation.stale_source_uri_models:
            _write_spec(renderer, paths.spec_file, spec_without_stale_source_uris(spec, preparation))

        models = unresolved_models(preparation)
        if models:
            _builder_call(renderer, lambda: resolve_models_via_builder(models, client))
        uploads = pending_uploads(preparation)

        public_nodes = public_node_projection(preparation.definition)
        if public_nodes:
            imported = _builder_call(
                renderer,
                lambda: client.resolve_snapshot(snapshot_from_definition({"customNodes": public_nodes})),
            )
            checked = (imported.get("definition") or {}).get("customNodes") or []
            try:
                missing = public_node_identities(public_nodes) - public_node_identities(
                    public_node_projection({"customNodes": checked})
                )
            except BuildSpecInvalidError as error:
                renderer.error(code=error.code, message=str(error), details={"path": str(paths.spec_file)})
                raise typer.Exit(code=1) from error
            if missing:
                labels = sorted(f"{identity.kind}:{identity.value}@{identity.version or '-'}" for identity in missing)
                renderer.error(
                    code="build_registry_pin_missing",
                    message="the builder could not resolve these custom node identities: " + ", ".join(labels),
                    hint="edit the spec to name a published registry version or normalized repository, or remove the node",
                )
                raise typer.Exit(code=1)

        # Checkpoint the reconciled spec after every blob so an interrupted push
        # resumes instead of restarting: `prepare_push` skips entries that
        # already carry a `blobId`, and until this lands on disk the ids exist
        # only in memory — a crash would re-upload the same bytes under new ids
        # and orphan the ones the builder already stored.
        uploaded = _builder_call(
            renderer,
            lambda: upload_assets(
                preparation, client, lambda: _write_spec(renderer, paths.spec_file, preparation.spec)
            ),
        )
    wire_definition = project_wire_definition(preparation.definition)
    target_id = build_id or stored_id
    name = str(spec["name"])
    description = str(spec["description"])
    if target_id is None:
        target_id = _builder_call(renderer, lambda: client.create_build(name, wire_definition, description))
        saved = _builder_call(renderer, lambda: client.get_build(target_id))
        created = True
    elif force:
        saved = _force_update(renderer, client, target_id, wire_definition, name, description)
        created = False
    else:
        try:
            saved = _builder_call(
                renderer,
                lambda: _update_for_push(
                    client,
                    target_id,
                    wire_definition,
                    str(spec["syncedRevision"]) if spec.get("syncedRevision") is not None else None,
                    name,
                    description,
                ),
            )
        except _StaleBuildError:
            _raise_build_spec_stale(renderer, f"build {target_id} changed since this spec was last synchronized.")
        created = False

    preparation.spec["id"] = target_id
    preparation.spec["syncedRevision"] = saved["updatedAt"]
    _write_spec(renderer, paths.spec_file, preparation.spec)
    payload.update(
        {
            "id": target_id,
            "syncedRevision": saved["updatedAt"],
            "created": created,
            "upload_count": len(uploads),
            "upload_bytes": sum(upload.size_bytes for upload in uploads),
            "uploaded": uploaded,
            "deduped": len(uploads) - uploaded,
        }
    )
    release_summary: dict[str, str] | None = None
    if release:
        requested = [item.as_wire() for item in targets]
        release_id, status_url = _builder_call(renderer, lambda: client.create_release(target_id, requested))
        release_summary = {"releaseId": release_id, "statusUrl": status_url}
        payload["targets"] = requested
        payload["release"] = release_summary
    if renderer.is_pretty():
        deduped = len(uploads) - uploaded
        summary = f"{uploaded} upload(s)" + (f", {deduped} already stored" if deduped else "")
        renderer.success(f"Pushed build {target_id} ({summary})")
        if release_summary is not None:
            renderer.print(f"  release: {release_summary['releaseId']}")
            renderer.print(f"  status:  {release_summary['statusUrl']}")
    renderer.emit(payload, command="build push", changed=True)


def _prompt_build_id(renderer, client) -> str | None:
    from comfy_cli.ui import prompt_select

    choices = []
    for remote in _builder_call(renderer, client.list_builds):
        build_id = remote.get("id")
        if not isinstance(build_id, str) or not build_id.strip():
            continue
        name = remote.get("name")
        label = f"{name} ({build_id})" if isinstance(name, str) and name else build_id
        choices.append({"name": label, "value": build_id})
    selected = prompt_select("Build", choices) if choices else None
    return selected if isinstance(selected, str) and selected else None


@app.command("pull", help="Replace the local spec with a fetched Build while retaining local asset identities.")
@tracking.track_command("build")
def pull_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    build_id: Annotated[str | None, typer.Option("--id", help="Pull this Build id instead of the spec's id.")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Overwrite the local spec without confirming.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Fetch the Build, print the diff, and leave the spec file untouched."),
    ] = False,
    models_dir: Annotated[
        str | None,
        typer.Option("--models-dir", help="Models folder used to recompute model content identities."),
    ] = None,
    custom_nodes_dir: Annotated[
        str | None,
        typer.Option("--custom-nodes-dir", help="Custom-nodes folder used to recompute node content identities."),
    ] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    try:
        paths = resolve_build_paths(
            path,
            overrides=InstallOverrides.from_options(models_dir, custom_nodes_dir),
        )
    except BuildSpecNotFoundError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    spec = _read_spec(renderer, paths.spec_file)
    target_id = require_option(
        "--id",
        build_id or spec.get("id"),
        prompt_fn=lambda: _prompt_build_id(renderer, client),
        error_code="build_id_unknown",
        ctx=ctx,
    )
    remote = _builder_call(renderer, lambda: client.get_build(target_id))
    try:
        # No `package_dir`: pull wants only the reconciled spec and the skip
        # report, never `prepared.uploads`, so it materializes no archive.
        prepared = prepare_push(spec, paths, ModelDigestCache(_sha256_file), package_dir=None)
        pulled = merge_pulled_spec(prepared.spec, remote, target_id)
    except NodePackageError as error:
        _raise_node_package_error(renderer, error)
    except UnsyncedDefinitionError as error:
        renderer.error(
            code=error.code,
            message=str(error),
            details={"path": str(paths.spec_file), "id": target_id, "fields": list(error.fields)},
        )
        raise typer.Exit(code=1) from error
    except BuildSpecInvalidError as error:
        renderer.error(code=error.code, message=str(error), details={"path": str(paths.spec_file)})
        raise typer.Exit(code=1) from error

    # Baselined on the definition on disk, never `prepared`'s: `prepare_push`
    # recomputes model `sha256` and node `localDigest`, and this write lands
    # those too, so diffing the reconciled copy would hide them.
    diff = diff_definitions(spec["definition"], pulled.definition)
    summary = summarize_definition_diff(diff)
    if renderer.is_pretty():
        render_definition_diff(renderer, diff)

    payload = {
        "spec_file": str(paths.spec_file),
        "id": target_id,
        "syncedRevision": pulled.spec["syncedRevision"],
        "dry_run": dry_run,
        "written": False,
        "summary": summary,
        "diff": diff.as_json(),
        "definition": pulled.definition,
    }
    # `pull` carries the local localDigest forward into the spec it writes
    # (`build_pull._NODE_LOCAL_FIELDS`), so it mints the same committed identity
    # `init`, `update` and `push` do and owes the same skip report — but only
    # for the nodes that survive the merge, renumbered into the merged order.
    reported = _warn_skipped_symlinks(
        renderer, _relocate_skipped_symlinks(prepared.skipped_symlinks, pulled.definition)
    )
    if reported:
        payload["skipped_symlinks"] = reported

    # Before the confirmation, not after: --dry-run promises to write nothing,
    # and a prompt whose only outcome is a write it will not perform is noise.
    if dry_run:
        if renderer.is_pretty():
            renderer.info(f"--dry-run: {paths.spec_file} left untouched.")
        renderer.emit(payload, command="build pull", changed=False)
        return

    if not confirm(
        f"Pull build {target_id} and overwrite the local spec at {paths.spec_file} ({summary})?",
        yes=yes,
        error_code="build_pull_needs_confirm",
        ctx=ctx,
    ):
        # Declining needs a prompt, and a prompt needs pretty mode, where `emit`
        # is a no-op; a machine caller is refused above with
        # `build_pull_needs_confirm` rather than reaching this branch.
        renderer.info("Aborted.")
        return

    _write_spec(renderer, paths.spec_file, pulled.spec)
    payload["written"] = True
    if renderer.is_pretty():
        renderer.success(f"Pulled build {target_id} → {paths.spec_file}")
    renderer.emit(payload, command="build pull", changed=True)


def _optional_str(value: object) -> str | None:
    """Proven, never coerced: ``str()`` on a mistyped wire field would render a
    stringified ``{...}`` as a perfectly plausible revision."""
    return value if isinstance(value, str) and value else None


def _render_status(renderer, payload: dict, drift: DefinitionDiff | None) -> None:
    build = payload["build"]
    remote = payload["remote"]
    renderer.print(f"build:  {build['id']}" + (f"  {build['name']}" if build["name"] else ""))
    renderer.print(f"spec:   {payload['spec']['path']}")
    renderer.print(f"remote: {'behind' if remote['behind'] else 'in sync'} (revision {remote['revision'] or '—'})")
    # "not compared" rather than a dash: silence here would read as "no drift".
    local = summarize_definition_diff(drift) if drift is not None else f"not compared ({payload['local']['reason']})"
    renderer.print(f"local:  {local}")
    if payload["hint"]:
        renderer.warn("the remote Build has moved since this spec was synchronized", hint=payload["hint"])


@app.command("status", help="Report how far the local spec is from the remote Build and from the install.")
@tracking.track_command("build")
def status_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    build_id: Annotated[
        str | None, typer.Option("--id", help="Report on this Build id instead of the spec's id.")
    ] = None,
    models_dir: Annotated[
        str | None,
        typer.Option("--models-dir", help="Models folder to rescan. Default: PATH/models/."),
    ] = None,
    custom_nodes_dir: Annotated[
        str | None,
        typer.Option("--custom-nodes-dir", help="Custom-nodes folder to rescan. Default: PATH/custom_nodes/."),
    ] = None,
    python: Annotated[
        str | None,
        typer.Option("--python", help="ComfyUI's Python executable, to capture pip deps for the rescan."),
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
    no_scan: Annotated[
        bool,
        typer.Option("--no-scan", help="Report spec-vs-remote only; never scan the install."),
    ] = False,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    # Auth first, before the spec is even read: a signed-out `status` could still
    # compute local drift, and reporting only that half silently answers the less
    # interesting question (build design lines 327-329, 346-349).
    client = _builder_client(renderer, builder_url)
    try:
        paths = resolve_build_paths(
            path,
            overrides=InstallOverrides.from_options(models_dir, custom_nodes_dir, python),
        )
    except BuildSpecNotFoundError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    spec = _read_spec(renderer, paths.spec_file)
    target_id = require_option(
        "--id",
        build_id or spec.get("id"),
        prompt_fn=lambda: _prompt_build_id(renderer, client),
        error_code="build_id_unknown",
        ctx=ctx,
    )
    remote = _builder_call(renderer, lambda: client.get_build(target_id))
    # The wire schema carries `updatedAt`, never a `revision` field, and
    # `updatedAt` is exactly the token `syncedRevision` mirrors — so the report
    # renders it under `remote.revision` but the comparison reads `updatedAt`.
    revision = _optional_str(remote.get("updatedAt"))
    behind = _optional_str(spec.get("syncedRevision")) != revision

    stored = spec["definition"]
    # Independent halves: only the drift one needs an install, so a hand-authored
    # spec must still be comparable against its remote.
    drift, local = _status_drift(
        renderer, ctx, paths, stored, no_scan=no_scan, comfy_version=comfy_version, comfy_url=comfy_url
    )

    payload = {
        "build": {"id": target_id, "name": _optional_str(remote.get("name"))},
        "spec": {"path": str(paths.spec_file), "syncedRevision": _optional_str(spec.get("syncedRevision"))},
        "remote": {"revision": revision, "behind": behind},
        "local": local,
        # In the payload rather than only on stderr: `emit` has no hint field,
        # and the agent reading this envelope is who needs the next step.
        "hint": "run `comfy build pull` to take the remote changes" if behind else None,
    }
    if renderer.is_pretty():
        _render_status(renderer, payload, drift)
    renderer.emit(payload, command="build status", changed=False)


def _status_drift(renderer, ctx, paths, stored: dict, *, no_scan: bool, comfy_version, comfy_url):
    """Spec-vs-install drift, or a stated reason there is none to compute.

    Returns ``(diff | None, local_payload)``. ``local.scanned`` is the flag a
    consumer branches on: ``false`` means the drift half was not computed and
    ``local.reason`` says why, NOT that the spec and the install agree.
    """
    if no_scan:
        return None, {"scanned": False, "reason": "--no-scan was passed", "drift": None}
    try:
        scanned = _scan_install(
            renderer, ctx, ScanRequest(paths, comfy_version=comfy_version, comfy_url=comfy_url), optional=True
        ).definition
    except ScanUnavailableError as error:
        renderer.warn(f"Skipping spec-vs-install drift: {error}", hint=error.hint)
        return None, {"scanned": False, "reason": str(error), "drift": None}
    # Merged before diffing, exactly as `update` does, so drift means "an
    # `update` would rewrite the spec": the cached keys a scan never reports
    # (`blobId`, a resolver's `sourceUri`) must not read as differences.
    drift = diff_definitions(stored, merge_definition(stored, scanned))
    return drift, {"scanned": True, "drift": drift.as_drift()}


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


def _report_builder_error(renderer, e) -> None:
    """Emit one error envelope for a builder failure. Prefers the limited-beta 403,
    then the builder's own error body (e.g. `INVALID_DEFINITION: …` or
    `SUBSCRIPTION_REQUIRED: …`) over urllib's opaque "HTTP Error 400", then the
    generic transport error.

    It carries no Build id, because the old `create` verb's orphan case is gone:
    `push` writes the id into the spec on disk before it cuts, so a cut that
    fails afterwards leaves the id in the user's own file rather than only in
    this envelope."""
    import urllib.error

    from comfy_cli.http import tls_trust_hint, tls_verification_failed

    # Ahead of the transport clause below, whose hint ("check the builder URL and
    # your access") points away from a failure that is neither: the endpoint and
    # the credential are both fine and the CA store is the problem.
    if tls_verification_failed(e):
        renderer.error(
            code="tls_verify_failed", message=f"TLS certificate verification failed: {e}", hint=tls_trust_hint()
        )
        return
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
            )
            return
        detail = _builder_msg(body) or getattr(e, "reason", None) or str(e)
        renderer.error(
            code="build_builder_error",
            message=f"builder call failed ({e.code}): {detail}",
            details={"status": e.code, "body": body[:1000]},
        )
        return
    renderer.error(code="build_builder_error", message=f"builder call failed: {e}")


def _builder_call(renderer, fn):
    """Run a builder API call, mapping every failure class to one error envelope
    + exit(1) via _report_builder_error.

    Never package inside *fn*. ``NodePackageError`` is a ``ValueError``, so the
    clause below would relabel a packaging failure as ``build_missing_input``
    and drop the node path from ``details`` — the contract
    ``_raise_node_package_error`` exists to hold. Package before the call.
    """
    import urllib.error

    from comfy_cli.http import ResponseTooLarge

    try:
        return fn()
    except ResponseTooLarge as e:
        # Mostly the build log outgrowing even the raised logs cap; a clear message
        # beats an unhandled traceback.
        renderer.error(code="build_builder_error", message=f"builder response exceeded the client size cap ({e})")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        renderer.error(code="build_missing_input", message=str(e))
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


@dataclass(frozen=True, slots=True)
class _BuildScope:
    ctx: typer.Context
    path: str | None
    build_id: str | None


def _resolve_build_id(renderer, client, scope: _BuildScope) -> str:
    if scope.build_id is not None:
        return scope.build_id
    try:
        paths = resolve_build_paths(scope.path)
    except BuildSpecNotFoundError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    spec = _read_spec(renderer, paths.spec_file)
    return require_option(
        "--id",
        spec.get("id"),
        prompt_fn=lambda: _prompt_build_id(renderer, client),
        error_code="build_id_unknown",
        ctx=scope.ctx,
    )


@app.command("ls", help="List the workspace's builds.")
@tracking.track_command("build")
def ls_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    builds = _builder_call(renderer, client.list_builds)
    if renderer.is_pretty():
        if not builds:
            renderer.info("No builds yet.")
        for build in builds:
            renderer.print(f"  {build.get('id', '?')}  {build.get('name', '')}")
    renderer.emit({"builds": builds}, command="build ls")


@app.command("show", help="Show a build and its full definition.")
@tracking.track_command("build")
def show_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    build_id: Annotated[str | None, typer.Option("--id", help="Show this Build id instead of the spec's id.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    selected_build_id = _resolve_build_id(renderer, client, _BuildScope(ctx, path, build_id))
    dist = _builder_call(renderer, lambda: client.get_build(selected_build_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(dist))
    renderer.emit(dist, command="build show")


_RELEASE_POLL_SECONDS = 2.0


#: Sorts below every usable timestamp, so a release the builder dated badly loses
#: the tiebreak instead of winning it on a value nobody can read. A Go zero
#: ``time.Time`` marshals to ``0001-01-01T00:00:00Z`` and ties with it, which is
#: the same answer: neither row carries a date worth ordering on.
_UNDATED: Final = datetime.min.replace(tzinfo=timezone.utc)


def _release_order(release: dict) -> tuple[int, datetime]:
    """Order releases by version, breaking ties on the instant they were cut.

    The instant, not its spelling: the builder marshals Go ``time.Time``, whose
    trailing-zero trimming leaves the fractional part a variable width, and
    comparing those as text sorts a whole-second ``...:16Z`` above the strictly
    later ``...:16.5Z`` — ``.`` precedes ``Z`` in ASCII.
    """
    version = release.get("version")
    created_at = release.get("createdAt")
    try:
        created = parse_rfc3339(created_at) if isinstance(created_at, str) else _UNDATED
    except ValueError:
        created = _UNDATED
    return (version if isinstance(version, int) else -1, created)


def _newest_release_id(renderer, client, build_id: str) -> str:
    releases = _builder_call(renderer, lambda: client.list_releases(build_id))
    if not releases:
        renderer.error(
            code="build_release_not_found",
            message=f"Build {build_id} has no releases.",
            hint=f"run `comfy build release create --target {TARGET_FORM}` first",
            details={"buildId": build_id},
        )
        raise typer.Exit(code=1)
    release_id = max(releases, key=_release_order).get("id")
    if not isinstance(release_id, str) or not release_id:
        renderer.error(
            code="build_builder_error",
            message="the builder returned a release without an id.",
            details={"buildId": build_id},
        )
        raise typer.Exit(code=1)
    return release_id


def _selected_release_id(renderer, client, scope: _BuildScope, release: str | None) -> str:
    if release is not None:
        return release
    return _newest_release_id(renderer, client, _resolve_build_id(renderer, client, scope))


def _poll_release(renderer, client, release_id: str) -> dict:
    while True:
        release = _builder_call(renderer, lambda: client.get_release(release_id))
        status = release.get("status")
        if renderer.is_pretty():
            renderer.info(f"release {release_id}: {status}")
        match status:
            case "complete":
                return release
            case "queued" | "building":
                time.sleep(_RELEASE_POLL_SECONDS)
            case _:
                renderer.error(
                    code="build_builder_error",
                    message=f"the builder returned unknown release status {status!r}.",
                    details={"releaseId": release_id, "status": status},
                )
                raise typer.Exit(code=1)


def _release_failed(release: dict) -> bool:
    counts = release.get("artifactCounts")
    failed = counts.get("failed") if isinstance(counts, dict) else None
    return isinstance(failed, int) and failed > 0


@release_app.command("create", help="Cut a release from the current Build.")
@tracking.track_command("build")
def release_create(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    build_id: Annotated[
        str | None, typer.Option("--id", help="Cut from this Build id instead of the spec's id.")
    ] = None,
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            help=f"Build target, as {TARGET_FORM} (e.g. linux/nvidia). Repeatable and required.",
        ),
    ] = None,
    watch: Annotated[bool, typer.Option("--watch", help="Poll the release until every target is terminal.")] = False,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    targets = _parse_release_targets(renderer, target or ())
    if not targets:
        targets = _parse_release_targets(
            renderer,
            require_option(
                "--target",
                None,
                prompt_fn=lambda: _prompt_release_targets(renderer, builder_url),
                error_code="build_missing_input",
                ctx=ctx,
            ),
        )
    client = _builder_client(renderer, builder_url)
    selected_build_id = _resolve_build_id(renderer, client, _BuildScope(ctx, path, build_id))
    requested = [item.as_wire() for item in targets]
    release_id, status_url = _builder_call(
        renderer,
        lambda: client.create_release(selected_build_id, requested),
    )
    payload = {
        "buildId": selected_build_id,
        "releaseId": release_id,
        "statusUrl": status_url,
        "targets": requested,
        "watched": watch,
    }
    if renderer.is_pretty():
        renderer.success(f"Cut release {release_id}")
        renderer.print(f"  status: {status_url}")
    if watch:
        release = _poll_release(renderer, client, release_id)
        payload["release"] = release
        if _release_failed(release):
            renderer.error(
                code="build_builder_error",
                message=f"release {release_id} completed with failed targets.",
                details={"releaseId": release_id, "artifactCounts": release.get("artifactCounts")},
            )
            raise typer.Exit(code=1)
    renderer.emit(payload, command="build release create", changed=True)


@release_app.command("ls", help="List the current Build's releases.")
@tracking.track_command("build")
def release_ls(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    build_id: Annotated[str | None, typer.Option("--id", help="List releases for this Build id.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    selected_build_id = _resolve_build_id(renderer, client, _BuildScope(ctx, path, build_id))
    releases = _builder_call(renderer, lambda: client.list_releases(selected_build_id))
    if renderer.is_pretty():
        if not releases:
            renderer.info("No releases yet.")
        for release in releases:
            renderer.print(f"  {release.get('id', '?')}  {release.get('status', '')}")
    renderer.emit({"buildId": selected_build_id, "releases": releases}, command="build release ls")


@release_app.command("show", help="Show a release's per-target build status.")
@tracking.track_command("build")
def release_show(
    ctx: typer.Context,
    release: Annotated[
        str | None, typer.Argument(help="Release id. Default: the current Build's newest release.")
    ] = None,
    path: Annotated[str | None, typer.Argument(help="Build spec path used when RELEASE is omitted.")] = None,
    build_id: Annotated[str | None, typer.Option("--id", help="Resolve the newest release from this Build id.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    release_id = _selected_release_id(renderer, client, _BuildScope(ctx, path, build_id), release)
    detail = _builder_call(renderer, lambda: client.get_release(release_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(detail))
    renderer.emit(detail, command="build release show")


@release_app.command("logs", help="Read one target's release build log.")
@tracking.track_command("build")
def release_logs(
    ctx: typer.Context,
    release: Annotated[
        str | None, typer.Argument(help="Release id. Default: the current Build's newest release.")
    ] = None,
    path: Annotated[str | None, typer.Argument(help="Build spec path used when RELEASE is omitted.")] = None,
    target: Annotated[
        str | None,
        typer.Option("--target", help=f"Target whose log to read, as {TARGET_FORM} (e.g. linux/nvidia)."),
    ] = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Tail until every target is terminal.")] = False,
    build_id: Annotated[str | None, typer.Option("--id", help="Resolve the newest release from this Build id.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    target_value = require_option(
        "--target",
        target,
        prompt_fn=lambda: None,
        error_code="build_missing_input",
        ctx=ctx,
    )
    selected_target = _parse_release_targets(renderer, [target_value])[0]
    client = _builder_client(renderer, builder_url)
    release_id = _selected_release_id(renderer, client, _BuildScope(ctx, path, build_id), release)
    previous_log = ""
    content: dict = {}
    while True:
        content = _builder_call(
            renderer,
            lambda: client.get_release_logs(release_id, os=selected_target.os, gpu=selected_target.gpu),
        )
        log = content.get("log", "")
        if renderer.is_pretty() and isinstance(log, str) and log:
            renderer.print(log[len(previous_log) :] if log.startswith(previous_log) else log)
            previous_log = log
        if not follow:
            break
        detail = _builder_call(renderer, lambda: client.get_release(release_id))
        match detail.get("status"):
            case "complete":
                break
            case "queued" | "building":
                time.sleep(_RELEASE_POLL_SECONDS)
            case status:
                renderer.error(
                    code="build_builder_error",
                    message=f"the builder returned unknown release status {status!r}.",
                    details={"releaseId": release_id, "status": status},
                )
                raise typer.Exit(code=1)
    if renderer.is_pretty() and not previous_log:
        renderer.print("(no build log captured yet)")
    if content.get("truncated"):
        renderer.warn("log truncated (head and tail kept; middle dropped)")
    # The body is the builder's, and a server predating the version-to-release
    # rename keys the id `versionId`. `build_release_logs.json` requires
    # `releaseId`, so emitting the body untouched would publish a payload that
    # fails this CLI's own schema. The id is already known either way.
    content.pop("versionId", None)
    renderer.emit({**content, "releaseId": release_id}, command="build release logs")


@app.command("validate", help="Validate the local comfy-build spec without contacting the builder.")
@tracking.track_command("build")
def validate_cmd(
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    remote: Annotated[
        bool,
        typer.Option("--remote", help="Also look up public model-source candidates. Requires sign-in."),
    ] = False,
    models_dir: Annotated[
        str | None,
        typer.Option("--models-dir", help="Models folder used to resolve model localPath values."),
    ] = None,
    custom_nodes_dir: Annotated[
        str | None,
        typer.Option("--custom-nodes-dir", help="Custom-nodes folder used to resolve node localPath values."),
    ] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url) if remote else None
    try:
        paths = resolve_build_paths(
            path,
            overrides=InstallOverrides.from_options(models_dir, custom_nodes_dir),
        )
    except BuildSpecNotFoundError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    spec = _read_spec(renderer, paths.spec_file)
    try:
        wire_definition = validate_local_build_spec(spec, paths)
    except BuildSpecInvalidError as error:
        renderer.error(code=error.code, message=str(error), details={"path": str(paths.spec_file)})
        raise typer.Exit(code=1) from error

    result = {
        "spec_file": str(paths.spec_file),
        "remote": remote,
        "wire_definition": wire_definition,
        "model_lookups": [],
    }
    if client is not None:
        lookups = _builder_call(renderer, lambda: lookup_public_model_sources(wire_definition, client.resolve_models))
        result["model_lookups"] = [lookup.as_json() for lookup in lookups]
    if renderer.is_pretty():
        renderer.success(f"Build spec is valid → {paths.spec_file}")
        for lookup in result["model_lookups"]:
            label = lookup["filename"] or lookup["entry"]
            detail = f": {lookup['error']}" if lookup.get("error") else ""
            renderer.info(f"{lookup['state']}  {label}{detail}")
    renderer.emit(result, command="build validate")


@app.command("delete", help="Delete a build (soft-delete).")
@tracking.track_command("build")
def delete_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    build_id: Annotated[str | None, typer.Option("--id", help="Delete this Build id instead of the spec's id.")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    selected_build_id = _resolve_build_id(renderer, client, _BuildScope(ctx, path, build_id))
    if not confirm(
        f"Delete build {selected_build_id}?",
        yes=yes,
        error_code="build_delete_needs_confirm",
        details={"buildId": selected_build_id},
        ctx=ctx,
    ):
        # Declining needs a prompt, and a prompt needs pretty mode, where `emit`
        # is a no-op; a machine caller is refused above with
        # `build_delete_needs_confirm` rather than reaching this branch.
        renderer.info("Aborted.")
        return
    _builder_call(renderer, lambda: client.delete_build(selected_build_id))
    if renderer.is_pretty():
        renderer.success(f"Deleted build {selected_build_id}")
    renderer.emit({"buildId": selected_build_id, "deleted": True}, command="build delete", changed=True)


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


@refs_app.command("resolve", help="Resolve model filenames to public download candidates (HF/CivitAI).")
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
    renderer.emit({"results": results}, command="build refs resolve")


@refs_app.command("base-images", help="List the curated base images a build can build on.")
@tracking.track_command("build")
def base_images_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    images = _builder_call(renderer, client.list_base_images)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(images))
    renderer.emit({"baseImages": images}, command="build refs base-images")


@refs_app.command("build-targets", help="List the build targets (os/gpu) a version can be cut for.")
@tracking.track_command("build")
def build_targets_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    targets = _builder_call(renderer, client.list_build_targets)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(targets))
    renderer.emit({"targets": targets}, command="build refs build-targets")


@refs_app.command("model-dirs", help="List the model directories a model may be placed in.")
@tracking.track_command("build")
def model_dirs_cmd(builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    dirs = _builder_call(renderer, client.list_model_directories)
    if renderer.is_pretty():
        for d in dirs:
            renderer.print(f"  {d}")
    renderer.emit({"directories": dirs}, command="build refs model-dirs")


@release_app.command("manifest", help="Show a release's models and runtime policies.")
@tracking.track_command("build")
def release_manifest(
    ctx: typer.Context,
    release: Annotated[
        str | None, typer.Argument(help="Release id. Default: the current Build's newest release.")
    ] = None,
    path: Annotated[str | None, typer.Argument(help="Build spec path used when RELEASE is omitted.")] = None,
    build_id: Annotated[str | None, typer.Option("--id", help="Resolve the newest release from this Build id.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    release_id = _selected_release_id(renderer, client, _BuildScope(ctx, path, build_id), release)
    manifest = _builder_call(renderer, lambda: client.get_release_manifest(release_id))
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(manifest))
    renderer.emit(manifest, command="build release manifest")


@blob_app.command("ls", help="List the workspace's private blobs.")
@tracking.track_command("build")
def blob_ls(
    kind: Annotated[str | None, typer.Option("--kind", help="Filter by blob kind: model or node_zip.")] = None,
    builder_url: Annotated[str | None, _BUILDER_URL_OPT] = None,
):
    from comfy_cli.builder_pagination import blob_listing_is_clamped

    renderer = get_renderer()
    client = _builder_client(renderer, builder_url)
    blobs = _builder_call(renderer, lambda: client.list_blobs(kind))
    # Unlike builds and releases this endpoint serves one page and mints no
    # cursor, so there is nothing to walk — but a workspace past the server's
    # clamp then reads as a complete listing that simply has no older blobs.
    truncated = blob_listing_is_clamped(blobs)
    if renderer.is_pretty():
        if not blobs:
            renderer.info("No blobs.")
        for b in blobs:
            renderer.print(f"  {b.get('blobId', b.get('id', '?'))}  {b.get('filename', '')}")
        if truncated:
            renderer.warn(
                f"showing the newest {len(blobs)} blobs; the builder serves this list in a single page",
                hint="older blobs are not reachable from this command",
            )
    renderer.emit({"blobs": blobs, "truncated": truncated}, command="build blob ls")
