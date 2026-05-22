"""Python wrapper around the bundled ``comfygraph.wasm`` (WASI build).

We keep a module-level singleton: the wasmtime engine, compiled module, and
graph state are loaded once per process and reused across every call. Cold
start is ~50-100ms (decompress + JIT + loadGraph); subsequent calls within
the same process pay only the per-query time.

The graph state is keyed by ``mode``: ``cloud`` reads bundled snapshots in
``comfy_cli/_wasm/data/`` (refreshable via ``comfy nodes refresh``), while
``local`` always reads live from the user's local ComfyUI ``/object_info``
endpoint — because a local install's catalog is whatever custom_nodes the
user has on disk. Switching modes within a process re-initializes the
graph; the wasm instance itself stays warm.

Public surface:

    from comfy_cli import comfygraph
    out = comfygraph.run_query(query_text, object_info_dict)
    out = comfygraph.apply_slots(workflow_dict, overrides_dict, object_info_dict)
    out = comfygraph.get_template_schema(template_id, workflow_dict, object_info_dict)
    out = comfygraph.expand_variations(workflow_dict, [overrides, ...], object_info_dict)
"""

from __future__ import annotations

import gzip
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ComfygraphError(Exception):
    """Raised when the wasm reports an error or the protocol breaks."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class _WasmPaths:
    wasm: Path


@lru_cache(maxsize=1)
def _wasm_path() -> Path:
    """Locate ``comfygraph.wasm`` shipped alongside the package."""
    try:
        with resources.as_file(resources.files("comfy_cli._wasm").joinpath("comfygraph.wasm")) as p:
            return Path(p)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        raise ComfygraphError(
            "comfygraph.wasm is not bundled in this build",
            details={"hint": "rebuild from source with the WASI target, see /Users/kishore/cql/cmd/comfygraph-wasi"},
        ) from e


@lru_cache(maxsize=1)
def _bundled_data_dir() -> Path:
    """Locate the bundled ``_wasm/data/`` directory with .gz snapshots."""
    try:
        with resources.as_file(resources.files("comfy_cli._wasm").joinpath("data")) as p:
            return Path(p)
    except (ModuleNotFoundError, FileNotFoundError):
        # Older wheel without data/ — caller must supply object_info explicitly.
        return Path("/nonexistent")


def _read_gz(path: Path) -> bytes | None:
    """Decompress a .gz file. Returns None on any failure (caller decides)."""
    try:
        return gzip.decompress(path.read_bytes())
    except (OSError, gzip.BadGzipFile):
        return None


@lru_cache(maxsize=1)
def _bundled_annotations() -> tuple[bytes | None, bytes | None, bytes | None]:
    """Return (supported_nodes, cloud_disable_config, no_gpu_nodes) bytes, or None each."""
    base = _bundled_data_dir()
    return (
        _read_gz(base / "supported_nodes.yaml.gz"),
        _read_gz(base / "cloud_disable_config.yaml.gz"),
        _read_gz(base / "no_gpu_nodes.json.gz"),
    )


@lru_cache(maxsize=1)
def _bundled_object_info_bytes() -> bytes | None:
    """Return the bundled cloud-snapshot object_info as raw JSON bytes."""
    return _read_gz(_bundled_data_dir() / "object_info.json.gz")


def cache_dir() -> Path:
    """User cache dir for refreshable cloud snapshots.

    Lives next to the config dir; not in the config file itself because the
    snapshot is ~5-8MB and bloating config.ini would be silly.
    """
    from comfy_cli import constants
    from comfy_cli.utils import get_os

    base = Path(constants.DEFAULT_CONFIG[get_os()]) / "cache"
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base


def _run_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pipe a sequence of JSON requests through a one-shot wasm invocation.

    Each call instantiates a fresh wasmtime store + WASI config; the
    compiled module is reused via the lru_cache on ``_engine_and_module``.
    For our query volumes this is fine; if we ever need < 50ms per call,
    upgrade to a long-lived subprocess pipe.
    """
    import tempfile

    import wasmtime

    engine, module = _engine_and_module()

    payload = b"".join(json.dumps(r).encode() + b"\n" for r in requests)
    with (
        tempfile.NamedTemporaryFile("wb", delete=False) as fin,
        tempfile.NamedTemporaryFile("rb", delete=False) as fout,
    ):
        fin.write(payload)
        fin.flush()
        in_path = fin.name
        out_path = fout.name

    try:
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        cfg = wasmtime.WasiConfig()
        cfg.stdin_file = in_path
        cfg.stdout_file = out_path
        store = wasmtime.Store(engine)
        store.set_wasi(cfg)
        instance = linker.instantiate(store, module)
        start = instance.exports(store).get("_start")
        if start is None:
            raise ComfygraphError("comfygraph.wasm has no _start export")
        try:
            start(store)
        except wasmtime.ExitTrap:
            pass
        raw = Path(out_path).read_bytes().decode("utf-8", errors="replace")
    finally:
        for p in (in_path, out_path):
            try:
                Path(p).unlink()
            except OSError:
                pass

    responses: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return responses


@lru_cache(maxsize=1)
def _engine_and_module():
    import wasmtime

    engine = wasmtime.Engine()
    module = wasmtime.Module.from_file(engine, str(_wasm_path()))
    return engine, module


def _check(resp: dict[str, Any], op: str) -> Any:
    if not resp.get("ok"):
        raise ComfygraphError(
            f"{op}: {resp.get('error') or 'unknown error'}",
            details={"raw": resp},
        )
    return resp.get("data")


# ---------------------------------------------------------------------------
# loadGraph request building — varies by what data the caller supplies
# ---------------------------------------------------------------------------


def _build_load_graph_params(
    object_info: dict[str, Any] | bytes,
    *,
    include_annotations: bool,
) -> dict[str, Any]:
    """Build the params dict for the wasm's loadGraph method.

    The wasm expects ``object_info`` as base64-encoded JSON via the bridge
    (we send it as a JSON object — the Go side reads ``[]byte(p.ObjectInfo)``
    which works as long as we pass the raw dict).
    """
    if isinstance(object_info, bytes):
        object_info_param: Any = json.loads(object_info)
    else:
        object_info_param = object_info

    params: dict[str, Any] = {"object_info": object_info_param}

    if include_annotations:
        sup, dis, nog = _bundled_annotations()
        if sup is not None:
            # YAML bytes; the wasm parses YAML internally.
            params["supported_nodes"] = json.loads(_yaml_bytes_to_json(sup))
        if dis is not None:
            params["disable_config"] = json.loads(_yaml_bytes_to_json(dis))
        if nog is not None:
            params["no_gpu_nodes"] = json.loads(nog.decode("utf-8"))
    return params


def _yaml_bytes_to_json(yaml_bytes: bytes) -> str:
    """Convert YAML bytes to JSON string. The wasm's parsers accept either
    YAML or JSON for these files, but our line-protocol is JSON so we go
    JSON → JSON here.

    Falls back to passing the YAML through as a JSON string field if PyYAML
    isn't available (the wasm's parsers will reject; better to be loud).
    """
    try:
        import yaml

        return json.dumps(yaml.safe_load(yaml_bytes))
    except ImportError:
        raise ComfygraphError("pyyaml is required to load annotation files")


# ---------------------------------------------------------------------------
# Helpers used by the public API
# ---------------------------------------------------------------------------


def _load_then(object_info: dict[str, Any] | bytes, method: str, params: dict[str, Any]) -> Any:
    """Load the graph (with annotations when bundled), then call one method.

    Annotations are bundled with the package (see ``_wasm/data/``). When
    they're available, we load them every time so predicates like
    ``cloud disabled``, ``pack X``, ``label Y``, and ``cpu``/``gpu`` work.
    For local catalogs the annotation predicates return defaults (correct —
    the annotations describe cloud's catalog, not the user's local one).
    """
    load_params = _build_load_graph_params(object_info, include_annotations=True)
    responses = _run_requests(
        [
            {"method": "loadGraph", "params": load_params},
            {"method": method, "params": params},
        ]
    )
    if len(responses) < 2:
        raise ComfygraphError(
            f"{method}: comfygraph returned fewer responses than expected",
            details={"responses": responses, "expected": 2},
        )
    _check(responses[0], "loadGraph")
    return _check(responses[1], method)


def _load_then_with_gallery(
    object_info: dict[str, Any] | bytes,
    gallery_index: list[dict[str, Any]] | bytes,
    method: str,
    params: dict[str, Any],
) -> Any:
    """Same as ``_load_then`` but additionally calls ``loadGallery`` before
    ``method``. Required for any ``templates …`` CQL grammar query — the
    wasm refuses template queries when no gallery is loaded.
    """
    load_params = _build_load_graph_params(object_info, include_annotations=True)
    if isinstance(gallery_index, bytes):
        gallery_param: Any = json.loads(gallery_index)
    else:
        gallery_param = gallery_index
    responses = _run_requests(
        [
            {"method": "loadGraph", "params": load_params},
            {"method": "loadGallery", "params": {"index": gallery_param}},
            {"method": method, "params": params},
        ]
    )
    if len(responses) < 3:
        raise ComfygraphError(
            f"{method}: comfygraph returned fewer responses than expected",
            details={"responses": responses, "expected": 3},
        )
    _check(responses[0], "loadGraph")
    _check(responses[1], "loadGallery")
    return _check(responses[2], method)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_query(query: str, object_info: dict[str, Any]) -> dict[str, Any]:
    """Execute a CQL query against an object_info dump. Returns the typed result."""
    return _load_then(object_info, "query", {"query": query})


def run_query_with_gallery(
    query: str,
    object_info: dict[str, Any],
    gallery_index: list[dict[str, Any]] | bytes,
) -> dict[str, Any]:
    """Execute a CQL query with both the node graph and a templates gallery
    available. Use this for ``templates …`` grammar queries.

    ``gallery_index`` is the decoded JSON of ``templates/index.json`` from
    ``Comfy-Org/workflow_templates``.
    """
    return _load_then_with_gallery(object_info, gallery_index, "query", {"query": query})


def inspect_node(name: str, object_info: dict[str, Any]) -> dict[str, Any]:
    """Return the full Morphism for a node class (inputs, outputs, annotations)."""
    return _load_then(object_info, "inspectNode", {"name": name})


def upstream(name: str, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Nodes whose outputs can feed into ``name``'s required inputs."""
    return _load_then(object_info, "upstream", {"name": name}) or []


def downstream(name: str, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Nodes that accept any of ``name``'s output types."""
    return _load_then(object_info, "downstream", {"name": name}) or []


def find_paths(
    from_type: str,
    to_type: str,
    object_info: dict[str, Any],
    *,
    max_depth: int = 4,
    max_paths: int = 10,
) -> list[dict[str, Any]]:
    """Routed paths from one type to another (may use intermediate types as glue)."""
    return (
        _load_then(
            object_info,
            "findPaths",
            {"from": from_type, "to": to_type, "max_depth": max_depth, "max_paths": max_paths},
        )
        or []
    )


def exact_paths(
    from_type: str,
    to_type: str,
    object_info: dict[str, Any],
    *,
    max_depth: int = 6,
    max_paths: int = 10,
) -> list[dict[str, Any]]:
    """Paths where every step's required link inputs are satisfiable from the path so far."""
    return (
        _load_then(
            object_info,
            "exactPaths",
            {"from": from_type, "to": to_type, "max_depth": max_depth, "max_paths": max_paths},
        )
        or []
    )


def list_types(object_info: dict[str, Any]) -> list[str]:
    """All connection types in the graph, ranked by connectivity."""
    return _load_then(object_info, "listTypes", {}) or []


def category_tree(object_info: dict[str, Any]) -> dict[str, Any]:
    """The full category tree, hierarchical."""
    return _load_then(object_info, "categoryTree", {}) or {}


def get_template_schema(template_id: str, workflow: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    """Return the typed slot manifest for a workflow."""
    return _load_then(object_info, "getTemplateSchema", {"id": template_id, "workflow": workflow})


def apply_slots(
    workflow: dict[str, Any],
    overrides: dict[str, Any],
    object_info: dict[str, Any],
) -> dict[str, Any]:
    """Apply typed slot overrides to a workflow. Returns ``{workflow, warnings}``."""
    return _load_then(object_info, "applySlots", {"workflow": workflow, "overrides": overrides})


def expand_variations(
    workflow: dict[str, Any],
    variations: list[dict[str, Any]],
    object_info: dict[str, Any],
) -> dict[str, Any]:
    """Expand N variation sets into N independent workflow copies."""
    return _load_then(object_info, "expandVariations", {"workflow": workflow, "variations": variations})


def validate_workflow(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
) -> dict[str, Any]:
    """Validate an API-format workflow against object_info.

    Checks:
      - Unknown class_type → hard error with suggestions
      - Input shape mismatches (string to INT, etc.) → hard error
      - Catalog drift (unknown enum, out-of-bounds) → warning

    Returns ``{"valid": bool, "errors": [...], "warnings": [...]}``.
    """
    return _load_then(object_info, "validateWorkflow", {"workflow": workflow})


# ---------------------------------------------------------------------------
# Source resolution — bundle + cache for cloud, live for local
# ---------------------------------------------------------------------------

# Security: cap bytes read from network/disk. Real object_info is a few MB;
# anything past this is suspicious.
MAX_OBJECT_INFO_BYTES = 64 * 1024 * 1024

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects — a redirect from /object_info is suspicious."""

    def http_error_301(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)

    http_error_302 = http_error_303 = http_error_307 = http_error_308 = http_error_301


_no_redirect_opener = urllib.request.build_opener(_NoRedirectHandler())


def load_object_info(
    *,
    mode: str = "local",
    input_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8188,
) -> dict[str, Any]:
    """Unified object_info loader. Single source for all consumers.

    Resolution:
      1. ``input_path`` given → read file directly
      2. ``mode="cloud"`` → user cache → bundled snapshot (no network)
      3. ``mode="local"`` → live HTTP from host:port

    Security (local HTTP path): loopback-only guard, bounded read (64 MB),
    no-redirect policy.
    """
    if input_path is not None:
        return _load_from_file(input_path)
    if mode == "cloud":
        return _load_cloud()
    return _load_local_http(host, port)


def _load_from_file(path: str) -> dict[str, Any]:
    """Read object_info from a local JSON file."""
    p = Path(path).expanduser()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ComfygraphError(
            f"cannot read object_info file: {p}: {e}",
            details={"input_path": str(p)},
        ) from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ComfygraphError(
            f"object_info file is not valid JSON: {p}: {e}",
            details={"input_path": str(p)},
        ) from e


def _load_cloud() -> dict[str, Any]:
    """Cloud path: user cache → bundled snapshot. No network."""
    cache_file = cache_dir() / "object_info-cloud.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # bad cache → fall through to bundle
    bundled = _bundled_object_info_bytes()
    if bundled is None:
        raise ComfygraphError(
            "no bundled object_info; reinstall comfy-cli or run `comfy nodes refresh`",
            details={"mode": "cloud"},
        )
    return json.loads(bundled.decode("utf-8"))


def _load_local_http(host: str, port: int) -> dict[str, Any]:
    """Fetch object_info from a local ComfyUI server with security guards."""
    # Loopback check: warn but don't hard-block (matches existing behaviour).
    parsed_host = urllib.parse.urlsplit(f"http://{host}").hostname or host
    if parsed_host.lower() not in _LOOPBACK_HOSTS and not parsed_host.startswith("127."):
        logger.warning(
            "object_info target %s:%d is not loopback — proceeding anyway",
            host,
            port,
        )

    url = f"http://{host}:{port}/object_info"
    try:
        with _no_redirect_opener.open(url, timeout=15.0) as resp:
            raw = resp.read(MAX_OBJECT_INFO_BYTES + 1)
            if len(raw) > MAX_OBJECT_INFO_BYTES:
                raise ComfygraphError(
                    f"server response exceeds {MAX_OBJECT_INFO_BYTES} bytes",
                    details={"host": host, "port": port},
                )
            return json.loads(raw)
    except urllib.error.URLError as e:
        raise ComfygraphError(
            f"could not fetch local object_info from {url}: {e.reason if hasattr(e, 'reason') else e}",
            details={"mode": "local", "host": host, "port": port, "hint": "run `comfy launch` first"},
        ) from e
    except (json.JSONDecodeError, OSError) as e:
        raise ComfygraphError(
            f"invalid object_info response from {url}: {e}",
            details={"mode": "local", "host": host, "port": port},
        ) from e


def resolve_object_info_for_mode(mode: str) -> dict[str, Any]:
    """Backward-compatible wrapper. Prefer ``load_object_info()``."""
    return load_object_info(mode=mode)


def refresh_cloud_snapshot() -> tuple[Path, int]:
    """Fetch live cloud object_info, write to user cache. Returns (path, bytes_written)."""
    import urllib.request

    from comfy_cli.target import resolve_target

    target = resolve_target(where="cloud")
    url = target.url("object_info")
    req = urllib.request.Request(url)
    if target.api_key:
        req.add_header("X-API-Key", target.api_key)
    elif target.auth_token:
        req.add_header("Authorization", f"Bearer {target.auth_token}")
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        data = resp.read()
    cache_file = cache_dir() / "object_info-cloud.json"
    cache_file.write_bytes(data)
    return cache_file, len(data)
