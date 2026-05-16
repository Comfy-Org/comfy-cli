"""Load the bundled openapi.yml and expose the curated image-endpoint registry.

Lookup order on disk:
1. ``~/.comfy/openapi-cache.yml`` if fresher than CACHE_TTL_DAYS
2. The vendored copy under ``comfy_cli/command/generate/spec/openapi.yml``

The parsed spec is cached in-process via functools.lru_cache so repeated lookups
inside a single CLI invocation don't re-parse the 30k-line YAML.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from yaml import CSafeLoader as _YamlLoader
except ImportError:
    from yaml import SafeLoader as _YamlLoader  # type: ignore[assignment]

PROXY_PREFIX = "/proxy/"
DEFAULT_BASE_URL = "https://api.comfy.org"
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

_BUNDLED_SPEC = Path(__file__).parent / "spec" / "openapi.yml"
_USER_CACHE = Path(os.path.expanduser("~/.comfy/openapi-cache.yml"))


@dataclass(frozen=True)
class Endpoint:
    """A single curated cloud API endpoint, resolved against the openapi spec."""

    id: str  # path with /proxy/ stripped, e.g. "openai/images/generations"
    path: str  # full openapi path, e.g. "/proxy/openai/images/generations"
    method: str  # "post" / "get"
    partner: str  # first path segment under /proxy/
    summary: str
    category: str  # "text-to-image", "image-edit", "upscale", "inpaint", ...
    request_schema: dict[str, Any]  # resolved (no $ref) request body schema
    request_content_type: str  # "application/json" | "multipart/form-data"
    response_schema: dict[str, Any]  # resolved 200 response schema
    polling: str | None  # "bfl" | "kling" | "luma" | "topaz" | None


# Short, creative-facing aliases mapping to the curated openapi paths below.
# Aliases are what end users actually type: `comfy generate flux-pro --prompt …`.
# The full openapi path remains accepted as a power-user escape hatch.
_ALIASES: dict[str, str] = {
    # Flux / BFL
    "flux-pro": "bfl/flux-pro-1.1/generate",
    "flux-ultra": "bfl/flux-pro-1.1-ultra/generate",
    "flux-2": "bfl/flux-2-pro/generate",
    "flux-kontext": "bfl/flux-kontext-pro/generate",
    "flux-kontext-max": "bfl/flux-kontext-max/generate",
    "flux-fill": "bfl/flux-pro-1.0-fill/generate",
    "flux-expand": "bfl/flux-pro-1.0-expand/generate",
    "flux-canny": "bfl/flux-pro-1.0-canny/generate",
    "flux-depth": "bfl/flux-pro-1.0-depth/generate",
    # Ideogram
    "ideogram": "ideogram/ideogram-v3/generate",
    "ideogram-edit": "ideogram/ideogram-v3/edit",
    "ideogram-remix": "ideogram/ideogram-v3/remix",
    "ideogram-reframe": "ideogram/ideogram-v3/reframe",
    "ideogram-bg": "ideogram/ideogram-v3/replace-background",
    # Stability
    "stability-ultra": "stability/v2beta/stable-image/generate/ultra",
    "stability-sd3": "stability/v2beta/stable-image/generate/sd3",
    "stability-upscale": "stability/v2beta/stable-image/upscale/conservative",
    "stability-upscale-creative": "stability/v2beta/stable-image/upscale/creative",
    "stability-upscale-fast": "stability/v2beta/stable-image/upscale/fast",
    # Recraft
    "recraft": "recraft/image_generation",
    "recraft-vectorize": "recraft/images/vectorize",
    "recraft-upscale": "recraft/images/crispUpscale",
    "recraft-upscale-creative": "recraft/images/creativeUpscale",
    "recraft-rmbg": "recraft/images/removeBackground",
    "recraft-replace-bg": "recraft/images/replaceBackground",
    "recraft-i2i": "recraft/images/imageToImage",
    "recraft-inpaint": "recraft/images/inpaint",
    # OpenAI / DALL·E
    "dalle": "openai/images/generations",
    "dalle-edit": "openai/images/edits",
    # xAI / Grok
    "grok": "xai/v1/images/generations",
    "grok-edit": "xai/v1/images/edits",
    # Reve
    "reve": "reve/v1/image/create",
    "reve-edit": "reve/v1/image/edit",
    # Runway
    "runway": "runway/text_to_image",
}

_PREFERRED_ALIAS: dict[str, str] = {v: k for k, v in _ALIASES.items()}


def aliases() -> dict[str, str]:
    """Return a copy of the alias → endpoint-id map (used for `list`)."""
    return dict(_ALIASES)


def preferred_alias(endpoint_id: str) -> str | None:
    """Return the short alias for an endpoint id, if any."""
    return _PREFERRED_ALIAS.get(endpoint_id)


def resolve_alias(target: str) -> str:
    """Map a user-typed model name to the canonical endpoint id.
    Accepts an alias, an endpoint id, or the full /proxy/... path."""
    if target in _ALIASES:
        return _ALIASES[target]
    if target.startswith(PROXY_PREFIX):
        return target[len(PROXY_PREFIX) :]
    return target


# Curated v1 image allowlist. Tuples of (endpoint_id, category, polling).
# Endpoint id is the openapi path with /proxy/ stripped.
_IMAGE_ALLOWLIST: list[tuple[str, str, str | None]] = [
    # OpenAI
    ("openai/images/generations", "text-to-image", None),
    ("openai/images/edits", "image-edit", None),
    # BFL / Flux — all async via polling_url
    ("bfl/flux-pro-1.1/generate", "text-to-image", "bfl"),
    ("bfl/flux-pro-1.1-ultra/generate", "text-to-image", "bfl"),
    ("bfl/flux-kontext-pro/generate", "image-edit", "bfl"),
    ("bfl/flux-kontext-max/generate", "image-edit", "bfl"),
    ("bfl/flux-2-pro/generate", "text-to-image", "bfl"),
    ("bfl/flux-pro-1.0-fill/generate", "inpaint", "bfl"),
    ("bfl/flux-pro-1.0-expand/generate", "outpaint", "bfl"),
    ("bfl/flux-pro-1.0-canny/generate", "controlnet", "bfl"),
    ("bfl/flux-pro-1.0-depth/generate", "controlnet", "bfl"),
    # Ideogram
    ("ideogram/ideogram-v3/generate", "text-to-image", None),
    ("ideogram/ideogram-v3/edit", "image-edit", None),
    ("ideogram/ideogram-v3/remix", "image-edit", None),
    ("ideogram/ideogram-v3/reframe", "image-edit", None),
    ("ideogram/ideogram-v3/replace-background", "image-edit", None),
    # Stability
    ("stability/v2beta/stable-image/generate/ultra", "text-to-image", None),
    ("stability/v2beta/stable-image/generate/sd3", "text-to-image", None),
    ("stability/v2beta/stable-image/upscale/conservative", "upscale", None),
    ("stability/v2beta/stable-image/upscale/creative", "upscale", None),
    ("stability/v2beta/stable-image/upscale/fast", "upscale", None),
    # Recraft
    ("recraft/image_generation", "text-to-image", None),
    ("recraft/images/vectorize", "vectorize", None),
    ("recraft/images/crispUpscale", "upscale", None),
    ("recraft/images/removeBackground", "background", None),
    ("recraft/images/imageToImage", "image-to-image", None),
    ("recraft/images/inpaint", "inpaint", None),
    ("recraft/images/replaceBackground", "background", None),
    ("recraft/images/creativeUpscale", "upscale", None),
    # xAI
    ("xai/v1/images/generations", "text-to-image", None),
    ("xai/v1/images/edits", "image-edit", None),
    # Reve
    ("reve/v1/image/create", "text-to-image", None),
    ("reve/v1/image/edit", "image-edit", None),
    # Runway
    ("runway/text_to_image", "text-to-image", None),
]


class SpecError(RuntimeError):
    pass


def _select_spec_path() -> Path:
    if _USER_CACHE.is_file():
        age = time.time() - _USER_CACHE.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return _USER_CACHE
    if not _BUNDLED_SPEC.is_file():
        raise SpecError(f"openapi.yml not found at {_BUNDLED_SPEC}")
    return _BUNDLED_SPEC


@lru_cache(maxsize=1)
def load_raw_spec() -> dict[str, Any]:
    path = _select_spec_path()
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_YamlLoader)


def base_url() -> str:
    override = os.environ.get("COMFY_API_BASE_URL")
    if override:
        return override.rstrip("/")
    spec = load_raw_spec()
    servers = spec.get("servers") or [{"url": DEFAULT_BASE_URL}]
    return str(servers[0]["url"]).rstrip("/")


def _resolve_ref(spec: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise SpecError(f"Only local $refs are supported: {ref}")
    parts = ref[2:].split("/")
    node: Any = spec
    for p in parts:
        node = node[p]
    return node


def _resolve(spec: dict[str, Any], node: Any, seen: frozenset[str] = frozenset()) -> Any:
    """Recursively inline $refs in a schema. Cycles are broken with a placeholder."""
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if ref in seen:
                return {"type": "object", "x-recursive-ref": ref}
            resolved = _resolve_ref(spec, ref)
            return _resolve(spec, resolved, seen | {ref})
        return {k: _resolve(spec, v, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(spec, item, seen) for item in node]
    return node


def _detect_polling(partner: str, response_schema: dict[str, Any]) -> str | None:
    """Heuristic: classify async polling style by partner + response shape."""
    props = response_schema.get("properties", {}) if isinstance(response_schema, dict) else {}
    if partner == "bfl" and "polling_url" in props:
        return "bfl"
    if partner == "kling" and "data" in props:
        return "kling"
    if partner == "luma" and ("state" in props or "id" in props):
        return "luma"
    if partner == "topaz" and "process_id" in props:
        return "topaz"
    return None


@lru_cache(maxsize=1)
def _registry() -> dict[str, Endpoint]:
    spec = load_raw_spec()
    paths = spec.get("paths") or {}
    registry: dict[str, Endpoint] = {}
    for endpoint_id, category, polling_hint in _IMAGE_ALLOWLIST:
        path = PROXY_PREFIX + endpoint_id
        node = paths.get(path)
        if not node:
            continue  # spec drift — skip silently, surfaced via `comfy api models`
        # All image endpoints are POST; pick the first defined method anyway.
        method = "post" if "post" in node else next(iter(node.keys()))
        op = node[method]
        partner = endpoint_id.split("/", 1)[0]

        req_body = op.get("requestBody") or {}
        content = req_body.get("content") or {}
        if "application/json" in content:
            ctype = "application/json"
        elif "multipart/form-data" in content:
            ctype = "multipart/form-data"
        else:
            ctype = next(iter(content.keys()), "application/json")
        req_schema = _resolve(spec, (content.get(ctype) or {}).get("schema") or {})

        # 200 response
        resp = (op.get("responses") or {}).get("200") or {}
        resp_content = resp.get("content") or {}
        resp_ctype = "application/json" if "application/json" in resp_content else next(iter(resp_content), "")
        resp_schema = _resolve(spec, (resp_content.get(resp_ctype) or {}).get("schema") or {}) if resp_ctype else {}

        polling = polling_hint or _detect_polling(partner, resp_schema)

        registry[endpoint_id] = Endpoint(
            id=endpoint_id,
            path=path,
            method=method,
            partner=partner,
            summary=str(op.get("summary") or op.get("description") or "").strip(),
            category=category,
            request_schema=req_schema if isinstance(req_schema, dict) else {},
            request_content_type=ctype,
            response_schema=resp_schema if isinstance(resp_schema, dict) else {},
            polling=polling,
        )
    return registry


def list_endpoints(
    partner: str | None = None,
    category: str | None = None,
    query: str | None = None,
) -> list[Endpoint]:
    out = list(_registry().values())
    if partner:
        out = [e for e in out if e.partner == partner.lower()]
    if category:
        out = [e for e in out if e.category == category]
    if query:
        q = query.lower()
        out = [e for e in out if q in e.id.lower() or q in e.summary.lower()]
    out.sort(key=lambda e: (e.partner, e.id))
    return out


def get_endpoint(endpoint_id: str) -> Endpoint:
    reg = _registry()
    canonical = resolve_alias(endpoint_id)
    if canonical in reg:
        return reg[canonical]
    raise SpecError(_unknown_endpoint_message(endpoint_id))


def _unknown_endpoint_message(endpoint_id: str) -> str:
    """Build a helpful error suggesting close matches."""
    import difflib

    candidates = list(_registry().keys()) + list(_ALIASES.keys())
    close = difflib.get_close_matches(endpoint_id, candidates, n=3, cutoff=0.5)
    msg = f"Unknown model: {endpoint_id!r}."
    if close:
        msg += "\nDid you mean: " + ", ".join(close) + "?"
    msg += "\nRun `comfy generate list` to see available models."
    return msg


def write_cache(yaml_text: str) -> Path:
    """Write `yaml_text` to the user cache, ensuring the parent dir exists."""
    _USER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _USER_CACHE.write_text(yaml_text, encoding="utf-8")
    # Invalidate in-process cache so the next load picks it up.
    load_raw_spec.cache_clear()
    _registry.cache_clear()
    return _USER_CACHE


def active_spec_path() -> Path:
    return _select_spec_path()
