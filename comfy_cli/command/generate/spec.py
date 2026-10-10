"""Load the bundled openapi.yml and expose the curated image-endpoint registry.

Lookup order on disk:
1. ``~/.comfy/openapi-cache.yml`` if fresher than CACHE_TTL_DAYS
2. The vendored copy under ``comfy_cli/command/generate/spec/openapi.yml``

The parsed spec is cached in-process via functools.lru_cache so repeated lookups
inside a single CLI invocation don't re-parse the 30k-line YAML.
"""

from __future__ import annotations

import os
import re as _re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class _YamlLoader(yaml.SafeLoader):
    """SafeLoader that strips YAML 1.1's bool aliases for ``on``/``off``/
    ``yes``/``no``/``y``/``n``.

    The vendored openapi uses unquoted ``[on, off]`` and similar as **string**
    enum values (e.g. Kling's ``sound`` field), but PyYAML's default resolvers
    promote them to ``True``/``False`` — which then breaks our flag-rendering
    (`'|'.join` on a list with booleans) and the upstream API contract. Limit
    bool resolution to the YAML 1.2 spelling (``true``/``false`` only)."""


_YamlLoader.yaml_implicit_resolvers = {
    k: [(t, r) for (t, r) in resolvers if t != "tag:yaml.org,2002:bool"]
    for k, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    _re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
# PyYAML's YAML 1.1 float resolver only recognizes scientific notation when a
# decimal point is present (e.g. ``1.0e6``). The remote spec is served as JSON,
# and ``json.dumps`` emits exponent literals WITHOUT a point for very large/small
# floats (e.g. ``1e+16``, ``1e-07``); without this those numeric defaults/bounds
# would silently parse as strings and leak that way into flag schemas. Add a
# resolver for the point-less exponent form so JSON floats round-trip correctly.
_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _re.compile(r"^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)

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
# The full openapi path remains accepted as an escape hatch.
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
    # Video — Kling
    "kling": "kling/v1/videos/text2video",
    "kling-i2v": "kling/v1/videos/image2video",
    "kling-extend": "kling/v1/videos/video-extend",
    "kling-lipsync": "kling/v1/videos/lip-sync",
    # Video — Luma Dream Machine
    "luma": "luma/generations",
    "luma-i2v": "luma/generations/image",
    # Video — MiniMax / Hailuo
    "hailuo": "minimax/video_generation",
    # Video — Runway Gen-3
    "runway-i2v": "runway/image_to_video",
    # Video — Moonvalley
    "moonvalley-t2v": "moonvalley/prompts/text-to-video",
    "moonvalley-i2v": "moonvalley/prompts/image-to-video",
    # Video — Pika
    "pika": "pika/generate/2.2/t2v",
    "pika-i2v": "pika/generate/2.2/i2v",
    # Video — Vidu
    "vidu": "vidu/text2video",
    "vidu-i2v": "vidu/img2video",
    "vidu-extend": "vidu/extend",
    # Video — xAI Grok
    "grok-video": "xai/v1/videos/generations",
    # Google Gemini Flash Image (nano-banana). The model variant lives in the
    # URL path; the adapter substitutes ``--model`` at send time.
    "nano-banana": "vertexai/gemini/{model}",
    # ByteDance Seedance (video).
    "seedance": "byteplus/api/v3/contents/generations/tasks",
}


# Used in the `list` table for endpoints whose openapi summary is empty or too
# generic to convey what the model is.
_SUMMARY_OVERRIDES: dict[str, str] = {
    "vertexai/gemini/{model}": (
        "Google Gemini Flash Image (nano-banana) — text-to-image and image edits "
        "from a prompt plus optional reference images."
    ),
    "byteplus/api/v3/contents/generations/tasks": (
        "ByteDance Seedance — text-to-video and image-to-video (3–12s clips, up to 1080p)."
    ),
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


# Curated endpoint allowlist. Tuples of (endpoint_id, category, polling).
# ``polling`` is the partner-key the poll registry uses (None = sync).
# Endpoint id is the openapi path with /proxy/ stripped.
_ENDPOINT_ALLOWLIST: list[tuple[str, str, str | None]] = [
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
    # Runway - image
    ("runway/text_to_image", "text-to-image", None),
    # Video — Kling
    ("kling/v1/videos/text2video", "text-to-video", "kling"),
    ("kling/v1/videos/image2video", "image-to-video", "kling"),
    ("kling/v1/videos/video-extend", "video-extend", "kling"),
    ("kling/v1/videos/lip-sync", "lipsync", "kling"),
    # Video — Luma
    ("luma/generations", "text-to-video", "luma"),
    ("luma/generations/image", "image-to-video", "luma"),
    # Video — MiniMax / Hailuo
    ("minimax/video_generation", "text-to-video", "minimax"),
    # Video — Runway
    ("runway/image_to_video", "image-to-video", "runway"),
    # Video — Moonvalley
    ("moonvalley/prompts/text-to-video", "text-to-video", "moonvalley"),
    ("moonvalley/prompts/image-to-video", "image-to-video", "moonvalley"),
    # Video — Pika
    ("pika/generate/2.2/t2v", "text-to-video", "pika"),
    ("pika/generate/2.2/i2v", "image-to-video", "pika"),
    # Video — Vidu
    ("vidu/text2video", "text-to-video", "vidu"),
    ("vidu/img2video", "image-to-video", "vidu"),
    ("vidu/extend", "video-extend", "vidu"),
    # Video — xAI Grok
    ("xai/v1/videos/generations", "text-to-video", "xai_video"),
    # Google Gemini Flash Image (nano-banana). Sync; adapter decodes inline data.
    ("vertexai/gemini/{model}", "image-edit", None),
    # ByteDance Seedance (video) — async, custom poller.
    ("byteplus/api/v3/contents/generations/tasks", "text-to-video", "seedance"),
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
    if not isinstance(ref, str):
        raise SpecError(f"Invalid non-string $ref: {ref!r}")
    if not ref.startswith("#/"):
        raise SpecError(f"Only local $refs are supported: {ref}")
    parts = ref[2:].split("/")
    node: Any = spec
    for p in parts:
        node = node[p]
    return node


def _resolve(
    spec: dict[str, Any],
    node: Any,
    seen: frozenset[str] = frozenset(),
    memo: dict[tuple[Any, ...], tuple[Any, bool]] | None = None,
    budget_limit: int | None = None,
    budget: list[int] | None = None,
) -> Any:
    """Recursively inline $refs in a schema. Cycles are broken with a placeholder.

    Cycle-bearing results are keyed by active ref ancestry, so a pruned subtree
    is not reused outside that cycle. Cycle-free results are shared regardless
    of ancestry, and object-identity keys collapse YAML alias DAGs whose shared
    inline nodes carry no ``$ref`` of their own.
    """
    value, _cyclic = _resolve_schema(
        spec,
        node,
        seen,
        memo if memo is not None else {},
        budget
        if budget is not None
        else [budget_limit if budget_limit is not None else _schema_resolution_budget(spec, node)],
    )
    return value


def _schema_resolution_budget(spec: dict[str, Any], node: Any = None) -> int:
    """A linear call budget for malformed ref graphs, including object cycles."""
    stack = [spec]
    if node is not None:
        stack.append(node)
    containers: set[int] = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, (dict, list)) or id(current) in containers:
            continue
        containers.add(id(current))
        stack.extend(current.values() if isinstance(current, dict) else current)
    return max(1_024, len(containers) * 64)


def _resolution_attempt_budget(limit: int, aggregate: list[int]) -> tuple[list[int], int]:
    """Allocate one isolated attempt without removing the operation-wide cap."""
    if aggregate[0] <= 0:
        raise SpecError("Schema resolution exceeded its aggregate safe traversal limit")
    allowance = min(limit, aggregate[0])
    return [allowance], allowance


def _charge_resolution_attempt(aggregate: list[int], allowance: int, remaining: list[int]) -> None:
    aggregate[0] -= allowance - remaining[0]


def _resolve_schema(
    spec: dict[str, Any],
    node: Any,
    seen: frozenset[str],
    memo: dict[tuple[Any, ...], tuple[Any, bool]],
    budget: list[int],
    active_inline: frozenset[int] = frozenset(),
) -> tuple[Any, bool]:
    """Return ``(resolved, contains_cycle_placeholder)`` for :func:`_resolve`."""
    if not isinstance(node, (dict, list)):
        return node, False
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str):
                raise SpecError(f"Invalid non-string $ref: {ref!r}")
            if ref in seen:
                return {"type": "object", "x-recursive-ref": ref}, True
            if id(node) in active_inline:
                return {"type": "object", "x-recursive-object": True}, True
            child_active = active_inline | {id(node)}
            shared_key = ("ref", ref)
            if shared_key in memo:
                return memo[shared_key]
            memo_key = (*shared_key, seen, active_inline)
            if memo_key in memo:
                return memo[memo_key]
            if budget[0] <= 0:
                raise SpecError("Schema resolution exceeded its safe traversal limit")
            budget[0] -= 1
            memo[memo_key] = ({"type": "object", "x-recursive-ref": ref}, True)
            try:
                resolved = _resolve_ref(spec, ref)
                result = _resolve_schema(spec, resolved, seen | {ref}, memo, budget, child_active)
            except BaseException:
                # The placeholder means "currently resolving", not "resolved
                # successfully". Shared registry/hint memos must never reuse it
                # after a missing ref, recursion error, or exhausted budget.
                memo.pop(memo_key, None)
                raise
            if result[1]:
                # An ancestry-scoped placeholder is useful only while this
                # branch is in progress. Retaining every completed cyclic key
                # pins its full seen/active frozensets and grows with paths.
                memo.pop(memo_key, None)
            else:
                memo[memo_key] = result
                memo[shared_key] = result
            return result
        if id(node) in active_inline:
            return {"type": "object", "x-recursive-object": True}, True
        shared_key = ("object", id(node))
        if shared_key in memo:
            return memo[shared_key]
        memo_key = ("object", id(node), seen, active_inline)
        if memo_key in memo:
            return memo[memo_key]
        if budget[0] <= 0:
            raise SpecError("Schema resolution exceeded its safe traversal limit")
        budget[0] -= 1
        memo[memo_key] = ({"type": "object", "x-recursive-object": True}, True)
        value: dict[str, Any] = {}
        cyclic = False
        try:
            for key, child in node.items():
                resolved_child, child_cyclic = _resolve_schema(
                    spec, child, seen, memo, budget, active_inline | {id(node)}
                )
                value[key] = resolved_child
                cyclic = cyclic or child_cyclic
        except BaseException:
            memo.pop(memo_key, None)
            raise
        result = (value, cyclic)
        if cyclic:
            memo.pop(memo_key, None)
        else:
            memo[memo_key] = result
            memo[shared_key] = result
        return result
    if isinstance(node, list):
        if id(node) in active_inline:
            return [], True
        shared_key = ("list", id(node))
        if shared_key in memo:
            return memo[shared_key]
        memo_key = ("list", id(node), seen, active_inline)
        if memo_key in memo:
            return memo[memo_key]
        if budget[0] <= 0:
            raise SpecError("Schema resolution exceeded its safe traversal limit")
        budget[0] -= 1
        memo[memo_key] = ([], True)
        value: list[Any] = []
        cyclic = False
        try:
            for item in node:
                resolved_item, item_cyclic = _resolve_schema(spec, item, seen, memo, budget, active_inline | {id(node)})
                value.append(resolved_item)
                cyclic = cyclic or item_cyclic
        except BaseException:
            memo.pop(memo_key, None)
            raise
        result = (value, cyclic)
        if cyclic:
            memo.pop(memo_key, None)
        else:
            memo[memo_key] = result
            memo[shared_key] = result
        return result
    raise AssertionError("unreachable schema node")


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


def _preferred_media_type(content: dict[str, Any]) -> str:
    if "application/json" in content:
        return "application/json"
    if "multipart/form-data" in content:
        return "multipart/form-data"
    return next(iter(content), "application/json")


@lru_cache(maxsize=1)
def _registry() -> dict[str, Endpoint]:
    spec = load_raw_spec()
    paths = spec.get("paths") or {}
    if not isinstance(paths, dict):
        return {}
    registry: dict[str, Endpoint] = {}
    resolution_budget_limit = _schema_resolution_budget(spec)
    aggregate_resolution_budget = [resolution_budget_limit * 4]
    resolution_memo: dict[tuple[Any, ...], tuple[Any, bool]] = {}
    for endpoint_id, category, polling_hint in _ENDPOINT_ALLOWLIST:
        if aggregate_resolution_budget[0] <= 0:
            break
        path = PROXY_PREFIX + endpoint_id
        node = paths.get(path)
        if not isinstance(node, dict) or not node:
            continue  # spec drift — skip silently, surfaced via `comfy generate list`
        # All image endpoints are POST; pick the first defined method anyway.
        method = (
            "post"
            if isinstance(node.get("post"), dict)
            else next(
                (
                    key
                    for key, value in node.items()
                    if isinstance(key, str)
                    and key.lower() in {"get", "put", "patch", "delete", "options", "head", "trace"}
                    and isinstance(value, dict)
                ),
                None,
            )
        )
        if method is None:
            continue
        op = node[method]
        partner = endpoint_id.split("/", 1)[0]

        req_body = op.get("requestBody") or {}
        if not isinstance(req_body, dict):
            continue
        content = req_body.get("content") or {}
        if not isinstance(content, dict):
            continue
        ctype = _preferred_media_type(content)
        request_media = content.get(ctype) or {}
        if not isinstance(request_media, dict):
            continue
        # Request and response share one allowance for this endpoint, but a
        # malformed endpoint cannot starve every endpoint that follows it. A
        # constant-factor operation-wide allowance prevents many independently
        # failing endpoints from multiplying traversal and memo growth.
        resolution_budget, resolution_allowance = _resolution_attempt_budget(
            resolution_budget_limit, aggregate_resolution_budget
        )
        try:
            req_schema = _resolve(
                spec,
                request_media.get("schema") or {},
                memo=resolution_memo,
                budget=resolution_budget,
            )
        except (AttributeError, KeyError, TypeError, SpecError, RecursionError):
            _charge_resolution_attempt(aggregate_resolution_budget, resolution_allowance, resolution_budget)
            continue

        # A malformed response affects only polling detection; it must not
        # remove an otherwise usable request endpoint from the generate catalog.
        resp_schema: Any = {}
        try:
            responses = op.get("responses") or {}
            resp = (responses.get("200") or {}) if isinstance(responses, dict) else {}
            resp_content = (resp.get("content") or {}) if isinstance(resp, dict) else {}
            if isinstance(resp_content, dict) and resp_content:
                resp_ctype = _preferred_media_type(resp_content)
                response_media = resp_content.get(resp_ctype) or {}
                if isinstance(response_media, dict):
                    resp_schema = _resolve(
                        spec,
                        response_media.get("schema") or {},
                        memo=resolution_memo,
                        budget=resolution_budget,
                    )
        except (AttributeError, KeyError, TypeError, SpecError, RecursionError):
            resp_schema = {}
        _charge_resolution_attempt(aggregate_resolution_budget, resolution_allowance, resolution_budget)

        polling = polling_hint or _detect_polling(partner, resp_schema)

        registry[endpoint_id] = Endpoint(
            id=endpoint_id,
            path=path,
            method=method,
            partner=partner,
            summary=_SUMMARY_OVERRIDES.get(endpoint_id)
            or str(op.get("summary") or op.get("description") or "").strip(),
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


def _extract_enum(
    prop: dict[str, Any],
    _memo: dict[int, list[str] | None] | None = None,
    _string_memo: dict[int, bool] | None = None,
) -> list[str] | None:
    """Pull a string enum out of a resolved property schema — directly, from
    ``items`` (array-typed fields), or from ``anyOf``/``oneOf``/``allOf``
    variants. ``anyOf``/``oneOf`` branches are unioned across their
    string-accepting branches; a free-form string branch makes that union
    unconstrained, while null/integer/object alternatives do not. ``allOf``
    branches and sibling keywords are intersected because every constraint
    must hold. Numeric members are coerced to their string form so an unquoted
    YAML value like ``3.5`` isn't silently dropped. Returns None when no
    finite, non-empty string enum is found."""
    if _memo is None:
        _memo = {}
    if _string_memo is None:
        _string_memo = {}
    memo_key = id(prop)
    if memo_key in _memo:
        return _memo[memo_key]
    # A resolved schema is a shared DAG. Mark this object before descending so
    # repeated branches are linear and a malformed object cycle is harmless.
    _memo[memo_key] = None

    def finish(value: list[str] | None) -> list[str] | None:
        _memo[memo_key] = value
        return value

    constraints: list[list[str]] = []
    const = prop.get("const")
    if isinstance(const, str):
        constraints.append([const])
    elif isinstance(const, int | float) and not isinstance(const, bool):
        constraints.append([str(const)])
    enum = prop.get("enum")
    if isinstance(enum, list):
        values = [str(v) if isinstance(v, int | float) and not isinstance(v, bool) else v for v in enum]
        values = [v for v in values if isinstance(v, str)]
        if values:
            constraints.append(values)
    items = prop.get("items")
    if isinstance(items, dict):
        found = _extract_enum(items, _memo, _string_memo)
        if found:
            constraints.append(found)
    for key in ("anyOf", "oneOf"):
        variants = prop.get(key)
        if not isinstance(variants, list):
            continue
        merged: list[str] = []
        merged_values: set[str] = set()
        merged_results: set[int] = set()
        branch_counts: dict[str, int] = {}
        unconstrained = False
        for variant in variants:
            if variant is False:
                # Boolean false accepts nothing, so it contributes no values
                # and cannot make a finite union unconstrained.
                continue
            if variant is True:
                unconstrained = True
                break
            found = _extract_enum(variant, _memo, _string_memo) if isinstance(variant, dict) else None
            if not found:
                if not isinstance(variant, dict) or _schema_admits_unconstrained_string(variant, _memo=_string_memo):
                    unconstrained = True
                    break
                continue
            if key == "oneOf":
                for value in set(found):
                    branch_counts[value] = branch_counts.get(value, 0) + 1
                for value in found:
                    if value not in merged_values:
                        merged_values.add(value)
                        merged.append(value)
            elif id(found) not in merged_results:
                merged_results.add(id(found))
                for value in found:
                    if value not in merged_values:
                        merged_values.add(value)
                        merged.append(value)
        if not unconstrained and merged:
            constraints.append(
                [value for value in merged if branch_counts.get(value) == 1] if key == "oneOf" else merged
            )
    all_of = prop.get("allOf")
    if isinstance(all_of, list):
        branch_results: set[int] = set()
        for variant in all_of:
            found = _extract_enum(variant, _memo, _string_memo) if isinstance(variant, dict) else None
            if found and id(found) not in branch_results:
                branch_results.add(id(found))
                constraints.append(found)
    if not constraints:
        return finish(None)
    allowed = set(constraints[0])
    for constraint in constraints[1:]:
        allowed.intersection_update(constraint)
    intersected = [value for value in constraints[0] if value in allowed]
    return finish(intersected or None)


def _schema_admits_unconstrained_string(
    schema: dict[str, Any],
    _active: set[int] | None = None,
    _memo: dict[int, bool] | None = None,
) -> bool:
    """Whether ``schema`` can accept arbitrary strings rather than a finite enum."""
    if _active is None:
        _active = set()
    if _memo is None:
        _memo = {}
    schema_id = id(schema)
    if schema_id in _memo:
        return _memo[schema_id]
    if schema_id in _active:
        return True
    _active.add(schema_id)
    try:
        if "const" in schema or "enum" in schema:
            result = False
        else:
            schema_type = schema.get("type")
            if isinstance(schema_type, str):
                result = schema_type == "string"
            elif isinstance(schema_type, list):
                result = "string" in schema_type
            else:
                result = "items" not in schema and "properties" not in schema
            # JSON Schema composition keywords are conjunctive with their
            # siblings. A free-form ``type: string`` does not override a finite
            # sibling ``allOf``, and an ``anyOf``/``oneOf`` only admits arbitrary
            # strings when at least one of its own branches does.
            for key in ("anyOf", "oneOf"):
                variants = schema.get(key)
                if isinstance(variants, list):
                    result = result and any(
                        variant is True
                        or (isinstance(variant, dict) and _schema_admits_unconstrained_string(variant, _active, _memo))
                        or (variant is not False and not isinstance(variant, dict))
                        for variant in variants
                    )
            all_of = schema.get("allOf")
            if isinstance(all_of, list) and all_of:
                result = result and all(
                    variant is True
                    or (isinstance(variant, dict) and _schema_admits_unconstrained_string(variant, _active, _memo))
                    for variant in all_of
                )
        _memo[schema_id] = result
        return result
    finally:
        _active.remove(schema_id)


def model_enum(endpoint_id: str, field: str = "model") -> list[str] | None:
    """Return the model-variant enum the active spec carries for
    ``endpoint_id``'s ``field`` request property, or None when the spec has no
    enum there (callers fall back to their hardcoded lists).

    The request schema is already ``$ref``-resolved by ``_resolve``, so a plain
    property walk suffices. Reading from the active spec (user cache when
    fresh, else the vendored copy) means a spec refresh surfaces new partner
    models with zero code changes."""
    try:
        endpoint = get_endpoint(endpoint_id)
    except SpecError:
        return None
    prop = _find_property(endpoint.request_schema or {}, field)
    if prop is None:
        return None
    return _extract_enum(prop)


def _schema_may_be_object(
    schema: Any,
    _active: set[int] | None = None,
    _memo: dict[int, bool] | None = None,
) -> bool:
    """Whether an OpenAPI/JSON Schema branch can describe a request object."""
    if schema is False:
        return False
    if schema is True or not isinstance(schema, dict):
        return True
    if _active is None:
        _active = set()
    if _memo is None:
        _memo = {}
    schema_id = id(schema)
    if schema_id in _memo:
        return _memo[schema_id]
    if schema_id in _active:
        return True
    _active.add(schema_id)
    try:
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            result = schema_type == "object"
        elif isinstance(schema_type, list):
            result = "object" in schema_type
        elif "const" in schema:
            result = isinstance(schema.get("const"), dict)
        else:
            enum = schema.get("enum")
            if isinstance(enum, list) and enum:
                result = any(isinstance(value, dict) for value in enum)
            else:
                result = True
        # Composition keywords constrain their siblings, including an
        # explicit ``type`` or ``const`` sibling.
        for key in ("anyOf", "oneOf"):
            variants = schema.get(key)
            if isinstance(variants, list):
                result = result and any(_schema_may_be_object(variant, _active, _memo) for variant in variants)
        all_of = schema.get("allOf")
        if isinstance(all_of, list) and all_of:
            result = result and all(_schema_may_be_object(variant, _active, _memo) for variant in all_of)
        _memo[schema_id] = result
        return result
    finally:
        _active.remove(schema_id)


def _find_property(
    schema: dict[str, Any],
    field: str,
    _memo: dict[tuple[int, str], dict[str, Any] | None] | None = None,
    _active: set[tuple[int, str]] | None = None,
    _object_memo: dict[int, bool] | None = None,
    _property_memo: dict[tuple[int, str], bool] | None = None,
) -> dict[str, Any] | None:
    """Locate ``field`` in ``schema['properties']``, descending into top-level
    ``allOf``/``anyOf``/``oneOf`` composition when the schema carries no direct
    match — a composed request body must not silently defeat the spec-derived
    enum and fall back to the hardcoded list."""
    if _memo is None:
        _memo = {}
    if _active is None:
        _active = set()
    if _object_memo is None:
        _object_memo = {}
    if _property_memo is None:
        _property_memo = {}
    memo_key = (id(schema), field)
    if memo_key in _memo:
        return _memo[memo_key]
    if memo_key in _active:
        return None
    _active.add(memo_key)
    candidates: list[dict[str, Any]] = []
    props = schema.get("properties")
    if isinstance(props, dict):
        prop = props.get(field)
        if isinstance(prop, dict):
            candidates.append(prop)
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for variant in all_of:
            if isinstance(variant, dict):
                found = _find_property(variant, field, _memo, _active, _object_memo, _property_memo)
                if found is not None:
                    candidates.append(found)
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if not isinstance(variants, list) or not variants:
            continue
        matches: list[dict[str, Any]] = []
        for variant in variants:
            found = (
                _find_property(variant, field, _memo, _active, _object_memo, _property_memo)
                if isinstance(variant, dict)
                else None
            )
            if found is None:
                if _schema_may_accept_property(variant, field, _object_memo=_object_memo, _memo=_property_memo):
                    matches = []
                    break
                continue
            matches.append(found)
        if matches:
            # ``oneOf`` is exclusive for the whole object. Two otherwise
            # distinct request variants may legally share this property's
            # value, so the lifted property union is inclusive.
            lifted_key = "anyOf" if key == "oneOf" else key
            candidates.append(matches[0] if len(matches) == 1 else {lifted_key: matches})
    _active.remove(memo_key)
    result = candidates[0] if len(candidates) == 1 else ({"allOf": candidates} if candidates else None)
    _memo[memo_key] = result
    return result


def _schema_may_accept_property(
    schema: Any,
    field: str,
    _active: set[tuple[int, str]] | None = None,
    _memo: dict[tuple[int, str], bool] | None = None,
    _object_memo: dict[int, bool] | None = None,
) -> bool:
    """Whether an object branch can contain ``field`` at all."""
    if schema is False:
        return False
    if schema is True or not isinstance(schema, dict):
        return True
    if _active is None:
        _active = set()
    if _memo is None:
        _memo = {}
    if _object_memo is None:
        _object_memo = {}
    key = (id(schema), field)
    if key in _memo:
        return _memo[key]
    if key in _active:
        return True
    if not _schema_may_be_object(schema, _memo=_object_memo):
        return False
    _active.add(key)
    try:
        props = schema.get("properties")
        result = (isinstance(props, dict) and field in props) or schema.get("additionalProperties") is not False
        for keyword in ("anyOf", "oneOf"):
            variants = schema.get(keyword)
            if isinstance(variants, list):
                result = result and any(
                    _schema_may_accept_property(variant, field, _active, _memo, _object_memo) for variant in variants
                )
        all_of = schema.get("allOf")
        if isinstance(all_of, list) and all_of:
            result = result and all(
                _schema_may_accept_property(variant, field, _active, _memo, _object_memo) for variant in all_of
            )
        _memo[key] = result
        return result
    finally:
        _active.remove(key)


# OpenAI's image request schema types `model` as a free string (no enum), so
# these names cannot be read off the spec the way other partners' models are
# (see _model_name_hint). They are the ids ComfyUI's OpenAI image nodes send
# to the same /proxy/openai/images/* routes the `dalle` aliases call. The value
# is the Cloud partner node that takes the id as its `model` (None: no node).
_OPENAI_IMAGE_MODELS: dict[str, str | None] = {
    "gpt-image-1": "OpenAIGPTImageNodeV2",
    "gpt-image-1.5": "OpenAIGPTImageNodeV2",
    "gpt-image-2": "OpenAIGPTImageNodeV2",
    "dall-e-2": None,
    "dall-e-3": None,
}


def _direct_route(alias: str, model_id: str, *, model_field: str = "model", suggest_partner_node: bool = True) -> str:
    """The matching `comfy generate <alias> --<field> <id>` line, saying whether it can
    also `--emit-workflow`. A caller building a workflow runs `generate` with
    `--emit-workflow`, so a route that cannot emit must say so or the hint is a
    dead end for it."""
    from comfy_cli.command.generate import emit

    option = model_field.replace("_", "-")
    route = f"`comfy generate {alias} --{option} {model_id} ...`"
    if emit.is_supported(alias):
        return f"{route} (also with --emit-workflow)"
    suffix = "; that alias has no --emit-workflow"
    if suggest_partner_node:
        suffix += ", so use a partner node for a workflow"
    return f"{route} runs it directly{suffix}"


def _model_name_hint(name: str) -> str | None:
    """Explain a name that is a partner MODEL rather than a `generate` alias.

    Agents ask for the model they know ("gpt-image-1", "seedream"), not the
    alias. `comfy generate <name>` used to give a bare "Unknown model" for the
    first, and for the second only "Did you mean: seedance", the VIDEO model.
    The workflow route (a partner node) leads; the direct `comfy generate`
    route follows. Returns ``None`` when the name matches no known model.
    """
    lowered = name.strip().lower()
    if lowered in _OPENAI_IMAGE_MODELS:
        node = _OPENAI_IMAGE_MODELS[lowered]
        direct = _direct_route("dalle", lowered, suggest_partner_node=node is not None)
        if node is None:
            return f"{lowered!r} is an OpenAI image model, not an alias. {direct}."
        return (
            f"{lowered!r} is an OpenAI image model, not an alias. For a workflow, add the partner node "
            f"{node} (`comfy nodes show {node}`) and set its model to {lowered!r}. Without a workflow: {direct} "
            "(image edits: `dalle-edit`)."
        )
    if len(lowered) < 4:
        return None
    # Every other partner declares its model ids as an enum on the request
    # body's `model` field. A name equal to or prefixing one of them is that
    # partner's model family, whichever route serves it.
    raw = load_raw_spec()
    paths = raw.get("paths") or {}
    if not isinstance(paths, dict):
        return None
    resolution_budget_limit = _schema_resolution_budget(raw)
    aggregate_resolution_budget = [resolution_budget_limit * 4]
    resolution_memo: dict[tuple[Any, ...], tuple[Any, bool]] = {}
    aliased = {v: k for k, v in _ALIASES.items()}
    hits: list[tuple[str, str, list[str]]] = []
    for path, node in paths.items():
        if aggregate_resolution_budget[0] <= 0:
            break
        if not str(path).startswith(PROXY_PREFIX) or not isinstance(node, dict):
            continue
        op = node.get("post")
        if not isinstance(op, dict):
            continue
        request_body = op.get("requestBody") or {}
        if not isinstance(request_body, dict):
            continue
        content = request_body.get("content") or {}
        if not isinstance(content, dict):
            continue
        ctype = _preferred_media_type(content)
        media = content.get(ctype) or {}
        if not isinstance(media, dict):
            continue
        schema = media.get("schema")
        if not schema:
            continue
        resolution_budget, resolution_allowance = _resolution_attempt_budget(
            resolution_budget_limit, aggregate_resolution_budget
        )
        try:
            resolved = _resolve(raw, schema, memo=resolution_memo, budget=resolution_budget)
            if not isinstance(resolved, dict):
                continue
            for field in ("model", "model_name", "model_id"):
                prop = _find_property(resolved, field)
                values = _extract_enum(prop) if prop else None
                matched = [v for v in values or [] if v.lower().startswith(lowered)]
                if matched:
                    hits.append((str(path)[len(PROXY_PREFIX) :], field, matched))
                    break
        except (KeyError, TypeError, SpecError, RecursionError):
            continue
        finally:
            _charge_resolution_attempt(aggregate_resolution_budget, resolution_allowance, resolution_budget)
    if not hits:
        return None
    lines = [
        f"Partner models matching {lowered!r} (for a workflow, find the partner node: `comfy nodes search {lowered}`):"
    ]
    for endpoint_id, field, values in hits:
        shown = ", ".join(values[:6])
        alias = aliased.get(endpoint_id)
        if alias is not None:
            lines.append(f"- {shown}: {_direct_route(alias, '<id>', model_field=field)}.")
        else:
            lines.append(f"- {shown}: served at {endpoint_id}, which has no `comfy generate` alias.")
    return "\n".join(lines)


def _unknown_endpoint_message(endpoint_id: str) -> str:
    """Build a helpful error: close alias matches first, then (when the name is
    a partner model id or prefix) where that model is served. The model hint is
    appended, never a replacement — "flux" prefixes recraft's flux1dev ids but
    is far more likely a typo of the flux-* aliases."""
    import difflib
    import re

    candidates = list(_registry().keys()) + list(_ALIASES.keys())
    close = difflib.get_close_matches(endpoint_id, candidates, n=3, cutoff=0.5)

    # Add family candidates keyed on the leading token.
    head = re.split(r"[-_/.]", endpoint_id.lower(), 1)[0]
    if len(head) >= 3:
        family = [c for c in candidates if c.lower().startswith(head) and c not in close]
        close = (close + sorted(family))[:6]

    msg = f"Unknown model: {endpoint_id!r}."
    if close:
        msg += "\nDid you mean: " + ", ".join(close) + "?"
    model_hint = _model_name_hint(endpoint_id)
    if model_hint is not None:
        msg += "\n" + model_hint
    msg += "\nRun `comfy generate list` to see available models."
    return msg


def validate_spec_text(text: str) -> dict[str, Any]:
    """Parse a raw openapi spec body with the same loader ``load_raw_spec`` uses
    and require a top-level ``paths`` mapping.

    The body may be YAML or JSON — JSON is a subset of YAML 1.2, and
    ``_YamlLoader`` only restricts bool resolution, so a JSON spec (as served at
    ``api.comfy.org/openapi``) parses. Raises :class:`SpecError` if the body does
    not parse or lacks ``paths``; callers use this to avoid caching a
    200-with-garbage response, which would poison the on-disk cache for
    ``CACHE_TTL_SECONDS``.
    """
    try:
        parsed = yaml.load(text, Loader=_YamlLoader)
    except yaml.YAMLError as e:
        raise SpecError(f"spec did not parse: {e}") from e
    if not isinstance(parsed, dict) or not isinstance(parsed.get("paths"), dict):
        raise SpecError("spec has no top-level 'paths' mapping")
    return parsed


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
