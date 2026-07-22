"""Per-endpoint adapters for partners whose request/response shapes don't fit
the generic schema-driven flag→JSON mold.

Two endpoints today:

- **Gemini Flash Image (nano-banana)** — Vertex AI's ``contents``/``parts``
  body, inline base64 image input, and inline base64 image output. The model
  variant lives in the URL path, not the body.
- **Seedance** (ByteDance) — assembles a ``content`` array of typed parts
  (``text`` + optional ``image_url``) and inlines its own knobs (resolution,
  duration, …) into the prompt string.

An adapter contributes three optional pieces:

- ``flags`` — replaces the schema-derived flag list for the model
- ``build_body`` — produces the JSON body from parsed flag values
- ``decode_sync`` — handles a sync response that ships inline blobs (Gemini)
- ``path_param`` — name of a flag whose value gets substituted into the URL
  path's ``{placeholder}`` (e.g. ``model`` for Gemini's templated path)
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from comfy_cli.command.generate import spec
from comfy_cli.command.generate.client import ApiError
from comfy_cli.command.generate.schema import FlagDef


@dataclass(frozen=True)
class Adapter:
    flags: list[FlagDef]
    build_body: Callable[[dict, str], dict]
    decode_sync: Callable[[dict, str, str], list[Path]] | None = None
    path_param: str | None = None


# ── Gemini / nano-banana ──────────────────────────────────────────────────

_GEMINI_ENDPOINT_ID = "vertexai/gemini/{model}"

# Fallback only — the active openapi spec's enum wins when it carries one
# (see _spec_model_flags). Gemini's model lives in the URL path, not the
# request body, so the spec has no body enum for it today and this tuple is
# the effective list.
GEMINI_IMAGE_MODELS = (
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-3-pro-image-preview",
)


def _inline_image(value: str) -> tuple[str, str]:
    """Return ``(mime_type, base64_str)`` for a local path, http(s) URL, or
    ``data:`` URI. Gemini accepts inline-only — there's no signed-URL path
    here, so we pull bytes locally rather than going through ``upload.py``."""
    if value.startswith("data:"):
        head, _, b64 = value.partition(",")
        mime = head.split(";", 1)[0].removeprefix("data:") or "image/png"
        return mime, b64
    if value.startswith(("http://", "https://")):
        with httpx.Client(timeout=60.0, follow_redirects=True) as c:
            r = c.get(value)
            r.raise_for_status()
            mime = (r.headers.get("content-type") or "image/png").split(";", 1)[0].strip()
            return mime, base64.b64encode(r.content).decode("ascii")
    path = Path(value).expanduser()
    if not path.is_file():
        raise ApiError(0, "", f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "image/png", base64.b64encode(path.read_bytes()).decode("ascii")


def _gemini_build_body(values: dict, api_key: str) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": str(values["prompt"])}]
    images = values.get("image") or []
    if isinstance(images, str):
        images = [images]
    for img in images:
        mime, b64 = _inline_image(str(img))
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }


def _gemini_decode_sync(body: dict, download: str, request_id: str) -> list[Path]:
    """Walk candidates[*].content.parts[*].inlineData; save each blob."""
    from comfy_cli.command.generate import output

    blobs: list[tuple[str, bytes]] = []
    for cand in body.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline:
                continue
            data_b64 = inline.get("data") or ""
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            try:
                raw = base64.b64decode(data_b64, validate=False)
            except (ValueError, TypeError):
                continue
            blobs.append((mime, raw))
    if not blobs:
        return []
    return output.save_inline_blobs(blobs, download, request_id)


_gemini_adapter = Adapter(
    flags=[
        FlagDef(
            name="prompt",
            kind="string",
            required=True,
            description="Text instruction. For edits, describe the change.",
        ),
        FlagDef(
            name="image",
            kind="array",
            item_kind="string",
            required=False,
            description="Optional reference image(s): local path, http(s) URL, or data URI.",
        ),
        FlagDef(
            name="model",
            kind="enum",
            required=False,
            default=GEMINI_IMAGE_MODELS[0],
            description="Gemini image-model variant.",
            enum=list(GEMINI_IMAGE_MODELS),
        ),
    ],
    build_body=_gemini_build_body,
    decode_sync=_gemini_decode_sync,
    path_param="model",
)


# ── Seedance ──────────────────────────────────────────────────────────────

_SEEDANCE_ENDPOINT_ID = "byteplus/api/v3/contents/generations/tasks"

# Fallback only — the active openapi spec's request-body enum wins when it
# carries one (see _spec_model_flags), so new Seedance releases need a spec
# refresh, not a CLI release.
SEEDANCE_MODELS = (
    "seedance-1-0-pro-250528",
    "seedance-1-0-pro-fast-251015",
    "seedance-1-5-pro-251215",
    "seedance-1-0-lite-t2v-250428",
    "seedance-1-0-lite-i2v-250428",
)

_SEEDANCE_INLINE_KEYS = ("resolution", "ratio", "duration", "fps", "seed", "camerafixed", "watermark")


def _seedance_text(values: dict) -> str:
    """Compose the ``text`` field, appending Seedance's inline ``--rs/--rt/…``
    style overrides for any flags the user set."""
    prompt = str(values["prompt"])
    extras: list[str] = []
    for key in _SEEDANCE_INLINE_KEYS:
        v = values.get(key)
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            v = "true" if v else "false"
        extras.append(f"--{key} {v}")
    return f"{prompt} {' '.join(extras)}".strip()


def _seedance_image_url(value: str, api_key: str) -> str:
    """Local paths get uploaded; data: and http(s) pass through verbatim."""
    if value.startswith(("http://", "https://", "data:")):
        return value
    from comfy_cli.command.generate import upload

    return upload.upload_path(Path(value).expanduser(), api_key).url


def _seedance_default_model() -> str:
    """Default model when the user didn't pass ``--model`` — the pinned default
    if the active spec's enum still lists it, else the enum's first entry."""
    models = spec.model_enum(_SEEDANCE_ENDPOINT_ID) or list(SEEDANCE_MODELS)
    return SEEDANCE_MODELS[0] if SEEDANCE_MODELS[0] in models else models[0]


def _seedance_build_body(values: dict, api_key: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": _seedance_text(values)}]
    image = values.get("image")
    if image:
        content.append({"type": "image_url", "image_url": {"url": _seedance_image_url(str(image), api_key)}})
    body: dict[str, Any] = {
        "model": values.get("model") or _seedance_default_model(),
        "content": content,
    }
    if "generate_audio" in values:
        body["generate_audio"] = bool(values["generate_audio"])
    if "return_last_frame" in values:
        body["return_last_frame"] = bool(values["return_last_frame"])
    return body


_seedance_adapter = Adapter(
    flags=[
        FlagDef(name="prompt", kind="string", required=True, description="Text prompt for the video."),
        FlagDef(
            name="image",
            kind="string",
            required=False,
            description="Optional first-frame image (URL, local path, or data URI). "
            "Local paths are auto-uploaded via /customers/storage.",
        ),
        FlagDef(
            name="model",
            kind="enum",
            required=False,
            default=SEEDANCE_MODELS[0],
            description="Seedance model variant.",
            enum=list(SEEDANCE_MODELS),
        ),
        FlagDef(name="resolution", kind="enum", required=False, enum=["480p", "720p", "1080p"]),
        FlagDef(
            name="ratio",
            kind="enum",
            required=False,
            enum=["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21", "adaptive"],
        ),
        FlagDef(name="duration", kind="integer", required=False, description="Length in seconds (3–12)."),
        FlagDef(name="fps", kind="integer", required=False, description="Frames per second (default 24)."),
        FlagDef(name="seed", kind="integer", required=False, description="RNG seed (-1 to 2^32-1)."),
        FlagDef(name="camerafixed", kind="boolean", required=False, description="Lock camera position."),
        FlagDef(name="watermark", kind="boolean", required=False, description="Include a watermark."),
        FlagDef(
            name="generate_audio",
            kind="boolean",
            required=False,
            description="Synthesize matching audio (Seedance 1.5 pro only).",
        ),
        FlagDef(
            name="return_last_frame",
            kind="boolean",
            required=False,
            description="Return the last-frame image alongside the video.",
        ),
    ],
    build_body=_seedance_build_body,
    decode_sync=None,
    path_param=None,
)


_ADAPTERS: dict[str, Adapter] = {
    _GEMINI_ENDPOINT_ID: _gemini_adapter,
    _SEEDANCE_ENDPOINT_ID: _seedance_adapter,
}


def _spec_model_flags(endpoint_id: str, flags: list[FlagDef]) -> list[FlagDef]:
    """Return ``flags`` with the ``model`` enum refreshed from the active spec.

    Resolved lazily at lookup time (not import) so a refreshed openapi cache
    surfaces new partner models with zero code changes; the hardcoded tuples
    above stay as the fallback when the spec carries no enum. The pinned
    default is kept while the derived enum still lists it; otherwise the first
    enum entry takes over (a spec that drops a deprecated default must not
    break the CLI)."""
    derived = spec.model_enum(endpoint_id)
    if not derived:
        return flags
    out: list[FlagDef] = []
    for f in flags:
        if f.name == "model" and f.kind == "enum":
            default = f.default if f.default in derived else derived[0]
            f = replace(f, enum=list(derived), default=default)
        out.append(f)
    return out


def get(endpoint_id: str) -> Adapter | None:
    adapter = _ADAPTERS.get(endpoint_id)
    if adapter is None:
        return None
    return replace(adapter, flags=_spec_model_flags(endpoint_id, adapter.flags))


def resolve_path(template: str, values: dict, adapter: Adapter) -> str:
    """Substitute ``adapter.path_param`` into the URL template, falling back to
    the flag's ``default`` when the user didn't pass it."""
    if not adapter.path_param:
        return template
    val = values.get(adapter.path_param)
    if not val:
        for f in adapter.flags:
            if f.name == adapter.path_param:
                val = f.default
                break
    if not val:
        raise ApiError(0, "", f"Missing --{adapter.path_param}: required to fill in the URL path.")
    # The value may come from a spec-derived enum (refreshable cache), so pin it
    # to a single path segment: percent-encode reserved characters ("/", "?",
    # "#", …) and reject dot segments outright — a tampered spec must not be
    # able to redirect the proxied request via path traversal.
    val = str(val)
    if val in (".", ".."):
        raise ApiError(0, "", f"Invalid --{adapter.path_param} value: {val!r}.")
    return template.replace("{" + adapter.path_param + "}", quote(val, safe=""))
