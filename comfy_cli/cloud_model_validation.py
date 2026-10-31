"""Resolve only verified private-model enum drift against authenticated assets.

The CQL engine stays pure and offline. Cloud callers may supplement a stale
static model enum with exact, accessible model records; an unavailable or
ambiguous lookup preserves the original validation diagnostics.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse

from comfy_cli.http import ResponseTooLarge, request_json
from comfy_cli.target import resolve_target

# Verified filename widgets from the frontend's MODEL_NODE_MAPPINGS. Do not
# generalize this to every combo widget or to id/preset/auto-loader widgets.
_MODEL_INPUTS = {
    ("CheckpointLoaderSimple", "ckpt_name"): "checkpoints",
    ("ImageOnlyCheckpointLoader", "ckpt_name"): "checkpoints",
    ("LoraLoader", "lora_name"): "loras",
    ("LoraLoaderModelOnly", "lora_name"): "loras",
    ("VAELoader", "vae_name"): "vae",
    ("ControlNetLoader", "control_net_name"): "controlnet",
    ("UNETLoader", "unet_name"): "diffusion_models",
    ("UpscaleModelLoader", "model_name"): "upscale_models",
    ("StyleModelLoader", "style_model_name"): "style_models",
    ("GLIGENLoader", "gligen_name"): "gligen",
    ("CLIPVisionLoader", "clip_name"): "clip_vision",
    ("CLIPLoader", "clip_name"): "text_encoders",
    ("AudioEncoderLoader", "audio_encoder_name"): "audio_encoders",
    ("ModelPatchLoader", "name"): "model_patches",
    ("LatentUpscaleModelLoader", "model_name"): "latent_upscale_models",
    ("IPAdapterModelLoader", "ipadapter_file"): "ipadapter",
    ("WanVideoLoraSelect", "lora"): "loras",
    ("LTXICLoRALoaderModelOnly", "lora_name"): "loras",
}
_MAX_ASSET_BYTES = 2 * 1024 * 1024
_MAX_LOOKUPS = 16


def _content_hash(value: object) -> str | None:
    # Cloud has returned both algorithm-prefixed hashes and storage filenames.
    match = re.fullmatch(r"(?:blake3:)?([a-f0-9]{64})(?:\.[A-Za-z0-9]+)?", value) if isinstance(value, str) else None
    return "blake3:" + match[1] if match else None


def _lookup(target, folder: str, name: str) -> dict | None:
    # A single tag is compatible across Cloud versions. Verify the required
    # folder tag locally; do not depend on multi-tag query encoding.
    query = urllib.parse.urlencode(
        [
            ("include_tags", "models"),
            ("include_public", "false"),
            ("name_contains", name),
            ("limit", 200),
        ]
    )
    status, body = request_json(target.url("assets") + "?" + query, target, max_bytes=_MAX_ASSET_BYTES, timeout=10)
    if status != 200 or not isinstance(body, dict) or body.get("has_more") is not False:
        return None
    rows = body.get("assets")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return None
    # The service search is substring-based. Only an exact name or explicit
    # server-provided loader_path can satisfy the workflow's unchanged value.
    matches = [row for row in rows if row.get("name") == name or row.get("loader_path") == name]
    if not matches:
        return None
    hashes = set()
    for row in matches:
        tags = row.get("tags")
        digest = _content_hash(row.get("hash"))
        if (
            not isinstance(row.get("id"), str)
            or not row["id"].strip()
            or not isinstance(tags, list)
            or not all(isinstance(tag, str) for tag in tags)
            or not {"models", folder}.issubset(tags)
            or digest is None
            or row.get("deleted_at") is not None
            or row.get("deletedAt") is not None
            or row.get("is_deleted") is True
        ):
            return None
        hashes.add(digest)
    if len(hashes) != 1:
        return None
    return {"asset_id": matches[0]["id"], "content_hash": hashes.pop(), "model_folder": folder}


def resolve_cloud_model_errors(validation: dict, workflow: dict, *, where: str | None, offline: bool = False) -> dict:
    """Supplement only Cloud model-name enum errors; leave other findings intact.

    Explicit catalog files and the environment-pinned catalog retain a strict
    no-network contract. No model value is substituted, and no asset is created.
    """
    if where != "cloud" or offline or os.environ.get("COMFY_OBJECT_INFO_FILE"):
        return validation
    errors = validation.get("errors")
    if not isinstance(errors, list):
        return validation
    candidates = []
    for index, error in enumerate(errors):
        if not isinstance(error, dict) or error.get("code") != "unknown_enum_value":
            continue
        node_id, field = error.get("node_id"), error.get("field")
        node = workflow.get(str(node_id))
        if not isinstance(node, dict) or not isinstance(field, str):
            continue
        folder = _MODEL_INPUTS.get((node.get("class_type"), field))
        inputs = node.get("inputs")
        value = inputs.get(field) if isinstance(inputs, dict) else None
        if folder and isinstance(value, str) and 0 < len(value) <= 1024 and not any(ord(c) < 32 for c in value):
            candidates.append((index, error, folder, value))
    if not candidates:
        return validation
    try:
        target = resolve_target(where="cloud", allow_clear=False)
        if not target.is_cloud or not (target.auth_token or target.api_key):
            return validation
        verified, cache = {}, {}
        for index, error, folder, value in candidates:
            key = folder, value
            if key not in cache:
                if len(cache) >= _MAX_LOOKUPS:
                    continue
                cache[key] = _lookup(target, folder, value)
            evidence = cache[key]
            if evidence is not None:
                verified[index] = {
                    "node_id": error["node_id"],
                    "field": error["field"],
                    "code": "cloud_model_asset_verified",
                    "message": f"{value!r} is an exact accessible Cloud model asset absent from the static enum",
                    **evidence,
                }
    except (OSError, ValueError, ResponseTooLarge):
        return validation
    if not verified:
        return validation
    remaining = [error for index, error in enumerate(errors) if index not in verified]
    return {
        **validation,
        "valid": not remaining,
        "errors": remaining,
        "warnings": [*validation.get("warnings", []), *verified.values()],
    }
