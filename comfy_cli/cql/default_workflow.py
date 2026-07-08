"""Bundled default text2img workflow + a direct API-format prompt injector.

``comfy run --prompt`` loads a pinned, **non-subgraphed** API-format graph
(``comfy_cli/cql/data/default_text2img.json``) with STABLE, KNOWN node ids and
writes a prompt straight into the node ``inputs`` map. This is deliberately NOT
``Graph.apply_slots`` (that operates on UI-format ``widgets_values`` and needs
``object_info``): a bundled API graph sets a field by a trivial dict write, so
no live server, no ``object_info``, and no ``apply_slots`` are involved.

The node ids below are constants because we OWN this graph — the gallery's
subgraphed template shifts its interior addresses across revisions, which is
exactly the fragility this pinned graph removes.
"""

from __future__ import annotations

import json
import math
from importlib import resources

# -- Pinned node ids (must match data/default_text2img.json) --
CHECKPOINT_LOADER_ID = "4"
POSITIVE_PROMPT_ID = "6"
NEGATIVE_PROMPT_ID = "7"
EMPTY_LATENT_ID = "5"
KSAMPLER_ID = "3"
VAE_DECODE_ID = "8"
SAVE_IMAGE_ID = "9"

# Documented convenience aliases → (node id, input field). These resolve
# against the KNOWN ids above so a caller never has to know the graph layout.
# The raw form ``NODE_ID.field=VALUE`` is always available too (see
# ``_resolve_address``), e.g. ``4.ckpt_name=NAME`` is equivalent to
# ``checkpoint=NAME``.
ALIASES: dict[str, tuple[str, str]] = {
    "prompt": (POSITIVE_PROMPT_ID, "text"),
    "positive": (POSITIVE_PROMPT_ID, "text"),
    "negative": (NEGATIVE_PROMPT_ID, "text"),
    "checkpoint": (CHECKPOINT_LOADER_ID, "ckpt_name"),
    "ckpt": (CHECKPOINT_LOADER_ID, "ckpt_name"),
    "seed": (KSAMPLER_ID, "seed"),
    "steps": (KSAMPLER_ID, "steps"),
    "cfg": (KSAMPLER_ID, "cfg"),
    "sampler": (KSAMPLER_ID, "sampler_name"),
    "scheduler": (KSAMPLER_ID, "scheduler"),
    "denoise": (KSAMPLER_ID, "denoise"),
    "width": (EMPTY_LATENT_ID, "width"),
    "height": (EMPTY_LATENT_ID, "height"),
    "batch_size": (EMPTY_LATENT_ID, "batch_size"),
    "filename_prefix": (SAVE_IMAGE_ID, "filename_prefix"),
}


class PromptInjectionError(Exception):
    """Raised for a malformed/unknown ``--prompt``/``--set`` override.

    ``code`` is a registered ``error_codes`` value so callers can forward it
    straight to ``renderer.error(code=e.code, ...)`` — no stack trace escapes
    to the user (acceptance criterion 3).
    """

    def __init__(self, message: str, *, code: str = "prompt_rejected", hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint


def load_default_workflow() -> dict:
    """Return a fresh deep copy of the bundled default text2img API graph.

    Uses the same ``importlib.resources`` package-data loader as
    ``engine._try_default_annotations``; ``data/*.json`` is already declared as
    package data in ``pyproject.toml``.

    A missing or corrupt bundle is a packaging fault, not user input, but it
    must still exit through the controlled envelope (no stack trace escapes):
    surface it as ``default_workflow_unavailable`` so ``build_default_workflow``'s
    caller catches it like any other ``PromptInjectionError``.
    """
    try:
        data = resources.files("comfy_cli.cql.data").joinpath("default_text2img.json").read_bytes()
        return json.loads(data)
    except (OSError, ModuleNotFoundError, ValueError) as e:
        raise PromptInjectionError(
            f"the bundled default text2img workflow could not be loaded: {e}",
            code="default_workflow_unavailable",
            hint="this is a comfy-cli packaging error; try reinstalling comfy-cli",
        ) from e


def _coerce(value: str, existing):
    """Coerce a string CLI value to the API-format type it should carry.

    If the address already holds a scalar in the graph, match its type (so
    ``seed=42`` becomes int ``42``, ``cfg=7.5`` becomes float, and a text field
    stays a string). For a brand-new field with no existing scalar, fall back to
    a JSON-scalar parse (``42`` → int) and finally the raw string.

    A list-valued ``existing`` is a graph connection edge (e.g. ``["6", 0]``),
    not a settable input — overwriting it with a scalar corrupts the topology,
    so those targets are rejected outright.
    """
    if isinstance(existing, list):
        raise PromptInjectionError(
            f"this field holds a graph connection, not a settable value; refusing to overwrite it with {value!r}",
            hint="--set only overrides scalar inputs (seed, cfg, text, ckpt_name, …), not wired node connections",
        )
    if isinstance(existing, bool):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise PromptInjectionError(f"expected a boolean for this field, got {value!r}")
    if isinstance(existing, int) and not isinstance(existing, bool):
        try:
            return int(value)
        except ValueError as e:
            raise PromptInjectionError(f"expected an integer for this field, got {value!r}") from e
    if isinstance(existing, float):
        try:
            result = float(value)
        except ValueError as e:
            raise PromptInjectionError(f"expected a number for this field, got {value!r}") from e
        # `float("nan"/"inf")` parses, but json.dumps emits non-standard
        # NaN/Infinity tokens that strict server-side parsers reject.
        if not math.isfinite(result):
            raise PromptInjectionError(f"expected a finite number for this field, got {value!r}")
        return result
    if isinstance(existing, str):
        return value
    # New field (existing is None): best-effort JSON scalar, else raw string.
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    if isinstance(parsed, bool) or isinstance(parsed, str):
        return parsed
    if isinstance(parsed, (int, float)):
        # json.loads accepts NaN/Infinity — fall back to the raw string rather
        # than inject a non-finite scalar that won't round-trip through JSON.
        return parsed if math.isfinite(parsed) else value
    return value


def _resolve_address(address: str, workflow: dict) -> tuple[str, str]:
    """Resolve a ``--set`` address to a (node id, field) pair in ``workflow``.

    Accepts an alias (``checkpoint``) or the raw ``NODE_ID.field`` form
    (``4.ckpt_name``). Raises ``PromptInjectionError`` for an unknown alias or a
    node id / raw form the pinned graph doesn't contain.
    """
    if "." in address:
        node_id, _, field = address.partition(".")
        if not node_id or not field:
            raise PromptInjectionError(
                f"invalid --set address {address!r}",
                hint="use NODE_ID.field=VALUE (e.g. 4.ckpt_name=model.safetensors) or an alias",
            )
    else:
        alias = ALIASES.get(address)
        if alias is None:
            raise PromptInjectionError(
                f"unknown --set field {address!r}",
                hint="known aliases: " + ", ".join(sorted(ALIASES)) + "; or use NODE_ID.field=VALUE",
            )
        node_id, field = alias

    node = workflow.get(node_id)
    if not isinstance(node, dict) or "class_type" not in node:
        raise PromptInjectionError(
            f"--set address {address!r} targets node {node_id!r}, which is not in the default workflow",
            hint="node ids in the bundled graph: " + ", ".join(sorted(workflow)),
        )
    # A raw ``NODE_ID.field`` typo (e.g. ``4.ckpt_naem``) would otherwise write a
    # junk key while the real input silently keeps its default. Every alias maps
    # to a real input too, so validating the resolved field against the node's
    # actual inputs guards both forms. (The bundled API graph carries every
    # settable input explicitly, so "not present" == "not a real input".)
    inputs = node.get("inputs")
    if not isinstance(inputs, dict) or field not in inputs:
        known = ", ".join(sorted(inputs)) if isinstance(inputs, dict) else "(none)"
        raise PromptInjectionError(
            f"--set address {address!r} targets field {field!r}, which node {node_id!r} "
            f"({node.get('class_type')}) has no such input",
            hint=f"inputs on this node: {known}",
        )
    return node_id, field


def _apply_set(workflow: dict, node_id: str, field: str, value: str) -> None:
    inputs = workflow[node_id].setdefault("inputs", {})
    inputs[field] = _coerce(value, inputs.get(field))


def build_default_workflow(*, prompt: str | None = None, overrides: list[str] | None = None) -> dict:
    """Build the injected API-format graph for ``comfy run --prompt``/``--set``.

    ``prompt`` writes the positive CLIPTextEncode ``text``; each ``overrides``
    entry is a ``node.field=VALUE`` / ``alias=VALUE`` string applied in order
    (later wins). Returns a ready-to-submit API-format workflow dict. Raises
    ``PromptInjectionError`` on any malformed/unknown override.
    """
    workflow = load_default_workflow()

    if prompt is not None:
        node_id, field = ALIASES["prompt"]
        _apply_set(workflow, node_id, field, prompt)

    for raw in overrides or []:
        if "=" not in raw:
            raise PromptInjectionError(
                f"invalid --set {raw!r}: expected node.field=VALUE",
                hint="e.g. --set checkpoint=model.safetensors or --set 3.seed=42",
            )
        address, _, value = raw.partition("=")
        address = address.strip()
        node_id, field = _resolve_address(address, workflow)
        _apply_set(workflow, node_id, field, value)

    return workflow
