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
from dataclasses import dataclass
from importlib import resources

# The name pinned in ``data/default_text2img.json`` node "4" ``ckpt_name``.
# Runtime resolution (``resolve_default_checkpoint``) swaps this for a checkpoint
# the target actually has when it's absent; keep the two in sync.
DEFAULT_CHECKPOINT_NAME = "v1-5-pruned-emaonly-fp16.safetensors"

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


def default_checkpoint(workflow: dict | None = None) -> str:
    """Return the ``ckpt_name`` the bundled default graph loads (``""`` if absent).

    ``comfy run --prompt`` silently depends on this checkpoint already being
    present in the target's ``models/checkpoints`` — comfy-cli neither bundles
    it nor downloads it on demand, so the run fails server-side with a bare
    validation error when it is missing. Callers surface the requirement up
    front instead. Pass an already-built graph to avoid re-reading the bundle.
    """
    graph = load_default_workflow() if workflow is None else workflow
    node = graph.get(CHECKPOINT_LOADER_ID)
    if not isinstance(node, dict):
        return ""
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return ""
    name = inputs.get("ckpt_name")
    return name if isinstance(name, str) else ""


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


def overrides_set_checkpoint(overrides: list[str] | None, workflow: dict) -> bool:
    """True if any ``--set`` override targets the checkpoint (node ``"4"``
    ``ckpt_name``), via an alias (``checkpoint``/``ckpt``) or the raw
    ``4.ckpt_name`` form.

    Used by the caller to decide whether the user pinned the checkpoint
    explicitly — if so, runtime resolution is skipped and the value is honored
    verbatim. ``workflow`` must be a built default graph so addresses resolve;
    malformed entries are ignored here (``build_default_workflow`` already
    validated/rejected them upstream).
    """
    for raw in overrides or []:
        if "=" not in raw:
            continue
        address = raw.partition("=")[0].strip()
        try:
            node_id, field = _resolve_address(address, workflow)
        except PromptInjectionError:
            continue
        if (node_id, field) == (CHECKPOINT_LOADER_ID, "ckpt_name"):
            return True
    return False


@dataclass(frozen=True)
class CheckpointResolution:
    """Outcome of :func:`resolve_default_checkpoint`.

    - ``note``/``substituted_to`` are set only when the pinned checkpoint was
      absent and a different one was substituted.
    - ``no_checkpoint`` is True ONLY when ``object_info`` positively enumerated
      an EMPTY checkpoint list (the target has zero checkpoints). It stays False
      when we can't tell (``object_info`` absent/empty, or it didn't enumerate
      ``CheckpointLoaderSimple.ckpt_name``) so callers fail open there.
    """

    note: str | None = None
    substituted_to: str | None = None
    no_checkpoint: bool = False


def _checkpoint_enum(object_info: dict) -> list | None:
    """Return the ``CheckpointLoaderSimple.ckpt_name`` option list from
    ``object_info``, or ``None`` when it isn't enumerated at all.

    The raw object_info shape is ``ckpt_name: [[<name>, …], {opts}]`` — the
    option list is element 0. ``None`` (not an empty list) means "can't tell"
    so the caller can distinguish a positively-empty enum from an absent one.
    """
    if not isinstance(object_info, dict):
        # A non-object /object_info payload (e.g. a hostile or misbehaving
        # server returning ``[]``) means "can't tell" — fail open, mirroring
        # Graph.from_object_info's isinstance guard rather than crashing.
        return None
    node = object_info.get("CheckpointLoaderSimple")
    if not isinstance(node, dict):
        return None
    inp = node.get("input")
    if not isinstance(inp, dict):
        return None
    req = inp.get("required")
    if not isinstance(req, dict):
        return None
    spec = req.get("ckpt_name")
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return spec[0]
    return None


def _basename(name: str) -> str:
    """Last path segment of a ComfyUI-enumerated model name.

    ``folder_paths`` builds these names with ``os.path.relpath``, so the
    separator is the SERVER's, not ours: a subfoldered checkpoint arrives as
    ``SD1.5/name.safetensors`` from a POSIX host and ``SD1.5\\name.safetensors``
    from a Windows one. Normalize both before comparing basenames — splitting on
    ``/`` alone would miss the Windows form and trigger a needless substitution.
    """
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def resolve_default_checkpoint(
    workflow: dict, object_info: dict, *, target: str = "the server"
) -> tuple[dict, CheckpointResolution]:
    """Resolve the bundled default's pinned checkpoint against a live target.

    Pure and offline-testable. Given the bundled default graph and a target's
    ``object_info``, decide what checkpoint node ``"4"`` should carry:

    - pinned name present in the target's enum → no change;
    - pinned absent but the enum is non-empty → substitute the first available
      checkpoint and return a ``note``;
    - enum positively empty → leave unchanged, flag ``no_checkpoint`` (the
      caller emits an actionable error);
    - enum absent / ``object_info`` empty → leave unchanged, no flag (fail open).

    Mutates ``workflow`` in place on substitution and returns it alongside the
    :class:`CheckpointResolution`. Callers must guard this to the bundled
    default graph only (``workflow_name == "default_text2img"``).
    """
    node = workflow.get(CHECKPOINT_LOADER_ID)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return workflow, CheckpointResolution()

    enum = _checkpoint_enum(object_info)
    if enum is None:
        # Not enumerated (fresh/unfetched object_info) — fail open.
        return workflow, CheckpointResolution()
    if not enum:
        # Positively empty: the target has zero checkpoints installed.
        return workflow, CheckpointResolution(no_checkpoint=True)

    pinned = inputs.get("ckpt_name")
    if pinned in enum:
        return workflow, CheckpointResolution()

    # ComfyUI enumerates checkpoints by their path relative to the models dir,
    # so a pinned bare filename won't exact-match the same file living in a
    # subfolder (e.g. ``SD1.5/v1-5-…safetensors``). Prefer a basename match to
    # the *intended* checkpoint before falling back to an arbitrary substitute.
    if isinstance(pinned, str):
        pinned_base = _basename(pinned)
        for entry in enum:
            if isinstance(entry, str) and _basename(entry) == pinned_base:
                inputs["ckpt_name"] = entry
                return workflow, CheckpointResolution()

    replacement = enum[0]
    inputs["ckpt_name"] = replacement
    note = (
        f"default checkpoint {pinned} not found on {target}; using {replacement} "
        f"instead (override with --set checkpoint=<name>)"
    )
    return workflow, CheckpointResolution(note=note, substituted_to=replacement)
