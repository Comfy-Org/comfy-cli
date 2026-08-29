"""Pure-Python CQL graph engine.

Parses ComfyUI's ``object_info.json``, builds an indexed compatibility graph,
and exposes upstream/downstream, path-finding, validation, annotations,
and widget-order resolution.

Port of the Go reference engine's ``nodegraph`` package.
"""

from __future__ import annotations

import copy
import difflib
import hashlib as _hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from comfy_cli.cql._net import is_loopback_host
from comfy_cli.http import NoRedirectHandler, build_http_only_opener

# ---------------------------------------------------------------------------
# Types — mirrors nodegraph/types.go
# ---------------------------------------------------------------------------

_IMPLICIT_WIDGET_TYPES = frozenset({"STRING", "INT", "FLOAT", "NUMBER", "BOOLEAN", "COMBO"})

# Uppercase custom types the frontend renders as a DOM widget that SERIALIZES
# into ``widgets_values`` (``ComponentWidgetImpl`` / ``DOMWidget``), so they
# occupy a positional slot exactly like an INT. Every other uppercase custom
# type is a link. Kept to types verified against saved workflows:
# ``Load3D.image`` / ``Load3DAdvanced.viewport_state`` (``LOAD_3D``),
# ``SaveGLB``/``Preview3D``'s injected ``image`` (``PREVIEW_3D``),
# ``LoadAudioUI.audioUI`` (``AUDIO_UI``). ``LOAD3D_CAMERA`` is deliberately
# absent: ``camera_info`` is ``serialize: false`` and writes no slot.
_FRONTEND_DOM_WIDGET_TYPES = frozenset(
    {"LOAD_3D", "LOAD_3D_ADVANCED", "PREVIEW_3D", "AUDIO_UI", "IMAGEUPLOAD", "AUDIOUPLOAD"}
)

# What a fresh node serializes in a DOM-widget slot. ``add_node`` must emit
# these for NON-trailing slots (``Load3D.image`` sits before ``width``), or the
# frontend reads ``width`` into the viewport slot.
_FRONTEND_DOM_WIDGET_DEFAULTS: dict[str, Any] = {"LOAD_3D": "", "LOAD_3D_ADVANCED": "", "PREVIEW_3D": ""}

# Widget names the FRONTEND injects into a node's inputs after object_info
# (``beforeRegisterNodeDef`` in ``uploadImage.ts``/``uploadAudio.ts``/
# ``load3d.ts``/``saveMesh.ts``). They have no schema port. Listed with
# ``control_after_generate`` because all three are marker slots a name<->index
# consumer must be able to name — but they differ as EDIT targets: the seed
# companion carries a real serialized user value (``fixed``/``randomize``/…)
# and stays writable, while the injected button/player/viewport slots (these
# two plus the ``PREVIEW_3D`` ``image`` of ``_PREVIEW_3D_CLASSES``) are refused
# by every write surface — see ``_WidgetEntry.frontend_injected``.
FRONTEND_MARKER_SLOTS = frozenset({"control_after_generate", "upload", "audioUI"})

# ``Comfy.AudioWidget`` appends an ``audioUI`` player to exactly these classes.
_AUDIO_UI_CLASSES = frozenset(
    {"LoadAudio", "SaveAudio", "PreviewAudio", "SaveAudioMP3", "SaveAudioOpus", "SaveAudioAdvanced"}
)
# ``Comfy.UploadImage`` attaches its upload button to the first required media
# COMBO carrying one of these flags (``isMediaUploadComboInput``). Audio has
# its own extension keyed on the ``audio`` input; ``file_upload`` (3D loaders)
# and ``mesh_upload`` attach nothing.
_IMAGE_UPLOAD_FLAGS = frozenset({"image_upload", "animated_image_upload", "video_upload"})
# File extensions that identify a COMBO's option list as a listing of the
# server's INPUT folder rather than an install-time vocabulary. The core
# loaders mark such ports with ``<kind>_upload``; a custom pack that ships its
# own upload button does not (VideoHelperSuite builds ``VHS_LoadVideo.video``
# from ``folder_paths.get_input_directory()`` filtered by its
# ``video_extensions`` and attaches the button by class name in ``VHS.core.js``,
# so object_info carries a bare ``[["bedroom.mp4"]]``). The listing itself is
# the evidence: every option is a media file name, which a model folder
# (``.safetensors``/``.ckpt``/``.gguf``) or a plain enum (``alpha``/``red``)
# never is. Families mirror what the frontend's upload widgets accept — image,
# animated image, video, audio, 3D model.
_INPUT_FILE_EXTENSIONS = frozenset(
    {
        # image / animated image
        "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "avif", "apng",
        # video (VHS ``video_extensions`` + core LoadVideo)
        "mp4", "webm", "mkv", "mov", "avi", "m4v",
        # audio
        "mp3", "wav", "ogg", "flac", "m4a", "aac", "opus",
        # 3D
        "glb", "gltf", "obj", "fbx", "stl", "ply", "usdz",
    }
)  # fmt: skip
# ``Comfy.Preview3D`` / ``Comfy.SaveGLB`` inject a ``PREVIEW_3D`` ``image`` widget.
_PREVIEW_3D_CLASSES = frozenset({"SaveGLB", "Preview3D"})

# Work budget for ``Graph.search_paths``: the number of frontier states it will
# expand before giving up and reporting ``truncated``. A full cloud catalog has
# thousands of nodes, so an unreachable target must fail fast rather than walk
# the whole type lattice.
_MAX_PATH_SEARCH_STATES = 20_000


@dataclass
class PortOptions:
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any = None
    multiline: bool = False
    control_after_generate: bool = False
    force_input: bool = False
    # For COMFY_DYNAMICCOMBO_V3: the raw options list ({key, inputs} dicts) so the
    # engine can expand key-dependent sub-widgets (e.g. model → model.resolution),
    # matching the converter. None for ordinary inputs.
    dynamic_options: list | None = None
    # For COMFY_AUTOGROW_V3: the raw ``template`` dict object_info carries for an
    # autogrow input (e.g. {"input": {...}, "prefix": "image", "min": 1, "max": 50}).
    # Use ``Port.autogrow_template`` to pull out just the naming fields.
    template: dict | None = None
    # True when object_info marked this input upload-backed — the frontend renders
    # an upload button and the declared options are the server's *installed input
    # files*, not an install-time enum. See ``Port.is_upload_backed``.
    upload: bool = False
    # True when object_info carried ANY ``<kind>_upload`` key, true or false —
    # an explicit declaration either way, which ``Port.is_upload_backed`` obeys
    # over the option-listing heuristic it otherwise falls back to.
    upload_declared: bool = False
    # The ``<kind>_upload`` flag names that were set (``("image_upload",)``),
    # so callers can tell WHICH frontend upload extension claims the input.
    upload_flags: tuple[str, ...] = ()
    # ``widgetType``: the frontend renders the widget for THIS type instead of
    # the declared socket type (``inputSpec.widgetType ?? inputSpec.type`` in
    # litegraphService), so a ``FLOAT,INT`` or ``STRING,FILE_3D_*`` input with
    # ``widgetType`` set is a widget slot even though its own type reads as a
    # link. None when the schema does not set it.
    widget_type: str | None = None


@dataclass
class Port:
    name: str
    type: str
    required: bool = False
    is_link: bool = False
    enum_values: list[Any] = field(default_factory=list)  # preserves the option's real type (int combos stay int)
    # True when object_info shipped an explicit choice list for this input —
    # *including an empty one*. An empty declared list means "the server has
    # zero options here" (a model folder with no files installed), which is a
    # different statement from "the choices aren't in object_info at all"
    # (remote/dynamic combos, whose options the frontend fetches at runtime).
    # Only the former is safe to validate against.
    enum_declared: bool = False
    options: PortOptions = field(default_factory=PortOptions)
    # V3 dynamic combos (COMFY_DYNAMICCOMBO_V3): the raw option dicts
    # ([{"key": ..., "inputs": {...}}]) retained for sub-input expansion.
    # enum_values carries just the selector keys.
    dynamic_options: list[dict] = field(default_factory=list)
    # The verbatim ``INPUT_TYPES`` spec this port was parsed from. Retained
    # (by reference — no copy) because a dynamic combo's sub-input schema lives
    # in the spec's ``options`` blocks and is NOT recoverable from the parsed
    # fields above. Output ports carry ``None``.
    raw_spec: Any = None

    @property
    def is_autogrow(self) -> bool:
        """V3 autogrow input (e.g. BatchImagesNode.images): the schema declares
        ONE input, but the server expects autogrown slot keys —
        ``images.image0``, ``images.image1``, … one per connection."""
        return self.type.startswith("COMFY_AUTOGROW")

    @property
    def is_dynamic_combo(self) -> bool:
        """V3 dynamic combo (e.g. ``COMFY_DYNAMICCOMBO_V3``): the schema declares
        ONE selector input, but the option the selector names carries its own
        ``INPUT_TYPES`` block, which the frontend — and ``convert_ui_to_api`` —
        lower into dotted slot keys (``model.size_preset``, ``model.width``, …).

        Deliberately the same test the converter applies
        (``workflow_to_api._is_widget_input``), so validation expands exactly the
        tree the converter lowered."""
        return self.type.startswith("COMFY_") and "COMBO" in self.type

    def autogrow_slot_example(self) -> str:
        """Slot-key example for hints: the first two slot names this group
        grows, from the schema template when the catalog carries one
        (``model.images.image_1, model.images.image_2, …``), else the
        historical singular-of-the-input-name guess (``images.image0, …``).

        ``self.name`` may itself be dotted (a dynamic-combo sub-input, e.g.
        ``model.images``) — the prefix fallback derives its stem from just the
        last segment so the example doesn't duplicate the dotted prefix
        (``model.images.image0``, not ``model.images.model.image0``)."""
        template = self.autogrow_element_template or {}
        names = template.get("names")
        if names:
            first = [f"{self.name}.{n}" for n in names[:2]]
        else:
            last = self.name.rsplit(".", 1)[-1]
            prefix = template.get("prefix") or (last[:-1] if last.endswith("s") else last)
            first = [f"{self.name}.{prefix}0", f"{self.name}.{prefix}1"]
        return ", ".join(first) + ", …"

    @property
    def autogrow_element_type(self) -> str | None:
        """The socket type every grown slot of this autogrow input carries —
        the single input the schema ``template`` declares (``IMAGE`` for
        ``model.images``, ``VIDEO`` for ``model.reference_videos``). Every
        group in the production catalog declares exactly one template input;
        a template with none (or a non-autogrow port) reads as ``None``, which
        callers treat as "accept the source type".
        """
        t = self.options.template
        if not self.is_autogrow or not isinstance(t, dict):
            return None
        inputs = t.get("input")
        if not isinstance(inputs, dict):
            return None
        for section in ("required", "optional"):
            section_def = inputs.get(section)
            if not isinstance(section_def, dict):
                continue
            for spec in section_def.values():
                type_id = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
                if isinstance(type_id, str) and type_id:
                    return type_id
        return None

    @property
    def autogrow_limits(self) -> tuple[int, int | None]:
        """``(min, max)`` slots for this autogrow input, exactly as the frontend
        derives them (``applyAutogrow``): ``min`` defaults to 1, ``max`` is the
        length of ``names`` when the template enumerates them, else the
        template's ``max`` (``None`` when the schema leaves it open — the
        frontend's own default there is 100, but that is a UI choice, not a
        server limit, so it is not asserted here)."""
        t = self.options.template if isinstance(self.options.template, dict) else {}
        lo = t.get("min")
        lo = int(lo) if isinstance(lo, (int, float)) and not isinstance(lo, bool) else 1
        names = t.get("names")
        if isinstance(names, list) and names:
            return lo, len(names)
        hi = t.get("max")
        return lo, (int(hi) if isinstance(hi, (int, float)) and not isinstance(hi, bool) else None)

    @property
    def autogrow_template(self) -> dict | None:
        """The V3 autogrow element-naming template from object_info, if the
        catalog carries one: ``{"names": [...]}`` verbatim, or ``{"prefix":
        "..."}`` — the two never co-occur (0/108 catalog cases). None when this
        port isn't autogrow, or the schema carries no template (older/partial
        catalogs, offline edits), so callers fall back to the historical
        ``{base[:-1]}{N}`` pluralization guess in :meth:`autogrow_slot_example`.
        """
        t = self.options.template
        if not self.is_autogrow or not isinstance(t, dict):
            return None
        names = t.get("names")
        if isinstance(names, list) and names:
            return {"names": list(names)}
        prefix = t.get("prefix")
        if isinstance(prefix, str) and prefix:
            return {"prefix": prefix}
        return None

    @property
    def autogrow_element_template(self) -> dict | None:
        """The element-naming template a caller should USE for this autogrow
        input — never None for an autogrow port.

        Identical to :attr:`autogrow_template` when object_info ships one;
        otherwise the historical pluralization fallback (``images`` → prefix
        ``image``) that ``workflow_ops._autogrow_elem_name`` applies when the
        schema is silent. Exporters need the *effective* answer: a consumer
        holding only the exported catalog has no object_info to fall back to,
        and omitting the entry would tell it the input does not autogrow at all.
        """
        if not self.is_autogrow:
            return None
        declared = self.autogrow_template
        if declared is not None:
            return declared
        last = self.name.rsplit(".", 1)[-1]
        stem = last[:-1] if last.endswith("s") else last
        return {"prefix": stem}

    @property
    def is_upload_backed(self) -> bool:
        """This COMBO's options are the server's *installed input files*, so the
        catalog snapshot is not authoritative for it.

        ComfyUI marks these inputs in ``object_info`` with a ``<kind>_upload``
        flag (``LoadImage.image`` → ``image_upload``, ``LoadAudio.audio`` →
        ``audio_upload``, ``LoadVideo.file`` → ``video_upload``,
        ``Load3D.model_file`` → ``file_upload``) — the same flag that makes the
        frontend render an upload button. Unlike a model folder (static, set at
        install time), this list is per-user and grows at RUN time: a file the
        user just uploaded can never be in the snapshot we validated against.
        Enum-checking it therefore produces guaranteed false rejections, so the
        port is left unconstrained and the real membership check is the
        server's at run time. The sibling ``LoadImageMask.channel`` carries no
        marker and stays a normal, constrained enum.

        The marker is the CORE loaders' contract. A custom pack that ships its
        own upload button (VideoHelperSuite: ``VHS_LoadVideo.video`` is the
        input folder filtered by ``video_extensions``, button attached by class
        name in ``VHS.core.js``) reaches object_info as a bare option list, and
        the enum check then refused every content hash an agent uploaded to
        Comfy Cloud (``'<64hex>.mp4' not in 1 known options for video``) — a
        whole multi-shot assembly killed at ``run`` preflight. When the catalog
        declares nothing, the listing itself decides: a list made entirely of
        media file names is a folder listing (see
        :data:`_INPUT_FILE_EXTENSIONS`). An explicit ``<kind>_upload: false``
        still wins over that heuristic.
        """
        if self.type != "COMBO":
            return False
        if self.options.upload_declared:
            return self.options.upload
        return _is_input_file_listing(self.enum_values)

    def canonical_combo(self, value: Any) -> Any | None:
        """Map a *mangled* COMBO value to the real option it clearly means, or
        None if it can't be resolved unambiguously.

        A model name is one of these enum options, but an LLM tends to rebuild it
        from memory — adding a directory prefix (``checkpoints/foo.safetensors``
        when the option is bare ``foo.safetensors``), dropping a subfolder, or
        drifting case. The filename is almost always right, so we match by
        basename (case-insensitive) and, only when EXACTLY ONE option matches,
        return it. Ambiguous or unmatched values return None so the caller still
        surfaces ``unknown_enum_value``. Exact values return None (nothing to do).

        Upload-backed ports are exempt for the same reason they are exempt from
        the enum check (see :attr:`is_upload_backed`): the option list is a
        stale directory listing, so "the real option it clearly means" is not a
        question this snapshot can answer. Rewriting there would silently swap a
        just-uploaded ``Beach.JPG`` for the sample ``beach.jpg`` and generate
        from the wrong file.
        """
        if self.type != "COMBO" or not self.enum_values or self.is_upload_backed:
            return None
        opts = [str(e) for e in self.enum_values]
        s = str(value)
        if s in opts:
            return None
        base = s.rsplit("/", 1)[-1].lower()
        matches = [o for o in opts if o.rsplit("/", 1)[-1].lower() == base]
        if len(matches) == 1:
            return matches[0]
        ci = [o for o in opts if o.lower() == s.lower()]
        if len(ci) == 1:
            return ci[0]
        return None

    def suggest_combo(self, value: Any, *, limit: int = 5) -> list[str]:
        """Closest real options to a rejected COMBO value, for a ``did_you_mean``
        hint — so an unavailable model points at the nearest available one the
        agent can substitute or offer, instead of a dead value."""
        if self.type != "COMBO" or not self.enum_values:
            return []
        import difflib

        opts = [str(e) for e in self.enum_values]
        base = str(value).rsplit("/", 1)[-1]
        bases = [o.rsplit("/", 1)[-1] for o in opts]
        out: list[str] = []
        for g in difflib.get_close_matches(base, bases, n=limit, cutoff=0.5):
            for o in opts:
                if o.rsplit("/", 1)[-1] == g and o not in out:
                    out.append(o)
                    break
        return out[:limit]

    def validate_shape(self, value: Any) -> str | None:
        """Hard-reject on JSON-shape mismatch. Returns error message or None."""
        if self.type == "INT":
            if isinstance(value, bool) or not isinstance(value, int | float):
                return f"{self.name}: expected INT, got {type(value).__name__} {value!r}"
            if isinstance(value, float) and value != int(value):
                return f"{self.name}: expected integer, got {value}"
        elif self.type in ("FLOAT", "NUMBER"):
            if isinstance(value, bool) or not isinstance(value, int | float):
                return f"{self.name}: expected {self.type}, got {type(value).__name__}"
        elif self.type == "STRING":
            if not isinstance(value, str):
                return f"{self.name}: expected STRING (string), got {type(value).__name__}"
        elif self.type == "COMBO":
            # COMBO options are usually strings, but the server also ships
            # int-valued combos (e.g. LTXV `duration`/`fps`). Accept any
            # scalar here; membership is the catalog enum check's job. Only
            # bool and container/None shapes are a true mismatch.
            if isinstance(value, bool) or not isinstance(value, str | int | float):
                return f"{self.name}: expected COMBO (string or number), got {type(value).__name__}"
        elif self.type == "BOOLEAN":
            if not isinstance(value, bool):
                return f"{self.name}: expected BOOLEAN, got {type(value).__name__}"
        return None

    def validate_catalog(self, value: Any) -> list[dict]:
        """Catalog findings for ``value``. Returns a list of finding dicts.

        Every finding carries ``code``, ``severity`` and ``value``:
        ``severity`` so a caller never infers fatality from prose, and ``value``
        so the offending operand is a field rather than something to regex back
        out of ``message``. Fatal codes are :data:`FATAL_FINDING_CODES`.
        """
        if self.validate_shape(value) is not None:
            return []
        warnings: list[dict] = []
        if self.is_upload_backed:
            # Upload-backed input file port: unconstrained, by design. Skipping
            # BOTH enum branches is deliberate — a freshly uploaded file is
            # absent from a POPULATED snapshot just as surely as from an empty
            # one, so gating only the empty-list case would still false-reject
            # (`LoadImage.image` typically ships a handful of sample images).
            pass
        elif self.type == "COMBO" and self.enum_values:
            # Membership compares on the stringified form BOTH ways, so a value
            # matches its option regardless of int/str (`8` ↔ "8", `8.0` ↔ "8").
            # This keeps validate lenient (never false-warns on a real value)
            # while the displayed schema keeps the option's true type. The
            # warning carries the FULL valid list (typed) so a rejection tells
            # the agent exactly what to pick — no truncation, no guessing.
            candidates = {str(value)}
            if isinstance(value, float) and value.is_integer():
                candidates.add(str(int(value)))
            enum_str = {str(e) for e in self.enum_values}
            if not (candidates & enum_str):
                warning = {
                    "code": "unknown_enum_value",
                    "field": self.name,
                    "message": f"{value!r} not in {len(self.enum_values)} known options for {self.name}",
                    "valid_options": list(self.enum_values),
                }
                suggestions = self.suggest_combo(value)
                if suggestions:
                    warning["did_you_mean"] = suggestions
                    warning["message"] += f" — closest: {', '.join(suggestions)}"
                warnings.append(warning)
        elif self.type == "COMBO" and self.enum_declared:
            # The server declared this field's choices and shipped NONE of them:
            # the folder backing it is empty — no models (or inputs) installed.
            # An empty option list is STRONGER evidence the value is unavailable
            # than a populated one, not weaker: the server rejects every value
            # against an empty list (`execution.validate_inputs` →
            # "Value not in list"), so skipping the check here just defers the
            # failure to run time — and it fails hardest on the fresh install
            # this check exists to serve, where the big downloads (UNETLoader,
            # CLIPLoader) have no built-in options to compare against.
            # Ports whose options are not in object_info at all keep
            # `enum_declared=False` and stay unconstrained (see the field).
            warnings.append(
                {
                    "code": "no_options_available",
                    "field": self.name,
                    "message": f"{value!r} is unavailable: the server reports 0 installed options for {self.name}",
                    "valid_options": [],
                }
            )
        if self.type in ("INT", "FLOAT", "NUMBER") and isinstance(value, int | float):
            if self.options.min is not None and value < self.options.min:
                warnings.append(
                    {
                        "code": "below_min",
                        "field": self.name,
                        "message": f"{self.name}={value} below catalog min {self.options.min}",
                    }
                )
            if self.options.max is not None and value > self.options.max:
                warnings.append(
                    {
                        "code": "above_max",
                        "field": self.name,
                        "message": f"{self.name}={value} above catalog max {self.options.max}",
                    }
                )
        # Stamp centrally: a finding added at any site above inherits its
        # severity from the code table, so none can ship without one.
        for w in warnings:
            w.setdefault("severity", finding_severity(w.get("code", "")))
            w.setdefault("value", value)
        return warnings


@dataclass
class Morphism:
    id: str
    display_name: str = ""
    description: str = ""
    category: str = ""
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    is_output_node: bool = False
    is_api_node: bool = False
    deprecated: bool = False
    experimental: bool = False
    search_aliases: list[str] = field(default_factory=list)
    pack: str = ""
    labels: list[str] = field(default_factory=list)
    cloud_disabled: bool = False

    def output_types(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self.outputs:
            if p.type not in seen:
                seen.add(p.type)
                out.append(p.type)
        return out

    def input_link_types(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self.inputs:
            if p.is_link and p.type not in seen:
                seen.add(p.type)
                out.append(p.type)
        return out

    def required_link_types(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self.inputs:
            if p.is_link and p.required and p.type not in seen:
                seen.add(p.type)
                out.append(p.type)
        return out

    def has_input(self, t: str) -> bool:
        return any(p.is_link and p.type == t for p in self.inputs)

    def has_output(self, t: str) -> bool:
        return any(p.type == t for p in self.outputs)

    def can_apply(self, available: set[str]) -> bool:
        return all(t in available for t in self.required_link_types())


# ---------------------------------------------------------------------------
# Parsing — mirrors nodegraph/parse.go
# ---------------------------------------------------------------------------


# Wildcard socket types: a port carrying one of these accepts/produces ANY type,
# so an edge touching it can never be a type mismatch.
#
# "*" is ComfyUI's classic wildcard. COMFY_MATCHTYPE_V3 is the V3 schema's
# match-type: a generic port whose concrete type is resolved at runtime from
# what it is wired to (ComfySwitchNode, ResizeImageMaskNode and friends). It was
# not recognised here, so every edge into or out of a V3 match-type port was
# reported as edge_type_mismatch — ~30 spurious warnings in a single 48h prod
# window, on graphs that were correct. The agent had to write a paragraph
# explaining them away in nearly every reply, which teaches it to discount
# validator output generally.
_WILDCARD_TYPE_PREFIX = "COMFY_MATCHTYPE"
_WILDCARD_TYPES = frozenset({"*"})


# Finding severity. Every catalog finding carries an explicit
# ``severity`` so a consumer never has to infer fatality from prose. The rule the
# codes below encode: a finding is an ERROR when the value cannot resolve at run
# time, which is precisely when `Graph.validate_workflow` already refuses it
# (see `_validate_catalog_value`) — the edit path was the only surface still
# demoting these to advisory warnings on an ``ok:true`` envelope.
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

#: Codes whose finding means "the server will reject this value". Callers must
#: treat these as fatal; `workflow_ops` refuses the edit outright rather than
#: writing the value and warning about it.
FATAL_FINDING_CODES = frozenset(
    {
        "unknown_enum_value",
        "no_options_available",
        "below_min",
        "above_max",
    }
)


def finding_severity(code: str) -> str:
    """Severity for a finding code. Unknown codes are advisory, never fatal."""
    return SEVERITY_ERROR if code in FATAL_FINDING_CODES else SEVERITY_WARNING


def _is_wildcard_type(type_id: str) -> bool:
    """True when a socket type accepts/produces any type.

    Matches COMFY_MATCHTYPE_V3 by prefix rather than exact string so a future
    match-type revision (V4, ...) does not silently reintroduce the false
    warnings this exists to prevent.
    """
    if not type_id:
        return False
    return type_id in _WILDCARD_TYPES or type_id.startswith(_WILDCARD_TYPE_PREFIX)


def _edge_types_compatible(src_type: str, dst_type: str) -> bool:
    """Whether an output socket of ``src_type`` may drive an input of ``dst_type``.

    A wildcard on either end (``*``, ``COMFY_MATCHTYPE_V3``) always matches.
    Otherwise defer to the converter's ``_is_valid_connection`` — the mirror of
    the frontend's ``isValidConnection`` — which expands a COMMA-SEPARATED
    union on both ends (``INT,FLOAT`` on a math operand, ``MESH,FILE_3D_GLB,…``
    on a 3D importer) before comparing. Comparing the raw strings reported a
    ``FLOAT`` output into an ``INT,FLOAT`` input as ``edge_type_mismatch``:
    12 of the 23 such warnings in a 3-day prod window were this shape, on
    edges the frontend draws and the server runs.
    """
    if _is_wildcard_type(src_type) or _is_wildcard_type(dst_type):
        return True
    # Lazy: workflow_to_api imports this module at load time.
    from comfy_cli.workflow_to_api import _is_valid_connection

    return _is_valid_connection(src_type, dst_type)


def _is_dynamic_combo_type(type_id: str) -> bool:
    """V3 dynamic-combo types (e.g. ``COMFY_DYNAMICCOMBO_V3``): a selector
    widget whose chosen option contributes its own sub-inputs. Same rule the
    UI→API converter applies (``workflow_to_api._is_widget_input``)."""
    return type_id.startswith("COMFY_") and "COMBO" in type_id


def _has_control_after_generate_slot(port: Port) -> bool:
    """True if the frontend places a ``control_after_generate`` marker widget
    right after this port — explicit (schema ``control_after_generate: True``)
    or implicit (the frontend's ``useIntWidget`` composable always companions
    an INT ``seed``/``noise_seed`` input, regardless of the schema flag).
    ``port.name`` may be dotted for a dynamic-combo sub-input (``model.seed``);
    the implicit rule keys off the leaf name, same as the converter.
    Mirrors ``workflow_to_api._has_control_after_generate_companion``'s
    schema-level test."""
    if port.options.control_after_generate:
        return True
    # Same seed-like rule as the converter's companion guard: partner nodes
    # name the widget every which way — ``image_seed``/``model_seed`` (Tripo),
    # ``Seed`` (Rodin3D), ``rand_seed``, ``noise_seed_sde``, ``variation_seed``
    # — and several ship it UNFLAGGED, yet the frontend still appends the
    # companion. An exact ``seed``/``noise_seed`` match here made the exported
    # widget catalog off by one for every such node, so a name<->index
    # consumer wrote into the marker slot.
    leaf_name = port.name.rsplit(".", 1)[-1]
    return port.type == "INT" and "seed" in leaf_name.lower()


def frontend_extra_widget_names(m: Morphism) -> list[str]:
    """Widget names the frontend injects into ``m``'s inputs AFTER object_info.

    ``getOrderedInputSpecs`` walks ``input_order.required``, then
    ``input_order.optional``, then every input not listed there — so an
    extension-injected input always serializes after every declared widget,
    optional ones included. Registration order decides the order among them:
    ``Comfy.AudioWidget`` (``audioUI``) before ``Comfy.UploadAudio`` /
    ``Comfy.UploadImage`` (``upload``); the lazily loaded 3D extensions
    (``PREVIEW_3D`` ``image``) last. A name the server already declares
    (``Load3DAdvanced.viewport_state``) is never injected twice.
    """
    declared = {p.name for p in m.inputs}
    class_id = getattr(m, "id", "")
    extras: list[str] = []
    if class_id in _AUDIO_UI_CLASSES:
        extras.append("audioUI")
    upload = False
    for p in m.inputs:
        if not p.required or p.is_link:
            continue
        flags = set(p.options.upload_flags)
        if p.name == "audio" and "audio_upload" in flags:
            upload = True
        elif p.is_upload_backed and flags & _IMAGE_UPLOAD_FLAGS:
            upload = True
    if upload:
        extras.append("upload")
    if class_id in _PREVIEW_3D_CLASSES:
        extras.append("image")
    return [e for e in extras if e not in declared]


def _is_link(type_id: str, is_enum: bool, force_input: bool, widget_type: str | None = None) -> bool:
    """Determine if an input participates in typed wiring (link) or is inline (widget)."""
    if is_enum:
        return False
    # ``widgetType`` overrides the socket type for widget selection (a
    # ``FLOAT,INT`` math input with ``widgetType: "STRING"``, Preview3D's
    # ``STRING,FILE_3D_*`` model_file with ``widgetType: "STRING"``).
    if widget_type and not force_input:
        return False
    # A dynamic combo is a widget port even when its options block is missing
    # or malformed — the frontend always renders the selector inline.
    if _is_dynamic_combo_type(type_id):
        return False
    if type_id in _FRONTEND_DOM_WIDGET_TYPES and not force_input:
        return False
    if type_id in _IMPLICIT_WIDGET_TYPES and not force_input and type_id != "*":
        return False
    return True


def _derive_pack(python_module: str) -> str:
    if not python_module:
        return "core"
    if (
        python_module.startswith("nodes")
        or python_module.startswith("comfy_extras")
        or python_module.startswith("comfy.comfy_types")
    ):
        return "core"
    if python_module.startswith("custom_nodes."):
        parts = python_module.split(".", 3)
        if len(parts) >= 2:
            return parts[1]
    return "core"


def _upload_marked(opts_raw: dict) -> bool:
    """True when the input's options dict carries an upload marker.

    ComfyUI has no single flag name — the marker is ``<kind>_upload`` and the
    kind varies by loader (``image_upload``, ``audio_upload``, ``video_upload``,
    ``file_upload`` are all present in the production catalog, and custom packs
    add their own). Matching the suffix rather than an allow-list keeps a new
    loader kind from silently regressing into false rejections.
    """
    return any(isinstance(k, str) and k.endswith("_upload") and bool(v) for k, v in opts_raw.items())


def _upload_declared(opts_raw: dict) -> bool:
    """True when the options dict carries an upload marker key at all — set
    either way. ``image_upload: false`` is a declaration, not an absence."""
    return any(isinstance(k, str) and k.endswith("_upload") for k in opts_raw)


def _is_input_file_listing(values: list[Any]) -> bool:
    """True when a COMBO's option list reads as a listing of the server's input
    folder: non-empty, and EVERY option is a file name carrying one of
    :data:`_INPUT_FILE_EXTENSIONS`. One non-file option (a ``none`` sentinel,
    a bare label) makes it a vocabulary that happens to contain file names.
    """
    if not values:
        return False
    for v in values:
        if not isinstance(v, str):
            return False
        name = v.rsplit("/", 1)[-1]
        stem, dot, ext = name.rpartition(".")
        if not dot or not stem or ext.lower() not in _INPUT_FILE_EXTENSIONS:
            return False
    return True


def _parse_port_options(opts_raw: dict) -> PortOptions:
    template_raw = opts_raw.get("template")
    return PortOptions(
        min=opts_raw.get("min"),
        max=opts_raw.get("max"),
        step=opts_raw.get("step"),
        default=opts_raw.get("default"),
        multiline=bool(opts_raw.get("multiline", False)),
        control_after_generate=_control_after_generate_set(opts_raw.get("control_after_generate")),
        force_input=bool(opts_raw.get("forceInput", False)),
        template=template_raw if isinstance(template_raw, dict) else None,
        upload=_upload_marked(opts_raw),
        upload_declared=_upload_declared(opts_raw),
        upload_flags=tuple(
            sorted(k for k, v in opts_raw.items() if isinstance(k, str) and k.endswith("_upload") and bool(v))
        ),
        widget_type=opts_raw.get("widgetType")
        if isinstance(opts_raw.get("widgetType"), str) and opts_raw.get("widgetType")
        else None,
    )


def _control_after_generate_set(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val != "" and val != "false"
    return True


def _parse_input_spec(spec: Any) -> tuple[str, bool, list[Any], PortOptions, bool, list[dict]]:
    """Returns (type_id, is_enum, enum_values, options, enum_declared, dynamic_options).

    ``enum_declared`` is True when the spec shipped an explicit choice list —
    *including an empty one* — so validation can tell "this server has zero
    options for the field" from "this field's options aren't in object_info".
    It is deliberately separate from ``is_enum`` (which stays truthiness-based)
    because ``is_enum`` also decides link-vs-widget, and an empty option list
    must not move a port between those.
    """
    if isinstance(spec, str):
        return spec, False, [], PortOptions(), False, []

    if not isinstance(spec, list) or len(spec) == 0:
        return "UNKNOWN", False, [], PortOptions(), False, []

    opts_raw = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    port_opts = _parse_port_options(opts_raw)

    first = spec[0]
    if isinstance(first, str):
        if _is_dynamic_combo_type(first):
            # V3 dynamic combo: options are [{"key": ..., "inputs": {...}}]
            # dicts, NOT membership choices. Expose the keys as the selector's
            # enum and retain the raw option dicts so slot extraction / writes
            # can expand the selected option's sub-inputs.
            options = opts_raw.get("options")
            dyn = [o for o in options if isinstance(o, dict) and "key" in o] if isinstance(options, list) else []
            return first, True, [o["key"] for o in dyn], port_opts, isinstance(options, list), dyn
        # V3 / partner-API combo dialect: the type is the literal string
        # "COMBO" and the choices live in the options dict, e.g.
        # ["COMBO", {"options": ["480p", "720p"]}].
        # Without this, dict-form combos lose their enum and validate can't
        # enum-check them — exactly the partner nodes (ByteDance, BFL, …)
        # where the choices array is the precision check.
        options = opts_raw.get("options")
        if isinstance(options, list) and all(_is_scalar_choice(v) for v in options):
            # Keep each option's real type: an int-valued combo (Sora-2/LTXV
            # `duration`) must stay [4, 8, 12], not ["4","8","12"], so `nodes
            # show` is truthful and agents pass the type the cloud accepts.
            # An EMPTY list is a declared-but-unpopulated combo — still not an
            # enum for wiring purposes (`is_enum` stays False, exactly as
            # before), but flagged as declared so validate can say "0 options
            # installed" instead of silently skipping the check.
            return first, bool(options), list(options), port_opts, True, []
        # Dynamic combo declared WITHOUT the V3 type name: options are
        # {key, inputs} dicts. It IS a widget (the frontend renders a selector
        # plus key-dependent sub-widgets) and the selector's choices are the
        # keys. Detected structurally rather than by type string so a node that
        # ships the option tree under a plain COMBO still expands its
        # sub-widgets; without this the sub-inputs vanish from widget_order and
        # `set-widget <id>.<sub>` reports the widget as unknown.
        # `enum_declared=True`: the choice list shipped, it just isn't scalar.
        if isinstance(options, list) and options and all(isinstance(v, dict) and "key" in v for v in options):
            return first, True, [v["key"] for v in options], port_opts, True, options
        # No usable `options` key at all: a remote/dynamic combo whose choices
        # the frontend fetches at runtime. Unknowable here — stay unconstrained.
        return first, False, [], port_opts, False, []

    if isinstance(first, list):
        # Same: preserve the option types for the classic list-form combo. The
        # list IS the declaration, so an empty one (a model folder with nothing
        # installed) is declared-but-empty, not unconstrained.
        return "COMBO", True, list(first), port_opts, True, []

    return "UNKNOWN", False, [], port_opts, False, []


def _is_scalar_choice(v: Any) -> bool:
    """A combo option is enumerable only if it's a scalar. Dynamic combos
    (COMFY_DYNAMICCOMBO_V3) carry dict options describing sub-inputs — those
    are not membership choices and must not be flattened into enum_values."""
    return isinstance(v, str | int | float) and not isinstance(v, bool)


def _ordered_names(raw: dict, order: list[str] | None) -> list[str]:
    """Return input names in declared order, falling back to alphabetical."""
    seen: set[str] = set()
    out: list[str] = []
    for name in order or []:
        if name in raw and name not in seen:
            out.append(name)
            seen.add(name)
    for name in sorted(raw.keys()):
        if name not in seen:
            out.append(name)
    return out


def _port_from_spec(name: str, spec: Any, required: bool) -> Port:
    """One place that turns an ``INPUT_TYPES`` entry into a :class:`Port`."""
    type_id, is_enum, enum_values, opts, enum_declared, dynamic_options = _parse_input_spec(spec)
    return Port(
        name=name,
        type=type_id,
        required=required,
        is_link=_is_link(type_id, is_enum, opts.force_input, opts.widget_type),
        enum_values=enum_values,
        enum_declared=enum_declared,
        options=opts,
        dynamic_options=dynamic_options,
        raw_spec=spec,
    )


def _parse_inputs(raw: dict, order: list[str] | None, required: bool) -> list[Port]:
    ports: list[Port] = []
    for name in _ordered_names(raw, order):
        ports.append(_port_from_spec(name, raw[name], required))
    return ports


def _unmarshal_string_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val]
    return []


def _parse_morphism(node_id: str, raw: dict) -> Morphism:
    input_block = raw.get("input") or {}
    input_order = raw.get("input_order") or {}
    req_raw = input_block.get("required") or {}
    opt_raw = input_block.get("optional") or {}
    req_order = input_order.get("required")
    opt_order = input_order.get("optional")

    inputs = _parse_inputs(req_raw, req_order, required=True)
    inputs += _parse_inputs(opt_raw, opt_order, required=False)

    raw_outputs = raw.get("output") or []
    output_names = _unmarshal_string_list(raw.get("output_name"))
    outputs: list[Port] = []
    for i, out in enumerate(raw_outputs):
        name = output_names[i] if i < len(output_names) else ""
        t = out if isinstance(out, str) else "COMBO"
        outputs.append(Port(name=name, type=t, required=True, is_link=True))

    # These are declared `str` and every consumer treats them as one (`.lower()`,
    # `.startswith()`, markup escaping). /object_info is server-supplied and a
    # custom node can put any JSON type here, so coerce rather than trust the
    # annotation — an int category used to crash `nodes search` with a raw
    # AttributeError instead of the structured error the command otherwise emits.
    return Morphism(
        id=node_id,
        display_name=str(raw.get("display_name") or node_id),
        description=str(raw.get("description") or ""),
        category=str(raw.get("category") or ""),
        inputs=inputs,
        outputs=outputs,
        is_output_node=bool(raw.get("output_node", False)),
        is_api_node=bool(raw.get("api_node", False)),
        deprecated=bool(raw.get("deprecated", False)),
        experimental=bool(raw.get("experimental", False)),
        search_aliases=_unmarshal_string_list(raw.get("search_aliases")),
        pack=_derive_pack(str(raw.get("python_module") or "")),
    )


# ---------------------------------------------------------------------------
# Annotations — mirrors nodegraph/annotations.go
# ---------------------------------------------------------------------------


def parse_supported_nodes(data: bytes) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse supported_nodes.yaml → (node_pack, node_labels)."""
    try:
        import yaml

        cfg = yaml.safe_load(data)
    except Exception:
        return {}, {}
    if not isinstance(cfg, dict):
        return {}, {}
    node_pack: dict[str, str] = {}
    node_labels: dict[str, list[str]] = {}
    for pack in cfg.get("node_packs") or []:
        if not isinstance(pack, dict):
            continue
        pack_name = pack.get("name", "")
        for node_name, labels in (pack.get("node_labels") or {}).items():
            node_pack[node_name] = pack_name
            node_labels[node_name] = list(labels) if isinstance(labels, list) else []
    return node_pack, node_labels


def parse_disable_config(data: bytes) -> set[str]:
    """Parse cloud_disable_config.yaml → set of labels that disable nodes."""
    try:
        import yaml

        cfg = yaml.safe_load(data)
    except Exception:
        return set()
    if not isinstance(cfg, dict):
        return set()
    disable = cfg.get("disable_nodes") or {}
    labels: set[str] = set()
    for rule in disable.get("or") or []:
        if isinstance(rule, dict):
            for label, enabled in rule.items():
                if enabled:
                    labels.add(label)
    return labels


# ---------------------------------------------------------------------------
# Graph — mirrors nodegraph/graph.go
# ---------------------------------------------------------------------------


class Graph:
    """Indexed compatibility graph over ComfyUI node classes."""

    def __init__(self) -> None:
        self._nodes: dict[str, Morphism] = {}
        self._producers: dict[str, list[Morphism]] = defaultdict(list)
        self._consumers: dict[str, list[Morphism]] = defaultdict(list)
        self._types: set[str] = set()
        self._annotated = False
        # Lazily-computed closure of types obtainable without wiring anything in
        # (see ``free_types``). Invalidated implicitly: graphs are built once.
        self._free_types: frozenset[str] | None = None
        # The raw ``/object_info`` payload this graph was built from. Retained
        # verbatim so callers that also need to lower a UI-format workflow to
        # API format (``convert_ui_to_api``) can reuse it without a second fetch.
        self._raw: dict[str, Any] = {}

    @property
    def object_info(self) -> dict[str, Any]:
        """The raw ``/object_info`` dict this graph was built from (``{}`` if
        the graph was constructed without one).

        Read-only: this is the graph's live internal schema state, returned by
        reference to avoid copying a large payload. Callers (e.g. the validate
        command handing it to ``convert_ui_to_api``) must not mutate it.
        """
        return self._raw

    @classmethod
    def from_object_info(cls, object_info: dict[str, Any]) -> Graph:
        if not isinstance(object_info, dict):
            raise LoadError(
                "object_info must be a JSON object",
                details={"top_level_type": type(object_info).__name__},
            )
        g = cls()
        g._raw = object_info
        for node_id, raw in object_info.items():
            if not isinstance(raw, dict):
                continue
            m = _parse_morphism(node_id, raw)
            g._nodes[m.id] = m
            # A deprecated class stays addressable by name (show, validate,
            # edits on a graph that already holds it) but is never a
            # discovery answer: upstream/downstream/path/free-producer all
            # read these indexes.
            for t in m.output_types():
                if not m.deprecated:
                    g._producers[t].append(m)
                g._types.add(t)
            for t in m.input_link_types():
                if not m.deprecated:
                    g._consumers[t].append(m)
                g._types.add(t)
        # Sort indexes for deterministic output
        for t in g._producers:
            g._producers[t].sort(key=lambda m: m.id)
        for t in g._consumers:
            g._consumers[t].sort(key=lambda m: m.id)
        return g

    # -- Annotation --

    def annotate(
        self,
        supported_nodes_yaml: bytes | None = None,
        cloud_disable_yaml: bytes | None = None,
    ) -> None:
        from comfy_cli.cql import annotations_source

        node_pack, node_labels, disable_labels = annotations_source.parsed_annotations(
            supported_nodes_yaml, cloud_disable_yaml
        )

        for nid, m in self._nodes.items():
            if nid in node_pack:
                m.pack = node_pack[nid]
            if nid in node_labels:
                m.labels = node_labels[nid]
            m.cloud_disabled = any(label in disable_labels for label in m.labels)
        self._annotated = True

    # -- Lookup --

    def node(self, name: str) -> Morphism | None:
        # A malformed workflow can supply an unhashable class_type (list/dict);
        # dict.get on an unhashable key raises TypeError, so screen it out here
        # rather than crash the reachability walk / lookups.
        if not isinstance(name, str):
            return None
        return self._nodes.get(name)

    def all_nodes(self) -> list[Morphism]:
        return sorted(self._nodes.values(), key=lambda m: m.id)

    def node_count(self) -> int:
        return len(self._nodes)

    # -- Traversal --

    def upstream(self, name: str) -> list[Morphism]:
        m = self._nodes.get(name)
        if m is None:
            return []
        seen: set[str] = set()
        result: list[Morphism] = []
        for t in m.input_link_types():
            for producer in self._producers.get(t, []):
                if producer.id != name and producer.id not in seen:
                    seen.add(producer.id)
                    result.append(producer)
        result.sort(key=lambda m: m.id)
        return result

    def downstream(self, name: str) -> list[Morphism]:
        m = self._nodes.get(name)
        if m is None:
            return []
        seen: set[str] = set()
        result: list[Morphism] = []
        for t in m.output_types():
            for consumer in self._consumers.get(t, []):
                if consumer.id != name and consumer.id not in seen:
                    seen.add(consumer.id)
                    result.append(consumer)
        result.sort(key=lambda m: m.id)
        return result

    def cloud_enabled_nodes(self) -> list[Morphism]:
        """All nodes that are enabled on Comfy Cloud."""
        return sorted([m for m in self._nodes.values() if not m.cloud_disabled], key=lambda m: m.id)

    def packs(self) -> list[str]:
        """All known pack names, sorted."""
        return sorted(set(m.pack for m in self._nodes.values() if m.pack))

    def free_types(self) -> frozenset[str]:
        """Types obtainable without wiring anything in — the fixpoint closure
        over nodes whose required link inputs are already satisfied (loaders,
        primitives, text-to-X API nodes, and whatever those unlock).

        The exact walker uses this to decide whether a step's *other* required
        inputs (a ``VAE`` for ``VAEDecode``, say) could be supplied by a support
        node. Support nodes are reported per path rather than routed through, so
        they never masquerade as steps on the requested path.
        """
        if self._free_types is None:
            free: set[str] = set()
            changed = True
            while changed:
                changed = False
                for m in self._nodes.values():
                    # Same rule as the producer index: a deprecated loader
                    # must not make its type look obtainable.
                    if m.deprecated or not m.can_apply(free):
                        continue
                    for t in m.output_types():
                        if t != "*" and t not in free:
                            free.add(t)
                            changed = True
            self._free_types = frozenset(free)
        return self._free_types

    def search_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        exact: bool = True,
        max_depth: int = 6,
        max_paths: int = 10,
        max_states: int = _MAX_PATH_SEARCH_STATES,
    ) -> dict:
        """Routed paths from ``from_type`` to ``to_type``, with honest bounds.

        Every step consumes the type the previous step produced — the first step
        consumes ``from_type`` — through a **declared link input of that type**,
        so a node that merely owns a widget *named* like the type (the COMBO
        ``model`` on the partner-API image nodes) is never routed through. Path
        length (the number of steps) is bounded by ``max_depth``.

        In ``exact`` mode a step is only taken when the node's other required
        link inputs are satisfiable — from types produced earlier on the path, or
        from a support node needing no wiring of its own (``free_types``). Those
        support inputs are reported per path under ``support`` instead of being
        spliced into ``steps``. Loose mode skips the satisfiability check and
        reports no support.

        Returns ``{"paths", "truncated", "truncated_by", "depth_limited",
        "collapsed", "not_searched", "not_searched_reason"}``:

        - ``not_searched`` — the walk declined the query outright and never ran,
          so the empty result is an abstention rather than an answer.
          ``not_searched_reason`` names the only shape that does this:
          ``"degenerate_bounds"`` (``max_depth`` or ``max_paths`` below 1, a
          bound no path can satisfy). A same-type query is *answered*, not
          declined — self-returning routes such as
          ``MODEL -> LoraLoaderModelOnly -> MODEL`` are real, and the no-op rule
          below exempts the terminal hop so they are found like any other.
        - ``truncated`` — the walk stopped early (``max_paths`` reached, or the
          internal state budget exhausted), so paths exist that are not listed.
        - ``depth_limited`` — the frontier was still expanding at ``max_depth``,
          so longer paths may exist beyond the requested bound.
        - ``collapsed`` — the walk reached some intermediate state by more than
          one route and explored it only once, so alternate chains through that
          state are not listed. Reachability is unaffected (the surviving route
          explores exactly the same continuations), which is why an **empty**
          result with all four flags false is a proof that no path exists —
          but a non-empty one is a sample of the routes, not the full set.

        A caller may only treat the listing as exhaustive when all four are
        false. Each errs toward true: hitting ``max_paths`` exactly is reported
        as truncated even when nothing further existed, and a revisited state is
        reported as collapsed even when its alternate route led nowhere.
        """
        result: dict = {
            "paths": [],
            "truncated": False,
            "truncated_by": None,
            "depth_limited": False,
            "collapsed": False,
            "not_searched": False,
            "not_searched_reason": None,
        }
        # The one query shape the walk declines outright. The empty result it
        # yields is an abstention, not a proof, so it has to say so — otherwise
        # it reads as "no route exists" with every limit flag reassuringly false.
        if max_depth < 1 or max_paths < 1:
            result["not_searched"] = True
            result["not_searched_reason"] = "degenerate_bounds"
            return result

        free = self.free_types() if exact else frozenset()
        paths: list[dict] = result["paths"]
        # state: (current_type, types produced by the path so far, steps[])
        queue: list[tuple[str, frozenset[str], list[dict]]] = [(from_type, frozenset(), [])]
        visited: set[tuple[str, frozenset[str]]] = {(from_type, frozenset())}
        states = 0

        while queue and len(paths) < max_paths:
            next_queue: list[tuple[str, frozenset[str], list[dict]]] = []
            for cur_type, produced, steps in queue:
                consumers = self._consumers.get(cur_type, [])
                if len(steps) >= max_depth:
                    if consumers:
                        result["depth_limited"] = True
                    continue
                available = free | produced | {from_type}
                for consumer in consumers:
                    if exact and not consumer.can_apply(available):
                        continue
                    outs = [t for t in consumer.output_types() if t != "*"]
                    # Loose mode ignores availability, so keeping ``produced``
                    # empty there collapses the visited key back to the type
                    # alone — the pruning loose path-finding has always used.
                    new_produced = produced | frozenset(outs) if exact else produced
                    for out_t in outs:
                        # A step that hands back the type it consumed is a no-op
                        # hop — except when that type is the target, where it is
                        # the terminal step and the only one that can answer the
                        # query (``MODEL -> LoraLoaderModelOnly -> MODEL``). The
                        # exemption is confined to same-type queries: any state
                        # whose output matched ``to_type`` was recorded as a
                        # completed path and never queued, so ``cur_type ==
                        # to_type`` can only hold for the initial frontier item,
                        # i.e. exactly when ``from_type == to_type``.
                        if out_t == cur_type and out_t != to_type:
                            continue
                        step = {"node": consumer.id, "input_type": cur_type, "output_type": out_t}
                        new_steps = steps + [step]
                        if out_t == to_type:
                            paths.append(self._path_record(from_type, to_type, new_steps, free if exact else None))
                            if len(paths) >= max_paths:
                                result["truncated"] = True
                                result["truncated_by"] = "max_paths"
                                return result
                            continue
                        key = (out_t, new_produced)
                        if key in visited:
                            # A second route into a state already queued. Its
                            # continuations are covered by the first one, so
                            # dropping it costs no reachability — but the chains
                            # it would have printed are lost, so the listing can
                            # no longer be called complete.
                            result["collapsed"] = True
                            continue
                        if states >= max_states:
                            result["truncated"] = True
                            result["truncated_by"] = "max_states"
                            return result
                        states += 1
                        visited.add(key)
                        next_queue.append((out_t, new_produced, new_steps))
            queue = next_queue
        return result

    def find_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        max_depth: int = 4,
        max_paths: int = 10,
    ) -> list[dict]:
        """Loose (routing-only) paths — see ``search_paths``."""
        return self.search_paths(from_type, to_type, exact=False, max_depth=max_depth, max_paths=max_paths)["paths"]

    def exact_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        max_depth: int = 6,
        max_paths: int = 10,
    ) -> list[dict]:
        """Satisfiability-aware paths — see ``search_paths``."""
        return self.search_paths(from_type, to_type, exact=True, max_depth=max_depth, max_paths=max_paths)["paths"]

    def _path_record(self, from_type: str, to_type: str, steps: list[dict], free: frozenset[str] | None) -> dict:
        record = {"from": from_type, "to": to_type, "steps": steps}
        if free is not None:
            record["support"] = self._support_for(from_type, steps, free)
        return record

    def _support_for(self, from_type: str, steps: list[dict], free: frozenset[str]) -> list[dict]:
        """Required link inputs a routed path needs *besides* the routed type,
        each with a node that can supply it without wiring of its own."""
        available: set[str] = {from_type}
        support: list[dict] = []
        seen: set[str] = set()
        for step in steps:
            m = self._nodes.get(step["node"])
            if m is None:
                continue
            for t in m.required_link_types():
                if t in available or t in seen:
                    continue
                seen.add(t)
                support.append({"type": t, "node": self._free_producer(t, free)})
            available.update(m.output_types())
        return support

    def _free_producer(self, type_id: str, free: frozenset[str]) -> str | None:
        """A node producing ``type_id`` that needs no incoming links, preferring
        one with no link inputs at all. ``None`` when the type can only be
        obtained by wiring something up first."""
        if type_id not in free:
            return None
        producers = self._producers.get(type_id, [])
        for m in producers:
            if not m.required_link_types():
                return m.id
        for m in producers:
            if m.can_apply(free):
                return m.id
        return None

    # -- Browse --

    def list_types(self) -> list[str]:
        return sorted(self._types)

    def category_tree(self) -> dict:
        """Build a hierarchical category tree with node counts."""
        counts: dict[str, int] = defaultdict(int)
        for m in self._nodes.values():
            if m.category:
                counts[m.category] += 1

        root: dict[str, Any] = {"FullPath": "", "Count": 0, "Children": {}}
        for path, count in sorted(counts.items()):
            parts = path.split("/")
            node = root
            for i, part in enumerate(parts):
                full = "/".join(parts[: i + 1])
                if part not in node["Children"]:
                    node["Children"][part] = {"FullPath": full, "Count": 0, "Children": {}}
                child = node["Children"][part]
                child["Count"] += count
                node = child
        return {"Root": root}

    # -- Widget order --

    def widget_order(self, class_name: str) -> list[str]:
        """Value-independent widget order. A dynamic combo contributes only its
        selector slot; use ``widget_order_for_node`` when the node's current
        ``widgets_values`` are available so the selected option's sub-inputs
        expand into their real positions."""
        m = self._nodes.get(class_name)
        if m is None:
            return []
        order: list[str] = []
        for p in m.inputs:
            if p.is_link:
                continue
            order.append(p.name)
            if _has_control_after_generate_slot(p):
                order.append("control_after_generate")
        order.extend(frontend_extra_widget_names(m))
        return order

    def widget_order_default(self, class_name: str) -> list[str]:
        """Static order with every dynamic combo expanded at its FIRST key.

        :meth:`widget_order` is deliberately value-independent — a combo
        contributes only its selector, because which sub-inputs exist depends on
        the node's current selection. A CATALOG has no node and no selection, but
        its consumers still need the sub-input names in order to address them
        (``set-widget <id>.model.resolution``). So the catalog publishes the order
        a FRESH node would have, which is the first key — the same option
        ``add_node`` materializes via :meth:`widget_defaults`.
        """
        m = self._nodes.get(class_name)
        if m is None:
            return []
        return [e.name for e in _expand_widget_entries(m, [], first_key=True)]

    def widget_order_for_node(self, class_name: str, widgets_values: list[Any] | None) -> list[str]:
        """Value-aware widget order: like ``widget_order`` but at each dynamic
        combo the current selector (read from its positional slot) picks an
        option whose widget-like sub-inputs are expanded in place as
        ``<name>.<sub>`` — matching how the frontend inlines the selected
        option's sub-values into the flat positional ``widgets_values``."""
        m = self._nodes.get(class_name)
        if m is None:
            return []
        return [e.name for e in _expand_widget_entries(m, widgets_values or [])]

    def editable_widget_names(self, class_name: str, widgets_values: list[Any] | None = None) -> list[str]:
        """The subset of :meth:`widget_order_for_node` a write may target —
        schema-backed slots only, i.e. what ``comfy workflow slots`` advertises."""
        m = self._nodes.get(class_name)
        if m is None:
            return []
        return _editable_widget_names(_expand_widget_entries(m, widgets_values or []))

    def frontend_injected_widget_names(self, class_name: str) -> list[str]:
        """Names in the widget order that the frontend injects with no schema
        port (``upload``, ``audioUI``, ``PREVIEW_3D`` ``image``). They own a
        positional slot but are never an edit target."""
        m = self._nodes.get(class_name)
        if m is None:
            return []
        return frontend_extra_widget_names(m)

    def widget_defaults(self, class_name: str) -> dict[str, Any]:
        """Default value per widget-order name — including dynamic-combo selectors
        (first key), their sub-widgets, and control_after_generate. Used by
        ``add-node`` so a fresh node is runtime-valid, aligned with the converter."""
        m = self._nodes.get(class_name)
        if m is None:
            return {}
        out: dict[str, Any] = {}
        # Same walk as ``widget_order_default`` (first key at every dynamic
        # combo, link-only sub-inputs skipped), so the two can never disagree
        # about which names own a slot.
        for entry in _expand_widget_entries(m, [], first_key=True):
            if entry.port is None:
                if entry.name == "control_after_generate":
                    out[entry.name] = "fixed"
                # Frontend-injected marker slots (``upload``, ``audioUI``) are
                # trailing and ``serialize: false`` on current frontends: no
                # default, no value.
                continue
            out[entry.name] = _widget_default(entry.port)
        # Frontend-injected slots: the ``upload``/``audioUI`` buttons are
        # ``serialize: false`` on current frontends — no default, no value, so
        # a fresh node ends before them (``_build_node`` drops trailing names
        # with no default). The injected PREVIEW_3D ``image`` (SaveGLB /
        # Preview3D) IS a DOM widget the frontend serializes as ``""`` — the
        # captured shape is ``["mesh/ComfyUI", ""]`` — so it gets the same
        # default a declared PREVIEW_3D port gets.
        if m.id in _PREVIEW_3D_CLASSES and "image" not in out and "image" in frontend_extra_widget_names(m):
            out["image"] = _FRONTEND_DOM_WIDGET_DEFAULTS["PREVIEW_3D"]
        return out

    def autogrow_groups(self, class_name: str, widgets_values: list[Any] | None) -> list[Port]:
        """Every ``COMFY_AUTOGROW_V3`` group a node of ``class_name`` exposes
        for its CURRENT selection, in declaration order: top-level groups
        (``BatchImagesNode.images``) and those nested under the option each
        dynamic combo currently selects, named the way the frontend names
        them — dotted under the combo (``model.reference_images``).

        ``widgets_values`` supplies the selectors (read positionally, exactly
        as ``widget_order_for_node`` does), so a node switched to an option
        without the group stops exposing it. Unknown class → ``[]``.
        """
        m = self._nodes.get(class_name)
        if m is None:
            return []
        sub_links: list[Port] = []
        _expand_widget_entries(m, widgets_values or [], sub_links=sub_links)
        top = [p for p in m.inputs if p.is_autogrow]
        nested = [p for p in sub_links if p.is_autogrow]
        return top + nested

    # -- Validation --

    def validate_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Validate an API-format workflow. Returns {valid, errors, warnings}."""
        errors: list[dict] = []
        warnings: list[dict] = []
        all_names = list(self._nodes.keys())
        # No-outputs check: the server rejects any prompt with zero output nodes
        # (execution.py:1155-1162, prompt_no_outputs). Track whether any
        # recognized node is an output node.
        has_output_node = False
        # Server parity: ComfyUI's validate_prompt only validates output nodes
        # and their transitive input ancestors — any node not reachable from an
        # output is pruned and never validated (execution.py). Restrict the
        # promoted hard checks (required-input presence, autogrow-required, and
        # below_min/above_max ranges) to that reachable set, so a
        # disconnected/incomplete node the server would silently drop isn't
        # hard-rejected here. Edge/shape/enum checks are left as-is (pre-existing
        # behavior, out of scope for this change).
        reachable = _output_reachable_node_ids(workflow, self)

        for node_id, node_data in workflow.items():
            # `_meta` is the compose/run provenance block (schema/blueprint/items),
            # stripped before submit — not a node and not a mistake. `comfy workflow compose`
            # adds it itself, so warning here is self-inflicted noise.
            if node_id == "_meta":
                continue
            if not isinstance(node_data, dict):
                warnings.append(
                    {
                        "node_id": node_id,
                        "field": node_id,
                        "code": "non_node_key",
                        "message": f"key {node_id!r} is not a workflow node (expected a dict with class_type)",
                    }
                )
                continue
            class_type = node_data.get("class_type", "")
            # A non-string class_type (list/dict from malformed JSON) is unhashable
            # and would crash the self._nodes.get(class_type) lookup below; treat it
            # as absent so it flows to the structured non_node_key path instead.
            if not isinstance(class_type, str):
                class_type = ""
            if not class_type:
                warnings.append(
                    {
                        "node_id": node_id,
                        "field": node_id,
                        "code": "non_node_key",
                        "message": f"key {node_id!r} has no class_type and will be ignored by the server",
                    }
                )
                continue

            m = self._nodes.get(class_type)
            if m is None:
                close = difflib.get_close_matches(class_type, all_names, n=3, cutoff=0.6)
                errors.append(
                    {
                        "node_id": node_id,
                        "code": "unknown_class_type",
                        "message": f"class_type {class_type!r} not found in object_info",
                        "hint": f"did you mean: {', '.join(close)}?"
                        if close
                        else "run `comfy nodes search <name>` to find available classes",
                        "suggestions": close,
                    }
                )
                continue

            if m.is_output_node:
                has_output_node = True

            port_by_name = {p.name: p for p in m.inputs}
            # V3 autogrow inputs are declared once (e.g. `images`) but wired as
            # slot keys (`images.image0`, `images.image1`, …). Track which
            # autogrow ports actually received a slot so the required-but-empty
            # case surfaces here instead of as a cryptic server reject.
            autogrow_ports = {p.name: p for p in m.inputs if p.is_autogrow}
            autogrow_seen: set[str] = set()
            node_inputs = node_data.get("inputs")
            # A truthy non-dict `inputs` (e.g. a string/list from malformed JSON)
            # sails through `or {}` and crashes `.items()`; treat it as empty so
            # required-input checks flag the absence instead of raising.
            if not isinstance(node_inputs, dict):
                node_inputs = {}
            # Dynamic combos are checked up front so the generic loop below can
            # exempt STALE dynamic sub-keys (left over from a previous
            # selection): the server ignores unknown sub-keys, so a hard edge
            # check on one would false-error; the unknown_input warning from
            # _check_dynamic_combos already covers it. Sub-keys under an
            # unresolved selection keep the old generic checks.
            dyn_port_names = {p.name for p in m.inputs if p.is_dynamic_combo}
            dyn_errors, dyn_warnings, dyn_valid_keys, dyn_unresolved = _check_dynamic_combos(
                node_id, class_type, m, node_inputs
            )
            for input_name, value in node_inputs.items():
                if (
                    "." in input_name
                    and input_name.split(".", 1)[0] in dyn_port_names
                    and input_name not in dyn_valid_keys
                    and not any(input_name.startswith(prefix) for prefix in dyn_unresolved)
                ):
                    continue
                if autogrow_ports and "." in input_name:
                    base = input_name.split(".", 1)[0]
                    if base in autogrow_ports:
                        autogrow_seen.add(base)
                if input_name in autogrow_ports and isinstance(value, list) and len(value) == 2:
                    port = autogrow_ports[input_name]
                    errors.append(
                        {
                            "node_id": node_id,
                            "field": input_name,
                            "code": "autogrow_bare_input",
                            "message": (
                                f"input {input_name!r} is an autogrow input ({port.type}) and cannot be "
                                f"wired as a single connection — the server expects one slot key per "
                                f"connection"
                            ),
                            "hint": f"wire one key per connection: {port.autogrow_slot_example()} "
                            f'(e.g. "{input_name}.{input_name[:-1] if input_name.endswith("s") else input_name}0": '
                            f"[{value[0]!r}, {value[1]!r}])",
                        }
                    )
                    continue
                # Link references: [source_node_id, output_index]
                if isinstance(value, list) and len(value) == 2:
                    src_id = str(value[0])
                    out_idx = value[1] if isinstance(value[1], int) else None

                    # (i) source node exists in workflow
                    src_data = workflow.get(src_id)
                    if not isinstance(src_data, dict) or not src_data.get("class_type"):
                        errors.append(
                            {
                                "node_id": node_id,
                                "field": input_name,
                                "code": "dangling_edge",
                                "message": f"input {input_name!r} references node {src_id!r} which does not exist",
                                "hint": f"add node {src_id!r} to the workflow, or rewire this input to an existing node",
                            }
                        )
                        continue

                    src_class = src_data["class_type"]
                    # Route through the guarded lookup: a referenced node with an
                    # unhashable class_type (malformed JSON) would otherwise crash
                    # the dict.get here.
                    src_m = self.node(src_class)
                    if src_m is None:
                        # Source class_type already flagged by the outer loop
                        continue

                    # (ii) output index in range
                    if out_idx is None or out_idx < 0 or out_idx >= len(src_m.outputs):
                        valid_indices = ", ".join(f"[{i}]={p.type}" for i, p in enumerate(src_m.outputs))
                        errors.append(
                            {
                                "node_id": node_id,
                                "field": input_name,
                                "code": "output_index_out_of_range",
                                "message": (
                                    f"input {input_name!r} references {src_class}[{value[1]}] "
                                    f"but {src_class} has {len(src_m.outputs)} output(s)"
                                ),
                                "hint": f"valid indices for {src_class}: {valid_indices}",
                            }
                        )
                        continue

                    # (iii) type compatibility — advisory only.
                    # ComfyUI allows cross-type wiring via reroutes, converters,
                    # and wildcard ports; the server is the authoritative validator.
                    port = port_by_name.get(input_name)
                    if port is not None:
                        src_type = src_m.outputs[out_idx].type
                        dst_type = port.type
                        if not _edge_types_compatible(src_type, dst_type):
                            # Find the correct index for the expected type
                            correct = [
                                f"[{i}]"
                                for i, p in enumerate(src_m.outputs)
                                if _edge_types_compatible(p.type, dst_type)
                            ]
                            hint = (
                                f"use {src_class}{correct[0]} instead"
                                if correct
                                else f"run `comfy nodes ls --produces {dst_type}` to find a source"
                            )
                            warnings.append(
                                {
                                    "node_id": node_id,
                                    "field": input_name,
                                    "code": "edge_type_mismatch",
                                    "message": (
                                        f"input {input_name!r} expects {dst_type} but "
                                        f"{src_class}[{out_idx}] produces {src_type}"
                                    ),
                                    "hint": hint,
                                }
                            )
                    continue

                port = port_by_name.get(input_name)
                if port is None:
                    continue
                # Shape check (hard error)
                shape_err = port.validate_shape(value)
                if shape_err:
                    errors.append(
                        {
                            "node_id": node_id,
                            "field": input_name,
                            "code": "shape_mismatch",
                            "message": shape_err,
                            "hint": f"expected {port.type}; check the value type",
                        }
                    )
                    continue
                # Catalog checks (enum membership, etc.). Range violations are a
                # hard reject only on a node the server will actually run; on a
                # pruned (output-unreachable) node they stay advisory warnings.
                cat_errors, cat_warnings = _validate_catalog_value(
                    node_id, class_type, input_name, port, value, range_is_error=node_id in reachable
                )
                errors.extend(cat_errors)
                warnings.extend(cat_warnings)

            # dyn_warnings are advisory (stray-dotted-key / catalog-membership
            # notices), the same class as the ungated shape/catalog checks the
            # generic per-input loop above already runs for every node
            # regardless of reachability — so they apply here too. This also
            # restores the diagnostic the stale-sub-key exemption above (which
            # itself is NOT reachability-gated, since it needs to run for
            # every node) removes from the generic loop: without it, a stale
            # dangling link on an unreachable node's dynamic sub-key would
            # silently lose its dangling_edge error with nothing standing in
            # for it.
            warnings.extend(dyn_warnings)
            # Required-presence checks apply only to output-reachable nodes: the
            # server prunes unreachable nodes without validating them, so
            # enforcing required inputs on a disconnected node over-rejects a
            # prompt the server would run. dyn_errors carries the same class
            # of required-presence-type hard errors (missing/unknown
            # dynamic-combo selection), so it's gated alongside them.
            if node_id in reachable:
                errors.extend(_check_autogrow_required(node_id, autogrow_ports, autogrow_seen, node_data))
                errors.extend(_check_required_present(node_id, m, node_data))
                errors.extend(dyn_errors)

        # No-outputs check: the server rejects any prompt with zero output
        # nodes (execution.py:1155-1162, prompt_no_outputs) — including an
        # empty/node-less prompt. Suppress it only when an unknown_class_type
        # error is present: that node could be the real (custom) output node we
        # just can't see, so the missing-output message would be misleading
        # noise on top of the unknown-class error the user must resolve first.
        has_unknown_class = any(e.get("code") == "unknown_class_type" for e in errors)
        if not has_output_node and not has_unknown_class:
            errors.append(
                {
                    # workflow-level error: no owning node, hence None (keeps the
                    # node_id/field keys every other error carries, for schema
                    # consistency).
                    "node_id": None,
                    "field": None,
                    "code": "prompt_no_outputs",
                    "message": "workflow has no output nodes — the server will reject it (prompt_no_outputs)",
                    "hint": "add an output node such as SaveImage/PreviewImage",
                }
            )

        # A node the server will silently PRUNE (not reachable from any output)
        # is almost always a wiring mistake: the author added it and forgot to
        # route its result onward. Because pruned nodes are skipped by every
        # promoted check above, such a graph could validate as
        # "0 errors, 0 warnings" while doing nothing the author intended.
        #
        # Observed in prod: a depth-ControlNet whose output was never wired into
        # the sampler validated completely clean; the graph then ran twice,
        # producing an image with no pose applied, and cost two paid GPU runs and
        # three turns of "it does nothing" before the dangling link was found.
        #
        # Advisory, not an error: a scratch node parked mid-build is legitimate,
        # and the server does run the graph. It only has to be VISIBLE.
        if has_output_node:
            for node_id, node_data in workflow.items():
                if node_id == "_meta" or node_id in reachable:
                    continue
                if not isinstance(node_data, dict):
                    continue
                class_type = node_data.get("class_type")
                m = self._nodes.get(class_type) if class_type else None
                # Note-style nodes legitimately feed nothing.
                if m is not None and not m.outputs:
                    continue
                warnings.append(
                    {
                        "node_id": node_id,
                        "field": None,
                        "code": "node_not_reachable_from_output",
                        "message": (
                            f"node {node_id} ({class_type}) feeds no output node — the server prunes it, "
                            f"so it will not run and has no effect on the result"
                        ),
                        "hint": (
                            "wire its output into the chain that reaches a save/preview node, or delete it; "
                            "a node that reaches no output is skipped entirely"
                        ),
                    }
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # -- Workflow slot editing --

    def get_template_schema(self, template_id: str, workflow: dict) -> dict:
        """Extract the slot manifest from a frontend-format workflow."""
        return {"id": template_id, "slots": _extract_frontend_slots(workflow, self)}

    def apply_slots(self, workflow: dict, overrides: dict[str, Any]) -> tuple[dict, list[dict]]:
        """Apply slot overrides. Returns (modified_workflow, warnings)."""
        import copy

        wf = copy.deepcopy(workflow)
        warnings: list[dict] = []
        for addr, value in overrides.items():
            warnings.extend(_apply_one_slot(wf, addr, value, self))
        return wf, warnings

    def expand_variations(self, workflow: dict, variations: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        """Apply N override sets, return N independent workflow copies."""
        results: list[dict] = []
        all_warnings: list[dict] = []
        for overrides in variations:
            modified, warnings = self.apply_slots(workflow, overrides)
            results.append(modified)
            all_warnings.extend(warnings)
        return results, all_warnings

    # -- Source loading (local / cloud / file) --

    @classmethod
    def load(
        cls,
        *,
        mode: str = "local",
        input_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        supported_nodes_yaml: bytes | None = None,
        cloud_disable_yaml: bytes | None = None,
    ) -> Graph:
        """Unified entry point: resolve object_info, build graph, annotate.

        Both local and cloud are the same: fetch ``/object_info`` from the
        resolved target. The only differences are base URL, path prefix,
        and auth headers — all handled by ``_load_from_target()``.

        Resolution order:
          1. ``input_path`` → read from local JSON file
          2. ``mode`` → resolve a Target via the CLI's routing chain,
             fetch ``/object_info`` over HTTP (local or cloud).

        Security: loopback-only guard for local, HTTPS-only for cloud,
        bounded read (64 MB), no-redirect policy.
        """
        if input_path is not None:
            raw = _load_from_file(input_path)
        else:
            raw = _load_from_target(mode=mode, host=host, port=port)

        g = cls.from_object_info(raw)
        if supported_nodes_yaml or cloud_disable_yaml:
            g.annotate(supported_nodes_yaml, cloud_disable_yaml)
        else:
            # ``--input <dump>`` is the offline path: the caller handed us a
            # local file precisely so nothing goes over the wire. An incidental
            # annotation lookup must not be the one thing that reaches out.
            g._try_default_annotations(allow_network=input_path is None)
        return g

    def _try_default_annotations(self, *, allow_network: bool = True) -> None:
        """Load node annotation data from Comfy-Org/comfy-complete.

        Resolves via :mod:`comfy_cli.cql.annotations_source`, which prefers a
        TTL-fresh local cache, falls back to a live fetch from the public repo
        (bounded, negative-cached, and skipped entirely when ``allow_network``
        is false), and finally to the package-bundled snapshot — so the data
        stays fresh without a ``pip install -U`` while remaining offline-safe.

        The annotations enrich every node with:
          - pack membership (which custom-node pack it belongs to)
          - behavioral labels (ReadsArbitraryFile, NetworkAccess, etc.)
          - cloud_disabled (whether this node is disabled on cloud)

        Useful for BOTH local and cloud: an agent building a workflow on a
        local server still needs to know which nodes will work on cloud.
        Local-only custom nodes not in comfy-complete simply get no labels
        and cloud_disabled=False (safe default).
        """
        try:
            from comfy_cli.cql import annotations_source

            sup, dis = annotations_source.load_annotation_bytes(allow_network=allow_network)
            if sup or dis:
                self.annotate(sup, dis)
        except Exception:
            pass  # missing data / network is non-fatal

    # -- Serialization helpers for CLI compat --

    def morphism_to_dict(self, m: Morphism) -> dict[str, Any]:
        return {
            "id": m.id,
            "name": m.id,
            "display_name": m.display_name,
            "description": m.description,
            "category": m.category,
            "output_types": m.output_types(),
            "output_node": m.is_output_node,
            "is_api_node": m.is_api_node,
            "deprecated": m.deprecated,
            "pack": m.pack,
            "labels": m.labels,
            "cloud_disabled": m.cloud_disabled,
            "inputs": [_input_payload(p, 0) for p in m.inputs],
            "outputs": [{"name": p.name, "type": p.type} for p in m.outputs],
        }


def _widget_default(p: Port) -> Any:
    """The value a fresh node carries for widget port ``p`` — a dynamic
    combo's first key, else the schema default, else the first choice, else
    the DOM-widget placeholder, else ``None``."""
    if p.dynamic_options:
        return p.enum_values[0] if p.enum_values else None
    if p.options.default is not None:
        return p.options.default
    if p.enum_values:
        return p.enum_values[0]
    if p.type in _FRONTEND_DOM_WIDGET_TYPES:
        return _FRONTEND_DOM_WIDGET_DEFAULTS.get(p.type)
    return None


def _input_payload(p: Port, depth: int) -> dict[str, Any]:
    """One ``nodes show`` input entry. A dynamic combo lists every option
    with that option's sub-inputs (dotted, recursively); an autogrow input —
    top-level or nested — names its element type, its slot vocabulary and the
    first keys to wire, so an agent can address ``model.images.image_1``
    without first failing a connect to learn the name."""
    entry: dict[str, Any] = {
        "name": p.name,
        "type": p.type,
        "required": p.required,
        "is_link": p.is_link,
        "section": "required" if p.required else "optional",
        # V3 dynamic combos keep their selection keys in enum_values
        # internally (e.g. so widget_defaults can pick the first key), but
        # those keys are exposed to callers as `selection_keys` /
        # `dynamic_options[].key` below, not as flat `choices` — they are not
        # membership choices for the field (see _is_scalar_choice).
        "choices": [] if p.is_dynamic_combo else p.enum_values,
        "options": {
            "min": p.options.min,
            "max": p.options.max,
            "step": p.options.step,
            "default": p.options.default,
        },
    }
    if p.is_autogrow:
        lo, hi = p.autogrow_limits
        template = p.autogrow_element_template or {}
        entry["autogrow"] = True
        entry["element_type"] = p.autogrow_element_type
        entry["slots"] = {**template, "min": lo, "max": hi}
        entry["wire_as"] = p.autogrow_slot_example()
    if p.is_dynamic_combo:
        entry["selection_keys"] = list(p.enum_values)
    if p.dynamic_options and _is_dynamic_combo_type(p.type) and depth < _MAX_DYNAMIC_COMBO_DEPTH:
        entry["dynamic_options"] = [
            {
                "key": key,
                "inputs": [
                    _input_payload(sub, depth + 1) for sub in _dynamic_combo_sub_ports(p.dynamic_options, key, p.name)
                ],
            }
            for key in p.enum_values
        ]
    return entry


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
#
# Context-independent checks factored out of Graph.validate_workflow so the
# driver loop reads as the connection/class_type walk it is. Each returns the
# error/warning dicts for the caller to append — no shared state is threaded.


def _output_reachable_node_ids(workflow: dict[str, Any], graph: Graph) -> set[str]:
    """Node ids the server would actually validate: output nodes and their
    transitive input ancestors.

    Mirrors ComfyUI's execution.py::validate_prompt, which validates only the
    output nodes (``OUTPUT_NODE``) and everything reachable by walking their
    input links backward — any node not reachable from an output is pruned and
    never validated. We reproduce that reachable set so the promoted hard checks
    (required_input_missing, autogrow_no_slots, below_min/above_max) don't
    reject a disconnected node the server would silently drop.

    An input value shaped ``[source_node_id, output_index]`` is a link edge (the
    same predicate the per-input link walk uses); we follow those edges backward
    from every output node. A reference to a node absent from the workflow (a
    dangling edge, flagged separately) simply isn't traversed.
    """
    reachable: set[str] = set()
    stack: list[str] = []
    for node_id, node_data in workflow.items():
        if node_id == "_meta" or not isinstance(node_data, dict):
            continue
        m = graph.node(node_data.get("class_type", ""))
        if m is not None and m.is_output_node and node_id not in reachable:
            reachable.add(node_id)
            stack.append(node_id)
    while stack:
        node_data = workflow.get(stack.pop())
        if not isinstance(node_data, dict):
            continue
        node_inputs = node_data.get("inputs")
        # Guard against a truthy non-dict `inputs` from malformed JSON, which
        # would slip past `or {}` and raise AttributeError on `.values()`.
        if not isinstance(node_inputs, dict):
            continue
        for value in node_inputs.values():
            if isinstance(value, list) and len(value) == 2:
                src_id = str(value[0])
                if src_id in workflow and src_id not in reachable:
                    reachable.add(src_id)
                    stack.append(src_id)
    return reachable


def _validate_catalog_value(
    node_id: str, class_type: str, input_name: str, port: Port, value: Any, *, range_is_error: bool = True
) -> tuple[list[dict], list[dict]]:
    """Enum-membership and other catalog checks for one scalar input value.

    Returns (errors, warnings): unknown-enum and out-of-range values are hard
    errors (the server rejects them); every other catalog finding is a
    namespaced warning.

    ``range_is_error`` gates the below_min/above_max promotion: the
    server only range-checks nodes it actually runs, so on a pruned
    (output-unreachable) node these are demoted back to advisory warnings.
    """
    errors: list[dict] = []
    warnings: list[dict] = []
    for w in port.validate_catalog(value):
        if w["code"] == "unknown_enum_value":
            top = port.enum_values[:8]
            errors.append(
                {
                    "node_id": node_id,
                    "field": input_name,
                    "code": "unknown_enum_value",
                    "message": w["message"],
                    "hint": f"valid options include: {', '.join(str(v) for v in top)}"
                    + (
                        f" (and {len(port.enum_values) - 8} more — see valid_options)"
                        if len(port.enum_values) > 8
                        else ""
                    ),
                    "suggestions": port.enum_values[:20],
                    # full, typed list — never truncated, so the agent
                    # can pick a real value instead of guessing.
                    "valid_options": list(port.enum_values),
                }
            )
        elif w["code"] == "no_options_available":
            # Declared-but-empty option list: the server has nothing installed
            # for this field, so it rejects EVERY value (value_not_in_list
            # against an empty list — execution.py:1035-1067). Same hard-error
            # tier as unknown_enum_value, and for the same reason; the only
            # difference is that there is no option to suggest. This is the
            # fresh-install case the membership check used to skip in silence:
            # UNETLoader/CLIPLoader ship no built-in options, so on a bare
            # install the two largest missing downloads went unreported while a
            # half-populated install got warned.
            errors.append(
                {
                    "node_id": node_id,
                    "field": input_name,
                    "code": "no_options_available",
                    "message": w["message"],
                    "hint": (
                        f"the server has no files installed for {input_name!r} on {class_type} — "
                        f"install it (e.g. `comfy model download`) or point this input at an installed file"
                    ),
                    "suggestions": [],
                    "valid_options": [],
                }
            )
        elif w["code"] in ("below_min", "above_max"):
            # The server hard-rejects out-of-range values
            # (value_smaller_than_min / value_bigger_than_max, execution.py:1008-1033),
            # so promote these to errors — same as unknown_enum_value above. One
            # theoretical over-strictness: the server skips its built-in range
            # checks for args covered by a node's custom VALIDATE_INPUTS
            # (execution.py:1007); we accept that trade-off exactly as the
            # unknown_enum_value hard error does.
            bound = port.options.min if w["code"] == "below_min" else port.options.max
            op = ">=" if w["code"] == "below_min" else "<="
            entry = {
                "node_id": node_id,
                "field": input_name,
                "code": w["code"],
                "message": w["message"],
                "hint": f"use a value {op} {bound}",
            }
            # Only a hard reject on a node the server will run; on a pruned node
            # it stays advisory (matching pre-promotion behavior for that node).
            if range_is_error:
                errors.append(entry)
            else:
                # Demoted to an advisory warning: match the fully-qualified
                # `field` schema every other warning uses (preflight renders
                # w["field"]), rather than leaking the bare input_name.
                entry["field"] = f"{node_id}.{class_type}.{input_name}"
                warnings.append(entry)
        else:
            w["field"] = f"{node_id}.{class_type}.{w['field']}"
            warnings.append(w)
    return errors, warnings


def _check_required_present(node_id: str, m: Morphism, node_data: dict) -> list[dict]:
    """Required inputs that are absent from ``node_data["inputs"]`` entirely.

    Mirrors the server (execution.py:884-900): any required input the frontend
    didn't serialize is a hard reject (required_input_missing). The per-input
    loop only inspects keys that ARE present, so this catches the absent ones.
    Skipped, because each is handled by its own path (and would otherwise
    double-error): autogrow ports (``_check_autogrow_required``), dynamic combos
    (``_check_dynamic_combos``, which reports the absent selector itself so it can
    also name the valid options), and COMFY_DYNAMICSLOT (always optional
    server-side anyway). The server has NO exemption for required inputs that
    carry a default — the frontend always serializes widget values, so absence
    is a genuine authoring error; we do not skip ports with defaults.
    """
    present = node_data.get("inputs") or {}
    errors: list[dict] = []
    for port in m.inputs:
        if not port.required or port.is_autogrow:
            continue
        if port.is_dynamic_combo or port.type.startswith("COMFY_DYNAMICSLOT"):
            continue
        if port.name in present:
            continue
        errors.append(
            {
                "node_id": node_id,
                "field": port.name,
                "code": "required_input_missing",
                "message": (
                    f"required input {port.name!r} is missing — "
                    f"the server will reject this node (required_input_missing)"
                ),
                "hint": f"add {port.name!r} to inputs"
                + (
                    f" (e.g. a {port.type} value)"
                    if not port.is_link
                    else f" (wire a {port.type} link: [<node_id>, <output_index>])"
                ),
            }
        )
    return errors


# A dynamic combo's option may declare another dynamic combo among its
# sub-inputs. Mirrors ``workflow_to_api._MAX_DYNAMIC_COMBO_DEPTH`` so validation
# walks the same bounded tree the converter expanded.
_MAX_DYNAMIC_COMBO_DEPTH = 16


def _dynamic_combo_options(spec: Any) -> list[dict]:
    """Every option block a dynamic-combo spec declares, in schema order.

    Shape: ``["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": …, "inputs":
    {"required": {…}, "optional": {…}}}, …]}]``. Non-dict entries are dropped
    rather than raising — object_info is server-supplied and we never want a
    malformed option block to take down validation of the whole workflow.
    (Counterpart of ``workflow_to_api._dynamic_combo_selected_subs``, which
    resolves the same structure for the conversion side.)
    """
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        return []
    options_meta = spec[1] if isinstance(spec[1], dict) else {}
    return [o for o in (options_meta.get("options") or []) if isinstance(o, dict)]


def _check_dynamic_combos(
    node_id: str, class_type: str, m: Morphism, node_inputs: dict
) -> tuple[list[dict], list[dict], set[str], set[str]]:
    """Validate every dynamic-combo input against the option its selector names.

    Mirrors the server: ``DynamicCombo._expand_schema_for_dynamic``
    (comfy_api/latest/_io.py) reads the submitted selector, finds the option
    whose ``key`` equals it, and folds THAT option's inputs — and only that
    one's — into the node's finalized required/optional sets under dotted names
    (``model.width``). ``validate_inputs`` (execution.py:886-900) then walks
    those finalized names, so a missing one is a real ``required_input_missing``
    and a present one gets the same type/range/enum checks as any input.

    object_info never declares those dotted keys, so the driver loop skips them
    (``port_by_name`` misses) and ``_check_required_present`` exempts the parent
    selector. Without this walk a workflow that omits or mistypes a sub-input
    validates clean and is then rejected at ``/prompt`` — validation giving
    false confidence right before a paid run.

    Also flags present dotted keys that match no sub-input of the resolved
    selection — a warning, not an error, because the server ignores extra
    keys; that pass lives in ``_unknown_dotted_key_warnings``. Returns ``(errors, warnings, valid_keys, unresolved)`` — the caller
    uses ``valid_keys``/``unresolved`` to exempt stale (server-ignored)
    dynamic sub-keys from the generic edge checks, which would otherwise
    hard-error on e.g. a dangling link left over from a previous selection.
    """
    errors: list[dict] = []
    warnings: list[dict] = []
    present = node_inputs
    dynamic_ports = [p for p in m.inputs if p.is_dynamic_combo]
    if not dynamic_ports:
        return errors, warnings, set(), set()

    valid_keys: set[str] = set()
    unresolved: set[str] = set()
    resolved: dict[str, Any] = {}
    for port in dynamic_ports:
        e, w, v, u = _check_dynamic_combo_input(
            node_id, class_type, port.name, port.raw_spec, port.required, present, resolved
        )
        errors.extend(e)
        warnings.extend(w)
        valid_keys |= v
        unresolved |= u

    warnings.extend(_unknown_dotted_key_warnings(node_id, present, dynamic_ports, valid_keys, unresolved, resolved))
    return errors, warnings, valid_keys, unresolved


def _unknown_dotted_key_warnings(
    node_id: str,
    present: dict,
    dynamic_ports: list[Port],
    valid_keys: set[str],
    unresolved: set[str],
    resolved: dict[str, Any],
) -> list[dict]:
    """Dotted keys on the node that no resolved selection accepts.

    Split out of ``_check_dynamic_combos`` so its two distinct jobs — driving
    per-port validation, and judging the dotted keys left over afterwards —
    stay separately readable and testable.

    Warnings only, never errors: the server silently drops keys outside the
    expanded schema, so the run still proceeds — what the user loses is the
    stale value, which is worth saying out loud but is not a failure.

    Under an unresolved prefix (a required selector that's absent, or one
    that's invalid or link-valued) the sub-keys can't be judged at all — the
    primary error already covers it, so don't pile on.
    """
    warnings: list[dict] = []
    port_specs = {p.name: p.raw_spec for p in dynamic_ports}
    for key in present:
        if "." not in key or key in valid_keys:
            continue
        base = key.split(".", 1)[0]
        if base not in port_specs:
            continue
        if any(key.startswith(prefix) for prefix in unresolved):
            continue
        # Attribute the stray key to the DEEPEST resolved combo prefix, so a
        # stray `model.mode.bogus` under a resolved `model.mode` names
        # `model.mode`'s selection (and lists ITS sub-keys), not `model`'s.
        anchor = max((n for n in resolved if key.startswith(f"{n}.")), key=len, default=base)
        if anchor not in resolved and anchor not in present:
            # An OPTIONAL selector nobody set — the one path that reaches here
            # with no `resolved` entry (a required one exits as an error with
            # its prefix unresolved, and is skipped above). The schema never
            # expanded, so say exactly that: rendering the missing selection
            # as `anchor=None` reads as "you set it to null" and sends the
            # reader hunting for a value they never wrote.
            opts = [str(k) for o in _dynamic_combo_options(port_specs.get(anchor)) if (k := o.get("key")) is not None]
            message = f"input {key!r} sits under {anchor!r}, which is not set — the server will ignore it"
            hint = (
                f"set {anchor!r} to the option declaring this sub-input: "
                + ", ".join(opts[:8])
                + (f" (and {len(opts) - 8} more)" if len(opts) > 8 else "")
                if opts
                else f"set {anchor!r} for this sub-input to apply"
            )
        else:
            selection = resolved.get(anchor, present.get(anchor))
            known = sorted(k for k in valid_keys if k.startswith(f"{anchor}."))
            # Same truncation as the sibling required/unknown-enum hints — an
            # autogrow sub-input's slot count is workflow-specific and can run
            # long, so don't dump the whole list into one line.
            known_hint = ", ".join(known[:8]) + (f" (and {len(known) - 8} more)" if len(known) > 8 else "")
            message = f"input {key!r} matches no sub-input of {anchor}={selection!r} — the server will ignore it"
            hint = (
                f"valid sub-keys for this selection: {known_hint}"
                if known
                else f"selection {selection!r} takes no sub-inputs"
            )
        warnings.append(
            {
                "node_id": node_id,
                "field": key,
                "code": "unknown_input",
                "message": message,
                "hint": hint,
            }
        )
    return warnings


def _check_dynamic_combo_input(
    node_id: str,
    class_type: str,
    name: str,
    spec: Any,
    required: bool,
    present: dict,
    resolved: dict[str, Any],
    depth: int = 0,
) -> tuple[list[dict], list[dict], set[str], set[str]]:
    """One dynamic-combo input: resolve its selected option, check its sub-inputs.

    Returns ``(errors, warnings, valid_keys, unresolved)`` — ``valid_keys`` is
    every dotted key this (and any nested, resolved) selection accepts;
    ``unresolved`` is the set of ``"<name>."`` prefixes whose sub-keys can't
    be judged, so the caller skips unknown-key warnings there. ``resolved``
    (mutated) records ``name -> selected key`` for every combo level that DID
    resolve, so the caller can attribute a stray key to the deepest resolved
    prefix.
    """
    errors: list[dict] = []
    warnings: list[dict] = []
    options = _dynamic_combo_options(spec)
    # Only a scalar `key` is a real membership choice: an option dict missing
    # `key` would otherwise contribute `None` into `valid_options`/`suggestions`
    # (agent-facing "pick one of these" lists), which is worse than omitting it —
    # an agent could try to set the field to `null`.
    keys = [k for o in options if (k := o.get("key")) is not None]

    if name not in present:
        # ``_check_required_present`` deliberately exempts the dynamic types
        # ("handled by its own path") — this is that path, so the absent
        # selector is reported here and nowhere else (no double-error).
        #
        # NOTE the failure is *later* than a plain missing input. With no
        # selector the server's schema expansion is a no-op
        # (``DynamicCombo._expand_schema_for_dynamic`` returns early), so the
        # selector never enters ``valid_inputs`` and ``/prompt`` does NOT emit
        # ``required_input_missing`` for it — the node's inputs are dropped and
        # it fails at EXECUTION instead, i.e. after a paid node has been
        # entered. Still a hard error here: the workflow cannot run.
        if required:
            # `keys` can be empty here too (an unreadable options container),
            # in which case "set X to one of its options: " with nothing after
            # the colon is self-refuting — say plainly that the schema
            # couldn't be read instead of dangling an empty enumeration.
            hint = (
                f"set {name!r} to one of its options: "
                + ", ".join(str(k) for k in keys[:8])
                + (f" (and {len(keys) - 8} more — see valid_options)" if len(keys) > 8 else "")
                if keys
                else f"{name!r} is required, but its option schema didn't parse — check object_info for this node"
            )
            errors.append(
                {
                    "node_id": node_id,
                    "field": name,
                    "code": "required_input_missing",
                    "message": (
                        f"required dynamic-combo input {name!r} is missing — the server cannot resolve "
                        f"which option's sub-inputs apply, so this node fails at execution"
                    ),
                    "hint": hint,
                    "suggestions": keys[:20],
                    "valid_options": keys,
                }
            )
            return errors, warnings, set(), {f"{name}."}
        # Optional and absent: no error at all for the selector itself — this
        # is a legitimate "not using this option" state, not a problem. Unlike
        # the required-absent case above (whose own error already explains
        # why any of its dotted sub-keys are meaningless) or a link-valued
        # selector (whose expansion is genuinely unknown until execution),
        # here the schema definitely did NOT expand and nothing else says so
        # — so a stray dotted sub-key needs the same treatment as one under a
        # resolved-but-unmatched selection: no false dangling_edge, but still
        # an informational unknown_input warning rather than dead silence.
        return errors, warnings, set(), set()

    if not keys:
        # No option schema parsed (absent/unreadable `options`, or every entry
        # was missing `key`) — we have no basis to judge the selection, so stay
        # lenient rather than hard-erroring on a schema we can't read. Matches
        # the "only strict where the option schema genuinely parsed" rule:
        # unparseable individual entries already degrade this way; an
        # unparseable options container must too (forward-compat with a newer
        # ComfyUI option shape this parser doesn't recognize yet).
        return errors, warnings, set(), {f"{name}."}

    selected = present[name]
    if isinstance(selected, list) and len(selected) == 2:
        # Wired as a link: which option expands is only known at execution time,
        # so there is no static sub-input set to check. The edge itself is
        # already validated by the driver loop.
        return errors, warnings, set(), {f"{name}."}

    option = next((o for o in options if o.get("key") == selected), None)
    if option is None:
        # Strict ``==`` on the key — the same test the server
        # (``DynamicCombo._expand_schema_for_dynamic``) and the converter both
        # apply, so all three agree on which option expands.
        #
        # Same late failure as the absent selector above: an unmatched key
        # expands to nothing, so the server drops this node's inputs rather
        # than rejecting the prompt, and the run dies at execution. The
        # converter ALSO failed to expand it, silently misaligning every
        # following widget value — surfacing that is the whole point.
        errors.append(
            {
                "node_id": node_id,
                "field": name,
                "code": "unknown_enum_value",
                "message": (
                    f"{selected!r} not in {len(keys)} known options for {name} — its sub-inputs "
                    f"cannot be resolved, so this node fails at execution"
                ),
                "hint": f"valid options: {', '.join(str(k) for k in keys[:8])}"
                + (f" (and {len(keys) - 8} more — see valid_options)" if len(keys) > 8 else ""),
                "suggestions": keys[:20],
                "valid_options": keys,
            }
        )
        return errors, warnings, set(), {f"{name}."}

    resolved[name] = selected
    if depth >= _MAX_DYNAMIC_COMBO_DEPTH:
        # Pathological nesting — the converter stops here too. The selection
        # itself DID resolve, but we can't enumerate its sub-inputs, so — same
        # as every other "resolved but can't judge" exit — mark the prefix
        # unresolved rather than leaving real sub-keys exposed to a false
        # unknown_input warning.
        return errors, warnings, set(), {f"{name}."}

    valid_keys: set[str] = set()
    nested_unresolved: set[str] = set()
    sub_def = option.get("inputs")
    if not isinstance(sub_def, dict):
        # The resolved option carries no readable sub-input schema — same
        # "resolved but can't judge" case as the depth cap above.
        return errors, warnings, valid_keys, {f"{name}."}
    for section, sub_required in (("required", True), ("optional", False)):
        section_def = sub_def.get(section)
        if not isinstance(section_def, dict):
            continue
        for sub_name, sub_spec in section_def.items():
            dotted = f"{name}.{sub_name}"
            valid_keys.add(dotted)
            e, w, v, u = _check_dynamic_combo_sub(
                node_id, class_type, dotted, sub_spec, sub_required, present, resolved, depth
            )
            errors.extend(e)
            warnings.extend(w)
            valid_keys |= v
            nested_unresolved |= u
    return errors, warnings, valid_keys, nested_unresolved


def _check_dynamic_combo_sub(
    node_id: str,
    class_type: str,
    dotted: str,
    sub_spec: Any,
    sub_required: bool,
    present: dict,
    resolved: dict[str, Any],
    depth: int,
) -> tuple[list[dict], list[dict], set[str], set[str]]:
    """Presence + shape + catalog checks for one expanded sub-input.

    The sub-input spec is a plain ``INPUT_TYPES`` entry, so it goes through the
    same ``_parse_input_spec`` / :class:`Port` machinery as a top-level input and
    inherits identical shape and enum/range semantics.
    """
    port = _port_from_spec(dotted, sub_spec, sub_required)

    if port.is_dynamic_combo:
        # Nested dynamic combo: its own selector/presence rules apply one level down.
        return _check_dynamic_combo_input(
            node_id, class_type, dotted, sub_spec, sub_required, present, resolved, depth + 1
        )

    if port.is_autogrow:
        # An autogrow sub-input wires as `<dotted>.<slot>` keys and routinely
        # declares `min: 0` even inside the `required` section (Seedream's
        # `model.images`), so absence is NOT a server reject — the converter
        # emits no key at all for a zero-slot autogrow. Nothing to presence- or
        # shape-check here. Any slot keys actually present are accepted
        # wholesale (not counted, not edge-checked here) so they don't
        # surface as unknown_input noise; the generic driver loop still
        # edge-checks whichever slot keys ARE present.
        #
        # A bare `dotted: [src, idx]` link, though, is the exact same mistake
        # the top-level autogrow_bare_input check catches — the server expects
        # one slot key per connection, not the base name wired directly. The
        # top-level driver loop never sees this dotted key (it's not in
        # `autogrow_ports`, which only tracks top-level names), so it has to
        # be caught here or it silently validates as an ordinary link.
        if isinstance(present.get(dotted), list) and len(present[dotted]) == 2:
            return (
                [
                    {
                        "node_id": node_id,
                        "field": dotted,
                        "code": "autogrow_bare_input",
                        "message": (
                            f"input {dotted!r} is an autogrow input ({port.type}) and cannot be "
                            f"wired as a single connection — the server expects one slot key per "
                            f"connection"
                        ),
                        "hint": f"wire one key per connection: {port.autogrow_slot_example()}",
                    }
                ],
                [],
                set(),
                set(),
            )
        slot_prefix = f"{dotted}."
        return [], [], {k for k in present if k.startswith(slot_prefix)}, set()

    if dotted not in present:
        if not sub_required:
            return [], [], set(), set()
        return (
            [
                {
                    "node_id": node_id,
                    "field": dotted,
                    "code": "required_input_missing",
                    "message": (
                        f"required input {dotted!r} is missing — the server will reject this node (required_input_missing)"
                    ),
                    "hint": f"add {dotted!r} to inputs"
                    + (
                        f" (e.g. a {port.type} value)"
                        if not port.is_link
                        else f" (wire a {port.type} link: [<node_id>, <output_index>])"
                    ),
                }
            ],
            [],
            set(),
            set(),
        )

    value = present[dotted]
    if isinstance(value, list) and len(value) == 2:
        # A wired sub-input — the driver loop already ran the dangling-edge and
        # output-index checks on this same key.
        return [], [], set(), set()

    shape_err = port.validate_shape(value)
    if shape_err:
        return (
            [
                {
                    "node_id": node_id,
                    "field": dotted,
                    "code": "shape_mismatch",
                    "message": shape_err,
                    "hint": f"expected {port.type}; check the value type",
                }
            ],
            [],
            set(),
            set(),
        )

    errs, warns = _validate_catalog_value(node_id, class_type, dotted, port, value)
    return errs, warns, set(), set()


def _check_autogrow_required(
    node_id: str, autogrow_ports: dict[str, Port], autogrow_seen: set[str], node_data: dict
) -> list[dict]:
    """Required autogrow inputs that received no connected slots.

    The server would reject such a node, so surface it here instead of as a
    cryptic downstream reject.
    """
    inputs = node_data.get("inputs") or {}
    errors: list[dict] = []
    for base, port in autogrow_ports.items():
        if port.required and base not in autogrow_seen and base not in inputs:
            errors.append(
                {
                    "node_id": node_id,
                    "field": base,
                    "code": "autogrow_no_slots",
                    "message": (
                        f"required autogrow input {base!r} has no connected slots — the server will reject this node"
                    ),
                    "hint": f"wire one key per connection: {port.autogrow_slot_example()}",
                }
            )
    return errors


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

_MAX_OBJECT_INFO_BYTES = 64 * 1024 * 1024


_opener = build_http_only_opener(NoRedirectHandler())


class LoadError(Exception):
    """Failed to load object_info from a source."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def _load_from_file(path: str) -> dict[str, Any]:
    """Read object_info from a local JSON file."""
    from pathlib import Path

    p = Path(path).expanduser()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise LoadError(f"cannot read object_info: {p}: {e}", details={"path": str(p)}) from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LoadError(f"invalid JSON in {p}: {e}", details={"path": str(p)}) from e


def _load_from_target(*, mode: str = "local", host: str | None = None, port: int | None = None) -> dict[str, Any]:
    """Fetch /object_info from the resolved target — local or cloud.

    Both paths are the same HTTP fetch; only the base URL, path prefix,
    and auth headers differ. The Target abstraction handles all three.

    Security:
      - Local: loopback-only guard (warn on non-loopback)
      - Cloud: HTTPS enforced by Target + auth headers
      - Both: bounded read (64 MB), no-redirect policy
    """
    from comfy_cli.target import resolve_target

    target = resolve_target(where=mode, host=host, port=port)
    url = target.url("object_info")

    # Loopback guard for local targets
    if not target.is_cloud:
        parsed_host = urllib.parse.urlsplit(url).hostname or ""
        if not is_loopback_host(parsed_host):
            raise LoadError(
                f"Refusing to fetch object_info from non-loopback host {parsed_host!r} "
                f"in local mode (potential SSRF). Use --where cloud for remote targets."
            )

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    # Auth headers (cloud only — local has no auth)
    from comfy_cli.http import target_auth_headers

    for k, v in target_auth_headers(target).items():
        req.add_header(k, v)

    try:
        with _opener.open(req, timeout=30.0) as resp:
            raw = resp.read(_MAX_OBJECT_INFO_BYTES + 1)
            if len(raw) > _MAX_OBJECT_INFO_BYTES:
                raise LoadError(
                    f"response exceeds {_MAX_OBJECT_INFO_BYTES} bytes",
                    details={"url": url},
                )
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        hint = "run `comfy cloud login`" if target.is_cloud else "run `comfy launch` first"
        raise LoadError(
            f"HTTP {e.code} from {url}: {body[:200]}",
            details={"url": url, "status": e.code, "hint": hint},
        ) from e
    except urllib.error.URLError as e:
        hint = "run `comfy cloud login`" if target.is_cloud else "run `comfy launch` first"
        raise LoadError(
            f"cannot reach {url}: {e.reason if hasattr(e, 'reason') else e}",
            details={"url": url, "mode": mode, "hint": hint},
        ) from e
    except (json.JSONDecodeError, OSError) as e:
        raise LoadError(f"invalid response from {url}: {e}", details={"url": url}) from e


# ---------------------------------------------------------------------------
# Slot editing helpers — port of nodegraph/frontend.go + runtemplate.go
# ---------------------------------------------------------------------------

# Defends slot recursion against pathological / cyclic subgraph nesting.
_MAX_SUBGRAPH_DEPTH = 32


# Delimiter that separates subgraph-nesting levels in a slot address. The
# final ``.`` separates the (possibly nested) node path from the input name.
# Examples:
#   ``3.seed``          top-level node 3, widget ``seed``
#   ``10/9.prompt``     subgraph instance 10 → interior node 9, widget ``prompt``
#   ``10/3/7.value``    instance 10 → interior subgraph node 3 → interior node 7
# UUID subgraph class_types contain ``-`` but never ``/`` or top-level ``.`` in
# an instance id, so the delimiters stay unambiguous.
_SUBGRAPH_PATH_SEP = "/"


def _subgraph_defs_by_id(workflow: dict) -> dict[str, dict]:
    """Index subgraph definitions so an instance's ``type`` resolves to its def.

    A subgraph *instance* node's ``type`` is normally the UUID ``id`` of its
    definition, so the UUID is the primary key. Real ComfyUI saves can also
    carry several distinct defs sharing the cosmetic ``name`` "New Subgraph";
    keying by name alone would silently map instances onto the wrong def, so id
    always wins. We still register ``name`` as a *fallback* key (only when it
    doesn't shadow an id and isn't ambiguous across defs) to support older
    name-typed templates that predate UUID ids.
    """
    defs = (workflow.get("definitions") or {}).get("subgraphs") or []
    by_id: dict[str, dict] = {}
    name_counts: dict[str, int] = {}
    name_first: dict[str, dict] = {}
    for sg in defs:
        if not isinstance(sg, dict):
            continue
        sg_id = sg.get("id")
        if isinstance(sg_id, str) and sg_id:
            by_id[sg_id] = sg
        name = sg.get("name")
        if isinstance(name, str) and name:
            name_counts[name] = name_counts.get(name, 0) + 1
            name_first.setdefault(name, sg)
    for name, count in name_counts.items():
        if count == 1 and name not in by_id:
            by_id[name] = name_first[name]
    return by_id


def _widgets_as_list(widgets_values: Any) -> list[Any]:
    """Normalize ``widgets_values`` to a list positionally indexable by widget order.

    ComfyUI's own convention is a positional LIST, but some custom nodes —
    VideoHelperSuite's ``VHS_*`` family (e.g. ``VHS_LoadVideo``) — serialize it
    as a NAMED DICT instead: ``{"video": "...", "force_rate": 0, ...}``. Every
    call site in this module indexes ``widgets_values`` by INTEGER position
    against the schema's widget ``order``; a dict is truthy (so a bare
    ``widgets_values or []`` guard doesn't catch it) and indexing it with an int
    raises ``KeyError``, while ``.extend()`` on it raises ``AttributeError``.
    Anything that isn't a list reads as "no positional values known" — mirrors
    ``workflow_to_api.py``'s non-list handling
    (``test_tolerates_non_list_widgets_values``) so the two code paths agree;
    a node whose widgets can't be positionally read shows as unset rather than
    crashing slot extraction or a set-widget write.
    """
    return list(widgets_values) if isinstance(widgets_values, list) else []


def _widgets_as_positional(widgets_values: Any, graph: Graph | None, class_type: str) -> list[Any]:
    """Positional view of ``widgets_values`` that PRESERVES the named-dict form.

    Lists pass through unchanged. The VHS-style dict serialization is projected
    onto the class's default widget order when the catalog knows it — each
    named value lands at its schema position, names the schema doesn't know are
    dropped (they have no positional home), and missing names read as ``None``.
    Without a catalog (or an unknown class) this degrades to
    :func:`_widgets_as_list`'s "no positional values known" behavior.

    Use this wherever a graph is in scope (set-widget, apply/replay, capture,
    slot writes): reading a dict as ``[]`` there meant one write silently
    destroyed every sibling value on the node (``{"width": 768, "height": 512,
    "batch_size": 1}`` + set ``batch_size`` → ``[None, None, 4]``).
    """
    if isinstance(widgets_values, list):
        return list(widgets_values)
    if isinstance(widgets_values, dict) and graph is not None:
        order = graph.widget_order_default(class_type)
        if order:
            return [widgets_values.get(name) for name in order]
    return _widgets_as_list(widgets_values)


# A dynamic combo may nest another dynamic combo among its sub-inputs. Real
# schemas are one or two levels deep; the cap defends the expansion walk
# against a pathological/malicious object_info entry.
_MAX_DYNAMIC_COMBO_DEPTH = 16


@dataclass
class _WidgetEntry:
    """One positional ``widgets_values`` slot in a node's value-aware order.

    ``port`` is ``None`` for a marker slot with no schema port: the
    ``control_after_generate`` seed companion, or a frontend-injected input
    (``frontend_extra_widget_names``). ``frontend_injected`` tells the two
    apart — the companion carries a real user value and is writable; an
    injected ``upload``/``audioUI``/``PREVIEW_3D`` slot is a button, player or
    viewport state with nothing to validate against, and every write surface
    refuses it by name (``frontend_injected_widget_error``). ``owner`` is the
    dotted name of the dynamic combo whose selected option contributed this
    entry (``None`` for top-level inputs) — used to size a combo's sub-span
    when its selector changes.
    """

    name: str
    port: Port | None
    owner: str | None
    frontend_injected: bool = False


def _editable_widget_names(entries: list[_WidgetEntry]) -> list[str]:
    """The names a write surface advertises: every schema-backed slot, in
    positional order — exactly what ``comfy workflow slots`` lists. Marker
    slots are left out: injected ones are refused, and the writable
    ``control_after_generate`` companion is deliberately unadvertised."""
    return [e.name for e in entries if e.port is not None]


def frontend_injected_widget_error(node_type: str, widget: str, available: list[str]) -> ValueError:
    """The refusal every write surface raises for a frontend-injected slot.

    Worded without ``not found`` on purpose: the address resolved, so the
    sibling-suggestion enrichment (``_enrich_resolution_error``) must not fire.
    """
    return ValueError(
        f"widget {widget!r} on {node_type} is frontend-injected (no schema input; "
        f"`comfy workflow slots` never lists it) and is not editable; "
        f"available widgets: {', '.join(available) if available else '(none — all inputs are links)'}"
    )


def _dynamic_combo_sub_ports(dynamic_options: list[dict], selector: Any, prefix: str) -> list[Port]:
    """The selected option's sub-inputs as Ports, dotted under ``prefix``.

    Returns ``[]`` when the selector matches no option or the option block is
    malformed. Connection-only sub-inputs (e.g. ``COMFY_AUTOGROW_V3`` image
    lists) are included with ``is_link=True`` so callers can skip them.
    """
    option = next((o for o in dynamic_options if o.get("key") == selector), None)
    if option is None:
        return []
    sub_def = option.get("inputs")
    if not isinstance(sub_def, dict):
        return []
    ports: list[Port] = []
    for section in ("required", "optional"):
        section_def = sub_def.get(section) or {}
        if not isinstance(section_def, dict):
            continue
        for sub_name, sub_spec in section_def.items():
            ports.append(_port_from_spec(f"{prefix}.{sub_name}", sub_spec, section == "required"))
    return ports


def _expand_widget_entries(
    m: Morphism,
    widgets_values: list[Any],
    *,
    first_key: bool = False,
    sub_links: list[Port] | None = None,
) -> list[_WidgetEntry]:
    """Flatten a node's widget ports into one entry per ``widgets_values`` slot.

    Walks declared ports in order; at a dynamic combo it reads the current
    selector from that combo's own positional slot and expands the matching
    option's widget-like sub-inputs in place (recursing for nested dynamic
    combos; connection sub-inputs contribute no slot). A control-flagged input
    — top-level or sub — is followed by its ``control_after_generate`` marker
    slot, exactly as the frontend serializes it.

    ``first_key=True`` selects every dynamic combo's first option instead of
    reading ``widgets_values`` — the layout of a FRESH node, which is what the
    static catalog (``widget_order_default``) and ``add_node``
    (``widget_defaults``) publish. One walk for both keeps them from ever
    disagreeing with the value-aware order ``set-widget`` indexes by.

    ``sub_links`` collects the connection-only sub-inputs the selected options
    contribute (``COMFY_AUTOGROW_V3`` groups, ``GEMINI_INPUT_FILES``…) — they
    own no slot, but the wiring side needs to know they exist.
    """
    entries: list[_WidgetEntry] = []

    def emit(name: str, port: Port, owner: str | None, depth: int) -> None:
        entries.append(_WidgetEntry(name=name, port=port, owner=owner))
        if port.dynamic_options and _is_dynamic_combo_type(port.type):
            if depth >= _MAX_DYNAMIC_COMBO_DEPTH:
                return
            idx = len(entries) - 1
            if first_key:
                selector = port.enum_values[0] if port.enum_values else None
            else:
                selector = widgets_values[idx] if idx < len(widgets_values) else port.options.default
            for sub in _dynamic_combo_sub_ports(port.dynamic_options, selector, name):
                if sub.is_link:
                    if sub_links is not None:
                        sub_links.append(sub)
                    continue
                emit(sub.name, sub, name, depth + 1)
        elif _has_control_after_generate_slot(port):
            entries.append(_WidgetEntry(name="control_after_generate", port=None, owner=owner))

    for p in m.inputs:
        if p.is_link:
            continue
        emit(p.name, p, None, 0)
    for name in frontend_extra_widget_names(m):
        entries.append(_WidgetEntry(name=name, port=None, owner=None, frontend_injected=True))
    return entries


def _node_widget_slots(node: dict, prefix: str, graph: Graph) -> list[dict]:
    """Surface a regular node's widget inputs as slots under ``prefix``.

    ``prefix`` is the addressable node path (``"3"`` at top level, ``"10/9"``
    inside a subgraph). Returns one slot per widget input the schema knows
    about — including a dynamic combo's selector (enum = its option keys) and
    the selected option's ``<name>.<sub>`` sub-inputs. Returns ``[]`` for
    nodes whose type isn't in object_info.
    """
    node_type = node.get("type", "")
    m = graph.node(node_type)
    if m is None:
        return []
    widgets = _widgets_as_positional(node.get("widgets_values"), graph, node_type)
    slots: list[dict] = []
    for idx, entry in enumerate(_expand_widget_entries(m, widgets)):
        if entry.port is None:  # control_after_generate / injected marker — not a slot
            continue
        current = widgets[idx] if idx < len(widgets) else None
        slot = {
            "address": f"{prefix}.{entry.name}",
            "name": entry.name,
            "type": entry.port.type,
            "current_value": current,
            "instance_id": prefix,
            "node_type": node_type,
        }
        if entry.port.enum_values:
            slot["enum"] = list(entry.port.enum_values)
        # A widget converted to an input and wired: the link supplies the value
        # at run time and the stored widget value is inert — say so, so a
        # caller edits the source instead (the same rule promoted widgets
        # follow across a subgraph boundary).
        linked = next(
            (
                i.get("link")
                for i in node.get("inputs") or []
                if isinstance(i, dict)
                and isinstance(i.get("widget"), dict)
                and i["widget"].get("name") == entry.name
                and i.get("link") is not None
            ),
            None,
        )
        if linked is not None:
            slot["linked_from"] = linked
        slots.append(slot)
    return slots


def _extract_frontend_slots(workflow: dict, graph: Graph) -> list[dict]:
    """Walk workflow nodes and extract tweakable slots.

    For every node we surface its widget inputs at ``<nodePath>.<input>``. When
    a node is a *subgraph instance* (its ``type`` is a UUID matching a def under
    ``definitions.subgraphs``) we additionally recurse INTO the definition and
    surface every interior node's widget inputs under a nested, instance-scoped
    address (``<instanceId>/<interiorId>.<input>``, recursing for deeper
    subgraphs). This is what lets an agent slot-edit a fetched gallery template
    whose editable prompt/seed/image live inside an opaque UUID subgraph.

    Curated subgraph ``inputs[]`` (the proxy parameter list) are still exposed
    at ``<instanceId>.<name>`` for backward compatibility, but the recursion
    means an agent is never stranded when those proxies are missing or dangle
    (fetched templates routinely point proxyWidgets at deleted interior ids).
    """
    defs_by_id = _subgraph_defs_by_id(workflow)
    slots: list[dict] = []
    seen_addrs: set[str] = set()

    def add(slot: dict) -> None:
        addr = slot["address"]
        if addr in seen_addrs:
            return
        seen_addrs.add(addr)
        slots.append(slot)

    from comfy_cli.cql import promoted as _promoted

    def walk(nodes: list, prefix: str, depth: int, exclude: set[tuple[str, str]], scope: dict) -> None:
        if depth > _MAX_SUBGRAPH_DEPTH:
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", ""))
            node_path = f"{prefix}{_SUBGRAPH_PATH_SEP}{node_id}" if prefix else node_id
            node_type = node.get("type", "")
            sg = defs_by_id.get(node_type)
            if sg is None:
                for slot in _node_widget_slots(node, node_path, graph):
                    if (node_id, slot["name"]) not in exclude:
                        add(slot)
                continue

            # Subgraph instance. Its promoted widgets are advertised at the
            # instance address (``57.width``) — the value the frontend runs
            # lives on the host (cql.promoted), so the interior widget behind
            # a promotion is NOT advertised: ``57/13.width`` is exactly the
            # "layer 2" address whose edits the surface overrides. Every
            # other interior widget is reachable at ``<instance>/<inner>.<w>``
            # (recursing for nested instances). A legacy template whose
            # curated surface comes entirely from ``proxyWidgets`` (nothing
            # linked) keeps its hand-picked view without recursion.
            declared, fully_curated = _declared_subgraph_slots(node, sg, node_id, graph, workflow, scope)
            for slot in declared:
                if (node_id, slot["name"]) not in exclude:
                    add(slot)
            promoted = [p for p in _promoted.promoted_inputs(sg, defs_by_id) if p.is_widget]
            if fully_curated and not promoted:
                continue
            hidden = {(str(p.source_node), str(p.source_widget or p.source_input)) for p in promoted}
            # …and the interior widgets a pending legacy repair will own.
            hidden |= _promoted.planned_hidden_sources(workflow, node, graph, defs_by_id)
            walk(sg.get("nodes") or [], node_path, depth + 1, hidden, sg)

    walk(workflow.get("nodes") or [], "", 0, set(), workflow)
    return slots


# Sentinel returned by _resolve_proxy_value when a proxy entry is genuinely
# unresolvable (interior node missing, widget name not in the node's order, or
# index past the end of widgets_values).  Callers must use ``is _UNRESOLVED``
# to distinguish this from a legitimately-null widget value (e.g. seed saved as
# None, or an optional image input that has not yet been set).
_UNRESOLVED = object()


def _declared_subgraph_slots(
    instance: dict,
    sg: dict,
    instance_id: str,
    graph: Graph,
    workflow: dict | None = None,
    scope: dict | None = None,
) -> tuple[list[dict], bool]:
    """Build slots for a subgraph instance's promoted inputs.

    A declared input that a boundary link feeds into an interior widget is a
    promoted WIDGET (``cql.promoted``): its address is ``<instance>.<name>``
    and its current value is the one the frontend runs — the host instance's
    own value when materialized, else the interior default. When an outside
    link feeds the instance input, the slot says so (``linked_from``): a
    write must then go to that source node. Socket-only inputs (``VIDEO``…)
    are link slots, not widget slots. A declared input the definition does
    not back with a link falls back to the legacy ``proxyWidgets`` route.

    Returns ``(slots, fully_curated)`` where ``fully_curated`` is True only when
    at least one widget slot exists and EVERY one resolved to a value (so the
    curated surface is complete and the caller can skip recursion).
    """
    from comfy_cli.cql import promoted as _promoted

    declared: list[dict] = []
    any_declared = False
    all_resolved = True
    defs = _subgraph_defs_by_id(workflow) if workflow is not None else {}
    promoted_by_name = {p.name: p for p in _promoted.promoted_inputs(sg, defs)}
    for inp in sg.get("inputs") or []:
        if not isinstance(inp, dict):
            continue
        inp_name = inp.get("name", "")
        if not inp_name:
            continue
        pi = promoted_by_name.get(inp_name)
        linked_from = None
        if pi is not None and pi.is_widget and workflow is not None:
            current = _promoted.host_value(instance, pi)
            if current is _promoted.UNSET:
                current = _promoted.source_value(workflow, sg, pi, graph, defs)
            if current is _promoted.UNSET:
                current = _UNRESOLVED
            # Only a LIVE link counts (the workflow's for a top-level instance,
            # the containing definition's for a nested one); a dangling id is
            # what the frontend drops on load, so the host value runs.
            linked_from = _promoted.live_external_link(scope if scope is not None else workflow, instance, inp_name)
        elif pi is not None and pi.is_widget:
            current = _UNRESOLVED
        else:
            legacy = _resolve_proxy_value(instance, sg, inp_name, graph)
            if legacy is _UNRESOLVED and pi is not None and not pi.is_widget and pi.source_node is None:
                # Declared but unbacked and unproxied — a socket or a dangling
                # declaration. Not a widget slot either way.
                continue
            current = legacy
        any_declared = True
        if current is _UNRESOLVED:
            all_resolved = False
            continue
        inp_type = inp.get("type", {})
        slot = {
            "address": f"{instance_id}.{inp_name}",
            "name": inp_name,
            "type": inp_type if isinstance(inp_type, str) else str(inp_type),
            "current_value": current,
            "instance_id": instance_id,
            "node_type": instance.get("type", ""),
        }
        if linked_from is not None:
            slot["linked_from"] = linked_from
        declared.append(slot)
    if workflow is not None:
        # Legacy ``proxyWidgets`` promotions the definition does not back
        # yet: the frontend repairs them into linked inputs on load, so they
        # are advertised where they will live — ``<instance>.<name>`` with the
        # value the frontend shows (legacy host value, else the interior
        # source). Reads never mutate; the first write performs the repair.
        advertised = {s["name"] for s in declared}
        for entry in _promoted.plan_proxy_migration(workflow, instance, graph, defs):
            if not entry.repairable or entry.name in advertised:
                continue
            advertised.add(entry.name)
            any_declared = True
            current = _promoted.entry_effective_value(workflow, sg, entry, graph, defs)
            if current is _promoted.UNSET:
                all_resolved = False
                continue
            declared.append(
                {
                    "address": f"{instance_id}.{entry.name}",
                    "name": entry.name,
                    "type": str(entry.type),
                    "current_value": current,
                    "instance_id": instance_id,
                    "node_type": instance.get("type", ""),
                }
            )
    return declared, (any_declared and all_resolved)


def promoted_source_port(sg: dict, pi: Any, defs: dict[str, dict], graph: Graph) -> tuple[str, Port | None]:
    """``(interior class, Port)`` of the concrete widget behind promoted input
    ``pi`` — the schema a host value is validated against."""
    from comfy_cli.cql import promoted as _promoted

    target = _promoted.deepest_source(sg, pi, defs)
    if target is None:
        return "", None
    path, widget = target
    current_sg = sg
    node: dict | None = None
    for seg in path:
        node = next((n for n in current_sg.get("nodes") or [] if str(n.get("id")) == seg), None)
        if node is None:
            return "", None
        nxt = defs.get(str(node.get("type", "")))
        if nxt is not None:
            current_sg = nxt
    class_type = str(node.get("type", "")) if node else ""
    m = graph.node(class_type)
    if m is None:
        return class_type, None
    return class_type, next((p for p in m.inputs if p.name == widget), None)


def _resolve_proxy_value(instance: dict, subgraph: dict, input_name: str, graph: Graph):
    """Navigate proxyWidgets to find the current widget value.

    Returns the widget's current value (which may be ``None`` for a
    legitimately-null widget) or the module-level ``_UNRESOLVED`` sentinel when
    the proxy entry points at a missing/dangling interior node, when the widget
    name is absent from the node's widget order, or when the index is past the
    end of ``widgets_values``.  Callers must use ``is _UNRESOLVED`` to test for
    the unresolvable case so that a real ``None`` value is preserved.
    """
    proxy = (instance.get("properties") or {}).get("proxyWidgets") or []
    for entry in proxy:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        name = entry[1] if isinstance(entry[1], str) else str(entry[1])
        if name != input_name:
            continue
        interior_id = str(entry[0])
        for inode in subgraph.get("nodes") or []:
            if not isinstance(inode, dict) or str(inode.get("id", "")) != interior_id:
                continue
            interior_class = inode.get("type", "")
            widgets = _widgets_as_positional(inode.get("widgets_values"), graph, interior_class)
            order = graph.widget_order_for_node(interior_class, widgets)
            try:
                idx = order.index(name)
            except ValueError:
                return _UNRESOLVED
            return widgets[idx] if idx < len(widgets) else _UNRESOLVED
        break
    return _UNRESOLVED


def _write_widget(node: dict, input_name: str, value: Any, graph: Graph, *, extend: bool) -> list[dict]:
    """Write ``value`` into ``node``'s ``widgets_values`` slot for ``input_name``.

    Validates against the node's schema and returns catalog warnings. ``extend``
    pads a short widget list for top-level direct edits (matches prior behavior);
    interior subgraph nodes always carry a full widget list and are not padded.

    Dynamic combos: the positional order is value-aware, so a selected option's
    sub-inputs (``model.size_preset``) address their real slots. Writing the
    selector itself to a different option rebuilds the node's sub-widget roster
    (see ``_write_dynamic_combo_selector``); a ``<combo>.<sub>`` address that
    doesn't exist under the current selector returns a warning without writing.
    """
    node_type = node.get("type", "")
    m = graph.node(node_type)
    if m is None:
        raise ValueError(f"unknown node type {node_type!r} for node {node.get('id')}")
    widgets = _widgets_as_positional(node.get("widgets_values"), graph, node_type)
    if not isinstance(node.get("widgets_values"), list):
        # Persist the positional projection so downstream re-reads (the
        # dynamic-combo selector path re-reads from the node) see the same
        # values this write is about to index against.
        node["widgets_values"] = widgets
    order = graph.widget_order_for_node(node_type, widgets)
    entries = _expand_widget_entries(m, widgets)
    if any(e.frontend_injected and e.name == input_name for e in entries):
        raise frontend_injected_widget_error(node_type, input_name, _editable_widget_names(entries))
    try:
        widget_idx = order.index(input_name)
    except ValueError:
        warning = _unknown_dynamic_sub_warning(m, input_name, order, widgets)
        if warning is not None:
            return [warning]
        avail = _editable_widget_names(entries)
        raise ValueError(
            f"widget {input_name!r} not found on {node_type}; "
            f"available widgets: {', '.join(avail) if avail else '(none — all inputs are links)'}"
        )

    port = next((e.port for e in entries if e.name == input_name), None)
    if port is None:
        # control_after_generate marker or an order override without matching
        # entries (tests monkeypatch widget_order_for_node) — fall back to the
        # declared port.
        port = next((p for p in m.inputs if p.name == input_name), None)

    if port is not None and _is_dynamic_combo_type(port.type) and port.dynamic_options:
        return _write_dynamic_combo_selector(node, port, input_name, widget_idx, value, entries, extend=extend)

    if widget_idx >= len(widgets):
        if not extend:
            raise ValueError(f"widget index {widget_idx} out of range for {node_type}")
        widgets.extend([None] * (widget_idx + 1 - len(widgets)))

    warnings: list[dict] = []
    if port:
        err = port.validate_shape(value)
        if err:
            raise ValueError(err)
        warnings = port.validate_catalog(value)

    widgets[widget_idx] = value
    node["widgets_values"] = widgets
    return warnings


def _unknown_dynamic_sub_warning(m: Morphism, input_name: str, order: list[str], widgets: list[Any]) -> dict | None:
    """Warning dict for a ``<combo>.<sub>`` address not present under the
    combo's CURRENT selector, or ``None`` when ``input_name`` isn't a
    dynamic-combo sub-address (caller falls through to the hard error)."""
    if "." not in input_name:
        return None
    base = input_name.split(".", 1)[0]
    base_port = next((p for p in m.inputs if p.name == base), None)
    if base_port is None or not _is_dynamic_combo_type(base_port.type) or base not in order:
        return None
    base_idx = order.index(base)
    selector = widgets[base_idx] if base_idx < len(widgets) else None
    valid = [n for n in order if n.startswith(f"{base}.")]
    return {
        "code": "unknown_dynamic_sub_input",
        "field": input_name,
        "message": (
            f"{input_name!r} does not exist under the current {base}={selector!r} selection; nothing was written"
        ),
        "hint": (
            f"valid {base}.* addresses: {', '.join(valid)}"
            if valid
            else f"{base}={selector!r} has no widget sub-inputs"
        )
        + f" — set {base}=<option> first to switch rosters",
        "valid_addresses": valid,
    }


def _write_dynamic_combo_selector(
    node: dict, port: Port, input_name: str, widget_idx: int, value: Any, entries: list[_WidgetEntry], *, extend: bool
) -> list[dict]:
    """Write a dynamic combo's selector, rebuilding the sub-widget roster when
    the selected option changes.

    The frontend inlines the selected option's widget sub-values right after
    the selector in ``widgets_values``, so switching options changes how many
    positional slots the combo owns. We keep every value before the combo,
    write the new selector, fill the new option's widget sub-inputs from schema
    defaults (option sub-spec ``default``; first enum option for combos), and
    keep the trailing values (seed/marker/watermark/…) aligned after them.
    """
    widgets = _widgets_as_list(node.get("widgets_values"))
    if value not in port.enum_values:
        valid = ", ".join(repr(k) for k in port.enum_values)
        raise ValueError(f"{input_name}: {value!r} is not a known option; valid options: {valid}")

    current = widgets[widget_idx] if widget_idx < len(widgets) else None
    if value == current:
        # Same option — the roster is unchanged; plain in-place write.
        if widget_idx >= len(widgets):
            if not extend:
                raise ValueError(f"widget index {widget_idx} out of range for {node.get('type')}")
            widgets.extend([None] * (widget_idx + 1 - len(widgets)))
        widgets[widget_idx] = value
        node["widgets_values"] = widgets
        return []

    # Sub-entries owned (directly or via nesting) by this combo sit contiguously
    # after it; their count is the positional span the OLD roster occupied.
    old_span = sum(
        1
        for e in entries[widget_idx + 1 :]
        if e.owner and (e.owner == input_name or e.owner.startswith(f"{input_name}."))
    )
    head = list(widgets[:widget_idx])
    if len(head) < widget_idx:
        if not extend:
            raise ValueError(f"widget index {widget_idx} out of range for {node.get('type')}")
        head.extend([None] * (widget_idx - len(head)))
    tail = list(widgets[widget_idx + 1 + old_span :])
    new_subs, new_names = _dynamic_combo_default_values(port.dynamic_options, value, input_name)
    node["widgets_values"] = head + [value] + new_subs + tail
    return [
        {
            "code": "dynamic_combo_roster_rebuilt",
            "field": input_name,
            "message": (
                f"{input_name}: {current!r} → {value!r} changed the sub-widget roster; "
                f"replaced {old_span} sub-value(s) with {len(new_subs)} schema default(s) "
                f"({', '.join(new_names) if new_names else 'none'}) — "
                f"values before and after the combo were preserved"
            ),
        }
    ]


def _dynamic_combo_default_values(
    dynamic_options: list[dict], selector: Any, prefix: str, depth: int = 0
) -> tuple[list[Any], list[str]]:
    """Schema-default ``widgets_values`` for the selected option's widget
    sub-inputs (positional, including nested combos' expansions and
    ``control_after_generate`` markers). Returns (values, dotted_names)."""
    values: list[Any] = []
    names: list[str] = []
    for sub in _dynamic_combo_sub_ports(dynamic_options, selector, prefix):
        if sub.is_link:
            continue
        default = sub.options.default
        if default is None and sub.enum_values:
            default = sub.enum_values[0]
        values.append(default)
        names.append(sub.name)
        if _is_dynamic_combo_type(sub.type) and sub.dynamic_options:
            if depth < _MAX_DYNAMIC_COMBO_DEPTH:
                nested_values, nested_names = _dynamic_combo_default_values(
                    sub.dynamic_options, default, sub.name, depth + 1
                )
                values.extend(nested_values)
                names.extend(nested_names)
        elif _has_control_after_generate_slot(sub):
            values.append("fixed")
            names.append("control_after_generate")
    return values, names


def _resolve_node_path(workflow: dict, segments: list[str], defs_by_id: dict[str, dict]) -> dict:
    """Walk a ``/``-separated node path into (possibly nested) subgraphs.

    The first segment names a top-level node; each subsequent segment names an
    interior node of the subgraph definition the previous segment instantiated.
    Returns the resolved (mutable) node dict, or raises ValueError describing
    the first hop that couldn't be found.

    Isolation: every non-terminal hop is forked (if its subgraph definition is
    shared) before descending, so interior writes at any depth can't alias
    sibling instances — not just the first hop.
    """
    nodes = workflow.get("nodes") or []
    node = next((n for n in nodes if isinstance(n, dict) and str(n.get("id", "")) == segments[0]), None)
    if node is None:
        raise ValueError(f"node {segments[0]} not found in workflow")
    for seg in segments[1:]:
        # ``node`` is a non-terminal hop we are about to descend into: fork its
        # shared definition first so the write below this hop stays isolated.
        _isolate_shared_subgraph(workflow, node, defs_by_id)
        defs_by_id = _subgraph_defs_by_id(workflow)  # rebuild: node.type may have changed
        sg = defs_by_id.get(node.get("type", ""))
        if sg is None:
            raise ValueError(f"node {node.get('id')} is not a subgraph; cannot descend to {seg!r}")
        inner = next((n for n in (sg.get("nodes") or []) if isinstance(n, dict) and str(n.get("id", "")) == seg), None)
        if inner is None:
            raise ValueError(f"interior node {seg} not found in subgraph {sg.get('id')}")
        node = inner
    return node


def _count_instances(workflow: dict, def_id: str) -> int:
    """Count nodes (top-level + interior-of-definitions) instantiating ``def_id``."""
    count = 0
    for n in workflow.get("nodes") or []:
        if isinstance(n, dict) and str(n.get("type", "")) == def_id:
            count += 1
    for sg in (workflow.get("definitions") or {}).get("subgraphs") or []:
        if isinstance(sg, dict):
            for n in sg.get("nodes") or []:
                if isinstance(n, dict) and str(n.get("type", "")) == def_id:
                    count += 1
    return count


def _isolate_shared_subgraph(workflow: dict, instance: dict, defs_by_id: dict[str, dict]) -> None:
    """If ``instance``'s subgraph definition is shared with another instance,
    deep-copy it under a fresh id and repoint ``instance`` so an interior write
    can't alias sibling instances. No-op when the instance already owns its def.

    The fork id is DERIVED DETERMINISTICALLY from ``(definition id, instance id)``
    — never a random UUID — so two replicas replaying the same op produce
    byte-identical graphs (a convergence requirement of the op model in
    :mod:`comfy_cli.workflow_ops`).
    """
    def_id = str(instance.get("type", ""))
    sg = defs_by_id.get(def_id)
    if sg is None or _count_instances(workflow, def_id) <= 1:
        return
    new_sg = copy.deepcopy(sg)
    new_id = _deterministic_fork_id(def_id, instance.get("id"))
    new_sg["id"] = new_id
    workflow.setdefault("definitions", {}).setdefault("subgraphs", []).append(new_sg)
    instance["type"] = new_id


def _deterministic_fork_id(def_id: str, instance_id: Any) -> str:
    """A stable id for the isolated copy of ``def_id`` owned by ``instance_id``.
    Deterministic across processes (``hashlib``, not the salted builtin ``hash``)
    so replaying the same op anywhere yields the same id. SHA-256 (not SHA-1) —
    this isn't a security boundary, but there's no reason to reach for a broken
    hash, and it keeps the scanners quiet."""
    seed = f"{def_id}\x00{instance_id}".encode()
    return "sg-" + _hashlib.sha256(seed).hexdigest()[:32]


def _suggest_slots_for_input(workflow: dict, input_name: str, graph: Graph, *, limit: int = 6) -> list[str]:
    """Real slot addresses whose widget name matches ``input_name``.

    Turns an unresolvable address into an actionable correction: an agent that
    named the right widget but the wrong node or separator (e.g.
    ``285/288.vae_name`` or ``285:288.vae_name`` when the VAELoader is ``285/29``)
    is pointed at the address that actually carries ``vae_name``. Best-effort —
    any extraction failure yields no suggestions rather than masking the error.
    """
    if not input_name:
        return []
    try:
        slots = _extract_frontend_slots(workflow, graph)
    except Exception:
        return []
    out: list[str] = []
    for s in slots:
        if s.get("name") == input_name:
            addr = s.get("address") or ""
            node_type = s.get("node_type") or ""
            out.append(f"{addr} ({node_type})" if node_type else addr)
            if len(out) >= limit:
                break
    return out


def _apply_one_slot(workflow: dict, addr: str, value: Any, graph: Graph) -> list[dict]:
    """Apply one slot override, enriching *not-found* errors with real address
    suggestions so a mistargeted edit self-corrects in one step.

    An LLM that reconstructs an interior address from memory (rather than copying
    it from ``slots``) tends to hit a real *sibling* node — e.g. writing
    ``285/288.vae_name`` (a CLIPLoader) when the VAELoader is ``285/29``. The
    intended widget name is almost always right, so on a not-found failure we
    scan the workflow for the address that actually carries that widget and name
    it in the error. Shape/enum errors (the target resolved fine) pass through
    unchanged.
    """
    try:
        return _apply_one_slot_impl(workflow, addr, value, graph)
    except ValueError as e:
        if "not found" not in str(e):
            raise
        input_name = addr.split(".", 1)[1] if "." in addr else ""
        suggestions = _suggest_slots_for_input(workflow, input_name, graph)
        if not suggestions:
            raise
        raise ValueError(
            f"{e}. Did you mean: {'; '.join(suggestions)}? "
            "Copy the address verbatim from `comfy workflow slots` — never rebuild it."
        ) from e


def _apply_one_slot_impl(workflow: dict, addr: str, value: Any, graph: Graph) -> list[dict]:
    """Apply a single slot override. Returns warnings. Raises ValueError on hard errors.

    Address forms (see ``_extract_frontend_slots`` / ``_SUBGRAPH_PATH_SEP``):
      * ``<id>.<input>``                 — top-level node widget (direct mode).
      * ``<id>.<declaredInput>``         — curated subgraph proxy input, routed
                                           through ``proxyWidgets`` to its interior
                                           node (legacy template mode).
      * ``<instanceId>/<innerId>.<input>`` (and deeper) — a widget on an interior
                                           node reached by descending into the
                                           subgraph definition(s).
    """
    if "." not in addr:
        raise ValueError(f"invalid slot address {addr!r} (expected 'instance_id.input_name')")
    # Node paths use '/' as separator; node IDs are numeric or UUID (no '.').
    # Input names may legitimately contain dots (e.g. 'images.image0').
    # Always split on the FIRST dot so multi-dot input names are preserved.
    node_path, input_name = addr.split(".", 1)
    segments = node_path.split(_SUBGRAPH_PATH_SEP)
    defs_by_id = _subgraph_defs_by_id(workflow)

    from comfy_cli.cql import promoted as _promoted

    # One resolution for every address form — the same one set-widget uses —
    # so a promoted widget's value lands on the host (or the outside node that
    # feeds it), an unpromoted interior widget lands in the definition, and a
    # plain node lands on itself. See ``cql.promoted.resolve_write``.
    target = _promoted.resolve_write(workflow, graph, segments, input_name)
    if target.kind == "host":
        # Fork every shared definition along the path before writing so a
        # nested host's siblings stay independent (a top-level host has no
        # non-terminal hop and is not forked — its values are its own).
        instance = _resolve_node_path(workflow, target.segments, defs_by_id)
        sg = _subgraph_defs_by_id(workflow).get(str(instance.get("type", "")))
        if target.repair is not None:
            pi = target.repair.as_promoted_input(sg)
        else:
            pi = _promoted.find_promoted(sg, _subgraph_defs_by_id(workflow), target.widget)
        _class, port = promoted_source_port(sg, pi, _subgraph_defs_by_id(workflow), graph)
        warnings: list[dict] = []
        if port is not None:
            err = port.validate_shape(value)
            if err:
                raise ValueError(err)
            warnings = [dict(w, field=addr) for w in port.validate_catalog(value)]
        if target.repair is not None:
            # A legacy ``proxyWidgets`` promotion: run the frontend's forward
            # migration on this instance first (forking a shared definition,
            # since the repair mutates it), exactly as the op path does.
            _isolate_shared_subgraph(workflow, instance, _subgraph_defs_by_id(workflow))
            _promoted.flush_proxy_migration(workflow, instance, graph, instance_path=list(target.segments))
        _promoted.set_host_value(workflow, instance, target.widget, value, graph)
        return warnings
    if target.kind == "interior":
        inner = _resolve_node_path(workflow, target.segments, defs_by_id)
        return _write_widget(inner, target.widget, value, graph, extend=False)
    if target.kind == "legacy_primitive":
        node = target.node
        values = node.get("widgets_values")
        values = list(values) if isinstance(values, list) else []
        if not values:
            values.append(None)
        values[0] = value
        node["widgets_values"] = values
        return []
    # --- Direct mode: a regular top-level node (or the primitive feeding a promoted input). ---
    return _write_widget(target.node, target.widget, value, graph, extend=True)
