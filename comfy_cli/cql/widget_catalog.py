"""The widget catalog: a derived projection of ``object_info`` that says, per
node class, **which widgets exist and in what positional order**.

WHY IT EXISTS. A ComfyUI workflow stores widget values POSITIONALLY, in each
node's ``widgets_values`` array. The CRDT document the cloud agent and the
frontend co-edit stores them BY NAME, because an index is not a stable identity
across a schema change (add one widget to a class and every later index moves).
Something has to convert between the two, in both directions, and that
something needs exactly one fact per class: the widget order.

That fact is already computed here — :meth:`cql.engine.Graph.widget_order` is
what every edit primitive in this CLI resolves ``set-widget <id>.<name>``
against, including the two shapes a naive projection gets wrong:

* ``control_after_generate`` — a synthetic widget the frontend injects after a
  seed. It occupies a real ``widgets_values`` slot and is in no ``input_order``.
* ``COMFY_DYNAMICCOMBO_V3`` — one declared selector that expands into
  key-dependent sub-widgets (``model`` → ``model``, ``model.resolution``).
* Frontend-injected inputs — ``upload`` (the upload button on every media
  loader), ``audioUI`` (the audio player), ``image`` (the ``PREVIEW_3D``
  viewport on ``SaveGLB``/``Preview3D``). Extensions add them to the node
  definition after ``object_info``; they serialize AFTER every declared
  widget, optional ones included, and are ``serialize: false`` on current
  frontends (older ones wrote ``"image"``/``null``). The order names them so
  a workflow saved by either frontend decomposes, and a fresh node omits them.
* DOM-widget inputs the schema declares under an uppercase custom type
  (``Load3D.image`` is ``LOAD_3D``, ``LoadAudioUI.audioUI`` is ``AUDIO_UI``)
  and inputs whose ``widgetType`` option overrides a link-shaped socket type
  (``FLOAT,INT`` with ``widgetType: "STRING"``) — both occupy a slot.

So the catalog is exported from here rather than recomputed by each consumer:
a second implementation of widget order is a second answer, and the two would
diverge silently — as a wrong index, i.e. a widget value written into the wrong
field of the user's canvas.

WHAT IT IS NOT. It is not ``object_info``. It carries no types, no defaults, no
enum choices, no descriptions — only what a name↔index converter needs. That
keeps it small enough to hand to a sidecar on every call.

SHAPE (``envelope/1`` ``data`` of ``comfy nodes widget-catalog``)::

    {
      "catalog_version": "sha256:<64 hex>",
      "class_count": 1234,
      "types": {
        "KSampler": {"widget_order": ["seed", "control_after_generate", ...]},
        "BatchImagesNode": {
          "widget_order": [],
          "autogrow_templates": {"images": {"prefix": "image"}}
        },
        "ImageBatchMulti": {
          "widget_order": ["inputcount"],
          "inputcount": {"widget": "inputcount", "elements": ["image"]}
        }
      }
    }

``catalog_version`` is the SHA-256 of the canonical JSON encoding of ``types``
alone (sorted keys, no whitespace, UTF-8), prefixed ``sha256:``. It excludes
itself and ``class_count`` so a consumer that stored only the catalog can
recompute and verify the pin it was given. Identical ``object_info`` in ⇒
identical version out, regardless of key iteration order; any change to any
class's widget order or grow family moves it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CATALOG_VERSION_PREFIX = "sha256:"


def build_types(graph) -> dict[str, dict[str, Any]]:
    """The ``types`` map: one entry per class the graph knows, in class order.

    Every class gets an entry, including widget-less ones (``VAEDecode`` →
    ``{"widget_order": []}``). A missing entry and an empty order are different
    statements — "I have never heard of this class" vs. "this class has no
    widgets" — and a converter must be able to tell them apart.
    """
    # Imported inside the function: workflow_ops sits above cql in the import
    # graph (it consumes cql.engine), so a module-level import here would make
    # the dependency circular.
    from comfy_cli.workflow_ops import INPUTCOUNT_WIDGET, inputcount_family_elements

    types: dict[str, dict[str, Any]] = {}
    for m in graph.all_nodes():
        entry: dict[str, Any] = {"widget_order": list(graph.widget_order_default(m.id))}

        # V3 autogrow (COMFY_AUTOGROW_V3): one declared input, one wire slot per
        # connection (`images` → `images.image0`, `images.image1`, …). The
        # effective template is used, so a schema that ships none still tells the
        # consumer the input grows (see Port.autogrow_element_template).
        autogrow = {p.name: p.autogrow_element_template for p in m.inputs if p.is_autogrow}
        if autogrow:
            entry["autogrow_templates"] = autogrow

        # kijai `inputcount` family (ImageBatchMulti, JoinStringMulti, …): NOT
        # autogrow-typed — fixed `{elem}_N` inputs plus an INT `inputcount`
        # widget the node reads at runtime. Growing a slot means bumping that
        # widget, which is a widget write, so the converter has to know.
        elements = inputcount_family_elements(graph, m.id)
        if elements:
            entry["inputcount"] = {"widget": INPUTCOUNT_WIDGET, "elements": elements}

        types[m.id] = entry
    return types


def catalog_version(types: dict[str, Any]) -> str:
    """``sha256:<hex>`` over the canonical JSON encoding of ``types``.

    Canonical = sorted keys, no insignificant whitespace, non-ASCII kept
    verbatim. Deterministic for identical input on any platform and any Python
    build, which is what makes it usable as a consumer-side cache key.
    """
    canonical = json.dumps(types, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return CATALOG_VERSION_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_catalog(graph) -> dict[str, Any]:
    """The full emitted payload — ``types`` plus its pin and cardinality."""
    types = build_types(graph)
    return {
        "catalog_version": catalog_version(types),
        "class_count": len(types),
        "types": types,
    }
