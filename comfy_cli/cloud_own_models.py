"""Models a workspace uploaded to its Comfy Cloud library count as valid loader values.

Cloud's ``object_info`` dropdowns list Cloud's own models only, never the ones a
workspace uploaded with ``comfy model upload``. Cloud still runs a workflow that
names an upload: it resolves the saved value against the workspace's library when
it validates the job. The CLI's enum check reads those dropdowns, so before this
module every ``--where cloud`` run and edit refused every upload.

So, before refusing a model file value on Cloud, ``comfy run`` and ``comfy workflow
set-widget`` ask the library about it, with the lookup and matching rule ``comfy
workflow check`` uses (:func:`comfy_cli.workflow_coverage._pick_upload`), and admit
a value the library holds into that input's options for the one command
(:meth:`comfy_cli.cql.engine.Graph.admit_combo_value`). A value nobody holds stays a
hard error, and its hint says the library was checked. So does any value of an input
that picks no model file (:func:`loads_a_model`): an upload named ``*.safetensors``
is not a sampler or a scheduler.

Only a value that would otherwise be refused is looked up, so a workflow that names
only Cloud's own models makes no extra request.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from comfy_cli.workflow_coverage import _basename, _pick_upload, is_model_value

#: Catalog refusals a held upload answers: a value missing from the options, and an
#: input whose folder Cloud ships no models for at all.
ADMITTABLE_CODES = frozenset({"unknown_enum_value", "no_options_available"})


def loads_a_model(graph, class_type: str, input_name: str) -> bool:
    """Whether ``class_type``'s COMBO input ``input_name`` picks a model file.

    Read from the input's own catalog options: any one of them a model file name
    (``VAELoader.vae_name`` also lists ``taesd``, so not all). Not from the input's
    name, which misleads both ways (a partner node's ``model`` is no file). An input
    with no options at all shows no role and counts as one: that is a model folder
    Cloud ships empty, since the non-model lists (``sampler_name``, ``scheduler``)
    are filled server side and never come back empty.
    """
    m = graph.node(class_type)
    port = next((p for p in m.inputs if p.name == input_name), None) if m is not None else None
    if port is None or port.type != "COMBO":
        return False
    options = port.enum_values or []
    return not options or any(is_model_value(o) for o in options)


class OwnModels:
    """Where a model value sits in the workspace's library, one lookup per file name.

    ``lookup(basename)`` returns every uploaded model asset whose name contains
    ``basename``, as ``/api/assets`` rows; it may raise. A failed lookup is
    recorded, never raised: the refusal it would have lifted stands.
    """

    def __init__(self, lookup: Callable[[str], list[dict[str, Any]]]):
        self._lookup = lookup
        self._asked: dict[str, list[dict[str, Any]] | None] = {}
        self.errors: dict[str, str] = {}

    def place(self, value: Any) -> tuple[str, str | None, list[str]]:
        """``(status, library_name, candidates)`` for ``value``.

        ``status`` is ``present`` (Cloud runs the saved value unchanged), ``remap``
        (one upload has the file name under another path: ``library_name``),
        ``missing`` (``candidates`` names the uploads when several fit and Cloud
        would not guess), ``unknown`` (the lookup failed) or ``not_a_model``.
        """
        if not is_model_value(value):
            return "not_a_model", None, []
        name = _basename(value)
        if name not in self._asked:
            try:
                self._asked[name] = [row for row in self._lookup(name) if isinstance(row, dict)]
            except Exception as e:  # noqa: BLE001 — any failure leaves the refusal standing
                self._asked[name] = None
                self.errors[name] = f"{type(e).__name__}: {e}"
        uploads = self._asked[name]
        if uploads is None:
            return "unknown", None, []
        return _pick_upload(value, uploads)

    def holds(self, value: Any) -> bool:
        return self.place(value)[0] == "present"

    def refusal_note(self, value: Any) -> str | None:
        """What the library said about a refused ``value``, for the refusal's hint."""
        status, picked, candidates = self.place(value)
        if status == "remap":
            return f"your Comfy Cloud model library has this file as {picked!r}: set that"
        if status == "missing" and candidates:
            return f"several models in your Comfy Cloud model library fit, so Cloud would not pick one: {', '.join(candidates)}"
        if status == "missing":
            return (
                f"no model in your Comfy Cloud model library is named {value!r} either: "
                "upload it with `comfy model upload`, or pick one of Cloud's"
            )
        if status == "unknown":
            return (
                f"could not check your Comfy Cloud model library: {self.errors.get(_basename(value), 'lookup failed')}"
            )
        return None


def for_cloud() -> OwnModels:
    """:class:`OwnModels` over the signed-in workspace's ``/api/assets``.

    Credentials are resolved on the first lookup, not here, so a command that never
    meets an unlisted model never touches them.
    """
    target = None

    def lookup(name: str) -> list[dict[str, Any]]:
        nonlocal target
        from comfy_cli.command.workflow_check import _fetch_model_assets
        from comfy_cli.target import resolve_target

        if target is None:
            # A side lookup: a refresh that fails here must not sign the person out.
            target = resolve_target(where="cloud", allow_clear=False)
        return _fetch_model_assets(target, name, include_public=False)

    return OwnModels(lookup)


def _refused_value(workflow: dict, err: dict) -> tuple[str, str, Any] | None:
    """``(class_type, input_name, value)`` behind one API-format catalog refusal."""
    if err.get("code") not in ADMITTABLE_CODES:
        return None
    node = workflow.get(str(err.get("node_id")))
    inputs = node.get("inputs") if isinstance(node, dict) else None
    field = err.get("field")
    if not isinstance(inputs, dict) or not isinstance(field, str) or field not in inputs:
        return None
    class_type = node.get("class_type")
    return (class_type, field, inputs[field]) if isinstance(class_type, str) else None


def admit(graph, class_type: str, input_name: str, value: Any, own: OwnModels) -> bool:
    """Admit ``value`` into that input of ``graph`` when the input picks a model file
    and the library holds the value. Asks the library about nothing else."""
    return (
        loads_a_model(graph, class_type, input_name)
        and own.holds(value)
        and graph.admit_combo_value(class_type, input_name, value)
    )


def note_for(graph, class_type: str, input_name: str, value: Any, own: OwnModels) -> str | None:
    """What the library said about a refused ``value``, when the input picks a model file."""
    return own.refusal_note(value) if loads_a_model(graph, class_type, input_name) else None


def admit_refused(graph, workflow: dict, errors: list[dict], own: OwnModels) -> list[dict]:
    """Admit into ``graph`` each refused model value of an API-format ``workflow``
    that the library holds. Returns one row per admitted input."""
    admitted: list[dict] = []
    for err in errors:
        refused = _refused_value(workflow, err)
        if refused is None:
            continue
        class_type, field, value = refused
        if admit(graph, class_type, field, value, own):
            admitted.append(
                {"node_id": str(err.get("node_id")), "class_type": class_type, "field": field, "value": value}
            )
    return admitted


def note_refusals(graph, workflow: dict, errors: list[dict], own: OwnModels) -> None:
    """Add what the library said to each refused model value, in place: to its
    ``hint``, and as ``library`` for the envelope's own hint.

    Reads only what :func:`admit_refused` already looked up, so it sends nothing."""
    for err in errors:
        refused = _refused_value(workflow, err)
        note = note_for(graph, *refused, own) if refused is not None else None
        if note:
            err["library"] = note
            err["hint"] = f"{err['hint']} — {note}" if err.get("hint") else note
