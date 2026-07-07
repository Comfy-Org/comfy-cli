"""Workflow-file loading and shape classification.

Pure functions over a single workflow JSON: detect format (UI vs API),
validate the parsed shape, and load from disk with typed errors.
"""

from __future__ import annotations

import json
import os

# Maximum bytes of a server response body we surface to the user (or
# embed in a `failed.error.body` field). Anything longer is truncated.
_MAX_BODY_PREVIEW = 500


def _node_errors_to_list(node_errors) -> list[dict]:
    """Transform ComfyUI's dict-keyed `node_errors` payload into a list of self-contained records.
    Each record carries `node_id` as a field, so agents can iterate the result
    directly without indirecting through dict keys."""
    if not isinstance(node_errors, dict):
        return []
    result = []
    for node_id, record in node_errors.items():
        if not isinstance(record, dict):
            continue
        entry = {"node_id": str(node_id)}
        entry.update(record)
        result.append(entry)
    return result


def is_ui_workflow(workflow) -> bool:
    return (
        isinstance(workflow, dict)
        and isinstance(workflow.get("nodes"), list)
        and isinstance(workflow.get("links"), list)
    )


def _classify_api_workflow(workflow):
    """Classify a parsed JSON object as API workflow / empty / invalid.

    Returns one of:
        ("ok", workflow_dict)   — well-formed API workflow with ≥1 node
        ("empty", None)         — empty dict (caller routes to workflow_empty)
        ("invalid", None)       — not a dict, or no value has class_type
    """
    if not isinstance(workflow, dict):
        return ("invalid", None)
    if not workflow:
        return ("empty", None)
    for val in workflow.values():
        if isinstance(val, dict) and "class_type" in val:
            return ("ok", workflow)
    return ("invalid", None)


def pop_compose_meta(workflow: dict) -> dict | None:
    """Pop and return the compose-embedded ``_meta`` provenance block.

    ``comfy workflow compose`` writes ``workflow["_meta"] = {"schema":
    "compose/1", "blueprint": …, "items": …}`` into the compiled JSON. The
    server would treat that key as a (broken) node, so ``run`` strips it
    before preflight validation and submit.

    Only a dict WITHOUT a ``class_type`` key is stripped — a node that is
    legitimately keyed ``"_meta"`` (i.e. has a class_type) is left alone.
    Per-node ``_meta: {title}`` blocks live inside nodes and are never
    touched. Returns the popped block, or ``None`` when nothing was popped.
    """
    if not isinstance(workflow, dict):
        return None
    meta = workflow.get("_meta")
    if isinstance(meta, dict) and "class_type" not in meta:
        del workflow["_meta"]
        return meta
    return None


class WorkflowLoadError(Exception):
    """Raised by ``_load_workflow_file`` for pre-flight file errors.

    ``code`` is a registered ``error_codes`` value — callers pass it straight
    to ``renderer.error(code=e.code, ...)``.
    """

    def __init__(self, *, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint


def _load_workflow_file(path: str) -> tuple[dict, str, bool]:
    """Load and validate a workflow JSON file.

    Returns (raw_workflow_dict, absolute_path, is_ui_format).
    Raises ``WorkflowLoadError`` on file-not-found, read error, or invalid JSON.
    """
    workflow_name = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(workflow_name):
        raise WorkflowLoadError(
            code="workflow_not_found",
            message=f"Specified workflow file not found: {workflow_name}",
            hint="check the path; pass the API-format JSON exported from ComfyUI",
        )

    try:
        with open(workflow_name, encoding="utf-8") as f:
            raw_workflow = json.load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise WorkflowLoadError(
            code="workflow_read_error",
            message=f"Unable to read workflow file: {e}",
        ) from e
    except json.JSONDecodeError as e:
        raise WorkflowLoadError(
            code="workflow_invalid_json",
            message=f"Specified workflow file is not valid JSON: {e}",
            hint="re-export the workflow from ComfyUI (File > Export (API))",
        ) from e

    return raw_workflow, workflow_name, is_ui_workflow(raw_workflow)
