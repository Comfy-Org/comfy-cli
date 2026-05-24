"""NDJSON event emitter for ``comfy run --json``.

Every ``emit_*`` method is a no-op when ``json_mode=False``, so the same
call sites work for both modes. See ``docs/json-output.md``.
"""

from __future__ import annotations

import json
import time

import typer

from comfy_cli.command.run.loader import SCHEMA_VERSION
from comfy_cli.output import rprint as pprint


class JsonEmitter:
    def __init__(self, json_mode: bool):
        self.json_mode = json_mode
        self.start_time = time.monotonic()
        self.client_id: str | None = None
        self.prompt_id: str | None = None
        self.workflow: dict | None = None
        self.cached_node_ids: list[str] = []
        self.executed_node_ids: list[str] = []
        self.outputs: list[dict] = []

    def set_workflow(self, workflow):
        self.workflow = workflow

    def set_client_id(self, client_id):
        self.client_id = client_id

    def _elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def get_title(self, node_id):
        if not isinstance(self.workflow, dict):
            return str(node_id)
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return str(node_id)
        meta = node.get("_meta")
        if isinstance(meta, dict):
            title = meta.get("title")
            if isinstance(title, str) and title:
                return title
        class_type = node.get("class_type")
        return class_type if isinstance(class_type, str) and class_type else str(node_id)

    def get_class_type(self, node_id):
        if not isinstance(self.workflow, dict):
            return ""
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return ""
        return node.get("class_type", "")

    def _emit(self, event: dict) -> None:
        if not self.json_mode:
            return
        line = json.dumps(event, ensure_ascii=True)
        print(line, flush=True)

    def emit_converted(self, node_count: int) -> None:
        self._emit(
            {
                "event": "converted",
                "schema_version": SCHEMA_VERSION,
                "node_count": node_count,
            }
        )

    def workflow_manifest(self) -> list[dict]:
        """Build the `nodes` array for the `queued` event — one entry per
        node in the submitted (post-conversion) workflow."""
        if not isinstance(self.workflow, dict):
            return []
        manifest: list[dict] = []
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            class_type = class_type if isinstance(class_type, str) else ""
            manifest.append(
                {
                    "node_id": str(node_id),
                    "class_type": class_type,
                    "title": self.get_title(node_id),
                }
            )
        return manifest

    def emit_prompt_preview(self, prompt: dict) -> None:
        self._emit(
            {
                "event": "prompt_preview",
                "schema_version": SCHEMA_VERSION,
                "prompt": prompt,
            }
        )

    def emit_queued(self, prompt_id: str, validation_warnings: list[dict]) -> None:
        self.prompt_id = prompt_id
        self._emit(
            {
                "event": "queued",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": prompt_id,
                "client_id": self.client_id,
                "validation_warnings": validation_warnings,
                "nodes": self.workflow_manifest(),
            }
        )

    def emit_node_cached(self, node_id) -> None:
        node_id = str(node_id)
        self.cached_node_ids.append(node_id)
        self._emit(
            {
                "event": "node_cached",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
            }
        )

    def emit_node_executing(self, node_id) -> None:
        node_id = str(node_id)
        # `executed_node_ids` aggregates everything the executor touched —
        # including intermediate nodes that never fire a server-side `executed` WS event.
        if node_id not in self.executed_node_ids:
            self.executed_node_ids.append(node_id)
        self._emit(
            {
                "event": "node_executing",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
            }
        )

    def emit_node_progress(self, node_id, value, max_val) -> None:
        node_id = str(node_id)
        self._emit(
            {
                "event": "node_progress",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
                "value": value,
                "max": max_val,
            }
        )

    def emit_node_executed(self, node_id, outputs: list[dict]) -> None:
        node_id = str(node_id)
        if node_id not in self.executed_node_ids:
            self.executed_node_ids.append(node_id)
        self.outputs.extend(outputs)
        self._emit(
            {
                "event": "node_executed",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
                "outputs": outputs,
            }
        )

    def emit_completed(self) -> None:
        self._emit(
            {
                "event": "completed",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": self.prompt_id,
                "client_id": self.client_id,
                "elapsed_seconds": self._elapsed(),
                "outputs": self.outputs,
                "cached_node_ids": self.cached_node_ids,
                "executed_node_ids": self.executed_node_ids,
            }
        )

    def fail(self, kind: str, message: str, *, rich_message: str | None = None, **extras) -> typer.Exit:
        """Emit a `failed` event (in JSON mode) or print a red text message
        (otherwise), then return the `typer.Exit(code=1)` for the caller to
        raise. Returning rather than raising keeps `raise ... from e`
        chaining clean at call sites. `rich_message` overrides `message`
        for the human-readable text only — it is auto-wrapped in
        `[bold red]...[/bold red]`. Sites that need multi-colour Rich
        markup should emit the failure event explicitly."""
        self.emit_failed(kind, message, **extras)
        if not self.json_mode:
            pprint(f"[bold red]{rich_message if rich_message is not None else message}[/bold red]")
        return typer.Exit(code=1)

    def emit_failed(self, kind: str, message: str, **extras) -> None:
        error = {"kind": kind, "message": message}
        error.update(extras)
        self._emit(
            {
                "event": "failed",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": self.prompt_id,
                "client_id": self.client_id,
                "elapsed_seconds": self._elapsed(),
                "error": error,
            }
        )
