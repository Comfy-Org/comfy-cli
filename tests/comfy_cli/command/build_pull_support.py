from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from build_push_support import RecordingBuilder
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app


class PullBuilder(RecordingBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.remote_builds: dict[str, dict] = {}

    def get_build(self, build_id: str) -> dict:
        if build_id not in self.remote_builds:
            return super().get_build(build_id)
        remote = deepcopy(self.remote_builds[build_id])
        revision = remote["updatedAt"]
        assert isinstance(revision, str)
        self.remote_revisions[build_id] = revision
        self.calls.append({"method": "get_build", "id": build_id, "updatedAt": revision})
        return remote

    def list_builds(self) -> list[dict]:
        self.calls.append({"method": "list_builds"})
        return [{"id": build_id, "name": remote["name"]} for build_id, remote in self.remote_builds.items()]


def serve(client: PullBuilder, build_id: str, definition: dict) -> None:
    client.remote_builds[build_id] = {
        "id": build_id,
        "name": f"Remote {build_id}",
        "description": f"Description for {build_id}",
        "updatedAt": f"revision-{build_id}",
        "definition": definition,
    }


def invoke_pull(root: Path, *args: str, agentic: bool = True):
    return CliRunner(mix_stderr=False).invoke(
        cli_app,
        ["build", "pull", *args, str(root)],
        env={
            "AI_AGENT": "1" if agentic else None,
            # A human at a terminal gets pretty output; `None` here would fall
            # through to the non-tty rule and hand a "TTY human" test a JSON
            # renderer, which no real interactive caller ever has.
            "COMFY_OUTPUT": "json" if agentic else "pretty",
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": None,
            "COMFY_BUILDER_URL": "https://builder.test",
        },
    )
