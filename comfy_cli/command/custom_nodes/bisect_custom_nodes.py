from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, NamedTuple

import typer
from typing_extensions import Annotated

from comfy_cli.command.custom_nodes.cm_cli_util import execute_cm_cli
from comfy_cli.command.launch import launch as launch_command

bisect_app = typer.Typer()

# File to store the state of bisect
default_state_file = Path("bisect_state.json")


class BisectState(NamedTuple):
    status: Literal["idle", "running", "resolved"]

    # All nodes in the current bisect session
    all: list[str]

    # The range of nodes that contains the bad node
    range: list[str]

    # The active set of nodes to test
    active: list[str]

    # The arguments to pass to the ComfyUI launch command
    launch_args: list[str] = []

    def good(self) -> BisectState:
        """The active set of nodes is good, narrowing down the potential problem area."""
        if self.status != "running":
            raise ValueError("No bisect session running.")

        new_range = list(set(self.range) - set(self.active))

        if len(new_range) == 1:
            return BisectState(
                status="resolved",
                all=self.all,
                launch_args=self.launch_args,
                range=new_range,
                active=[],
            )

        return BisectState(
            status="running",
            all=self.all,
            launch_args=self.launch_args,
            range=new_range,
            active=new_range[len(new_range) // 2 :],
        )

    def bad(self) -> BisectState:
        """The active set of nodes is bad, indicating the problem is within this set."""
        if self.status != "running":
            raise ValueError("No bisect session running.")

        new_range = self.active

        if len(new_range) == 1:
            return BisectState(
                status="resolved",
                all=self.all,
                launch_args=self.launch_args,
                range=new_range,
                active=[],
            )

        return BisectState(
            status="running",
            all=self.all,
            launch_args=self.launch_args,
            range=new_range,
            active=new_range[len(new_range) // 2 :],
        )

    def save(self, state_file=None):
        self.set_custom_node_enabled_states()
        state_file = state_file or default_state_file
        with state_file.open("w") as f:
            json.dump(self._asdict(), f)  # pylint: disable=no-member

    def reset(self):
        BisectState(
            "idle",
            all=self.all,
            launch_args=self.launch_args,
            range=self.all,
            active=self.all,
        ).set_custom_node_enabled_states()
        return BisectState("idle", self.all, self.all, self.all, self.launch_args)

    @classmethod
    def load(cls, state_file=None) -> BisectState:
        state_file = state_file or default_state_file
        if state_file.exists():
            with state_file.open() as f:
                return BisectState(**json.load(f))
        return BisectState("idle", [], [], [])

    @property
    def inactive_nodes(self) -> list[str]:
        return list(set(self.all) - set(self.active))

    def set_custom_node_enabled_states(self):
        if self.active:
            execute_cm_cli(["enable", *self.active])
        if self.inactive_nodes:
            execute_cm_cli(["disable", *self.inactive_nodes])

    def __str__(self):
        active_list = "\n".join([f"{i + 1:3}. {node}" for i, node in enumerate(self.active)])
        return f"""BisectState(status={self.status})
set of nodes with culprit: {len(self.range)}
set of nodes to test: {len(self.active)}
--------------------------
{active_list}"""


@bisect_app.command(
    help="Start a new bisect session with optionally pinned nodes to always enable, and optional ComfyUI launch args."
    + "?[--pinned-nodes PINNED_NODES]"
    + "?[-- <extra args ...>]"
)
def start(
    pinned_nodes: Annotated[str, typer.Option(help="Pinned nodes always enable during the bisect")] = "",
    extra: list[str] = typer.Argument(None),
):
    """Start a new bisect session. The initial state is bad with all custom nodes
    enabled, good with all custom nodes disabled."""

    if BisectState.load().status != "idle":
        typer.echo("A bisect session is already running.")
        raise typer.Exit()

    pinned_nodes = {s.strip() for s in pinned_nodes.split(",") if s}

    cm_output: str | None = execute_cm_cli(["simple-show", "enabled"])
    if cm_output is None:
        typer.echo("Failed to fetch the list of nodes.")
        raise typer.Exit()

    nodes_list = [
        line.strip()
        for line in cm_output.strip().split("\n")
        if not line.startswith("FETCH DATA") and line.strip() not in pinned_nodes
    ]
    state = BisectState(
        status="running",
        all=nodes_list,
        range=nodes_list,
        active=nodes_list,
        launch_args=extra or [],
    )
    state.save()

    typer.echo(f"Bisect session started.\n{state}")
    if pinned_nodes:
        typer.echo(f"Pinned nodes: {', '.join(pinned_nodes)}")

    bad()


@bisect_app.command(help="Mark the current active set as good, indicating the problem is outside the test set.")
def good():
    state = BisectState.load()
    if state.status != "running":
        typer.echo("No bisect session running or no active nodes to process.")
        raise typer.Exit()

    new_state = state.good()

    if new_state.status == "resolved":
        assert len(new_state.range) == 1
        typer.echo(f"Problematic node identified: {new_state.range[0]}")
        reset()
    else:
        new_state.save()
        typer.echo(new_state)
        launch_command(background=False, extra=state.launch_args)


@bisect_app.command(help="Mark the current active set as bad, indicating the problem is within the test set.")
def bad():
    state = BisectState.load()
    if state.status != "running":
        typer.echo("No bisect session running or no active nodes to process.")
        raise typer.Exit()

    new_state = state.bad()

    if new_state.status == "resolved":
        assert len(new_state.range) == 1
        typer.echo(f"Problematic node identified: {new_state.range[0]}")
        reset()
    else:
        new_state.save()
        typer.echo(new_state)
        launch_command(background=False, extra=state.launch_args)


@bisect_app.command(help="Reset the current bisect session.")
def reset():
    if default_state_file.exists():
        BisectState.load().reset()
        os.unlink(default_state_file)
        typer.echo("Bisect session reset.")
    else:
        typer.echo("No bisect session to reset.")
