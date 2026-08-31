"""The live `comfy build` command tree, read from Typer rather than a literal."""

from __future__ import annotations

import typer

from comfy_cli.command import build


def leaf_commands() -> set[str]:
    """Every leaf under `comfy build`, hidden ones (``blob ls``) included.

    Paths are relative to ``build`` — ``"release ls"``, not ``"comfy build release ls"``.
    """
    leaves: set[str] = set()

    def walk(commands: dict[str, object], prefix: tuple[str, ...]) -> None:
        for name, child in commands.items():
            subcommands = getattr(child, "commands", None)
            if subcommands:
                walk(subcommands, (*prefix, name))
            else:
                leaves.add(" ".join((*prefix, name)))

    walk(typer.main.get_command(build.app).commands, ())
    return leaves
