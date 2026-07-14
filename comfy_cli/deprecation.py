"""Shared helper for wiring hidden, deprecated command aliases.

A deprecation alias keeps an old command/group spelling working while steering
users to the new one. Three things happen to an alias:

  (a) it is hidden from ``--help`` (so the surface advertises only the new name);
  (b) its group help is prefixed with a ``[DEPRECATED — use <new>]`` banner; and
  (c) invoking any leaf under it prints a single ``[yellow]`` warning line to
      stderr.

The warning is never silenced — discoverability of the rename is the whole
point, and stderr keeps it out of the one-envelope-on-stdout contract.

Introduced for the ``models`` → ``model`` merge (BE-2999); the sibling noun
consolidations reuse :func:`add_deprecated_alias`.
"""

from __future__ import annotations

import typer

# Use the output shim so the warning lands on stderr (not stdout) in JSON mode,
# preserving the one-envelope-on-stdout contract.
from comfy_cli.output import rprint


def deprecated_help(new_name: str, original_help: str | None = None) -> str:
    """Prefix ``original_help`` with the deprecation banner.

    ``new_name`` is the command path *without* the ``comfy`` prog prefix
    (e.g. ``"model"``); the banner renders it as ``comfy <new_name>``.
    """
    banner = f"[DEPRECATED — use `comfy {new_name}`]"
    original = (original_help or "").strip()
    return f"{banner} {original}".strip()


def warn_deprecated(old_name: str, new_name: str) -> None:
    """Emit the one-line yellow deprecation warning to stderr."""
    rprint(f"[yellow]'comfy {old_name} …' is deprecated; use 'comfy {new_name} …'[/yellow]")


def add_deprecated_alias(
    parent: typer.Typer,
    source_app: typer.Typer,
    *,
    old_name: str,
    new_name: str,
) -> typer.Typer:
    """Mount a hidden, deprecated alias of ``source_app`` on ``parent``.

    The alias reuses ``source_app``'s command implementations verbatim (no logic
    duplication) and adds a group callback that prints one yellow warning to
    stderr before any leaf runs. ``source_app`` itself is left untouched — the
    alias is a fresh Typer that borrows the registered commands — so the canonical
    mount of the same commands stays warning-free.

    Returns the alias app (useful for tests).
    """
    alias = typer.Typer(
        no_args_is_help=True,
        help=deprecated_help(new_name, source_app.info.help),
    )
    # Borrow the exact command registrations — same CommandInfo objects, so the
    # alias and the canonical surface stay in lockstep with zero duplication.
    alias.registered_commands.extend(source_app.registered_commands)

    @alias.callback()
    def _emit_deprecation_warning() -> None:
        warn_deprecated(old_name, new_name)

    parent.add_typer(alias, name=old_name, hidden=True)
    return alias
