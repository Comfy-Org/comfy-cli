"""Shared helper for wiring hidden, deprecated command aliases.

A deprecation alias keeps an old command/group spelling working while steering
users to the new one. Three things happen to an alias:

  (a) it is hidden from ``--help`` (so the surface advertises only the new name);
  (b) its group help is prefixed with a ``[DEPRECATED — use <new>]`` banner; and
  (c) invoking any leaf under it prints a single ``[yellow]`` warning line via
      :func:`rprint` — stdout in pretty mode, stderr in JSON/NDJSON mode.

The warning is never silenced — discoverability of the rename is the whole
point — and in JSON mode it lands on stderr, keeping it out of the
one-envelope-on-stdout contract.

Introduced for the ``models`` → ``model`` merge; the sibling noun
consolidations reuse :func:`add_deprecated_alias`.
"""

from __future__ import annotations

import functools
import typing

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
    """Emit the one-line yellow deprecation warning (stderr in JSON mode)."""
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

    Both leaf commands *and* nested sub-groups (``add_typer``) are carried over,
    and if ``source_app`` declares its own group callback the alias composes it
    with the deprecation warning rather than dropping it — so the helper stays
    correct as the sibling noun consolidations reuse it against richer command
    trees.

    Returns the alias app (useful for tests).
    """
    alias = typer.Typer(
        no_args_is_help=True,
        help=deprecated_help(new_name, source_app.info.help),
    )
    # Borrow the exact command + sub-group registrations — same CommandInfo /
    # TyperInfo objects, so the alias and the canonical surface stay in lockstep
    # with zero duplication.
    alias.registered_commands.extend(source_app.registered_commands)
    alias.registered_groups.extend(source_app.registered_groups)

    # The alias always needs a group callback to emit the deprecation warning.
    # If the source app declared one of its own (with its own options / setup),
    # compose the two — warn first, then delegate — preserving the source's
    # callback signature so its CLI options still resolve on the alias.
    source_cb_info = source_app.registered_callback
    source_cb = getattr(source_cb_info, "callback", None) if source_cb_info else None

    if source_cb is None:

        @alias.callback()
        def _emit_deprecation_warning() -> None:
            warn_deprecated(old_name, new_name)
    else:

        @alias.callback()
        @functools.wraps(source_cb)
        def _emit_deprecation_warning(*args, **kwargs):
            warn_deprecated(old_name, new_name)
            return source_cb(*args, **kwargs)

        # ``functools.wraps`` copied ``source_cb``'s *string* annotations (source
        # modules use ``from __future__ import annotations``) but the wrapper's
        # ``__globals__`` still point at this module. Typer resolves a callback's
        # hints with ``get_type_hints``, so any source-local type would be looked
        # up in ``deprecation.py`` and raise ``NameError`` at CLI startup.
        # Pre-resolve the hints against ``source_cb``'s own globals and store the
        # concrete types (``include_extras`` keeps typer.Option/Argument metadata).
        try:
            _emit_deprecation_warning.__annotations__ = typing.get_type_hints(source_cb, include_extras=True)
        except Exception:
            # Unresolvable source hints are a problem with the source callback
            # itself (it would fail canonically too), not with this alias — leave
            # the copied annotations untouched so behaviour matches the canonical
            # mount rather than masking the error here.
            pass

    parent.add_typer(alias, name=old_name, hidden=True)
    return alias
