"""Lazy-import helpers for the CLI's startup path.

Every ``comfy`` invocation pays ``cmdline.py``'s import graph before doing
any work, and agents pay it on every tool call. Most of that graph is the
subcommand modules, of which a single invocation uses one. These helpers let
``cmdline.py`` keep its flat, module-level names (tests patch
``comfy_cli.cmdline.run_inner`` and friends) while deferring the actual import
to first use.

Three pieces:

* :class:`LazyModule` — a module proxy. Attribute reads import the module on
  first touch; reads, writes and deletes all forward to the real module, so
  ``patch("comfy_cli.cmdline.run_inner.execute")`` patches the real function
  exactly as it did when ``run_inner`` was the module itself.
* :func:`lazy_attr` — a callable proxy for one name in a module
  (``EnvChecker``, ``launch``). Resolved on every call, so a patch applied to
  the source module is honoured too.
* :class:`LazyTyperGroup` — a ``TyperGroup`` whose subgroups are declared as
  ``(module, attr)`` pairs and built on first lookup with typer's own
  ``get_group_from_info``, so help text, hidden flags and callbacks behave
  exactly as ``app.add_typer(...)`` did. ``list_commands`` keeps the eager
  commands first and the lazy groups in table order, so ``comfy --help`` is
  byte-identical to the eager wiring.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import click
import typer
from typer.core import TyperGroup
from typer.models import CommandInfo, TyperInfo


class LazyModule:
    """Proxy for a module that is imported on first attribute access.

    ``get``/``set``/``del`` of attributes all forward to the real module, and
    ``__dict__`` is the real module's namespace, so ``unittest.mock.patch``
    and ``monkeypatch.setattr`` treat the proxy as the module.
    """

    __slots__ = ("_lazy_path", "_lazy_target")

    def __init__(self, path: str) -> None:
        object.__setattr__(self, "_lazy_path", path)
        object.__setattr__(self, "_lazy_target", None)

    def _lazy_resolve(self) -> ModuleType:
        target = object.__getattribute__(self, "_lazy_target")
        if target is None:
            target = importlib.import_module(object.__getattribute__(self, "_lazy_path"))
            object.__setattr__(self, "_lazy_target", target)
        return target

    @property
    def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
        return self._lazy_resolve().__dict__

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lazy_resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._lazy_resolve(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._lazy_resolve(), name)

    def __dir__(self) -> list[str]:
        return dir(self._lazy_resolve())

    def __repr__(self) -> str:
        return f"<LazyModule {object.__getattribute__(self, '_lazy_path')!r}>"


def lazy_module(path: str) -> Any:
    """``run_inner = lazy_module("comfy_cli.command.run")``."""
    return LazyModule(path)


class _LazyAttr:
    __slots__ = ("_path", "_name")

    def __init__(self, path: str, name: str) -> None:
        self._path = path
        self._name = name

    def _resolve(self) -> Any:
        return getattr(importlib.import_module(self._path), self._name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return f"<lazy_attr {self._path}.{self._name}>"


def lazy_attr(path: str, name: str) -> Any:
    """``EnvChecker = lazy_attr("comfy_cli.env_checker", "EnvChecker")``.

    Only for names that are *called*. Do not use for typer ``callback=``
    functions: typer inspects the callback's signature at decoration time and
    a proxy has none.
    """
    return _LazyAttr(path, name)


@dataclass(frozen=True)
class LazySubcommand:
    """One deferred ``app.add_typer(...)``: the module holding the sub-``Typer``
    and the ``add_typer`` keyword arguments that were passed with it."""

    module: str
    attr: str = "app"
    help: str | None = None
    hidden: bool = False
    callback: Callable[..., Any] | None = None


@dataclass(frozen=True)
class LazyCommand:
    """One deferred ``app.command(name, ...)(fn)``. ``attr`` names the function
    in ``module``. ``register`` instead names a ``register_with(app)`` hook on
    the module that registers the command onto a Typer app itself (``generate``
    does this because its command is a closure)."""

    module: str
    attr: str | None = None
    register: str | None = None
    help: str | None = None
    context_settings: dict[str, Any] | None = None


class LazyTyperGroup(TyperGroup):
    """Root group with subcommands declared in ``lazy_subcommands`` and built
    on first lookup. Set the table on a subclass::

        class RootGroup(LazyTyperGroup):
            lazy_subcommands = {"nodes": LazySubcommand("comfy_cli.command.nodes", help="...")}

        app = typer.Typer(cls=RootGroup)
    """

    lazy_subcommands: dict[str, LazySubcommand | LazyCommand] = {}
    # Mirrors ``typer.Typer()``'s default; the root app in cmdline.py uses it.
    pretty_exceptions_short: bool = True

    def __init__(self, **attrs: Any) -> None:
        super().__init__(**attrs)
        self._lazy_loaded: dict[str, click.Command] = {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        eager = list(super().list_commands(ctx))
        return eager + [name for name in self.lazy_subcommands if name not in self.commands]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        spec = self.lazy_subcommands.get(cmd_name)
        if spec is None:
            return None
        if cmd_name not in self._lazy_loaded:
            build = self._build_command if isinstance(spec, LazyCommand) else self._build_group
            self._lazy_loaded[cmd_name] = build(cmd_name, spec)
        return self._lazy_loaded[cmd_name]

    def _build_command(self, name: str, spec: LazyCommand) -> click.Command:
        module = importlib.import_module(spec.module)
        if spec.register is not None:
            scratch = typer.Typer()
            getattr(module, spec.register)(scratch)
            infos = [info for info in scratch.registered_commands if info.name == name]
            assert len(infos) == 1, f"{spec.module}.{spec.register} did not register exactly one {name!r} command"
            info = infos[0]
        else:
            assert spec.attr, "LazyCommand needs attr or register"
            info = CommandInfo(
                name=name,
                callback=getattr(module, spec.attr),
                help=spec.help,
                context_settings=spec.context_settings,
            )
        return typer.main.get_command_from_info(
            info,
            pretty_exceptions_short=self.pretty_exceptions_short,
            rich_markup_mode=self.rich_markup_mode,
        )

    def _build_group(self, name: str, spec: LazySubcommand) -> click.Command:
        sub_app = getattr(importlib.import_module(spec.module), spec.attr)
        # Same TyperInfo ``add_typer`` would have appended: only the keywords
        # that were given, so the sub-app's own settings fill the rest.
        kwargs: dict[str, Any] = {"name": name}
        if spec.help is not None:
            kwargs["help"] = spec.help
        if spec.hidden:
            kwargs["hidden"] = True
        if spec.callback is not None:
            kwargs["callback"] = spec.callback
        return typer.main.get_group_from_info(
            TyperInfo(sub_app, **kwargs),
            pretty_exceptions_short=self.pretty_exceptions_short,
            rich_markup_mode=self.rich_markup_mode,
        )
