"""``comfy_cli._lazy``: the module proxy and the lazy root group.

``LazyModule`` exists so ``cmdline.py`` can keep flat module-level names that
tests patch (``patch("comfy_cli.cmdline.run_inner.execute")``) while the
import is deferred. The tests below pin exactly that: patching through a
proxy patches the real module, and the lazy group renders help identically
to ``add_typer``.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from comfy_cli._lazy import LazyCommand, LazyModule, LazySubcommand, LazyTyperGroup


def _fake_module(monkeypatch, name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


# --- LazyModule -------------------------------------------------------------


def test_lazy_module_imports_on_first_attribute_only(monkeypatch):
    real = types.ModuleType("_lazy_probe")
    real.answer = 41
    calls: list[str] = []
    original = importlib.import_module

    def spy(name, package=None):
        calls.append(name)
        if name == "_lazy_probe":
            return real
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", spy)
    proxy = LazyModule("_lazy_probe")
    assert calls == []
    assert proxy.answer == 41
    assert calls == ["_lazy_probe"]
    _ = proxy.answer
    assert calls == ["_lazy_probe"], "second access must reuse the imported module"


def test_lazy_module_forwards_set_delete_and_dict(monkeypatch):
    real = _fake_module(monkeypatch, "_lazy_probe2", value=1)
    proxy = LazyModule("_lazy_probe2")
    proxy.value = 2
    assert real.value == 2
    proxy.extra = "x"
    assert real.extra == "x"
    del proxy.extra
    assert not hasattr(real, "extra")
    assert vars(proxy) is real.__dict__
    assert "value" in dir(proxy)


def test_mock_patch_through_proxy_patches_the_real_module(monkeypatch):
    real = _fake_module(monkeypatch, "_lazy_probe3", run=lambda: "real")
    holder = _fake_module(monkeypatch, "_lazy_holder", inner=LazyModule("_lazy_probe3"))
    with patch("_lazy_holder.inner.run", return_value="patched"):
        assert real.run() == "patched"
        assert holder.inner.run() == "patched"
    assert real.run() == "real"


def test_monkeypatch_setattr_on_proxy_lands_on_the_real_module(monkeypatch):
    real = _fake_module(monkeypatch, "_lazy_probe4", run=lambda: "real")
    proxy = LazyModule("_lazy_probe4")
    monkeypatch.setattr(proxy, "run", lambda: "patched")
    assert real.run() == "patched"
    monkeypatch.undo()
    assert real.run() == "real"


# --- LazyTyperGroup ---------------------------------------------------------


def _sub_app(label: str) -> typer.Typer:
    sub = typer.Typer()

    @sub.command()
    def hello():
        typer.echo(f"hello from {label}")

    return sub


def _root_app(cls) -> typer.Typer:
    app = typer.Typer(cls=cls)

    @app.callback()
    def _root():
        pass

    @app.command()
    def eager():
        typer.echo("eager")

    return app


def _lazy_root(table: dict[str, LazySubcommand]) -> typer.Typer:
    class Root(LazyTyperGroup):
        lazy_subcommands = table

    return _root_app(Root)


def test_list_commands_keeps_eager_first_then_table_order(monkeypatch):
    _fake_module(monkeypatch, "_lz_a", app=_sub_app("a"))
    _fake_module(monkeypatch, "_lz_b", app=_sub_app("b"))
    app = _lazy_root({"zeta": LazySubcommand("_lz_a"), "alpha": LazySubcommand("_lz_b", hidden=True)})
    group = typer.main.get_command(app)
    assert group.list_commands(None) == ["eager", "zeta", "alpha"]


def test_group_module_is_imported_only_when_looked_up(monkeypatch):
    app = _lazy_root({"late": LazySubcommand("_lz_missing")})
    group = typer.main.get_command(app)
    assert "late" in group.list_commands(None)  # listing must not import
    _fake_module(monkeypatch, "_lz_missing", app=_sub_app("late"))
    cmd = group.get_command(None, "late")
    assert hasattr(cmd, "commands"), cmd
    assert cmd.name == "late"
    assert group.get_command(None, "late") is cmd, "built once, then cached"
    assert group.get_command(None, "nope") is None


def test_lazy_command_renders_exactly_like_app_command(monkeypatch):
    def greet(name: str = typer.Argument("world")):
        """Say hello."""
        typer.echo(f"hi {name}")

    def register_with(parent: typer.Typer) -> None:
        @parent.command(name="reg", help="Registered by hook.", context_settings={"allow_extra_args": True})
        def _reg(ctx: typer.Context):
            typer.echo(f"reg {list(ctx.args)}")

    _fake_module(monkeypatch, "_lz_d", greet=greet, register_with=register_with)
    lazy = _lazy_root(
        {
            "greet": LazyCommand("_lz_d", attr="greet", help="Greet someone."),
            "reg": LazyCommand("_lz_d", register="register_with"),
        }
    )
    eager = _root_app(None)
    eager.command("greet", help="Greet someone.")(greet)
    register_with(eager)

    runner = CliRunner()
    for argv in (["--help"], ["greet", "--help"], ["reg", "--help"], ["greet", "bob"], ["reg", "--", "x", "y"]):
        assert runner.invoke(lazy, argv).output == runner.invoke(eager, argv).output, argv
    assert "hi bob" in runner.invoke(lazy, ["greet", "bob"]).output
    assert "reg ['x', 'y']" in runner.invoke(lazy, ["reg", "--", "x", "y"]).output


def test_lazy_command_hidden_applies_on_register_branch_too(monkeypatch):
    def register_with(parent: typer.Typer) -> None:
        @parent.command(name="reg")
        def _reg():
            typer.echo("reg")

    _fake_module(monkeypatch, "_lz_e", register_with=register_with)
    lazy = _lazy_root({"reg": LazyCommand("_lz_e", register="register_with", hidden=True)})
    group = typer.main.get_command(lazy)
    cmd = group.get_command(None, "reg")
    assert cmd.hidden is True


def _callback_only_app() -> typer.Typer:
    """A Typer app that is one callback with a positional and an option —
    the ``code-search`` shape."""
    sub = typer.Typer()

    @sub.callback(invoke_without_command=True)
    def _only(query: str, repo: str = typer.Option(None, "--repo")):
        typer.echo(f"q={query} repo={repo}")

    return sub


def test_lazy_subcommand_refuses_a_callback_only_app(monkeypatch):
    """Mounting a single-callback app as a group is always a wiring bug: click
    then demands a nested COMMAND and rejects options after the positional.
    Fail at build time, naming LazyCommand, instead of shipping that parse."""
    _fake_module(monkeypatch, "_lz_cb", app=_callback_only_app())
    group = typer.main.get_command(_lazy_root({"one": LazySubcommand("_lz_cb")}))
    with pytest.raises(TypeError, match="LazyCommand"):
        group.get_command(None, "one")


def test_lazy_subcommand_refuses_a_name_that_is_not_a_typer_app(monkeypatch):
    """``spec.attr`` pointed at the callback itself (or anything else that
    isn't a Typer app) should raise a message naming the actual mis-wiring,
    not a bare AttributeError from touching registered_commands."""
    _fake_module(monkeypatch, "_lz_notapp", only=lambda: None)
    group = typer.main.get_command(_lazy_root({"one": LazySubcommand("_lz_notapp", attr="only")}))
    with pytest.raises(TypeError, match="not a typer.Typer instance"):
        group.get_command(None, "one")


def test_lazy_subcommand_refuses_a_truly_empty_app(monkeypatch):
    """An app with no commands, no subgroups, and no callback at all has
    nothing to mount — distinct wording from the callback-only case, since
    there is no callback to redirect the author toward."""
    _fake_module(monkeypatch, "_lz_empty", app=typer.Typer())
    group = typer.main.get_command(_lazy_root({"one": LazySubcommand("_lz_empty")}))
    with pytest.raises(TypeError, match="no callback"):
        group.get_command(None, "one")


def test_lazy_subcommand_accepts_a_nested_lazy_group_with_a_custom_cls(monkeypatch):
    """A sub-app that is itself a nested LazyTyperGroup with no eager commands
    of its own has empty registered_commands/registered_groups by design —
    its real commands build on lookup. The guard must tell that apart from a
    bare callback by checking for the custom ``cls``, not just emptiness."""

    class InnerRoot(LazyTyperGroup):
        lazy_subcommands = {"hello": LazyCommand("_lz_inner", attr="hello")}

    inner = typer.Typer(cls=InnerRoot)
    _fake_module(monkeypatch, "_lz_inner", hello=lambda: typer.echo("hello"))
    _fake_module(monkeypatch, "_lz_nested", app=inner)
    group = typer.main.get_command(_lazy_root({"nested": LazySubcommand("_lz_nested")}))
    built = group.get_command(None, "nested")
    assert built is not None
    assert built.get_command(None, "hello") is not None


def test_lazy_subcommand_accepts_a_custom_cls_set_on_the_callback(monkeypatch):
    """``cls`` can be declared on ``@sub_app.callback(cls=...)`` instead of on
    ``typer.Typer(cls=...)``. ``typer.main.solve_typer_info_defaults`` reads the
    callback's value first, so a sub-app with a plain ``typer.Typer()`` but a
    custom ``cls`` on its callback is a real custom group too, not a bare
    callback — the guard must check both, not just ``sub_app.info.cls``."""

    class InnerRoot(LazyTyperGroup):
        lazy_subcommands = {"hello": LazyCommand("_lz_inner2", attr="hello")}

    inner = typer.Typer()

    @inner.callback(cls=InnerRoot)
    def _inner_callback():
        pass

    _fake_module(monkeypatch, "_lz_inner2", hello=lambda: typer.echo("hello"))
    _fake_module(monkeypatch, "_lz_nested2", app=inner)
    group = typer.main.get_command(_lazy_root({"nested": LazySubcommand("_lz_nested2")}))
    built = group.get_command(None, "nested")
    assert built is not None
    assert built.get_command(None, "hello") is not None


def test_lazy_command_accepts_options_on_either_side_of_the_positional(monkeypatch):
    """The regression this guard protects: with the callback mounted as a
    LazyCommand, every flag order parses to the same result."""

    def only(query: str, repo: str = typer.Option(None, "--repo")):
        typer.echo(f"q={query} repo={repo}")

    _fake_module(monkeypatch, "_lz_cb2", only=only)
    lazy = _lazy_root({"one": LazyCommand("_lz_cb2", attr="only")})

    runner = CliRunner()
    for argv in (["one", "Q", "--repo", "X"], ["one", "--repo", "X", "Q"]):
        result = runner.invoke(lazy, argv)
        assert result.exit_code == 0, (argv, result.output)
        assert "q=Q repo=X" in result.output, argv


def test_every_mounted_root_subcommand_builds():
    """Walk the real root table so a future mis-mount fails in CI rather than
    in an installed CLI. Building is what runs the guard above."""
    from click import Context

    from comfy_cli.cmdline import app as root_app

    root_command = typer.main.get_command(root_app)
    ctx = Context(root_command)
    for name in root_command.lazy_subcommands:
        assert root_command.get_command(ctx, name) is not None, name


def test_lazy_group_renders_exactly_like_add_typer(monkeypatch):
    seen: list[str] = []
    sub = _sub_app("c")
    _fake_module(monkeypatch, "_lz_c", app=sub)

    def on_hidden():
        seen.append("cb")

    lazy = _lazy_root(
        {
            "vis": LazySubcommand("_lz_c", help="Visible help."),
            "hid": LazySubcommand("_lz_c", hidden=True, callback=on_hidden),
        }
    )
    eager = _root_app(None)
    eager.add_typer(sub, name="vis", help="Visible help.")
    eager.add_typer(sub, name="hid", hidden=True, callback=on_hidden)

    runner = CliRunner()
    assert runner.invoke(lazy, ["--help"]).output == runner.invoke(eager, ["--help"]).output
    assert runner.invoke(lazy, ["vis", "--help"]).output == runner.invoke(eager, ["vis", "--help"]).output
    out = runner.invoke(lazy, ["--help"]).output
    assert "Visible help." in out
    assert "hid" not in out

    result = runner.invoke(lazy, ["hid", "hello"])
    assert result.exit_code == 0, result.output
    assert "hello from c" in result.output
    assert seen == ["cb"], "add_typer(callback=...) semantics: the group callback runs before the subcommand"


def test_factory_kwargs_track_the_installed_typer_signature():
    """typer 0.20 added ``suggest_commands`` as a required, defaultless keyword;
    typer 0.12 has no such parameter. Pass exactly what this version accepts."""
    group = typer.main.get_command(_lazy_root({}))
    for factory in (typer.main.get_command_from_info, typer.main.get_group_from_info):
        params = inspect.signature(factory).parameters
        kwargs = group._factory_kwargs(factory)
        assert set(kwargs) <= set(params), factory
        assert ("suggest_commands" in kwargs) == ("suggest_commands" in params), factory


def test_root_group_reads_pretty_exceptions_short_off_the_app():
    from comfy_cli import cmdline

    assert cmdline._RootGroup.pretty_exceptions_short == cmdline.app.pretty_exceptions_short
