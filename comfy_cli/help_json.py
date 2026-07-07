"""Machine-readable help for the Typer/Click command tree.

``comfy --help-json`` writes a single JSON document describing every command,
its options, types, defaults, and examples. The shape is the contract for
agents that drive the CLI without reading source.

Examples are looked up from ``HELP_EXAMPLES`` (a module-level dict keyed by
fully-qualified command path) so we can attach one-liners without bloating
docstrings. Phase 2 will hoist this alongside ``comfy discover`` and add a
``schemas`` field that points at each command's output schema.
"""

from __future__ import annotations

from typing import Any

import click
import typer

# One-liner examples per fully-qualified command path. Agents copy/paste.
# Keys match the full ``path`` we emit (``"comfy <subcommand> ..."``).
HELP_EXAMPLES: dict[str, list[str]] = {
    "comfy env": ["comfy env", "comfy --json env"],
    "comfy which": ["comfy which", "comfy --json which"],
    "comfy discover": [
        "comfy --json discover",
        "comfy --json discover --schemas-only",
    ],
    "comfy query": [
        "comfy --json query --query 'from nodes select name'",
        "comfy query --input object_info.json --query 'from nodes where category=\"loaders\"'",
    ],
    "comfy auth": ["comfy auth list", "comfy auth set civitai --key ..."],
    "comfy auth set": ["comfy auth set comfy-cloud --key sk-…"],
    "comfy auth list": ["comfy auth list", "comfy --json auth list"],
    "comfy auth remove": ["comfy auth remove civitai"],
    "comfy install": [
        "comfy install --nvidia",
        "comfy install --cpu --skip-manager",
    ],
    "comfy launch": [
        "comfy launch",
        "comfy launch -- --listen 0.0.0.0 --port 8188",
    ],
    "comfy run": [
        "comfy run --workflow path/to/workflow_api.json",
        "comfy --json-stream run --workflow path/to/workflow_api.json",
    ],
    "comfy stop": ["comfy stop"],
    "comfy logs": ["comfy logs", "comfy --json logs --tail 50"],
    "comfy set-default": ["comfy set-default /path/to/ComfyUI"],
    "comfy update": ["comfy update", "comfy update all"],
    "comfy standalone": ["comfy standalone"],
    "comfy feedback": ["comfy feedback"],
    "comfy model": ["comfy model download --url ..."],
    "comfy model download": [
        "comfy model download --url https://civitai.com/api/download/models/12345",
    ],
    "comfy node": ["comfy node show installed"],
    "comfy manager": ["comfy manager disable-gui"],
}


def _param_to_dict(param: click.Parameter) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": param.name,
        "param_kind": param.param_type_name,  # "option" or "argument"
        "required": bool(param.required),
        "hidden": bool(getattr(param, "hidden", False)),
        "default": _coerce_default(param.default),
        "help": getattr(param, "help", None),
        "is_flag": bool(getattr(param, "is_flag", False)),
        "multiple": bool(getattr(param, "multiple", False)),
    }
    type_info = _type_to_dict(param.type)
    info["type"] = type_info["type"]
    if "choices" in type_info:
        info["choices"] = type_info["choices"]
    # Duck-type "is this an option?" via ``param_type_name`` rather than
    # ``isinstance(param, click.Option)``: typer >= 0.13 ships ``TyperOption`` /
    # ``TyperArgument`` whose MRO is ``-> click.Parameter`` (they no longer
    # subclass click.Option/Argument), so the isinstance check silently dropped
    # ``flags``/``envvar`` from every option in the help JSON contract.
    if param.param_type_name == "option":
        info["flags"] = list(param.opts) + list(param.secondary_opts)
        info["envvar"] = param.envvar
    return info


def _type_to_dict(t: click.ParamType) -> dict[str, Any]:
    name = getattr(t, "name", t.__class__.__name__).lower()
    out: dict[str, Any] = {"type": name}
    if isinstance(t, click.Choice):
        out["choices"] = list(t.choices)
    return out


def _coerce_default(default: Any) -> Any:
    # Typer wraps defaults in OptionInfo / ArgumentInfo; unwrap.
    if hasattr(default, "default"):
        default = default.default
    if callable(default):
        return None
    if default is ... or default is None:
        return None
    try:
        import json

        json.dumps(default)
        return default
    except TypeError:
        return str(default)


def _command_to_dict(cmd: click.Command, *, path: list[str]) -> dict[str, Any]:
    fqp = " ".join(path)
    params = [_param_to_dict(p) for p in cmd.params]
    entry: dict[str, Any] = {
        "name": cmd.name,
        "path": fqp,
        "help": (cmd.help or "").strip() or None,
        "short_help": (cmd.short_help or "").strip() or None,
        "hidden": bool(getattr(cmd, "hidden", False)),
        "params": params,
        "examples": HELP_EXAMPLES.get(fqp, []),
    }
    # Duck-type the group capability instead of ``isinstance(cmd, click.Group)``.
    # typer >= 0.13 / click >= 8.2 ship a ``TyperGroup`` that no longer subclasses
    # ``click.Group`` (its MRO is ``TyperGroup -> click.Command``), so the isinstance
    # check silently returned no subcommands and produced an empty command tree.
    if hasattr(cmd, "list_commands") and hasattr(cmd, "get_command"):
        subs: dict[str, Any] = {}
        for sub_name in sorted(cmd.list_commands(ctx=None)):  # type: ignore[arg-type]
            sub = cmd.get_command(None, sub_name)  # type: ignore[arg-type]
            if sub is None:
                continue
            subs[sub_name] = _command_to_dict(sub, path=path + [sub_name])
        entry["subcommands"] = subs
    return entry


def build_help_json(app: typer.Typer, *, prog_name: str = "comfy") -> dict[str, Any]:
    """Walk the Typer app and produce a help JSON document."""
    click_command = typer.main.get_command(app)
    root = _command_to_dict(click_command, path=[prog_name])
    return {
        "prog": prog_name,
        "commands": root.get("subcommands", {}),
        "root": {
            "help": root.get("help"),
            "params": root.get("params", []),
        },
    }


def iter_command_paths(app: typer.Typer, *, prog_name: str = "comfy") -> list[str]:
    """Return every leaf-command path as a flat list. Useful for tests."""
    doc = build_help_json(app, prog_name=prog_name)
    out: list[str] = []

    def walk(node: dict[str, Any], path: list[str]) -> None:
        subs = node.get("subcommands")
        if subs:
            for name, child in subs.items():
                walk(child, path + [name])
        else:
            out.append(" ".join(path))

    walk({"subcommands": doc["commands"]}, [prog_name])
    return out
