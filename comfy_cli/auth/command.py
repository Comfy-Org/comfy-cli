"""``comfy auth`` — manage API tokens for third-party model hosts.

Tokens here are used by ``comfy model download`` when fetching gated
checkpoints / LoRAs / VAEs from Civitai or Hugging Face. Stored locally,
never transmitted except to the issuing provider.

Comfy Cloud sign-in lives in a separate namespace — see ``comfy cloud``.
"""

from __future__ import annotations

from typing import Annotated

import typer

from comfy_cli import tracking
from comfy_cli.auth import store
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(
    no_args_is_help=True,
    help="Manage API tokens for model hosts (Civitai, Hugging Face).",
)


@app.command("list", help="List third-party API-key providers (civitai, huggingface).")
@tracking.track_command("auth")
def list_cmd():
    renderer = get_renderer()
    records = store.list_records()
    if renderer.is_pretty():
        _render_pretty_list(records=records)
    renderer.emit(
        {
            "providers": [r.to_dict(redact=True) for r in records],
            "supported": list(store.SUPPORTED_PROVIDERS),
            "path": str(store.secrets_path()),
            "action": "list",
        },
        command="auth list",
    )


@app.command("set", help="Set or replace the API token for a third-party model host.")
@tracking.track_command("auth")
def set_cmd(
    provider: Annotated[
        str,
        typer.Argument(help="Provider name — `civitai` or `huggingface`. Comfy Cloud uses `comfy cloud login`."),
    ],
    key: Annotated[
        str,
        typer.Option(
            "--key",
            show_default=False,
            help="The API token. Stored locally; never sent except to the provider.",
        ),
    ],
):
    renderer = get_renderer()
    if provider == "comfy-cloud":
        renderer.error(
            code="auth_use_login_for_cloud",
            message="Comfy Cloud uses OAuth — `auth set --key` is not supported for `comfy-cloud`.",
            hint="run: comfy cloud login   (or `comfy cloud set-key` for the API-key path)",
            details={"provider": provider},
        )
        raise typer.Exit(code=1)
    if not key:
        renderer.error(code="auth_invalid_key", message="--key cannot be empty.")
        raise typer.Exit(code=1)
    try:
        record = store.set(provider, key)
    except ValueError as e:
        renderer.error(code="auth_invalid_key", message=str(e))
        raise typer.Exit(code=1)
    if renderer.is_pretty():
        rprint(f"[bold green]Stored token for {record.provider}[/bold green] ({record.to_dict()['key']})")
        if provider not in store.SUPPORTED_PROVIDERS:
            rprint(f"[yellow]Note:[/yellow] {provider!r} is not a well-known provider; stored anyway.")
    renderer.emit(
        {
            "providers": [r.to_dict(redact=True) for r in store.list_records()],
            "supported": list(store.SUPPORTED_PROVIDERS),
            "path": str(store.secrets_path()),
            "action": "set",
        },
        command="auth set",
        changed=True,
    )


@app.command("remove", help="Remove a stored third-party API token.")
@tracking.track_command("auth")
def remove_cmd(
    provider: Annotated[str, typer.Argument(help="Provider name.")],
):
    renderer = get_renderer()
    if provider == "comfy-cloud":
        renderer.error(
            code="auth_use_logout_for_cloud",
            message="Comfy Cloud uses OAuth — use `comfy cloud logout` to clear the session.",
            hint="run: comfy cloud logout",
            details={"provider": provider},
        )
        raise typer.Exit(code=1)
    removed = store.remove(provider)
    if not removed:
        renderer.error(
            code="auth_not_found",
            message=f"No stored key for {provider!r}.",
            hint="run: comfy auth list",
            details={"provider": provider},
        )
        raise typer.Exit(code=1)
    if renderer.is_pretty():
        rprint(f"[bold]Removed token for {provider}[/bold]")
    renderer.emit(
        {
            "providers": [r.to_dict(redact=True) for r in store.list_records()],
            "supported": list(store.SUPPORTED_PROVIDERS),
            "path": str(store.secrets_path()),
            "action": "remove",
        },
        command="auth remove",
        changed=True,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _render_pretty_list(*, records):
    """Pretty-render the auth list (third-party provider tokens only)."""
    from rich.console import Group
    from rich.text import Text

    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.output.branding import branded_panel
    from comfy_cli.output.panels import auth_empty_panel, auth_list_table

    renderer = get_renderer()
    path = str(store.secrets_path())

    if not records:
        # Empty-state has its own panel; brand it via the canonical wrapper.
        body = auth_empty_panel(supported=list(store.SUPPORTED_PROVIDERS), path=path)
    else:
        redacted = [r.to_dict(redact=True) for r in records]
        body = auth_list_table(redacted, supported=list(store.SUPPORTED_PROVIDERS), path=path)

    hint = Text("Comfy Cloud sign-in lives under `comfy cloud whoami`.", style="dim")
    group = Group(body, Text(""), hint)

    renderer.console().print(branded_panel(group, title="auth", version=ConfigManager().get_cli_version()))
