"""``comfy cloud`` — sign in to Comfy Cloud (OAuth) and manage cloud routing config.

The ``cloud`` namespace is the explicit prefix for everything that talks to
Comfy Cloud:

- ``cloud login``       — Authorization Code + PKCE flow against the cloud
- ``cloud logout``      — clear the local OAuth session
- ``cloud whoami``      — inspect the current session + base URL + auth path
- ``cloud set-base-url`` — pin a non-prod env (PR preview, staging, …)
- ``cloud set-key``     — testing path: persist a Comfy Cloud API key
                            (canonical sign-in is ``cloud login``)

Third-party API tokens (Civitai, Hugging Face) for ``comfy model download``
live under ``comfy auth`` since they're not cloud-specific.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

import typer

from comfy_cli import tracking
from comfy_cli.auth import store
from comfy_cli.cloud import CLIENT_ID, CLIENT_NAME, CONFIG_KEY_BASE_URL, get_base_url, get_resource_url, get_scopes
from comfy_cli.cloud.oauth import OAuthError, run_login
from comfy_cli.config_manager import ConfigManager
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(
    no_args_is_help=True,
    help="Comfy Cloud — sign in, route commands, inspect session.",
)


# ---------------------------------------------------------------------------
# login / logout / whoami
# ---------------------------------------------------------------------------


@app.command("login", help="Sign in to Comfy Cloud via your browser (OAuth + PKCE).")
@tracking.track_command("cloud")
def login_cmd(
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Don't try to open the browser automatically — just print the URL.",
        ),
    ] = False,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            help="Seconds to wait for the browser callback before giving up.",
        ),
    ] = 300,
):
    renderer = get_renderer()
    client_id = CLIENT_ID

    base_url = get_base_url()
    if renderer.is_pretty():
        rprint(f"Signing in to [bold cyan]Comfy Cloud[/bold cyan] ([dim]{base_url}[/dim])")

    def _on_url(url: str) -> None:
        if not renderer.is_pretty():
            return
        if no_browser:
            rprint("\nOpen this URL in your browser to sign in:")
            rprint(f"  [cyan]{url}[/cyan]\n")
        else:
            rprint("[dim]Opening browser… (if it doesn't appear, copy this URL)[/dim]")
            rprint(f"[dim]  {url}[/dim]")

    try:
        result = run_login(
            base_url=base_url,
            resource=get_resource_url(),
            scopes=get_scopes(),
            client_id=client_id,
            client_name=CLIENT_NAME,
            open_browser=not no_browser,
            timeout_s=float(timeout),
            on_url_ready=_on_url,
        )
    except OAuthError as e:
        renderer.error(
            code=e.code,
            message=str(e),
            hint=e.hint,
            details=e.details,
        )
        raise typer.Exit(code=1)

    session = store.save_cloud_session(
        base_url=result.base_url,
        resource=result.resource,
        client_id=result.client_id,
        scope=result.scope,
        access_token=result.tokens.access_token,
        refresh_token=result.tokens.refresh_token,
        token_type=result.tokens.token_type,
        expires_at=result.tokens.expires_at,
    )

    if renderer.is_pretty():
        from comfy_cli.output.branding import welcome_banner

        renderer.console().print("")
        renderer.console().print(
            welcome_banner(
                base_url=session.base_url,
                scope=session.scope,
                client_id=session.client_id,
                expires_at=session.expires_at,
                cta="comfy run --workflow <path> --where cloud",
            )
        )

    renderer.emit(
        {
            "session": session.to_dict(redact=True),
            "action": "login",
        },
        command="cloud login",
        changed=True,
    )


@app.command("logout", help="Sign out of Comfy Cloud (clears the local OAuth session).")
@tracking.track_command("cloud")
def logout_cmd():
    renderer = get_renderer()
    removed = store.clear_cloud_session()
    if not removed:
        renderer.error(
            code="auth_not_signed_in",
            message="No active Comfy Cloud session.",
            hint="run: comfy cloud login",
        )
        raise typer.Exit(code=1)
    if renderer.is_pretty():
        rprint("[bold]Signed out of Comfy Cloud.[/bold]")
    renderer.emit({"action": "logout"}, command="cloud logout", changed=True)


@app.command("whoami", help="Show the current Comfy Cloud sign-in status.")
@tracking.track_command("cloud")
def whoami_cmd():
    from comfy_cli.credentials import find_api_key, get_session

    renderer = get_renderer()
    # Refresh first so we report (and persist) a live token rather than a
    # token that quietly lapsed since the last command.
    session = get_session(refresh=True)
    configured_base_url = get_base_url()

    # Ambient API key (env / store), reported even when an OAuth session
    # outranks it. ``Credential.source`` is "env:<VAR>" / "stored:<provider>";
    # keep the short "env" / "store" labels the whoami envelope documents.
    ambient_key = find_api_key(purpose="cloud")
    api_key_source: str | None = None
    if ambient_key is not None:
        api_key_source = "env" if ambient_key.source.startswith("env:") else "store"

    # `signed_in` reflects *any* valid auth path — OAuth session or API key.
    # When both are present, OAuth wins (it's the credential the client sends);
    # the API key is only used as a fallback when no live session exists.
    has_api_key = api_key_source is not None
    has_oauth = session is not None
    expired = session.is_expired() if session else None
    oauth_usable = has_oauth and not expired
    # `signed_in` must reflect whether we can actually authenticate right now:
    # a live (non-expired) OAuth session, or an API key. An expired,
    # unrefreshable OAuth session is NOT signed in on its own.
    signed_in = oauth_usable or has_api_key
    auth_method: str | None
    if oauth_usable:
        auth_method = "oauth"
    elif has_api_key:
        auth_method = "api_key"
    else:
        auth_method = None
    stale_base_url = (session.base_url != configured_base_url) if session else False

    if renderer.is_pretty():
        if has_oauth:
            from comfy_cli.output.branding import whoami_banner

            renderer.console().print(
                whoami_banner(
                    base_url=session.base_url,
                    scope=session.scope,
                    client_id=session.client_id,
                    expires_at=session.expires_at,
                    expired=bool(expired),
                    version=ConfigManager().get_cli_version(),
                )
            )
            if stale_base_url:
                rprint(
                    f"[yellow]Session was minted for [bold]{session.base_url}[/bold] "
                    f"but current configured base URL is [bold]{configured_base_url}[/bold].[/yellow]"
                )
                rprint("[dim]→ run `comfy cloud login` to mint a fresh session against the new base URL.[/dim]")
            if has_api_key:
                if oauth_usable:
                    rprint(
                        f"[dim]OAuth session is preferred; X-API-Key from {api_key_source} present but unused.[/dim]"
                    )
                else:
                    rprint(f"[dim]OAuth session expired — falling back to X-API-Key from {api_key_source}.[/dim]")
        else:
            from comfy_cli.output.branding import signed_out_banner

            renderer.console().print(
                signed_out_banner(
                    base_url=configured_base_url,
                    version=ConfigManager().get_cli_version(),
                )
            )
            if has_api_key:
                rprint(f"[dim]Authenticated via X-API-Key from {api_key_source}.[/dim]")

    payload: dict[str, object | None] = {
        "signed_in": signed_in,
        "auth_method": auth_method,
        "base_url": session.base_url if session else configured_base_url,
        "configured_base_url": configured_base_url,
        "api_key_source": api_key_source,
    }
    if session is not None:
        payload["expired"] = expired
        payload["session"] = session.to_dict(redact=True)
        payload["stale_base_url"] = stale_base_url

    renderer.emit(payload, command="cloud whoami")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@app.command("set-base-url", help="Persist a custom Comfy Cloud base URL (e.g. a PR-preview env).")
@tracking.track_command("cloud")
def set_base_url_cmd(
    url: Annotated[
        str | None,
        typer.Argument(
            help="Base URL like 'https://fe-pr-12159.testenvs.comfy.org'. Pass '' or omit and use --clear to reset.",
        ),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="Clear the persisted base URL (falls back to env var, then default)."),
    ] = False,
):
    renderer = get_renderer()
    config = ConfigManager()
    if clear or not url:
        if clear:
            config.set(CONFIG_KEY_BASE_URL, "")
        if renderer.is_pretty():
            rprint(f"[bold]Cloud base URL[/bold] → [cyan]{get_base_url()}[/cyan] (config cleared)")
        renderer.emit(
            {"base_url": get_base_url(), "config_set": False},
            command="cloud set-base-url",
            changed=clear,
        )
        return
    cleaned = url.rstrip("/")
    config.set(CONFIG_KEY_BASE_URL, cleaned)

    from comfy_cli.credentials import get_session

    cleared_stale_session = False
    session = get_session(refresh=False)
    if session is not None and session.base_url != cleaned:
        if session.is_expired():
            store.clear_cloud_session()
            cleared_stale_session = True

    if renderer.is_pretty():
        rprint(f"[bold green]Persisted cloud base URL[/bold green] → [cyan]{cleaned}[/cyan]")
        if cleared_stale_session:
            rprint("[dim]Cleared an expired session that was pinned to the old URL.[/dim]")
        elif session is not None and session.base_url != cleaned:
            rprint(
                f"[yellow]Warning:[/yellow] an active session is still pinned to "
                f"[bold]{session.base_url}[/bold]. Run `comfy cloud logout` then "
                f"`comfy cloud login` to mint a fresh one against {cleaned}."
            )
        rprint("[dim]Next `comfy cloud login` will use this URL.[/dim]")
    renderer.emit(
        {
            "base_url": cleaned,
            "config_set": True,
            "cleared_stale_session": cleared_stale_session,
        },
        command="cloud set-base-url",
        changed=True,
    )


@app.command(
    "set-key",
    hidden=True,
    help="Persist a Comfy Cloud API key (testing path; canonical sign-in is `comfy cloud login`).",
)
@tracking.track_command("cloud")
def set_key_cmd(
    key: Annotated[str, typer.Option("--key", help="The Comfy Cloud API key.")],
):
    """Store a Comfy Cloud API key.

    The key is sent as ``X-API-Key`` on cloud requests and injected as
    ``api_key_comfy_org`` into ``extra_data`` so partner-API nodes can use
    it. Used as a fallback only — a live OAuth session takes precedence.
    Stored at ``secrets.json`` mode 0600.
    """
    from comfy_cli.target import CLOUD_API_KEY_PROVIDER

    renderer = get_renderer()
    if not key.strip():
        renderer.error(code="auth_invalid_key", message="--key cannot be empty.")
        raise typer.Exit(code=1)
    try:
        record = store.set(CLOUD_API_KEY_PROVIDER, key.strip())
    except ValueError as e:
        renderer.error(code="auth_invalid_key", message=str(e))
        raise typer.Exit(code=1) from e
    if renderer.is_pretty():
        rprint(f"[bold]Stored Comfy Cloud API key[/bold] [dim]({record.to_dict()['key']})[/dim]")
        rprint("[dim]→ stored as a fallback; a live OAuth session takes precedence[/dim]")
    renderer.emit(
        {
            "provider": CLOUD_API_KEY_PROVIDER,
            "path": str(store.secrets_path()),
            "action": "set-key",
        },
        command="cloud set-key",
        changed=True,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fmt_expiry(expires_at: int | None) -> str:
    if expires_at is None:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(expires_at, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return "unknown"
    return dt.isoformat(timespec="seconds")
