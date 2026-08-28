"""``comfy cloud`` — sign in to Comfy Cloud (OAuth) and manage cloud routing config.

The ``cloud`` namespace is the explicit prefix for everything that talks to
Comfy Cloud:

- ``cloud login``       — Authorization Code + PKCE flow against the cloud
- ``cloud logout``      — clear the local OAuth session
- ``cloud whoami``      — inspect the current session + base URL + auth path
- ``cloud status``      — cloud workspace, credit balance, subscription tier,
                            concurrency limit and upgrade suggestion
- ``cloud set-base-url`` — pin a non-prod env (PR preview, staging, …)
- ``cloud set-key``     — testing path: persist a Comfy Cloud API key
                            (canonical sign-in is ``cloud login``)

Third-party API tokens (Civitai, Hugging Face) for ``comfy model download``
live under ``comfy auth`` since they're not cloud-specific.
"""

from __future__ import annotations

import urllib.error
from datetime import datetime, timezone
from typing import Annotated, Any

import typer
from rich import markup as rich_markup

from comfy_cli import knowledge, tracking
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


@app.command(
    "login",
    help=(
        "Sign in to Comfy Cloud via your browser (OAuth + PKCE). Under --json/--json-stream "
        "the authorize URL is emitted as a `login_url` event before the command blocks on the "
        "browser callback, so an agent/MCP parent can open it; see docs/json-output.md."
    ),
)
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
        if renderer.is_pretty():
            # Escape the URL for Rich: an IPv6 base_url (e.g. `http://[::1]:8188`)
            # would otherwise make Rich parse `[::1]` as markup and raise.
            safe_url = rich_markup.escape(url)
            if no_browser:
                rprint("\nOpen this URL in your browser to sign in:")
                rprint(f"  [cyan]{safe_url}[/cyan]\n")
            else:
                rprint("[dim]Opening browser… (if it doesn't appear, copy this URL)[/dim]")
                rprint(f"[dim]  {safe_url}[/dim]")
            return
        # Machine mode: surface the authorize URL as a `login_url` event so an
        # agent/MCP parent driving `comfy --json cloud login` can open (or hand
        # off) the URL. run_login then blocks up to `timeout` seconds on the
        # loopback callback, so upgrade to the NDJSON stream and emit the line
        # now — `event()` writes + flushes, so the parent sees the URL *before*
        # we block on the wait. If the write raises (the piped parent hung up),
        # let it propagate: run_login re-raises OSError from the callback rather
        # than blocking the full timeout on a stream nobody is reading.
        renderer.force_stream()
        renderer.event("login_url", url=url, timeout_s=timeout)

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
    except OSError:
        # The machine stream broke while emitting `login_url` (the piped parent
        # hung up), so run_login bailed instead of blocking the full timeout.
        # We can't write an envelope to a dead stream — just fail fast.
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

    # Fill the knowledge cache while the user is online and freshly signed
    # in, so the cache-only attach path has a bundle on a brand-new install.
    knowledge.refresh_if_stale()

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
# status commands - billing, plan, concurrency
# ---------------------------------------------------------------------------

# Billing payloads are small objects. The cap only exists to bound memory on a
# misbehaving server, so it is generous relative to the real bodies (~1 KiB)
# and still nowhere near enough to matter.
_STATUS_MAX_BYTES = 1 * 1024 * 1024


# Bound on the slice of a server error body we quote back in `details`. Read
# via `HTTPError.read(n)` so the cap is applied by the read itself; draining
# the whole body and slicing afterwards would let the server decide how much
# memory we spend on the error path, which is the one path least able to
# afford it.
_MAX_ERROR_BODY_BYTES = 1000


def _as_optional_str(value: Any) -> str | None:
    """Keep a value only if it is genuinely a string, else None.

    Dates and rail names are declared as nullable strings in
    ``cloud_status.json``; a backend sending a number or an object for one of
    them would otherwise produce an envelope that fails our own schema.
    """
    return value if isinstance(value, str) else None


def _read_error_body(e: urllib.error.HTTPError) -> str:
    """Best-effort read of a bounded slice of the server's error body.

    Mirrors ``comfy_cli/command/_cloud_errors.py::_read_error_body``. Failures
    degrade to an empty body rather than escaping: a reset or truncated stream
    while quoting an error must not pre-empt the very envelope the caller is in
    the middle of emitting, turning a transient fault into a traceback.
    """
    try:
        return (e.read(_MAX_ERROR_BODY_BYTES) or b"").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — a failed courtesy read must never mask the envelope
        return ""


def _billing_get(url: str, target, *, timeout: float = 30.0):
    """Authenticated GET returning the parsed body. Raises errors verbatim."""
    from comfy_cli.http import request_json

    _, body = request_json(url, target, timeout=timeout, max_bytes=_STATUS_MAX_BYTES)
    return body


def _billing_get_optional(url: str, target, *, timeout: float = 30.0) -> tuple[Any, bool]:
    """Fetch an optional endpoint, returning ``(body, failed)``.

    Used for every endpoint whose absence should degrade one row rather than
    fail the command: an older backend that 404s ``/api/workspaces/current``,
    or a features/plans call that times out, must still let the user see their
    balance and tier. The command's whole job is answering "what do I have",
    so partial data beats no answer.

    The ``failed`` flag exists because ``None`` is ambiguous: it means both
    "the call did not answer" and "the call answered with nothing useful". For
    the balance endpoint those must not be conflated, or a transient blip
    silently changes the trust guard's verdict. See
    :func:`billing.is_unconfirmed`.

    ``ValueError`` is caught alongside the HTTP families because
    ``assert_safe_url`` raises it for a non-loopback plaintext URL, and
    ``comfy cloud set-base-url`` / ``COMFY_CLOUD_BASE_URL`` both accept such a
    URL without validation. Note ``JSONDecodeError`` is a ``ValueError``
    subclass, so naming the parent covers it.
    """
    import urllib.error

    from comfy_cli.http import ResponseTooLarge

    try:
        return _billing_get(url, target, timeout=timeout), False
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
        ValueError,
        ResponseTooLarge,
    ):
        return None, True


@app.command(
    "status",
    help=(
        "Show the current Comfy Cloud workspace, credit balance, subscription tier, "
        "concurrency limit and upgrade suggestion. Read-only; spends nothing."
    ),
)
@tracking.track_command("cloud")
def status_cmd(
    plans: Annotated[
        bool,
        typer.Option("--plans", help="Include the full tier table, not just the current plan and one suggestion."),
    ] = False,
):
    import urllib.error

    from comfy_cli.cloud import billing
    from comfy_cli.http import ResponseTooLarge
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    target = resolve_target(where="cloud")
    renderer.where = target.kind

    # Fail before any request when there is no credential at all: an
    # unauthenticated GET would come back 401 and we would report "cloud
    # rejected your token" to someone who never had one.
    if not target.auth_token and not target.api_key:
        renderer.error(
            code="cloud_not_configured",
            message="No Comfy Cloud credential found.",
            hint="run `comfy cloud login`",
        )
        raise typer.Exit(code=1)

    manage_url = billing.manage_url(target.base_url)

    # /api/billing/status is the only required call: without it there is no
    # tier, no status and no rail, so there is nothing meaningful to print.
    status_url = target.url("billing", "status")
    try:
        status_body = _billing_get(status_url, target)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            renderer.error(
                code="cloud_unauthorized",
                message=f"Comfy Cloud rejected the credential (HTTP {e.code}).",
                hint="run `comfy cloud login`",
                details={"status": e.code, "url": status_url},
            )
            raise typer.Exit(code=1) from e
        renderer.error(
            code="cloud_billing_unavailable",
            message=f"HTTP {e.code} from {status_url}",
            hint="retry shortly; if it persists, contact Comfy support",
            details={"status": e.code, "body": _read_error_body(e)},
        )
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, OSError, ValueError, ResponseTooLarge) as e:
        # ValueError covers `assert_safe_url` rejecting a plaintext non-loopback
        # base URL, which a user can reach through the documented
        # `comfy cloud set-base-url` / COMFY_CLOUD_BASE_URL paths. It also
        # covers JSONDecodeError, which subclasses it.
        unsafe_url = not isinstance(e, (urllib.error.URLError, OSError, ResponseTooLarge))
        renderer.error(
            code="cloud_billing_unavailable",
            message=f"failed to fetch {status_url}: {e}",
            hint=(
                "the cloud base URL must be https (or loopback); fix it with `comfy cloud set-base-url`"
                if unsafe_url
                else "check your network connection and try again"
            ),
        )
        raise typer.Exit(code=1) from e

    if not isinstance(status_body, dict):
        renderer.error(
            code="cloud_billing_unavailable",
            message=f"unexpected billing status payload from {status_url}",
            hint="retry shortly; if it persists, contact Comfy support",
        )
        raise typer.Exit(code=1)

    # Everything below degrades independently.
    balance_body, balance_failed = _billing_get_optional(target.url("billing", "balance"), target)
    features_body, _ = _billing_get_optional(target.url("features"), target)
    workspace_body, _ = _billing_get_optional(target.url("workspaces", "current"), target)
    plans_body, _ = _billing_get_optional(target.url("billing", "plans"), target)

    balance_micros = billing.select_effective_balance_micros(balance_body if isinstance(balance_body, dict) else None)
    balance_usd = billing.micros_to_usd(balance_micros)
    balance_credits = billing.usd_to_credits(balance_usd)

    subscription_status = status_body.get("subscription_status")
    # `subscription_tier` is absent entirely (not null) when there is no
    # subscription, so `.get` is the contract, not a convenience.
    subscription_tier = status_body.get("subscription_tier")
    billing_rail = status_body.get("billing_rail")
    has_funds = status_body.get("has_funds")

    unconfirmed = billing.is_unconfirmed(
        balance_micros=balance_micros,
        subscription_status=subscription_status,
        has_funds=has_funds,
        balance_fetch_failed=balance_failed,
    )
    # Distinct from `not unconfirmed`: a failed balance fetch alongside an
    # active subscription is not "unconfirmed" (we know the account is real)
    # but still has no figure to publish.
    balance_confirmed = billing.balance_is_confirmed(
        balance_micros=balance_micros,
        balance_fetch_failed=balance_failed,
        unconfirmed=unconfirmed,
    )
    legacy_rail = billing.is_legacy_rail(billing_rail)

    message: str | None = None
    if unconfirmed:
        message = billing.unconfirmed_message(manage_url)
        if legacy_rail:
            message = f"{message} {billing.LEGACY_RAIL_NOTE}"
    elif balance_failed:
        # A request-level failure, so say that plainly instead of borrowing the
        # mid-migration wording, which would be a claim about the account.
        message = billing.BALANCE_UNAVAILABLE_NOTE
        if legacy_rail:
            message = f"{message} {billing.LEGACY_RAIL_NOTE}"
    elif legacy_rail:
        message = billing.LEGACY_RAIL_NOTE

    cloud_workspace: dict[str, object | None] | None = None
    if isinstance(workspace_body, dict):
        cloud_workspace = {
            "id": _as_optional_str(workspace_body.get("id")),
            "name": _as_optional_str(workspace_body.get("name")),
            "type": _as_optional_str(workspace_body.get("type")),
            "role": _as_optional_str(workspace_body.get("role")),
        }

    # Every server-supplied scalar below goes through a coercion before it
    # enters the payload. `cloud_status.json` constrains these types and agents
    # are told to validate against it, so a backend sending "3" or
    # `is_active: "false"` would otherwise emit an envelope that fails our own
    # schema. Values that don't fit are dropped to null, the same way
    # `normalize_plans` already treats malformed plan entries.
    max_concurrent_jobs = None
    free_tier_balance = None
    if isinstance(features_body, dict):
        max_concurrent_jobs = billing.coerce_int(features_body.get("max_concurrent_jobs"))
        raw_free_tier = features_body.get("free_tier_balance")
        if isinstance(raw_free_tier, dict):
            free_tier_balance = {
                "allowance": billing.coerce_int(raw_free_tier.get("allowance")),
                "used": billing.coerce_int(raw_free_tier.get("used")),
                "remaining": billing.coerce_int(raw_free_tier.get("remaining")),
            }

    plan_list, current_plan_slug = billing.normalize_plans(plans_body)
    if current_plan_slug is None:
        slug = status_body.get("plan_slug")
        # Assign the STRIPPED slug, not the raw one: `normalize_plans` strips
        # every slug it emits, so a whitespace-padded value here would never
        # match in `suggest_upgrade` (silently dropping the suggestion) and the
        # padding would leak into the payload.
        current_plan_slug = slug.strip() if isinstance(slug, str) and slug.strip() else None
    upgrade_suggestion = billing.suggest_upgrade(plan_list, current_plan_slug)

    tier_default_concurrency = None
    if isinstance(subscription_tier, str):
        tier_default_concurrency = billing.TIER_CONCURRENCY.get(subscription_tier.strip().upper())

    payload: dict[str, object | None] = {
        "cloud_workspace": cloud_workspace,
        # Withhold the figure entirely when it isn't confirmed. A null here is
        # "we don't know"; printing 0.0 would assert a balance we cannot back up.
        "credit_balance_usd": balance_usd if balance_confirmed else None,
        "credit_balance_credits": balance_credits if balance_confirmed else None,
        "effective_balance_micros": balance_micros if balance_confirmed else None,
        "currency": "USD",
        "balance_confirmed": balance_confirmed,
        "subscription": {
            "tier": subscription_tier if isinstance(subscription_tier, str) else None,
            "status": subscription_status if isinstance(subscription_status, str) else None,
            "is_active": billing.coerce_bool(status_body.get("is_active")),
            "plan_slug": current_plan_slug,
            "renewal_date": _as_optional_str(status_body.get("renewal_date")),
            "cancel_at": _as_optional_str(status_body.get("cancel_at")),
        },
        "billing_rail": billing_rail if isinstance(billing_rail, str) else None,
        "max_concurrent_jobs": max_concurrent_jobs,
        "tier_default_concurrent_jobs": tier_default_concurrency,
        "free_tier_balance": free_tier_balance,
        "upgrade_suggestion": upgrade_suggestion,
        "blocked_upgrades": billing.blocked_upgrade_reasons(plan_list),
        "manage_url": manage_url,
        "message": message,
    }
    seats = {
        "max": billing.coerce_int(status_body.get("max_seats")),
        "occupied": billing.coerce_int(status_body.get("occupied_seats")),
    }
    if seats["max"] is not None or seats["occupied"] is not None:
        payload["seats"] = seats
    if plans:
        payload["plans"] = plan_list

    if renderer.is_pretty():
        from comfy_cli.output.panels import cloud_status_panel

        renderer.console().print(
            cloud_status_panel(
                payload=payload,
                version=ConfigManager().get_cli_version(),
                show_plans=plans,
            )
        )

    renderer.emit(payload, command="cloud status")


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
