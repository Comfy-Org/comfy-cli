"""Pretty-mode panel builders for the agent-aware commands.

Centralizes the Rich rendering for the new commands (``discover``, ``which``,
``auth list``) and the structured error path so the human surface matches the
quality of the existing ``install`` / ``launch`` / ``env`` flows.

Every helper returns a Rich renderable — the renderer is responsible for
choosing the right Console (stdout in pretty mode, stderr in JSON mode).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from comfy_cli.output.sanitize import sanitize_markup, sanitize_optional, sanitize_value

_TARGET_DIM_CHAR = "·"


def _kv_table(rows: Sequence[tuple[str, str]], *, label_style: str = "bold cyan", value_style: str = "") -> Table:
    """Two-column borderless table used inside panels to align label/value rows.

    ``Table.add_row`` parses Rich markup in a ``str`` cell, so any caller
    passing untrusted text must run it through ``sanitize_markup`` first.
    """
    tbl = Table.grid(padding=(0, 2), expand=False)
    tbl.add_column(justify="right", style=label_style, no_wrap=True)
    tbl.add_column(style=value_style, overflow="fold")
    for label, value in rows:
        tbl.add_row(label, value)
    return tbl


# ---------------------------------------------------------------------------
# Error panel
# ---------------------------------------------------------------------------


def error_panel(
    *,
    code: str,
    message: str,
    hint: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> Panel:
    """Render an error as a red-bordered Rich Panel.

    Layout::

        ╭─ error · <code> ─────────────────────────────────────────────╮
        │ <message>                                                     │
        │                                                               │
        │ → <hint>                                                      │
        │ details:                                                      │
        │   key=value                                                   │
        ╰───────────────────────────────────────────────────────────────╯

    ``code``, ``message``, ``hint`` and the stringified ``details`` are all
    sanitized here: any of them can carry server-supplied text, and this panel
    is the pretty-mode boundary where it would otherwise reach the terminal as
    live escape sequences (see ``comfy_cli.output.sanitize``).

    ``code``/``message``/``hint`` are rendered through ``Text``, which never
    parses markup, so stripping the control bytes is enough for them. The
    ``details`` rows go to ``Table.add_row``, which *does* parse markup, so
    those additionally need ``sanitize_markup``.
    """
    code = sanitize_value(code)
    message = sanitize_value(message)
    hint = sanitize_optional(hint)
    body: list[Any] = [Text(message, style="white")]
    if hint:
        body.append(Text(""))
        body.append(Text.assemble(("→ ", "bold yellow"), (hint, "yellow")))
    if details:
        body.append(Text(""))
        body.append(Text("details:", style="dim"))
        rows = []
        for k, v in details.items():
            if v is None:
                continue
            rows.append((sanitize_markup(k), sanitize_markup(v)))
        if rows:
            body.append(_kv_table(rows, label_style="dim", value_style="dim"))
    return Panel(
        Group(*body),
        title=Text.assemble(("error", "bold red"), (f" {_TARGET_DIM_CHAR} ", "dim"), (code, "red")),
        title_align="left",
        border_style="red",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Discover panel
# ---------------------------------------------------------------------------


def discover_panel(doc: Mapping[str, Any], *, command_count: int) -> Panel:
    version = doc.get("version") or "(dev)"
    commands = doc.get("commands") or {}
    schemas = doc.get("schemas") or {}
    error_codes = doc.get("error_codes") or []
    capabilities = doc.get("capabilities") or {}

    caps_text: list[Text] = []
    for cap, val in sorted(capabilities.items()):
        if isinstance(val, list):
            continue  # rendered separately below
        if val is True:
            caps_text.append(Text.assemble(("✓ ", "bold green"), (cap, "white")))
    caps_grid = Table.grid(padding=(0, 3))
    if caps_text:
        per_row = 3
        for i in range(0, len(caps_text), per_row):
            row = caps_text[i : i + per_row]
            while len(row) < per_row:
                row.append(Text(""))
            caps_grid.add_row(*row)
    else:
        caps_grid.add_row(Text("(none advertised)", style="dim"))

    where_targets = capabilities.get("where_targets") or []
    auth_providers = capabilities.get("auth_providers") or []
    # auth_providers may be a flat list (legacy) or a {kind: [names...]} map.
    if isinstance(auth_providers, dict):
        oauth_names = list(auth_providers.get("oauth") or [])
        key_names = list(auth_providers.get("api_key") or [])
        oauth_line = "OAuth: " + f" {_TARGET_DIM_CHAR} ".join(oauth_names) if oauth_names else None
        key_line = "API key: " + f" {_TARGET_DIM_CHAR} ".join(key_names) if key_names else None
        auth_value = "  ".join(p for p in (oauth_line, key_line) if p) or "—"
    else:
        auth_value = f" {_TARGET_DIM_CHAR} ".join(auth_providers) if auth_providers else "—"

    # Build a two-column command table: name (+ subcount) | description
    visible_cmds = {
        name: entry
        for name, entry in sorted(commands.items())
        if isinstance(entry, dict) and not entry.get("hidden", False)
    }
    cmd_table = Table.grid(padding=(0, 2))
    cmd_table.add_column(style="white", no_wrap=True)  # name
    cmd_table.add_column(style="dim")  # description
    for name, entry in visible_cmds.items():
        subs = entry.get("subcommands") or {}
        desc = entry.get("short_help") or entry.get("help") or ""
        # Truncate long descriptions to first sentence
        if desc:
            dot = desc.find(".")
            if dot > 0:
                desc = desc[: dot + 1]
            if len(desc) > 60:
                desc = desc[:57] + "…"
        if subs:
            sub_count = len([s for s in subs.values() if isinstance(s, dict) and not s.get("hidden", False)])
            label = Text.assemble((name, "white"), (f" ({sub_count})", "dim cyan"))
        else:
            label = Text(name, style="white")
        cmd_table.add_row(label, Text(desc, style="dim"))

    # Schema names list
    schema_names = sorted(schemas.keys())
    schema_line = f" {_TARGET_DIM_CHAR} ".join(schema_names) if schema_names else "—"

    summary = _kv_table(
        [
            ("Commands", f"{len(visible_cmds)} top-level, {command_count} total (incl. subcommands)"),
            ("Schemas", f"{len(schemas)} — {schema_line}"),
            ("Error codes", f"{len(error_codes)}"),
        ]
    )
    routing = _kv_table(
        [
            ("Where targets", f" {_TARGET_DIM_CHAR} ".join(where_targets) if where_targets else "—"),
            ("Auth providers", auth_value),
        ]
    )

    body = Group(
        Text("Commands", style="bold magenta"),
        cmd_table,
        Text(""),
        Text("Capabilities", style="bold magenta"),
        caps_grid,
        Text(""),
        Text("Routing", style="bold magenta"),
        routing,
        Text(""),
        Text("Summary", style="bold magenta"),
        summary,
        Text(""),
        Text.assemble(
            ("→ ", "bold yellow"),
            ("comfy --json discover", "yellow"),
            ("   for the full machine-readable document", "dim"),
        ),
    )
    from comfy_cli.output.branding import branded_panel

    return branded_panel(
        body,
        title="discover",
        version=str(version),
    )


# ---------------------------------------------------------------------------
# Which panel
# ---------------------------------------------------------------------------


def which_panel(
    *,
    workspace_path: str,
    workspace_type: str | None,
    python_executable: str | None = None,
    python_version: str | None = None,
    server_running: bool = False,
    server_url: str | None = None,
    version: str = "0.0.0",
) -> Panel:
    type_badge = Text(workspace_type or "—", style="bold green") if workspace_type else Text("—", style="dim")
    python_value = Text("—", style="dim")
    if python_executable:
        if python_version:
            python_value = Text.assemble((python_executable, "white"), (f"  ({python_version})", "dim"))
        else:
            python_value = Text(python_executable, style="white")

    if server_running:
        server_value = Text.assemble(
            ("● running", "bold green"),
            ("    ", ""),
            (server_url or "", "white"),
        )
    else:
        server_value = Text.assemble(
            ("○ stopped", "dim"),
            ("    ", ""),
            (server_url or "", "dim"),
        )

    rows: list[tuple[str, Any]] = [
        ("Path", Text(workspace_path, style="bold white")),
        ("Type", type_badge),
        ("Python", python_value),
        ("Server", server_value),
    ]
    tbl = Table.grid(padding=(0, 2), expand=False)
    tbl.add_column(justify="right", style="bold cyan", no_wrap=True)
    tbl.add_column(overflow="fold")
    for label, value in rows:
        tbl.add_row(label, value)
    from comfy_cli.output.branding import branded_panel

    return branded_panel(
        tbl,
        title="workspace",
        version=version,
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Cloud status panel (billing / plan / concurrency)
# ---------------------------------------------------------------------------


def _fmt_usd(usd: float | None) -> str | None:
    if usd is None:
        return None
    # Negative balances are real (an overdrawn account) and must read as
    # negative rather than as a parenthesised accounting figure.
    sign = "-" if usd < 0 else ""
    return f"{sign}${abs(usd):,.2f}"


def cloud_status_panel(*, payload: Mapping[str, Any], version: str, show_plans: bool = False) -> Panel:
    """Pretty-mode rendering of the ``comfy cloud status`` payload.

    Reads the already-assembled envelope payload rather than re-deriving
    anything, so the human surface and the JSON surface cannot disagree. In
    particular the trust guard is honored upstream: when ``balance_confirmed``
    is false the balance keys are null and this function prints no figure at
    all, which is the whole point of the guard.

    Server-supplied strings (workspace name, plan slugs, the message) go
    through ``Text`` or ``sanitize_markup`` so a crafted name cannot inject
    Rich markup into the panel.
    """
    rows: list[tuple[str, Any]] = []

    workspace = payload.get("cloud_workspace")
    if isinstance(workspace, Mapping):
        # "Cloud workspace" is spelled out everywhere: in comfy-cli a bare
        # "workspace" is the local ComfyUI install directory.
        name = sanitize_optional(workspace.get("name")) or "—"
        wtype = sanitize_optional(workspace.get("type"))
        role = sanitize_optional(workspace.get("role"))
        parts: list[tuple[str, str]] = [(str(name), "bold white")]
        if wtype:
            parts.append((f"  ({wtype})", "dim"))
        if role:
            parts.append((f"  {_TARGET_DIM_CHAR} {role}", "dim"))
        rows.append(("Cloud workspace", Text.assemble(*parts)))

    if payload.get("balance_confirmed"):
        usd = _fmt_usd(payload.get("credit_balance_usd"))
        credits = payload.get("credit_balance_credits")
        micros = payload.get("effective_balance_micros")
        negative = isinstance(micros, int) and micros < 0
        balance_parts: list[tuple[str, str]] = [(usd or "—", "bold red" if negative else "bold green")]
        if credits is not None:
            balance_parts.append((f"  ({credits:,} credits)", "dim"))
        rows.append(("Balance", Text.assemble(*balance_parts)))
    else:
        rows.append(("Balance", Text("not confirmed", style="yellow")))

    subscription = payload.get("subscription")
    if isinstance(subscription, Mapping):
        tier = sanitize_optional(subscription.get("tier"))
        status = sanitize_optional(subscription.get("status"))
        is_active = subscription.get("is_active")
        if tier or status:
            sub_parts: list[tuple[str, str]] = [(str(tier or "—"), "bold white")]
            if status:
                sub_parts.append((f"  {status}", "green" if is_active else "yellow"))
            rows.append(("Subscription", Text.assemble(*sub_parts)))
        plan_slug = sanitize_optional(subscription.get("plan_slug"))
        if plan_slug:
            rows.append(("Plan", Text(str(plan_slug), style="white")))
        renewal = sanitize_optional(subscription.get("renewal_date"))
        cancel_at = sanitize_optional(subscription.get("cancel_at"))
        if cancel_at:
            rows.append(("Cancels", Text(str(cancel_at), style="yellow")))
        elif renewal:
            rows.append(("Renews", Text(str(renewal), style="dim")))

    free_tier = payload.get("free_tier_balance")
    if isinstance(free_tier, Mapping):
        # Sanitized like every other server-supplied field here. `Text` does not
        # parse Rich markup but it does pass `\x1b` straight through, so an
        # unsanitized CSI/OSC payload in one of these numbers would reach the
        # terminal live. Reachable in practice: a user can point the CLI at a
        # hostile host via the documented `comfy cloud set-base-url`.
        remaining = sanitize_optional(free_tier.get("remaining"))
        allowance = sanitize_optional(free_tier.get("allowance"))
        if remaining is not None or allowance is not None:
            rows.append(
                (
                    "Free tier",
                    Text.assemble(
                        (remaining if remaining is not None else "—", "bold white"),
                        (f" of {allowance} remaining" if allowance is not None else " remaining", "dim"),
                    ),
                )
            )

    concurrency = payload.get("max_concurrent_jobs")
    tier_default_concurrency = payload.get("tier_default_concurrent_jobs")
    # Fall back to the tier default when /api/features was unavailable, rather
    # than printing no concurrency row at all. The payload ships that field
    # precisely to provide this context, and dropping the row hid a number we
    # already knew.
    shown_concurrency = concurrency if isinstance(concurrency, int) else None
    from_tier_default = False
    if shown_concurrency is None and isinstance(tier_default_concurrency, int):
        shown_concurrency = tier_default_concurrency
        from_tier_default = True
    if shown_concurrency is not None:
        conc_parts: list[tuple[str, str]] = [(f"{shown_concurrency}", "bold white")]
        conc_parts.append((f"  concurrent job{'s' if shown_concurrency != 1 else ''}", "dim"))
        if from_tier_default:
            conc_parts.append(("  (tier default)", "dim"))
        rows.append(("Concurrency", Text.assemble(*conc_parts)))

    upgrade = payload.get("upgrade_suggestion")
    if isinstance(upgrade, Mapping):
        slug = sanitize_optional(upgrade.get("plan_slug")) or "—"
        price = _fmt_usd(upgrade.get("price_usd"))
        up_parts: list[tuple[str, str]] = [(str(slug), "bold cyan")]
        if price:
            up_parts.append((f"  {price}", "white"))
        credits = upgrade.get("credits")
        if credits is not None:
            up_parts.append((f"  ({credits:,} credits)", "dim"))
        rows.append(("Upgrade", Text.assemble(*up_parts)))

    manage_url = payload.get("manage_url")
    if manage_url:
        rows.append(("Manage", Text(str(sanitize_value(manage_url)), style="cyan")))

    tbl = Table.grid(padding=(0, 2), expand=False)
    tbl.add_column(justify="right", style="bold cyan", no_wrap=True)
    tbl.add_column(overflow="fold")
    for label, value in rows:
        tbl.add_row(label, value)

    blocks: list[Any] = [tbl]

    message = payload.get("message")
    if message:
        blocks.append(Text(""))
        blocks.append(Text(str(sanitize_value(message)), style="yellow"))

    if shown_concurrency is not None:
        # API-key callers get the tier limit; a browser session can differ
        # because parallel execution is gated separately on the frontend.
        blocks.append(Text(""))
        blocks.append(
            Text(
                "Concurrency is the limit for this credential; browser sessions may differ.",
                style="dim",
            )
        )

    if show_plans:
        plans = payload.get("plans")
        if isinstance(plans, Sequence) and plans:
            plan_tbl = Table(box=None, pad_edge=False, show_edge=False)
            plan_tbl.add_column("Plan", style="white", no_wrap=True)
            plan_tbl.add_column("Price", justify="right")
            plan_tbl.add_column("Credits", justify="right")
            plan_tbl.add_column("Available")
            for plan in plans:
                if not isinstance(plan, Mapping):
                    continue
                credits = plan.get("credits")
                available = plan.get("available")
                reason = sanitize_optional(plan.get("unavailable_reason"))
                avail_text = (
                    Text("yes", style="green") if available else Text(f"no ({reason})" if reason else "no", style="dim")
                )
                plan_tbl.add_row(
                    Text(str(sanitize_value(plan.get("plan_slug") or "—"))),
                    Text(_fmt_usd(plan.get("price_usd")) or "—"),
                    Text(f"{credits:,}" if credits is not None else "—"),
                    avail_text,
                )
            blocks.append(Text(""))
            blocks.append(plan_tbl)

    from comfy_cli.output.branding import branded_panel

    return branded_panel(
        Group(*blocks),
        title="cloud status",
        version=version,
        where="cloud",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Auth list table / empty panel
# ---------------------------------------------------------------------------


def auth_list_table(records: Sequence[Mapping[str, Any]], *, supported: Sequence[str], path: str | None = None) -> Any:
    """Return a Group: title + table + supported-providers footer.

    ``records`` should be the redacted dicts from ``AuthRecord.to_dict(redact=True)``
    (provider / key / updated_at).
    """
    title = Text.assemble(
        ("Auth providers", "bold cyan"),
        (f"  ({len(records)} configured)", "dim"),
    )

    tbl = Table(show_header=True, header_style="bold magenta", border_style="cyan", pad_edge=False)
    tbl.add_column("Provider", style="bold white", no_wrap=True)
    tbl.add_column("Key", style="white", no_wrap=True)
    tbl.add_column("Updated", style="dim", no_wrap=True)
    for r in records:
        tbl.add_row(
            str(r.get("provider", "")),
            str(r.get("key", "")),
            str(r.get("updated_at") or "—"),
        )

    footer_parts: list[Text] = []
    if supported:
        footer_parts.append(
            Text.assemble(
                ("supported: ", "dim"),
                (f" {_TARGET_DIM_CHAR} ".join(supported), "white"),
            )
        )
    if path:
        footer_parts.append(Text.assemble(("store: ", "dim"), (path, "dim")))
    footer = Group(*footer_parts) if footer_parts else Text("")

    return Group(title, tbl, footer)


def auth_empty_panel(*, supported: Sequence[str], path: str | None = None) -> Panel:
    lines: list[Any] = [
        Text("No providers configured.", style="white"),
        Text(""),
        Text.assemble(
            ("→ ", "bold yellow"),
            ("comfy auth set <provider> --key <KEY>", "yellow"),
        ),
    ]
    if supported:
        lines.append(Text(""))
        lines.append(
            Text.assemble(
                ("Supported: ", "dim"),
                (f" {_TARGET_DIM_CHAR} ".join(supported), "white"),
            )
        )
    if path:
        lines.append(Text.assemble(("Store:     ", "dim"), (path, "dim")))
    return Panel(
        Group(*lines),
        title=Text("Auth providers", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )
