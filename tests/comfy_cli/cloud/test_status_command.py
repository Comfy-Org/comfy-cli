"""``comfy cloud status`` — billing / plan / concurrency snapshot.

All HTTP is mocked at ``comfy_cli.http.request_json``, keyed on the URL, so
each test declares only the endpoints it cares about and every unlisted
endpoint 404s. That is deliberate: the command is supposed to degrade one row
per unavailable endpoint rather than fail, and a fixture that silently serves
everything would never exercise that.

The two display rules with history behind them get the most coverage:

* the zero-masking balance chain, where a present-but-zero high-priority field
  must not mask a positive component below it;
* the trust guard (BE-1795 / BE-1798), where an all-silent backend must produce
  neutral copy instead of "$0.00 / Subscribe".
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import typer

from comfy_cli.cloud import billing, command
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode
from comfy_cli.target import Target

_BASE = "https://testcloud.comfy.org"
_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas" / "cloud_status.json"


@pytest.fixture(autouse=True)
def _no_tracking(monkeypatch):
    """Keep the ``@track_command`` decorator inert (no network / consent I/O)."""
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **k: None)


def _cloud_target(*, token: str | None = "tok", api_key: str | None = None) -> Target:
    return Target(
        kind="cloud",
        base_url=_BASE,
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        auth_token=token,
        api_key=api_key,
    )


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, f"HTTP {code}", {}, io.BytesIO(b"{}"))


def _install(monkeypatch, routes: dict[str, Any], *, target: Target | None = None) -> None:
    """Route ``request_json`` by URL suffix.

    A route value that is an ``Exception`` is raised; anything else is returned
    as a 200 body. An unlisted suffix raises 404, standing in for an older
    backend or a transient failure.
    """
    resolved = target if target is not None else _cloud_target()
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kwargs: resolved)

    def fake_request_json(url, _target, *, method="GET", body=None, timeout=30.0, max_bytes):
        for suffix, value in routes.items():
            if url.endswith(suffix):
                if isinstance(value, Exception):
                    raise value
                return 200, value
        raise _http_error(url, 404)

    monkeypatch.setattr("comfy_cli.http.request_json", fake_request_json)


def _run(monkeypatch, capsys, routes: dict[str, Any], *, plans: bool = False, target: Target | None = None) -> dict:
    _install(monkeypatch, routes, target=target)
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))
    command.status_cmd(plans=plans)
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert lines, "no envelope on stdout"
    return lines[-1]


# ---------------------------------------------------------------------------
# Fixture payloads, modeled on real cloud responses.
# ---------------------------------------------------------------------------

_ACTIVE_CREATOR = {
    "is_active": True,
    "has_funds": True,
    "subscription_status": "active",
    "subscription_tier": "CREATOR",
    "plan_slug": "creator-monthly",
    "billing_rail": "stripe",
    "renewal_date": "2026-09-19",
    "cancel_at": None,
}

_PLANS = {
    "current_plan_slug": "creator-monthly",
    "plans": [
        {"slug": "free", "price_cents": 0, "credits_cents": 0, "availability": {"available": True}},
        {"slug": "creator-monthly", "price_cents": 2000, "credits_cents": 2000, "availability": {"available": True}},
        {"slug": "pro-monthly", "price_cents": 5000, "credits_cents": 5500, "availability": {"available": True}},
        {"slug": "team-monthly", "price_cents": 20000, "credits_cents": 22000, "availability": {"available": True}},
        {
            "slug": "enterprise",
            "price_cents": 100000,
            "credits_cents": 120000,
            "availability": {"available": False, "reason": "requires_team"},
        },
    ],
}

_WORKSPACE = {"id": "w-abc123", "name": "Matt's Studio", "type": "personal", "role": "owner"}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_active_creator_positive_balance(monkeypatch, capsys):
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 12_500_000},
            "/features": {"max_concurrent_jobs": 3},
            "/workspaces/current": _WORKSPACE,
            "/billing/plans": _PLANS,
        },
    )
    assert env["ok"] is True
    data = env["data"]
    assert data["balance_confirmed"] is True
    assert data["credit_balance_usd"] == pytest.approx(12.5)
    assert data["credit_balance_credits"] == int(12.5 * billing.CREDITS_PER_USD)
    assert data["currency"] == "USD"
    assert data["cloud_workspace"] == _WORKSPACE
    assert data["subscription"]["tier"] == "CREATOR"
    assert data["subscription"]["is_active"] is True
    assert data["max_concurrent_jobs"] == 3
    assert data["tier_default_concurrent_jobs"] == 3
    assert data["manage_url"] == f"{_BASE}/billing"
    assert data["message"] is None
    # Cheapest available plan priced above creator-monthly (2000c).
    assert data["upgrade_suggestion"]["plan_slug"] == "pro-monthly"
    assert data["upgrade_suggestion"]["price_usd"] == pytest.approx(50.0)
    # The unavailable tier is reported with the server's reason, not silently dropped.
    assert {"plan_slug": "enterprise", "reason": "requires_team"} in data["blocked_upgrades"]


def test_plans_flag_includes_full_tier_table(monkeypatch, capsys):
    without = _run(monkeypatch, capsys, {"/billing/status": _ACTIVE_CREATOR, "/billing/plans": _PLANS})
    assert "plans" not in without["data"]

    with_plans = _run(monkeypatch, capsys, {"/billing/status": _ACTIVE_CREATOR, "/billing/plans": _PLANS}, plans=True)
    slugs = [p["plan_slug"] for p in with_plans["data"]["plans"]]
    assert slugs == ["free", "creator-monthly", "pro-monthly", "team-monthly", "enterprise"]


# ---------------------------------------------------------------------------
# The zero-masking chain
# ---------------------------------------------------------------------------


def test_zero_effective_balance_does_not_mask_positive_component(monkeypatch, capsys):
    """A literal 0 in the highest-priority field must not hide real credits."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {
                "effective_balance_micros": 0,
                "amount_micros": 0,
                "cloud_credit_balance_micros": 7_000_000,
            },
        },
    )
    data = env["data"]
    assert data["effective_balance_micros"] == 7_000_000
    assert data["credit_balance_usd"] == pytest.approx(7.0)
    assert data["balance_confirmed"] is True


def test_negative_effective_balance_is_reported_not_clamped(monkeypatch, capsys):
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": -2_250_000},
        },
    )
    data = env["data"]
    assert data["effective_balance_micros"] == -2_250_000
    assert data["credit_balance_usd"] == pytest.approx(-2.25)
    assert data["credit_balance_credits"] < 0
    assert data["balance_confirmed"] is True


def test_all_zero_balance_with_active_subscription_is_a_confirmed_zero(monkeypatch, capsys):
    """Zero is a real answer when the account is otherwise confirmed.

    The trust guard must not swallow a genuine zero balance on an active
    subscription, or a user who has simply run out of credits gets told we
    could not confirm anything.
    """
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 0, "amount_micros": 0},
        },
    )
    data = env["data"]
    assert data["balance_confirmed"] is True
    assert data["credit_balance_usd"] == 0
    assert data["message"] is None


# ---------------------------------------------------------------------------
# The trust guard
# ---------------------------------------------------------------------------


def test_neutral_copy_when_nothing_can_be_confirmed(monkeypatch, capsys):
    """All-zero balance + no subscription + !has_funds must not print $0.00."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {"is_active": False, "has_funds": False, "subscription_status": None},
            "/billing/balance": {},
        },
    )
    data = env["data"]
    assert data["balance_confirmed"] is False
    # The figures are withheld, not zeroed: null cannot be rendered as "$0.00".
    assert data["credit_balance_usd"] is None
    assert data["credit_balance_credits"] is None
    assert data["effective_balance_micros"] is None
    assert "Couldn't confirm an active subscription or balance" in data["message"]
    assert f"{_BASE}/billing" in data["message"]


def test_legacy_stripe_rail_is_annotated(monkeypatch, capsys):
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {
                "is_active": False,
                "has_funds": False,
                "subscription_status": None,
                "billing_rail": "legacy_stripe",
            },
            "/billing/balance": {},
        },
    )
    data = env["data"]
    assert data["balance_confirmed"] is False
    assert data["billing_rail"] == "legacy_stripe"
    assert "legacy billing rail" in data["message"]
    assert "Couldn't confirm" in data["message"]


def test_legacy_rail_annotated_even_when_balance_is_confirmed(monkeypatch, capsys):
    """The rail caveat is independent of the trust guard.

    A confirmed balance on the legacy rail can still be partial, so the note
    rides along without the neutral copy.
    """
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {**_ACTIVE_CREATOR, "billing_rail": "legacy_stripe"},
            "/billing/balance": {"effective_balance_micros": 5_000_000},
        },
    )
    data = env["data"]
    assert data["balance_confirmed"] is True
    assert data["message"] == billing.LEGACY_RAIL_NOTE
    assert "Couldn't confirm" not in data["message"]


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def test_missing_workspaces_current_degrades_to_null_not_failure(monkeypatch, capsys):
    """An older backend without GET /api/workspaces/current still works."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 1_000_000},
            "/features": {"max_concurrent_jobs": 3},
            # /workspaces/current deliberately absent -> 404
        },
    )
    assert env["ok"] is True
    assert env["data"]["cloud_workspace"] is None
    assert env["data"]["credit_balance_usd"] == pytest.approx(1.0)


def test_every_optional_endpoint_failing_still_emits_a_useful_envelope(monkeypatch, capsys):
    env = _run(monkeypatch, capsys, {"/billing/status": _ACTIVE_CREATOR})
    data = env["data"]
    assert env["ok"] is True
    assert data["cloud_workspace"] is None
    assert data["max_concurrent_jobs"] is None
    assert data["upgrade_suggestion"] is None
    # plan_slug falls back to the billing-status payload when /plans is gone.
    assert data["subscription"]["plan_slug"] == "creator-monthly"
    assert data["subscription"]["tier"] == "CREATOR"


def test_free_tier_balance_is_surfaced(monkeypatch, capsys):
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {
                "is_active": True,
                "has_funds": True,
                "subscription_status": "active",
                "subscription_tier": "FREE",
                "plan_slug": "free",
            },
            "/billing/balance": {"effective_balance_micros": 0, "amount_micros": 0},
            "/features": {
                "max_concurrent_jobs": 1,
                "free_tier_balance": {"allowance": 100, "used": 40, "remaining": 60},
            },
        },
    )
    data = env["data"]
    assert data["subscription"]["tier"] == "FREE"
    assert data["tier_default_concurrent_jobs"] == 1
    assert data["free_tier_balance"] == {"allowance": 100, "used": 40, "remaining": 60}


def test_seats_only_present_when_reported(monkeypatch, capsys):
    without = _run(monkeypatch, capsys, {"/billing/status": _ACTIVE_CREATOR})
    assert "seats" not in without["data"]

    with_seats = _run(
        monkeypatch,
        capsys,
        {"/billing/status": {**_ACTIVE_CREATOR, "max_seats": 25, "occupied_seats": 4}},
    )
    assert with_seats["data"]["seats"] == {"max": 25, "occupied": 4}


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_no_credential_fails_before_any_request(monkeypatch, capsys):
    """Never report "cloud rejected your token" to someone who has none."""
    calls: list[str] = []

    def fake_request_json(url, *a, **k):
        calls.append(url)
        raise AssertionError("should not have issued a request")

    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: _cloud_target(token=None, api_key=None))
    monkeypatch.setattr("comfy_cli.http.request_json", fake_request_json)
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))

    with pytest.raises(typer.Exit) as excinfo:
        command.status_cmd(plans=False)
    assert excinfo.value.exit_code == 1
    assert calls == []
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["ok"] is False
    assert env["error"]["code"] == "cloud_not_configured"


@pytest.mark.parametrize("status_code", [401, 403])
def test_rejected_credential_maps_to_cloud_unauthorized(monkeypatch, capsys, status_code):
    _install(monkeypatch, {"/billing/status": _http_error(f"{_BASE}/api/billing/status", status_code)})
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))

    with pytest.raises(typer.Exit):
        command.status_cmd(plans=False)
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["error"]["code"] == "cloud_unauthorized"


def test_billing_status_server_error_maps_to_cloud_billing_unavailable(monkeypatch, capsys):
    _install(monkeypatch, {"/billing/status": _http_error(f"{_BASE}/api/billing/status", 503)})
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))

    with pytest.raises(typer.Exit):
        command.status_cmd(plans=False)
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["error"]["code"] == "cloud_billing_unavailable"
    assert env["error"]["details"]["status"] == 503


def test_unparseable_billing_status_body_is_an_error_not_a_crash(monkeypatch, capsys):
    _install(monkeypatch, {"/billing/status": ["not", "an", "object"]})
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))

    with pytest.raises(typer.Exit):
        command.status_cmd(plans=False)
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["error"]["code"] == "cloud_billing_unavailable"


# ---------------------------------------------------------------------------
# Schema + pretty surface
# ---------------------------------------------------------------------------


def test_payload_validates_against_the_shipped_schema(monkeypatch, capsys):
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 12_500_000},
            "/features": {"max_concurrent_jobs": 3},
            "/workspaces/current": _WORKSPACE,
            "/billing/plans": _PLANS,
        },
        plans=True,
    )
    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(env["data"])


def test_unconfirmed_payload_also_validates(monkeypatch, capsys):
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {"is_active": False, "has_funds": False, "subscription_status": None},
            "/billing/balance": {},
        },
    )
    assert env["data"]["balance_confirmed"] is False
    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(env["data"])


def test_pretty_mode_renders_without_asserting_a_balance(monkeypatch, capsys):
    """The panel must not print a dollar figure for an unconfirmed workspace.

    This is the BE-1798 regression in its original surface: the JSON payload
    withholding the number is only half the fix if the panel prints one anyway.
    """
    _install(
        monkeypatch,
        {
            "/billing/status": {"is_active": False, "has_funds": False, "subscription_status": None},
            # An explicit empty body, not an absent route. The guard fires on a
            # backend that ANSWERED and reported nothing; a route that fails to
            # answer is a different case and is covered separately, since a
            # transient error is not evidence about the account.
            "/billing/balance": {},
        },
    )
    set_renderer(Renderer(mode=OutputMode.PRETTY, command="cloud status", version="test"))
    command.status_cmd(plans=False)
    out = capsys.readouterr().out
    assert "$" not in out
    assert "not confirmed" in out
    assert "Couldn't confirm" in out


def test_pretty_mode_renders_a_confirmed_balance(monkeypatch, capsys):
    _install(
        monkeypatch,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 12_500_000},
            "/features": {"max_concurrent_jobs": 3},
            "/workspaces/current": _WORKSPACE,
        },
    )
    set_renderer(Renderer(mode=OutputMode.PRETTY, command="cloud status", version="test"))
    command.status_cmd(plans=False)
    out = capsys.readouterr().out
    assert "$12.50" in out
    assert "CREATOR" in out


def test_pretty_mode_does_not_interpret_markup_from_a_server_supplied_name(monkeypatch, capsys):
    """A crafted workspace name must not inject Rich markup into the panel."""
    _install(
        monkeypatch,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 1_000_000},
            "/workspaces/current": {**_WORKSPACE, "name": "[red]pwned[/red]"},
        },
    )
    set_renderer(Renderer(mode=OutputMode.PRETTY, command="cloud status", version="test"))
    command.status_cmd(plans=False)
    out = capsys.readouterr().out
    # Rendered literally by Text(), so the tags survive as characters.
    assert "[red]pwned[/red]" in out


# ---------------------------------------------------------------------------
# Review regressions
#
# Each test below is a finding from the review of the original PR. They share a
# root cause: the original fixtures were all well-formed, so nothing exercised
# the shapes the billing endpoints actually permit.
# ---------------------------------------------------------------------------


class _ExplodingBodyHTTPError(urllib.error.HTTPError):
    """An HTTPError whose body read fails, as a reset stream would."""

    def __init__(self, url: str, code: int):
        super().__init__(url, code, f"HTTP {code}", {}, io.BytesIO(b""))
        self.read_calls: list[int | None] = []

    def read(self, amt=None):  # noqa: D102 - mirrors HTTPError.read
        self.read_calls.append(amt)
        raise OSError("connection reset while draining the error body")


def test_error_body_read_failure_does_not_escape_the_handler(monkeypatch, capsys):
    """A raise while quoting the body must not pre-empt the envelope.

    The original code called `e.read()` unguarded inside the except block, so a
    reset mid-drain turned a transient 503 into a traceback.
    """
    boom = _ExplodingBodyHTTPError(f"{_BASE}/api/billing/status", 503)
    _install(monkeypatch, {"/billing/status": boom})
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))

    with pytest.raises(typer.Exit):
        command.status_cmd(plans=False)
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["ok"] is False
    assert env["error"]["code"] == "cloud_billing_unavailable"
    assert env["error"]["details"]["body"] == ""


def test_error_body_read_is_bounded_by_the_cap_not_sliced_after(monkeypatch, capsys):
    """`e.read(n)` must carry the limit, so the server can't pick our memory use."""
    seen: list[int | None] = []

    class _RecordingHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(f"{_BASE}/api/billing/status", 503, "boom", {}, io.BytesIO(b"x" * 10_000))

        def read(self, amt=None):
            seen.append(amt)
            return b"x" * (amt or 10_000)

    _install(monkeypatch, {"/billing/status": _RecordingHTTPError()})
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))
    with pytest.raises(typer.Exit):
        command.status_cmd(plans=False)

    assert seen == [command._MAX_ERROR_BODY_BYTES]
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(env["error"]["details"]["body"]) <= command._MAX_ERROR_BODY_BYTES


def test_unsafe_base_url_produces_an_envelope_not_a_traceback(monkeypatch, capsys):
    """`assert_safe_url` raises a bare ValueError for plaintext non-loopback.

    Reachable through the documented `comfy cloud set-base-url` and
    COMFY_CLOUD_BASE_URL, neither of which validates the scheme.
    """
    _install(monkeypatch, {"/billing/status": ValueError("refusing to send request to non-https URL")})
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud status", version="test"))

    with pytest.raises(typer.Exit):
        command.status_cmd(plans=False)
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["error"]["code"] == "cloud_billing_unavailable"
    assert "https" in env["error"]["hint"]


def test_unsafe_base_url_on_an_optional_endpoint_degrades(monkeypatch, capsys):
    """The same ValueError on a non-required call must not kill the command."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": ValueError("refusing to send request to non-https URL"),
        },
    )
    assert env["ok"] is True
    assert env["data"]["balance_confirmed"] is False


# --- a failed balance fetch is not a verdict about the account -------------


def test_failed_balance_fetch_with_a_subscription_withholds_without_accusing(monkeypatch, capsys):
    """The schema-contradiction case.

    Previously this emitted `balance_confirmed: true` beside three null balance
    fields, violating cloud_status.json, which agents are told to validate
    against.
    """
    env = _run(
        monkeypatch,
        capsys,
        {"/billing/status": _ACTIVE_CREATOR, "/billing/balance": _http_error(f"{_BASE}/api/billing/balance", 503)},
    )
    data = env["data"]
    assert data["balance_confirmed"] is False
    assert data["credit_balance_usd"] is None
    assert data["effective_balance_micros"] is None
    # Says what actually happened, and does not imply a migration.
    assert data["message"] == billing.BALANCE_UNAVAILABLE_NOTE
    assert "mid-migration" not in data["message"]
    # The rest of the answer still lands.
    assert data["subscription"]["tier"] == "CREATOR"

    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(data)


def test_failed_balance_fetch_without_a_subscription_does_not_accuse_either(monkeypatch, capsys):
    """One blip must not flip the guard for a user who may hold real credit."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {"is_active": False, "has_funds": False, "subscription_status": None},
            "/billing/balance": _http_error(f"{_BASE}/api/billing/balance", 500),
        },
    )
    data = env["data"]
    assert data["balance_confirmed"] is False
    assert "Couldn't confirm an active subscription" not in (data["message"] or "")
    assert data["message"] == billing.BALANCE_UNAVAILABLE_NOTE


def test_balance_answered_with_nothing_still_trips_the_guard(monkeypatch, capsys):
    """The distinction: answered-and-empty is a verdict, failed-to-answer isn't."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {"is_active": False, "has_funds": False, "subscription_status": None},
            "/billing/balance": {},
        },
    )
    assert "Couldn't confirm an active subscription" in env["data"]["message"]


# --- malformed scalars must not violate our own schema --------------------


def test_wrongly_typed_server_scalars_are_coerced_or_dropped(monkeypatch, capsys):
    """A backend sending "3" or `is_active: "false"` must still validate."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {
                "is_active": "false",
                "has_funds": "true",
                "subscription_status": "active",
                "subscription_tier": "PRO",
                "plan_slug": "  pro-monthly  ",
                "max_seats": "40",
                "occupied_seats": 39.0,
                "renewal_date": 20260911,
            },
            "/billing/balance": {"effective_balance_micros": "1500000"},
            "/features": {"max_concurrent_jobs": "5", "free_tier_balance": {"allowance": "100", "remaining": "60"}},
            "/workspaces/current": {"id": "w-1", "name": 12345, "type": "team", "role": None},
        },
    )
    data = env["data"]
    assert data["subscription"]["is_active"] is False
    assert data["max_concurrent_jobs"] == 5
    assert data["seats"] == {"max": 40, "occupied": 39}
    assert data["free_tier_balance"]["allowance"] == 100
    # Non-string values in string-typed fields are dropped, not stringified.
    assert data["subscription"]["renewal_date"] is None
    assert data["cloud_workspace"]["name"] is None
    # The padded slug is stripped so it can match a plan and not leak padding.
    assert data["subscription"]["plan_slug"] == "pro-monthly"

    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(data)


def test_nan_balance_does_not_crash_the_command(monkeypatch, capsys):
    """`json.loads` accepts a bare NaN literal; int(nan) raises."""
    env = _run(
        monkeypatch,
        capsys,
        {"/billing/status": _ACTIVE_CREATOR, "/billing/balance": {"effective_balance_micros": float("nan")}},
    )
    assert env["ok"] is True
    assert env["data"]["credit_balance_usd"] is None


def test_padded_plan_slug_still_matches_a_plan_for_the_upgrade_suggestion(monkeypatch, capsys):
    """The strip fix, observed end to end: padding used to drop the suggestion."""
    env = _run(
        monkeypatch,
        capsys,
        {
            "/billing/status": {**_ACTIVE_CREATOR, "plan_slug": " creator-monthly "},
            "/billing/plans": {"plans": _PLANS["plans"]},
        },
    )
    assert env["data"]["upgrade_suggestion"]["plan_slug"] == "pro-monthly"


# --- panel hardening ------------------------------------------------------


def test_panel_sanitizes_escape_sequences_from_features(monkeypatch, capsys):
    """`Text` blocks markup but passes \\x1b through, so these need sanitizing.

    Reachable whenever a user points the CLI at a hostile host via the
    documented `comfy cloud set-base-url` flow.
    """
    _install(
        monkeypatch,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 1_000_000},
            "/features": {"max_concurrent_jobs": 1, "free_tier_balance": {"allowance": "\x1b]0;pwned\x07100"}},
        },
    )
    set_renderer(Renderer(mode=OutputMode.PRETTY, command="cloud status", version="test"))
    command.status_cmd(plans=False)
    out = capsys.readouterr().out
    assert "\x1b" not in out


def test_panel_falls_back_to_the_tier_default_concurrency(monkeypatch, capsys):
    """/api/features down should not hide a number the tier already tells us."""
    _install(
        monkeypatch,
        {
            "/billing/status": _ACTIVE_CREATOR,
            "/billing/balance": {"effective_balance_micros": 1_000_000},
            # /features absent -> 404
        },
    )
    set_renderer(Renderer(mode=OutputMode.PRETTY, command="cloud status", version="test"))
    command.status_cmd(plans=False)
    out = capsys.readouterr().out
    assert "Concurrency" in out
    assert "tier default" in out
