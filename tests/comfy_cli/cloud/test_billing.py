"""Unit tests for the pure billing rules behind ``comfy cloud status``.

The command-level tests in ``test_status_command.py`` cover these through the
envelope; these pin the arithmetic and the shape tolerance directly, where the
edge cases are cheap to enumerate. The shape tolerance is not academic: the
billing endpoints have sent micro-dollar fields as both JSON numbers and
decimal strings, and ``/api/billing/plans`` has been observed as both an object
and a bare list.
"""

from __future__ import annotations

import pytest

from comfy_cli.cloud import billing

# ---------------------------------------------------------------------------
# select_effective_balance_micros
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("balance", "expected"),
    [
        # Highest-priority field wins when it is non-zero.
        ({"effective_balance_micros": 5, "amount_micros": 9}, 5),
        # A present-but-zero field is skipped, not accepted.
        ({"effective_balance_micros": 0, "amount_micros": 9}, 9),
        ({"effective_balance_micros": 0, "amount_micros": 0, "prepaid_balance_micros": 7}, 7),
        ({"effective_balance_micros": 0, "cloud_credit_balance_micros": 3}, 3),
        # Negative is a real balance and is returned as-is.
        ({"effective_balance_micros": -100}, -100),
        # A lower-priority positive does not override a negative above it.
        ({"effective_balance_micros": -100, "amount_micros": 500}, -100),
        # Every present field is zero: a confirmed zero, not unknown.
        ({"effective_balance_micros": 0, "amount_micros": 0}, 0),
        # Nothing present at all: unknown, which must stay distinct from zero.
        ({}, None),
        ({"unrelated": 5}, None),
        (None, None),
        # Strings and floats are tolerated.
        ({"effective_balance_micros": "2500000"}, 2_500_000),
        ({"effective_balance_micros": 2_500_000.9}, 2_500_000),
        ({"effective_balance_micros": " 42 "}, 42),
        # Non-numeric junk is ignored rather than crashing the command.
        ({"effective_balance_micros": "abc", "amount_micros": 8}, 8),
        ({"effective_balance_micros": None, "amount_micros": 8}, 8),
        # A stray bool must not read as a one-micro balance.
        ({"effective_balance_micros": True, "amount_micros": 8}, 8),
    ],
)
def test_select_effective_balance_micros(balance, expected):
    assert billing.select_effective_balance_micros(balance) == expected


def test_unknown_and_zero_are_distinguishable():
    """The distinction the trust guard depends on."""
    assert billing.select_effective_balance_micros({}) is None
    assert billing.select_effective_balance_micros({"amount_micros": 0}) == 0


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def test_micros_and_credits_conversions():
    assert billing.micros_to_usd(12_500_000) == pytest.approx(12.5)
    assert billing.micros_to_usd(None) is None
    assert billing.usd_to_credits(1.0) == billing.CREDITS_PER_USD
    assert billing.usd_to_credits(None) is None


def test_credits_truncate_toward_zero_in_both_directions():
    """Truncation keeps a negative balance's credit figure negative."""
    assert billing.usd_to_credits(0.9) == int(0.9 * billing.CREDITS_PER_USD)
    assert billing.usd_to_credits(-2.25) < 0


# ---------------------------------------------------------------------------
# Trust guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, False),
        ("", False),
        ("none", False),
        ("NONE", False),
        ("no_subscription", False),
        ("active", True),
        # A lapsed subscription still confirms the account is real.
        ("past_due", True),
        ("canceled", True),
    ],
)
def test_has_subscription(status, expected):
    assert billing.has_subscription(status) is expected


def test_is_unconfirmed_requires_all_three_signals_negative():
    negative = {"balance_micros": None, "subscription_status": None, "has_funds": False}
    assert billing.is_unconfirmed(**negative) is True

    # Any single affirmative signal is enough to report normally.
    assert billing.is_unconfirmed(**{**negative, "balance_micros": 100}) is False
    assert billing.is_unconfirmed(**{**negative, "subscription_status": "active"}) is False
    assert billing.is_unconfirmed(**{**negative, "has_funds": True}) is False


def test_is_unconfirmed_treats_confirmed_zero_as_unconfirmed_only_when_alone():
    """A zero balance is not itself affirmative, but a subscription is."""
    assert billing.is_unconfirmed(balance_micros=0, subscription_status=None, has_funds=False) is True
    assert billing.is_unconfirmed(balance_micros=0, subscription_status="active", has_funds=False) is False


def test_unconfirmed_message_carries_the_env_specific_url():
    msg = billing.unconfirmed_message("https://preview.example/billing")
    assert "https://preview.example/billing" in msg
    assert "Couldn't confirm" in msg


def test_manage_url_derives_from_base_and_tolerates_a_trailing_slash():
    assert billing.manage_url("https://cloud.comfy.org") == "https://cloud.comfy.org/billing"
    assert billing.manage_url("https://cloud.comfy.org/") == "https://cloud.comfy.org/billing"


@pytest.mark.parametrize(
    ("rail", "expected"),
    [("legacy_stripe", True), ("LEGACY_STRIPE", True), ("stripe", False), (None, False), (7, False)],
)
def test_is_legacy_rail(rail, expected):
    assert billing.is_legacy_rail(rail) is expected


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


_PLAN_LIST = [
    {"slug": "free", "price_cents": 0, "credits_cents": 0},
    {"slug": "creator", "price_cents": 2000, "credits_cents": 2000},
    {"slug": "pro", "price_cents": 5000, "credits_cents": 5500},
    {"slug": "team", "price_cents": 20000, "credits_cents": 22000, "availability": {"available": True}},
    {"slug": "ent", "price_cents": 30000, "availability": {"available": False, "reason": "requires_team"}},
]


def test_normalize_plans_accepts_the_documented_object():
    plans, current = billing.normalize_plans({"current_plan_slug": "creator", "plans": _PLAN_LIST})
    assert current == "creator"
    assert [p["plan_slug"] for p in plans] == ["free", "creator", "pro", "team", "ent"]
    assert plans[1]["price_usd"] == pytest.approx(20.0)
    assert plans[1]["credits"] == billing.usd_to_credits(20.0)


def test_normalize_plans_accepts_a_bare_list():
    """Observed shape; tolerated rather than failing the whole command."""
    plans, current = billing.normalize_plans(_PLAN_LIST)
    assert current is None
    assert len(plans) == 5


@pytest.mark.parametrize("body", [None, {}, [], "nope", 7, {"plans": "nope"}])
def test_normalize_plans_degrades_on_junk(body):
    plans, current = billing.normalize_plans(body)
    assert plans == []
    assert current is None


def test_normalize_plans_skips_entries_without_a_usable_slug():
    plans, _ = billing.normalize_plans([{"price_cents": 100}, "junk", {"slug": "ok", "price_cents": 1}])
    assert [p["plan_slug"] for p in plans] == ["ok"]


def test_missing_availability_block_means_available():
    """An older backend that omits availability is not saying 'unavailable'."""
    plans, _ = billing.normalize_plans([{"slug": "creator", "price_cents": 2000}])
    assert plans[0]["available"] is True


def test_suggest_upgrade_picks_cheapest_above_current():
    plans, current = billing.normalize_plans({"current_plan_slug": "creator", "plans": _PLAN_LIST})
    suggestion = billing.suggest_upgrade(plans, current)
    assert suggestion["plan_slug"] == "pro"
    assert suggestion["price_usd"] == pytest.approx(50.0)


def test_suggest_upgrade_skips_unavailable_plans():
    plans, _ = billing.normalize_plans({"current_plan_slug": "team", "plans": _PLAN_LIST})
    # Only `ent` is priced above team, and it is unavailable.
    assert billing.suggest_upgrade(plans, "team") is None


def test_suggest_upgrade_on_the_top_plan_returns_none():
    plans, _ = billing.normalize_plans({"current_plan_slug": "ent", "plans": _PLAN_LIST})
    assert billing.suggest_upgrade(plans, "ent") is None


def test_suggest_upgrade_stays_quiet_when_the_current_price_is_unknown():
    """No suggestion beats a wrong one.

    Observed in prod: /api/billing/status reports plan_slug
    "team-pro-monthly" while /api/billing/plans carries no such slug. A
    "cheapest available" fallback then suggested a $0.00 per-credit plan as an
    upgrade from PRO, which is worse than saying nothing.
    """
    plans, _ = billing.normalize_plans(_PLAN_LIST)
    assert billing.suggest_upgrade(plans, None) is None
    assert billing.suggest_upgrade(plans, "team-pro-monthly-not-in-list") is None


def test_suggest_upgrade_from_a_zero_priced_plan_still_works():
    """A known price of 0 is resolvable, unlike an unmatched slug."""
    plans, _ = billing.normalize_plans(_PLAN_LIST)
    suggestion = billing.suggest_upgrade(plans, "free")
    assert suggestion["plan_slug"] == "creator"


def test_suggest_upgrade_never_suggests_a_cheaper_or_equal_plan():
    plans, _ = billing.normalize_plans(_PLAN_LIST)
    for current in ("free", "creator", "pro", "team", "ent"):
        suggestion = billing.suggest_upgrade(plans, current)
        if suggestion is None:
            continue
        current_price = next(p["price_cents"] for p in plans if p["plan_slug"] == current)
        suggested_price = next(p["price_cents"] for p in plans if p["plan_slug"] == suggestion["plan_slug"])
        assert suggested_price > current_price


def test_blocked_upgrade_reasons_reports_only_annotated_unavailable_plans():
    plans, _ = billing.normalize_plans(_PLAN_LIST)
    assert billing.blocked_upgrade_reasons(plans) == [{"plan_slug": "ent", "reason": "requires_team"}]


def test_tier_concurrency_covers_the_documented_tiers():
    assert billing.TIER_CONCURRENCY["FREE"] == 1
    assert billing.TIER_CONCURRENCY["CREATOR"] == 3
    assert billing.TIER_CONCURRENCY["PRO"] == 5
    assert billing.TIER_CONCURRENCY["TEAM"] == 25
