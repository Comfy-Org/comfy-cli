"""Billing arithmetic and display rules for ``comfy cloud status``.

Deliberately free of I/O. Every rule the command has to get right (the
zero-masking balance chain, the trust guard, the upgrade suggestion) is a pure
function here, so it can be unit-tested without mocking HTTP. ``command.py``
owns the fetching, the panel and the envelope.

The rules encoded here are not arbitrary. Two of them are scar tissue:

* The balance chain exists because a present-but-zero high-priority field used
  to mask a positive component underneath it, reporting "$0.00" to users who
  had credits.
* The trust guard exists because "$0.00 / Subscribe" was shown to paying
  customers whose billing rows were mid-migration (BE-1795 / BE-1798). When we
  cannot confirm state, we say so instead of asserting a balance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# $1 = 211 credits. Source of truth is
# ``ComfyUI_frontend/src/base/credits/comfyCredits.ts``, mirrored in
# ``comfy-cloud-mcp-server/src/pricing.ts``. Keep all three in lockstep.
CREDITS_PER_USD = 211

MICROS_PER_USD = 1_000_000
CENTS_PER_USD = 100

# Default concurrency per tier, shown as context beside the live
# ``max_concurrent_jobs`` from /api/features.
TIER_CONCURRENCY: Mapping[str, int] = {
    "FREE": 1,
    "STANDARD": 1,
    "FOUNDERS": 1,
    "CREATOR": 3,
    "PRO": 5,
    "TEAM": 25,
}

# Priority order for resolving a balance. A present-but-zero field is SKIPPED
# rather than accepted, so a literal 0 in a higher-priority field can never
# mask a positive component below it. Ported from comfy-cloud-mcp-server's
# ``selectEffectiveBalanceMicros`` (src/tools-feedback.ts).
BALANCE_FIELDS = (
    "effective_balance_micros",
    "amount_micros",
    "prepaid_balance_micros",
    "cloud_credit_balance_micros",
)

# Shown when we cannot confirm a subscription OR a balance. Never paired with a
# "$0.00" figure: asserting zero to a customer who is mid-migration is the
# exact BE-1798 failure this replaces.
UNCONFIRMED_MESSAGE = (
    "Couldn't confirm an active subscription or balance for this workspace. "
    "If you already subscribe, your account may be mid-migration; contact Comfy support. "
    "New to Comfy Cloud? Subscribe at {url}."
)

# Appended whenever the workspace is still on the pre-migration billing rail,
# whether or not the trust guard fired: figures on that rail can be partial.
LEGACY_RAIL_NOTE = "This workspace is on the legacy billing rail, so balance data may be incomplete pending migration."

# Values of ``subscription_status`` that mean "there is no subscription", as
# opposed to one that exists but is lapsed/cancelled. Kept tight on purpose:
# a lapsed subscription is confirmation the account is real, so it must not
# trip the trust guard.
_NO_SUBSCRIPTION_STATUSES = {"", "none", "null", "no_subscription"}

_LEGACY_RAIL = "legacy_stripe"


def _coerce_int(value: Any) -> int | None:
    """Best-effort int from a JSON scalar, or None if it isn't numeric.

    The billing endpoints have historically sent micro-dollar fields as both
    JSON numbers and decimal strings, so tolerate both rather than crashing on
    a shape the server is entitled to send. ``bool`` is rejected explicitly:
    it is an ``int`` subclass in Python, and a stray ``True`` coercing to 1
    would read as a one-micro balance.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (TypeError, ValueError):
            return None
    return None


def select_effective_balance_micros(balance: Mapping[str, Any] | None) -> int | None:
    """Resolve the balance in micro-dollars, or None when nothing is reported.

    Walks :data:`BALANCE_FIELDS` in priority order and returns the first
    NON-ZERO value. Returns 0 only when at least one field was present and
    every present field was zero (a real, confirmed zero balance). Returns
    None when no field was present at all, which is "unknown" and must not be
    rendered as "$0.00".

    A negative result is returned as-is: ``effective_balance_micros`` legitimately
    goes negative on an overdrawn account and the user needs to see that, so it
    is never clamped to zero.
    """
    if not isinstance(balance, Mapping):
        return None
    saw_numeric_field = False
    for name in BALANCE_FIELDS:
        value = _coerce_int(balance.get(name))
        if value is None:
            continue
        saw_numeric_field = True
        if value != 0:
            return value
    return 0 if saw_numeric_field else None


def micros_to_usd(micros: int | None) -> float | None:
    if micros is None:
        return None
    return micros / MICROS_PER_USD


def usd_to_credits(usd: float | None) -> int | None:
    """Convert USD to credits, rounding toward zero so we never overstate.

    Truncating keeps a negative balance's credit figure negative, which
    matches the USD figure it sits beside.
    """
    if usd is None:
        return None
    return int(usd * CREDITS_PER_USD)


def has_subscription(subscription_status: Any) -> bool:
    """True when ``subscription_status`` names a subscription that exists.

    A lapsed or cancelled subscription counts: it confirms the account is real,
    which is the only question the trust guard asks.
    """
    if subscription_status is None:
        return False
    return str(subscription_status).strip().lower() not in _NO_SUBSCRIPTION_STATUSES


def is_legacy_rail(billing_rail: Any) -> bool:
    return isinstance(billing_rail, str) and billing_rail.strip().lower() == _LEGACY_RAIL


def is_unconfirmed(
    *,
    balance_micros: int | None,
    subscription_status: Any,
    has_funds: Any,
) -> bool:
    """True when we cannot confirm either a subscription or a balance.

    All three signals must be negative before we withhold a figure, so a
    genuine free-tier user with a confirmed zero balance and a FREE
    subscription row still gets normal output. It is only the all-silent shape,
    where the backend told us nothing affirmative, that earns neutral copy.
    """
    balance_absent_or_zero = balance_micros is None or balance_micros == 0
    return balance_absent_or_zero and not has_subscription(subscription_status) and not bool(has_funds)


def unconfirmed_message(manage_url: str) -> str:
    return UNCONFIRMED_MESSAGE.format(url=manage_url)


def manage_url(base_url: str) -> str:
    """The subscription page for a resolved cloud base URL.

    One derivation, so a PR-preview or staging env points at its own billing
    page instead of sending the user to production.
    """
    return f"{base_url.rstrip('/')}/billing"


def _plan_slug(plan: Mapping[str, Any]) -> str | None:
    for key in ("slug", "plan_slug", "id", "name"):
        value = plan.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _plan_available(plan: Mapping[str, Any]) -> bool:
    """Whether the caller can actually buy this plan.

    Availability absent entirely is treated as available: the endpoint only
    started annotating plans later, and an older backend that omits the block
    is not saying "unavailable".
    """
    availability = plan.get("availability")
    if not isinstance(availability, Mapping):
        return True
    return bool(availability.get("available", True))


def _plan_unavailable_reason(plan: Mapping[str, Any]) -> str | None:
    availability = plan.get("availability")
    if not isinstance(availability, Mapping):
        return None
    reason = availability.get("reason")
    return reason.strip() if isinstance(reason, str) and reason.strip() else None


def normalize_plans(plans_body: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Normalize ``/api/billing/plans`` into (plans, current_plan_slug).

    Accepts either the documented object (``{current_plan_slug, plans: [...]}``)
    or a bare list of plans, since both shapes have been observed and neither
    is worth failing the whole command over.
    """
    current_slug: str | None = None
    raw_plans: Any = []
    if isinstance(plans_body, Mapping):
        current = plans_body.get("current_plan_slug")
        if isinstance(current, str) and current.strip():
            current_slug = current.strip()
        raw_plans = plans_body.get("plans")
        if not isinstance(raw_plans, list):
            raw_plans = []
    elif isinstance(plans_body, list):
        raw_plans = plans_body

    plans: list[dict[str, Any]] = []
    for entry in raw_plans:
        if not isinstance(entry, Mapping):
            continue
        slug = _plan_slug(entry)
        if slug is None:
            continue
        price_cents = _coerce_int(entry.get("price_cents"))
        credits_cents = _coerce_int(entry.get("credits_cents"))
        plans.append(
            {
                "plan_slug": slug,
                "price_cents": price_cents,
                "price_usd": (price_cents / CENTS_PER_USD) if price_cents is not None else None,
                "credits": usd_to_credits(credits_cents / CENTS_PER_USD) if credits_cents is not None else None,
                "available": _plan_available(entry),
                "unavailable_reason": _plan_unavailable_reason(entry),
            }
        )
    return plans, current_slug


def suggest_upgrade(plans: list[dict[str, Any]], current_plan_slug: str | None) -> dict[str, Any] | None:
    """Cheapest available plan priced above the current one, or None.

    Price is the ranking key rather than a hardcoded tier ladder: the tier
    names have churned (FOUNDERS, STANDARD) and ``price_cents`` comes straight
    from the endpoint, so ranking on it cannot go stale.

    When the current plan's price cannot be established we return None rather
    than guessing. Guessing is actively harmful here and prod proves it: the
    live team account reports ``plan_slug: "team-pro-monthly"`` from
    /api/billing/status while /api/billing/plans carries no such slug, and a
    "cheapest available plan" fallback then suggested a $0.00 per-credit plan
    as an *upgrade* from PRO. No suggestion is strictly better than a wrong
    one, so an unresolvable current price means we stay quiet.
    """
    priced = [p for p in plans if p["available"] and p["price_cents"] is not None]
    if not priced:
        return None

    current_price: int | None = None
    if current_plan_slug is not None:
        for plan in plans:
            if plan["plan_slug"] == current_plan_slug and plan["price_cents"] is not None:
                current_price = plan["price_cents"]
                break

    if current_price is None:
        return None
    candidates = [p for p in priced if p["price_cents"] > current_price]

    if not candidates:
        return None
    best = min(candidates, key=lambda p: p["price_cents"])
    return {
        "plan_slug": best["plan_slug"],
        "price_usd": best["price_usd"],
        "credits": best["credits"],
    }


def blocked_upgrade_reasons(plans: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Plans the caller cannot buy, with the server's stated reason.

    Surfaced so ``requires_team`` / ``exceeds_max_seats`` explain why a tier
    the user can see is not being suggested to them.
    """
    return [
        {"plan_slug": p["plan_slug"], "reason": p["unavailable_reason"]}
        for p in plans
        if not p["available"] and p["unavailable_reason"]
    ]
