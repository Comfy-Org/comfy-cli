"""What holds the workspace's release and active-deployment slots.

Neither service reports a used count, so both are counted here from listings:
releases by walking the Builds the caller can see, active deployments from the
deploy list. The deploy list shows each deployment's serving copy, while
comfy-deploy's limit counts a deployment with any undeleted copy, so a stopped
deployment whose replacement copy is queued holds a slot the list reads as free:
the active count is a lower bound, never the service's own count. Every read made after a refusal is best effort: the refusal is the
answer, and a read that fails leaves its envelope as it was.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final, TypeVar

from comfy_cli.utils import parse_rfc3339

T = TypeVar("T")

#: Builds whose releases are read, one call each. Every plan's build cap sits well
#: under this, so it bounds a listing that runs away rather than a real workspace.
MAX_BUILDS: Final = 100
#: What one release-labels call takes.
_MAX_LABELS: Final = 100
#: comfy-deploy's rule for a deployment that holds no active slot. A deleted one
#: holds none either.
_NO_SLOT: Final = frozenset({"failed", "stopped", "stop_failed"})
_UNDATED: Final = datetime.min.replace(tzinfo=timezone.utc)
#: The field each service's GET /v1/limits always carries; a body without it is unread.
_BUILDER_SHAPE: Final = "maxRunningBuilds"
_DEPLOY_SHAPE: Final = "maxActiveDeployments"


def best_effort(call: Callable[[], T]) -> T | None:
    """``call()``, or None when it raises anything at all.

    Only for reads made to enrich an answer the command already has, where any
    failure, a malformed body included, must leave that answer as it was.
    """
    try:
        return call()
    except Exception:
        return None


def deploy_client():
    """The deploy client a build command reads deployments with.

    Its credential comes in the order ``build._builder_client`` takes one: a
    Cloud JWT handed in as ``COMFY_BUILDER_TOKEN`` first, which comfy-deploy
    accepts as well, then the deploy commands' own order.
    """
    from comfy_cli.deploy_api import DeployClient

    token = (os.environ.get("COMFY_BUILDER_TOKEN") or "").strip()
    return DeployClient(None, token) if token else DeployClient.from_credentials()


def _instant(value: object) -> datetime:
    try:
        return parse_rfc3339(value) if isinstance(value, str) else _UNDATED
    except ValueError:
        return _UNDATED


def active_deployments(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The deployments the list shows holding an active slot: undeleted, in no status that frees one.

    A lower bound on the slot holders: the list's status is the serving copy's,
    and a deployment whose serving copy is stopped can still hold a slot through
    a queued replacement copy.
    """
    active = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("deletedAt") is not None or str(row.get("status")) in _NO_SLOT:
            continue
        entry = {key: row.get(key) for key in ("id", "name", "releaseId", "status", "createdAt")}
        active.append({key: value for key, value in entry.items() if value is not None})
    return active


def label_builds(builder, active: list[dict[str, Any]]) -> None:
    """Add the ``buildId`` of each active deployment's release, in place, where the builder names it."""
    ids = sorted({row["releaseId"] for row in active if isinstance(row.get("releaseId"), str)})[:_MAX_LABELS]
    labels = best_effort(lambda: builder.release_labels(ids)) if ids else None
    builds = {label.get("releaseId"): label.get("buildId") for label in labels or () if isinstance(label, dict)}
    for row in active:
        build_id = builds.get(row.get("releaseId"))
        if isinstance(build_id, str):
            row["buildId"] = build_id


def release_holders(builder, live: Mapping[str, int] | None) -> tuple[list[dict[str, Any]], int]:
    """Each visible Build, up to ``MAX_BUILDS``, with its release count and newest
    release; and how many Builds are visible.

    ``live`` counts the undeleted deployments on each release id; with it, each
    Build also says how many reference its releases. Raises on a failed read.
    """
    builds = builder.list_builds()
    holders = []
    for build in builds[:MAX_BUILDS]:
        releases = builder.list_releases(build["id"])
        newest = max(releases, key=lambda release: _instant(release.get("createdAt")), default=None)
        holder: dict[str, Any] = {"id": build["id"], "releases": len(releases)}
        if isinstance(build.get("name"), str):
            holder["name"] = build["name"]
        if newest is not None and isinstance(newest.get("createdAt"), str):
            holder["newestReleaseAt"] = newest["createdAt"]
        if live is not None:
            holder["deployments"] = sum(live.get(release.get("id"), 0) for release in releases)
            holder["inUse"] = holder["deployments"] > 0
        holders.append(holder)
    return holders, len(builds)


def live_by_release(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many undeleted deployments, stopped ones included, reference each release."""
    counts: dict[str, int] = {}
    for row in rows:
        release_id = row.get("releaseId") if isinstance(row, Mapping) else None
        if isinstance(release_id, str) and row.get("deletedAt") is None:
            counts[release_id] = counts.get(release_id, 0) + 1
    return counts


def release_limit_details(builder) -> dict[str, Any]:
    """What a release-limit refusal adds to ``details``: the Builds holding releases, or nothing.

    The deployment read is the lesser half: when it fails the Builds are still
    listed, without ``deployments`` and ``inUse``.
    """
    live = best_effort(lambda: live_by_release(deploy_client().list_all_deployments()))
    found = best_effort(lambda: release_holders(builder, live))
    if found is None:
        return {}
    holders, visible = found
    return {"builds": holders, **({"buildsTruncated": True} if visible > len(holders) else {})}


#: Said beside every count: GET /v1/builds shows a member outside the enterprise
#: plan only their own Builds, while the caps count every member's.
NOTE: Final = (
    "releases and builds are counted over the Builds you can see, your own outside the enterprise plan, "
    "while both limits count the whole workspace, so those two can read low; active deployments are counted "
    "from the deploy list, which can miss a slot held by a deployment moving to another release, so that count "
    "is at least the number shown"
)


def _usage(used: int, limits: object, key: str, *, shape: str) -> dict[str, Any]:
    """``used`` against ``limits[key]``.

    ``limitKnown`` is true only when ``limits`` has the service's shape, its
    always-present ``shape`` field an integer: an empty or unparsable body reaches
    here as ``{}``, and must not read as no limit. A null limit that is known: no
    cap of that kind applies.
    """
    known = isinstance(limits, dict) and _is_count(limits.get(shape))
    limit = limits.get(key) if known else None
    return {"used": used, "limit": limit if _is_count(limit) else None, "limitKnown": known}


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def quota_payload(
    holders: Sequence[Mapping[str, Any]], visible: int, active: int, build_limits: object, deploy_limits: object
) -> dict[str, Any]:
    releases = _usage(
        sum(holder["releases"] for holder in holders), build_limits, "maxTotalReleases", shape=_BUILDER_SHAPE
    )
    if visible > len(holders):
        releases["truncated"] = True
    # Counted from the deploy list, which can miss a slot held by a queued copy.
    deployments = {**_usage(active, deploy_limits, "maxActiveDeployments", shape=_DEPLOY_SHAPE), "exact": False}
    return {
        "releases": releases,
        "activeDeployments": deployments,
        "builds": _usage(visible, build_limits, "maxBuilds", shape=_BUILDER_SHAPE),
        "note": NOTE,
    }


def usage_line(label: str, usage: Mapping[str, Any]) -> str:
    if not usage["limitKnown"]:
        limit = "an unknown limit"
    else:
        limit = "no limit" if usage["limit"] is None else str(usage["limit"])
    used = f"at least {usage['used']}" if usage.get("exact") is False else str(usage["used"])
    line = f"{label}: {used} of {limit}"
    if usage.get("truncated") is True:
        line += f" (counted over the first {MAX_BUILDS} Builds only; more are visible)"
    return line
