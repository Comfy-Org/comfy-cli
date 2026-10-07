"""Move one deployment onto the release another one serves."""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass, replace

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_resolve import BuilderReleaseClient, DeployResolveError
from comfy_cli.command.deploy_types import (
    DeployUpClient,
    move_changed,
    optional_revision,
    release_summary,
    required_string,
)
from comfy_cli.command.deploy_up import raise_unless_landed, refuse_unmovable
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge


class PromoteUnavailableError(DeployResolveError):
    code = "deploy_updates_unavailable"

    def __init__(self, deployment_id: str, release_id: str) -> None:
        self.hint = (
            f"run `comfy deploy up --create --release {release_id} --gpu <class> --region <region>` "
            "from the source's Build folder to start a separate deployment on that release"
        )
        self.details = {"deployment_id": deployment_id}
        super().__init__(
            f"deployment {deployment_id} cannot move to another deployment's release, because deployment "
            "updates are not on for this workspace yet"
        )


@dataclass(frozen=True, slots=True)
class PromoteResult:
    deployment: JsonObject
    source_id: str
    release: JsonObject
    previous_release: JsonObject
    changed: bool

    @property
    def waiting(self) -> bool:
        return isinstance(self.deployment.get("pendingUpdate"), dict)

    def payload(self) -> JsonObject:
        deployment: JsonObject = {
            "id": required_string(self.deployment, "id"),
            "status": required_string(self.deployment, "status"),
        }
        revision = optional_revision(self.deployment)
        if revision is not None:
            deployment["revision"] = revision
        return {
            "deployment": deployment,
            "source": {"id": self.source_id},
            "release": self.release,
            "previousRelease": self.previous_release,
        }


def _moved_to(moved: JsonObject, source_release_id: str) -> str:
    """The release the reply says TARGET moves to.

    A reply with no revision is one the service's rollout check failed to
    answer, so it carries neither the waiting update nor the new release; the
    service resolved SOURCE's release, which was read just before.
    """
    pending = moved.get("pendingUpdate")
    if isinstance(pending, dict):
        return required_string(pending, "releaseId")
    if optional_revision(moved) is None:
        return source_release_id
    return required_string(moved, "releaseId")


def _late_release(builder: BuilderReleaseClient, release_id: str) -> JsonObject:
    # SOURCE moved between its read and the promote. The move is already
    # accepted, so a failed lookup costs the version in the output, not the watch.
    try:
        return release_summary(builder.get_release(release_id))
    except (DeployAPIError, ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError):
        return {"id": release_id}


def promote(builder: BuilderReleaseClient, client: DeployUpClient, source_id: str, target_id: str) -> PromoteResult:
    """Point TARGET at SOURCE's release; the service resolves which release that is."""
    source_release_id = required_string(client.get_deployment(source_id), "releaseId")
    target = client.get_deployment(target_id)
    base_revision = optional_revision(target)
    if base_revision is None:
        raise PromoteUnavailableError(target_id, source_release_id)
    refuse_unmovable(target, command="promote")
    previous = release_summary(builder.get_release(required_string(target, "releaseId")))
    source_release = (
        previous if source_release_id == previous["id"] else release_summary(builder.get_release(source_release_id))
    )
    moved = client.promote_deployment(target_id, base_revision, source_id)
    if optional_revision(moved) is None:
        # The reply says nothing about the move; one read usually does.
        moved = client.get_deployment(target_id)
    release_id = _moved_to(moved, source_release_id)
    changed = move_changed(moved, base_revision, release_id, previous["id"])
    known = {previous["id"]: previous, source_release["id"]: source_release}
    release = known.get(release_id) or _late_release(builder, release_id)
    return PromoteResult(moved, source_id, release, previous, changed)


def finish_promote(result: PromoteResult, watched: JsonObject) -> PromoteResult:
    """The watched result, once the move it followed landed."""
    raise_unless_landed(watched, result.release, result.previous_release)
    return replace(result, deployment=watched)
