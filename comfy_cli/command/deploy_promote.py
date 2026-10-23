"""Move one deployment onto the release another one serves."""

from __future__ import annotations

from comfy_cli.command.deploy_resolve import BuilderReleaseClient, DeployResolveError
from comfy_cli.command.deploy_types import (
    DeployUpClient,
    MoveResult,
    move_changed,
    moved_release,
    optional_revision,
    release_summary,
    required_string,
)
from comfy_cli.command.deploy_up import refuse_unmovable, release_or_id


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


def promote(builder: BuilderReleaseClient, client: DeployUpClient, source_id: str, target_id: str) -> MoveResult:
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
    # The service resolved SOURCE's release, which was read just before.
    release_id = moved_release(moved, source_release_id)
    changed = move_changed(moved, base_revision, release_id, previous["id"])
    known = {previous["id"]: previous, source_release["id"]: source_release}
    # SOURCE may have moved between its read and the promote.
    release = known.get(release_id) or release_or_id(builder, release_id)
    return MoveResult(moved, release, previous, changed, source_id=source_id)
