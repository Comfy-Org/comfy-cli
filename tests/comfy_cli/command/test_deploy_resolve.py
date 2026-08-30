from __future__ import annotations

import pytest

from comfy_cli.command.build_spec import BuildSpec, JsonObject
from comfy_cli.command.deploy_resolve import (
    _STATUS_RANK,
    AmbiguousDeploymentError,
    BuildNotPushedError,
    DeploymentReference,
    NoDeployableReleaseError,
    ReleaseReference,
    ReleaseResolveRequest,
    UnrelatedDeploymentError,
    resolve_deployment,
    resolve_release,
)
from comfy_cli.deploy_api_errors import DeployAPIError


class RecordingBuilder:
    def __init__(self, *, pages: list[list[JsonObject]] | None = None, release: JsonObject | None = None) -> None:
        self.pages = pages or []
        self.release = release or {}
        self.calls: list[tuple[str, str]] = []

    def get_release(self, release_id: str) -> JsonObject:
        self.calls.append(("get_release", release_id))
        return self.release

    def list_releases(self, build_id: str) -> list[JsonObject]:
        self.calls.append(("list_releases", build_id))
        return [release for page in self.pages for release in page]


class RecordingDeployments:
    def __init__(self, pages: list[list[JsonObject]]) -> None:
        self.pages = pages
        self.calls = 0
        self.page_request_count = 0

    def list_all_deployments(self) -> list[JsonObject]:
        self.calls += 1
        deployments: list[JsonObject] = []
        for page in self.pages:
            self.page_request_count += 1
            deployments.extend(page)
        return deployments


def _spec(build_id: str | None = "build-1") -> BuildSpec:
    return {
        "schema": "comfy-build/1",
        "id": build_id,
        "name": "example",
        "description": "",
        "definition": {},
    }


def _deployment(
    deployment_id: str,
    status: str = "ready",
    created_at: str = "2026-08-23T12:00:00Z",
) -> JsonObject:
    return {
        "id": deployment_id,
        "releaseId": "release-1",
        "status": status,
        "createdAt": created_at,
        "deletedAt": None,
    }


def _deleted_candidate_clients() -> tuple[RecordingBuilder, RecordingDeployments]:
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deleted = {**_deployment("deployment-deleted"), "deletedAt": "2026-08-23T12:30:00Z"}
    live = _deployment("deployment-live", "failed")
    return builder, RecordingDeployments([[deleted, live]])


def test_deployment_id_short_circuits_release_and_spec_resolution() -> None:
    # Given
    builder = RecordingBuilder(release={"id": "release-ignored"})
    request = ReleaseResolveRequest(
        deployment_id="deployment-explicit",
        release_id="release-ignored",
        spec=None,
    )

    # When
    resolved = resolve_release(builder, request)

    # Then
    assert resolved == DeploymentReference(deployment_id="deployment-explicit")
    assert builder.calls == []


def test_explicit_release_is_fetched_without_listing_the_specs_build() -> None:
    # Given
    release = {"id": "release-explicit", "buildId": "build-other", "version": 4, "deployable": True}
    builder = RecordingBuilder(pages=[[{"id": "release-ignored"}]], release=release)
    request = ReleaseResolveRequest(release_id="release-explicit", spec=_spec())

    # When
    resolved = resolve_release(builder, request)

    # Then
    assert resolved == ReleaseReference(release=release, build_id="build-other")
    assert builder.calls == [("get_release", "release-explicit")]


def test_spec_without_build_id_fails_before_any_builder_call() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-ignored"}]])
    spec = _spec()
    del spec["id"]
    request = ReleaseResolveRequest(spec=spec)

    # When / Then
    try:
        resolve_release(builder, request)
    except BuildNotPushedError as error:
        assert error.code == "deploy_build_not_pushed"
        assert error.hint == "run `comfy build push`"
    else:
        raise AssertionError("expected deploy_build_not_pushed")
    assert builder.calls == []


def test_newest_deployable_uses_highest_version_not_list_order() -> None:
    # Given
    releases = [
        {"id": "release-5", "version": 5, "deployable": True},
        {"id": "release-3", "version": 3, "deployable": True},
        {"id": "release-8", "version": 8, "deployable": False},
    ]
    builder = RecordingBuilder(pages=[releases])

    # When
    resolved = resolve_release(builder, ReleaseResolveRequest(spec=_spec()))

    # Then
    assert isinstance(resolved, ReleaseReference)
    assert resolved.release["id"] == "release-5"


def test_resolution_consumes_releases_spanning_three_pages() -> None:
    # Given
    builder = RecordingBuilder(
        pages=[
            [{"id": "release-2", "version": 2, "deployable": True}],
            [{"id": "release-7", "version": 7, "deployable": False}],
            [{"id": "release-9", "version": 9, "deployable": True}],
        ]
    )

    # When
    resolved = resolve_release(builder, ReleaseResolveRequest(spec=_spec()))

    # Then
    assert isinstance(resolved, ReleaseReference)
    assert resolved.release["id"] == "release-9"
    assert builder.calls == [("list_releases", "build-1")]


def test_zero_releases_hints_how_to_cut_the_first_linux_nvidia_release() -> None:
    # Given
    builder = RecordingBuilder()

    # When / Then
    try:
        resolve_release(builder, ReleaseResolveRequest(spec=_spec()))
    except NoDeployableReleaseError as error:
        assert error.code == "deploy_no_deployable_release"
        assert error.hint == "run `comfy build release create --target linux/nvidia`"
    else:
        raise AssertionError("expected deploy_no_deployable_release")


def test_cpu_only_releases_name_the_missing_linux_nvidia_artifact() -> None:
    # Given
    builder = RecordingBuilder(
        pages=[
            [
                {
                    "id": "release-cpu-1",
                    "version": 1,
                    "deployable": False,
                    "artifacts": [{"os": "linux", "gpu": "cpu"}],
                },
                {
                    "id": "release-cpu-2",
                    "version": 2,
                    "deployable": False,
                    "artifacts": [{"os": "linux", "gpu": "cpu"}],
                },
            ]
        ]
    )

    # When / Then
    try:
        resolve_release(builder, ReleaseResolveRequest(spec=_spec()))
    except NoDeployableReleaseError as error:
        assert error.code == "deploy_no_deployable_release"
        assert error.hint == (
            "no `linux/nvidia` artifact exists in this Build's releases; "
            "run `comfy build release create --target linux/nvidia`"
        )
    else:
        raise AssertionError("expected deploy_no_deployable_release")


def test_status_rank_contains_exactly_the_server_enum() -> None:
    # Given
    server_statuses = (
        "queued",
        "provisioning",
        "starting",
        "ready",
        "unhealthy",
        "stopping",
        "stop_failed",
        "stopped",
        "failed",
    )

    # When / Then
    assert set(_STATUS_RANK) == set(server_statuses)


def test_ready_beats_stopped() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deployments = RecordingDeployments(
        [[_deployment("deployment-stopped", "stopped"), _deployment("deployment-ready")]]
    )

    # When
    resolved = resolve_deployment(builder, deployments, "build-1")

    # Then
    assert resolved == _deployment("deployment-ready")


def test_unhealthy_beats_starting() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deployments = RecordingDeployments(
        [[_deployment("deployment-starting", "starting"), _deployment("deployment-unhealthy", "unhealthy")]]
    )

    # When
    resolved = resolve_deployment(builder, deployments, "build-1")

    # Then
    assert resolved == _deployment("deployment-unhealthy", "unhealthy")


def test_newest_created_at_breaks_a_status_tie() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    newest = _deployment("deployment-new", created_at="2026-08-23T13:00:00Z")
    deployments = RecordingDeployments([[newest, _deployment("deployment-old")]])

    # When
    resolved = resolve_deployment(builder, deployments, "build-1")

    # Then
    assert resolved == newest


def test_identical_rank_and_created_at_raise_structured_ambiguity() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deployments = RecordingDeployments([[_deployment("deployment-b"), _deployment("deployment-a")]])

    # When / Then
    try:
        resolve_deployment(builder, deployments, "build-1")
    except AmbiguousDeploymentError as error:
        assert error.code == "deploy_ambiguous_deployment"
        assert error.details["candidateIds"] == ["deployment-a", "deployment-b"]
        assert "deployment-a" in str(error)
        assert "deployment-b" in str(error)
        assert "--deployment" in error.hint
    else:
        raise AssertionError("expected deploy_ambiguous_deployment")


def test_a_named_deployment_settles_an_otherwise_ambiguous_build() -> None:
    """The hint on `deploy_ambiguous_deployment` promises this exact recovery."""
    # Given the tie that raises the ambiguity
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deployments = RecordingDeployments([[_deployment("deployment-b"), _deployment("deployment-a")]])

    # When
    resolved = resolve_deployment(builder, deployments, "build-1", deployment_id="deployment-a")

    # Then
    assert resolved == _deployment("deployment-a")


def test_a_named_deployment_is_taken_over_the_ranking() -> None:
    # Given a build whose ranking would pick the newer row
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    newest = _deployment("deployment-new", created_at="2026-08-23T13:00:00Z")
    deployments = RecordingDeployments([[newest, _deployment("deployment-old")]])

    # When
    resolved = resolve_deployment(builder, deployments, "build-1", deployment_id="deployment-old")

    # Then ranking past an id the user typed would act on the wrong deployment
    assert resolved == _deployment("deployment-old")


def test_a_deployment_from_another_build_is_refused_rather_than_acted_on() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deployments = RecordingDeployments([[_deployment("deployment-a")]])

    # When / Then
    with pytest.raises(UnrelatedDeploymentError) as caught:
        resolve_deployment(builder, deployments, "build-1", deployment_id="deployment-elsewhere")
    error = caught.value
    assert error.code == "deploy_unrelated_deployment"
    assert error.details["deploymentId"] == "deployment-elsewhere"
    assert error.details["candidateIds"] == ["deployment-a"]
    assert error.details["scope"] == "the live deployments of Build build-1"


def test_a_named_deployment_is_refused_when_the_build_has_none() -> None:
    """Naming an id must not be answered with "no deployment".

    On the `up` path that answer falls through to the create branch; here it
    exits 0 with a null payload, so a typo reads as an undeployed Build.
    """
    # Given a Build with no live deployment at all
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deployments = RecordingDeployments([[]])

    # When / Then
    with pytest.raises(UnrelatedDeploymentError) as caught:
        resolve_deployment(builder, deployments, "build-1", deployment_id="deployment-elsewhere")
    error = caught.value
    assert error.details["candidateIds"] == []
    assert "drop `--deployment`" in error.hint


def test_deleted_rows_are_excluded_by_default() -> None:
    # Given
    builder, deployments = _deleted_candidate_clients()

    # When
    resolved = resolve_deployment(builder, deployments, "build-1")

    # Then
    assert resolved is not None
    assert resolved["id"] == "deployment-live"


def test_deleted_rows_are_included_when_requested() -> None:
    # Given
    builder, deployments = _deleted_candidate_clients()

    # When
    resolved = resolve_deployment(builder, deployments, "build-1", include_deleted=True)

    # Then
    assert resolved is not None
    assert resolved["id"] == "deployment-deleted"


def test_zero_matching_deployments_returns_none() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    other_release = {**_deployment("deployment-other"), "releaseId": "release-other"}
    deployments = RecordingDeployments([[other_release]])

    # When
    resolved = resolve_deployment(builder, deployments, "build-1")

    # Then
    assert resolved is None


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_deployment("deployment-bad", created_at="not-a-timestamp"), id="unparsable-createdAt"),
        pytest.param(_deployment("deployment-bad", status="unheard-of"), id="unknown-status"),
    ],
)
def test_a_malformed_selection_field_is_a_server_shape_error_not_a_traceback(row: JsonObject) -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    deployments = RecordingDeployments([[row]])

    # When / Then
    with pytest.raises(DeployAPIError) as raised:
        resolve_deployment(builder, deployments, "build-1")
    assert raised.value.code == "deploy_server_error"


def test_a_naive_timestamp_never_makes_the_ranking_comparison_raise() -> None:
    """Both rows share a status so the ranks tie and ``max`` is forced onto the
    timestamps; mixing aware and naive there raises TypeError without the UTC
    normalization in ``deployment_selection_key``."""
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}]])
    aware = _deployment("deployment-aware", status="ready", created_at="2026-08-23T11:00:00Z")
    naive = _deployment("deployment-naive", status="ready", created_at="2026-08-23T12:00:00")
    deployments = RecordingDeployments([[aware, naive]])
    assert _STATUS_RANK[str(aware["status"])] == _STATUS_RANK[str(naive["status"])]

    # When
    resolved = resolve_deployment(builder, deployments, "build-1")

    # Then
    assert resolved is not None
    assert resolved["id"] == "deployment-naive"


def test_join_consumes_three_pages_from_each_exhaustive_collector() -> None:
    # Given
    builder = RecordingBuilder(pages=[[{"id": "release-1"}], [{"id": "release-2"}], [{"id": "release-page-3"}]])
    target = {**_deployment("deployment-target"), "releaseId": "release-page-3"}
    deployments = RecordingDeployments(
        [
            [{**_deployment("deployment-other-1"), "releaseId": "release-other-1"}],
            [{**_deployment("deployment-other-2"), "releaseId": "release-other-2"}],
            [target],
        ]
    )

    # When
    resolved = resolve_deployment(builder, deployments, "build-1")

    # Then
    assert resolved == target
    assert builder.calls == [("list_releases", "build-1")]
    assert deployments.calls == 1
    assert deployments.page_request_count == len(deployments.pages)
