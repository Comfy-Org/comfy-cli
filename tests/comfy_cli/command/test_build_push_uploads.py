"""What `pending_uploads` still owes the builder after blobs start landing.

`upload_assets` writes each ``blobId`` back into the definition as its blob
lands, and that definition — not the plan drawn before the first PUT — is the
record of what is left. A node archive used to be exempt from that re-read, so
asking a second time reported every completed node as still pending.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from comfy_cli.command.build_push import (
    PushPreparation,
    PushUpload,
    pending_uploads,
    upload_assets,
)


def _preparation(models: list[dict], nodes: list[dict], uploads: tuple[PushUpload, ...]) -> PushPreparation:
    spec = {"definition": {"models": models, "customNodes": nodes}}
    return PushPreparation(spec, uploads, ())


class _RecordingClient:
    def __init__(self, upload_url: str | None = "https://blobs.test/put") -> None:
        self.upload_url = upload_url
        self.created: list[tuple[str, str]] = []
        self.uploaded: list[str] = []

    def create_blob(self, kind: str, filename: str, sha256: str, size_bytes: int) -> tuple[str, str | None]:
        self.created.append((kind, filename))
        return f"blob-{filename}", self.upload_url

    def upload_blob(self, upload_url: str, path: Path) -> None:
        self.uploaded.append(upload_url)


def test_a_node_archive_that_already_landed_is_no_longer_pending(tmp_path: Path) -> None:
    # Given a node whose blob landed on an earlier run
    archive = tmp_path / "node-0.zip"
    archive.write_bytes(b"ZIP")
    uploads = (PushUpload("node_zip", "customNodes", 0, "n.zip", "d" * 64, 3, archive),)
    preparation = _preparation([], [{"source": "local", "blobId": "blob-n.zip"}], uploads)

    # When / Then
    assert pending_uploads(preparation) == ()


def test_a_node_archive_without_a_blob_is_still_pending(tmp_path: Path) -> None:
    # Given
    archive = tmp_path / "node-0.zip"
    archive.write_bytes(b"ZIP")
    uploads = (PushUpload("node_zip", "customNodes", 0, "n.zip", "d" * 64, 3, archive),)
    preparation = _preparation([], [{"source": "local"}], uploads)

    # When / Then
    assert pending_uploads(preparation) == uploads


def test_a_node_archive_is_not_resolved_by_a_sourceuri(tmp_path: Path) -> None:
    """Only a ``blobId`` satisfies a packaged archive.

    ``prepare_push`` plans a local node on ``blobId`` alone — a model is the one
    kind a public ``sourceUri`` can stand in for — so reading both fields here
    would drop an upload the plan still owes.
    """
    # Given
    archive = tmp_path / "node-0.zip"
    archive.write_bytes(b"ZIP")
    uploads = (PushUpload("node_zip", "customNodes", 0, "n.zip", "d" * 64, 3, archive),)
    preparation = _preparation([], [{"source": "local", "sourceUri": "https://example.test/n.zip"}], uploads)

    # When / Then
    assert pending_uploads(preparation) == uploads


@pytest.mark.parametrize("resolved_by", ["blobId", "sourceUri"])
def test_a_resolved_model_is_no_longer_pending(tmp_path: Path, resolved_by: str) -> None:
    # Given
    model_file = tmp_path / "m.safetensors"
    model_file.write_bytes(b"MODEL")
    uploads = (PushUpload("model", "models", 0, "m.safetensors", "a" * 64, 5, model_file),)
    resolved = {"blobId": "blob-1", "sourceUri": "https://example.test/m"}[resolved_by]
    preparation = _preparation([{"source": "local", resolved_by: resolved}], [], uploads)

    # When / Then
    assert pending_uploads(preparation) == ()


def test_a_second_pass_after_uploading_owes_the_builder_nothing(tmp_path: Path) -> None:
    """The regression: re-asking after a completed run must not requeue.

    A repeated ``create_blob`` for content the builder already stored mints a
    *fresh* id, orphaning the bytes held under the first one.
    """
    # Given
    archive = tmp_path / "node-0.zip"
    archive.write_bytes(b"ZIP")
    model_file = tmp_path / "m.safetensors"
    model_file.write_bytes(b"MODEL")
    uploads = (
        PushUpload("model", "models", 0, "m.safetensors", "a" * 64, 5, model_file),
        PushUpload("node_zip", "customNodes", 0, "n.zip", "d" * 64, 3, archive),
    )
    preparation = _preparation([{"source": "local"}], [{"source": "local"}], uploads)
    client = _RecordingClient()

    # When
    assert upload_assets(preparation, client) == 2
    remaining = pending_uploads(preparation)

    # Then
    assert remaining == ()
    assert upload_assets(preparation, client) == 0
    assert client.created == [("model", "m.safetensors"), ("node_zip", "n.zip")]


def test_a_deduplicated_blob_is_recorded_without_being_uploaded(tmp_path: Path) -> None:
    # Given a builder that already holds the content and returns no upload URL
    archive = tmp_path / "node-0.zip"
    archive.write_bytes(b"ZIP")
    uploads = (PushUpload("node_zip", "customNodes", 0, "n.zip", "d" * 64, 3, archive),)
    preparation = _preparation([], [{"source": "local"}], uploads)
    client = _RecordingClient(upload_url=None)

    # When
    transferred = upload_assets(preparation, client)

    # Then
    assert transferred == 0
    assert client.uploaded == []
    assert preparation.definition["customNodes"][0]["blobId"] == "blob-n.zip"
    assert pending_uploads(preparation) == ()
