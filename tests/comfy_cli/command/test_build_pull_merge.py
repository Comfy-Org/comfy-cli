"""Who owns an entry's content identity when a pull merges server onto local.

An entry whose ``source`` is ``local`` is described by bytes on this machine, so
its ``blobId``/digest/size are authoring data the server cannot know better.
Every other source is the server's to describe: letting local win there reverted
a server-side switch from a private blob to a public ``sourceUri`` on the next
push, and dropped the server's blob reference and hash under a local public
entry.
"""

from __future__ import annotations

import pytest

from comfy_cli.command.build_pull import merge_pull_definition
from comfy_cli.command.build_validation import project_wire_definition


def _definition(model: dict) -> dict:
    return {"models": [model], "customNodes": []}


def _merged_model(local: dict, server: dict) -> dict:
    merged = merge_pull_definition(_definition(local), _definition(server))
    assert len(merged["models"]) == 1
    return merged["models"][0]


def _wire_model(merged_model: dict) -> dict:
    return project_wire_definition(_definition(merged_model))["models"][0]


_LOCATION = {"type": "checkpoints", "filename": "m.safetensors"}


def test_a_local_entry_keeps_its_own_blob_digest_and_size() -> None:
    # Given a locally-authored model the server describes differently
    local = {**_LOCATION, "source": "local", "localPath": "m.safetensors", "blobId": "local-blob", "sha256": "a" * 64}
    server = {**_LOCATION, "blobId": "server-blob", "sha256": "b" * 64, "sizeBytes": 2}

    # When
    merged = _merged_model(local, server)

    # Then the bytes on this machine remain the truth
    assert merged["blobId"] == "local-blob"
    assert merged["sha256"] == "a" * 64
    assert merged["source"] == "local"
    assert merged["localPath"] == "m.safetensors"


def test_a_local_entry_still_drops_a_server_sourceuri_it_does_not_carry() -> None:
    # Given
    local = {**_LOCATION, "source": "local", "localPath": "m.safetensors", "sha256": "a" * 64}
    server = {**_LOCATION, "sha256": "a" * 64, "sourceUri": "https://cdn.test/m.safetensors"}

    # When
    merged = _merged_model(local, server)

    # Then
    assert "sourceUri" not in merged


def test_a_non_local_entry_keeps_the_servers_sourceuri() -> None:
    """The regression: a server-side switch from private blob to public URI.

    Asserted on the *wire* copy, because that is where reverting happens.
    ``MODEL_SOURCES`` is a precedence group — ``project_wire_definition`` emits
    the first source an entry names and drops the rest — so merely keeping the
    server's ``sourceUri`` proves nothing while a stale local ``blobId`` sits
    beside it outranking it.
    """
    # Given
    local = {**_LOCATION, "source": "url", "blobId": "stale-blob"}
    server = {**_LOCATION, "sourceUri": "https://cdn.test/m.safetensors"}

    # When
    merged = _merged_model(local, server)

    # Then
    assert merged["sourceUri"] == "https://cdn.test/m.safetensors"
    assert "blobId" not in merged
    assert _wire_model(merged) == {**_LOCATION, "sourceUri": "https://cdn.test/m.safetensors"}


def test_a_non_local_node_keeps_the_servers_registry_version() -> None:
    """The node shape of the same revert, which loses more when it happens.

    ``_project_node`` also drops ``gitRef``/``commit`` once ``blobId`` wins, so a
    refilled stale blob takes the git coordinates down with the registry pin.
    """
    # Given
    local = {"name": "n", "id": "pkg", "source": "registry", "blobId": "stale-node-blob"}
    server = {"name": "n", "id": "pkg", "registryVersion": "1.2.3"}

    # When
    merged = merge_pull_definition({"models": [], "customNodes": [local]}, {"models": [], "customNodes": [server]})
    node = merged["customNodes"][0]

    # Then
    assert node["registryVersion"] == "1.2.3"
    assert "blobId" not in node
    assert project_wire_definition(merged)["customNodes"][0] == {"name": "n", "id": "pkg", "registryVersion": "1.2.3"}


def test_a_non_local_entry_keeps_the_servers_blob_and_hash() -> None:
    # Given a public local entry pulled over a server-held blob
    local = {**_LOCATION, "source": "url", "blobId": "stale-blob", "sha256": "a" * 64, "sizeBytes": 1}
    server = {**_LOCATION, "blobId": "server-blob", "sha256": "b" * 64, "sizeBytes": 2}

    # When
    merged = _merged_model(local, server)

    # Then the server's reference survives instead of being clobbered
    assert merged["blobId"] == "server-blob"
    assert merged["sha256"] == "b" * 64
    assert merged["sizeBytes"] == 2


def test_a_non_local_entry_keeps_its_authoring_fields() -> None:
    # Given a server that describes none of the local authoring data
    local = {**_LOCATION, "source": "url", "localPath": "vendor/m.safetensors"}
    server = {**_LOCATION, "sourceUri": "https://cdn.test/m.safetensors"}

    # When
    merged = _merged_model(local, server)

    # Then re-owning content identity does not delete what only local knows
    assert merged["source"] == "url"
    assert merged["localPath"] == "vendor/m.safetensors"


@pytest.mark.parametrize("field", ["blobId", "sha256", "sizeBytes", "sourceUri"])
def test_a_non_local_entry_fills_a_field_the_server_named_no_source_for(field: str) -> None:
    """Server-owned means server-wins-if-present, never server-wins-by-absence.

    The server stores several definition shapes through ``ToMap``, whose
    ``omitempty`` drops empty fields entirely. The server entry here names *no*
    source at all, which is the only case where that absence is safe to fill —
    a server naming one is covered by the two revert tests above.
    """
    # Given
    value = {"blobId": "local-blob", "sha256": "a" * 64, "sizeBytes": 7, "sourceUri": "https://cdn.test/m"}[field]
    local = {**_LOCATION, "source": "url", field: value}
    server = dict(_LOCATION)

    # When
    merged = _merged_model(local, server)

    # Then
    assert merged[field] == value


@pytest.mark.parametrize("changed_to", ["https://cdn.test/NEW.safetensors", None], ids=["new_uri", "server_blob"])
def test_a_changed_source_does_not_keep_the_hash_of_the_old_one(changed_to: str | None) -> None:
    """A hash describes the bytes a source resolves to, so the two move together.

    Keeping the local ``sha256`` beside a source the server has since changed
    states an integrity claim that was never true, and the builder enforces it —
    the pull's damage would surface as a checksum mismatch at deploy staging,
    far from the command that wrote it. Dropping the hash is safe; it is
    optional and the builder recomputes.
    """
    # Given a local entry whose source and hash were taken together
    local = {**_LOCATION, "source": "url", "sourceUri": "https://cdn.test/OLD.safetensors", "sha256": "a" * 64}
    server = {**_LOCATION, "sourceUri": changed_to} if changed_to else {**_LOCATION, "blobId": "server-blob"}

    # When the server has moved the entry to a different source and states no hash
    merged = _merged_model(local, server)

    # Then the stale hash does not ride along with the new source
    assert "sha256" not in merged
    wire = _wire_model(merged)
    assert "sha256" not in wire
    assert wire == {**_LOCATION, ("sourceUri" if changed_to else "blobId"): changed_to or "server-blob"}


def test_a_server_that_names_no_source_still_keeps_the_hash() -> None:
    # Given a server that names no source at all, so nothing contradicts the hash
    local = {**_LOCATION, "source": "url", "sha256": "a" * 64}
    server = dict(_LOCATION)

    # When
    merged = _merged_model(local, server)

    # Then
    assert merged["sha256"] == "a" * 64


def test_even_an_unchanged_source_drops_a_hash_the_server_omits() -> None:
    """Matching source *strings* does not make the local hash describe them.

    The local ``sha256`` is the digest of the file on this machine, not of what
    the URI serves. The one flow that pairs the two — ``resolve_models_via_builder``
    — only assigns a ``sourceUri`` once it has confirmed the candidate's hash
    equals the local one, and it sends both together, so a server that holds
    that pairing hands the hash back and this fill never fires.

    Left over, an equal string is exactly the renamed-fine-tune hazard: two
    different files published under one URL. Dropping the hash means "do not
    verify" — weaker, but never wrong; the builder treats an absent hash as
    optional and `comfy build update` restores it from disk on the next scan.
    """
    # Given local and server naming the very same source, server stating no hash
    uri = "https://cdn.test/m.safetensors"
    local = {**_LOCATION, "source": "url", "sourceUri": uri, "sha256": "a" * 64}
    server = {**_LOCATION, "sourceUri": uri}

    # When
    merged = _merged_model(local, server)

    # Then
    assert "sha256" not in merged
    assert merged["sourceUri"] == uri


@pytest.mark.parametrize("blank", ["   ", "", 123, [], None])
def test_a_pull_does_not_carry_an_unusable_sourceuri_through(blank: object) -> None:
    """A value the wire projection would reject must not reach the spec.

    ``project_wire_definition`` raises on a non-string ``sourceUri``, so writing
    one here leaves the workspace failing its own validation on the next push.
    """
    # Given
    local = {**_LOCATION, "source": "local", "localPath": "m.safetensors", "sourceUri": blank}
    server = dict(_LOCATION)

    # When
    merged = _merged_model(local, server)

    # Then
    assert "sourceUri" not in merged
    assert _wire_model(merged) == _LOCATION
