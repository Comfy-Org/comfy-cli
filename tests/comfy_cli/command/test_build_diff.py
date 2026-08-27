"""The rescan merge and diff that `comfy build update` is built on."""

from __future__ import annotations

from comfy_cli.command.build_diff import diff_definitions, merge_definition, summarize_definition_diff


def _definition(*, models=(), nodes=(), **extra) -> dict:
    return {
        "schema": "distribution-definition/0",
        "models": list(models),
        "customNodes": list(nodes),
        "pipDependencies": "example==1.0.0\n",
        "environment": {"os": "Linux", "arch": "x86_64", "pythonVersion": "3.12.0", "torch": None},
        **extra,
    }


def _model(filename: str = "ae.safetensors", *, sha256: str = "d0", **extra) -> dict:
    return {
        "type": "vae",
        "filename": filename,
        "localPath": f"vae/{filename}",
        "sha256": sha256,
        "sizeBytes": 4,
        "source": "local",
        **extra,
    }


def _node(name: str = "pack", *, digest: str = "n0", **extra) -> dict:
    return {
        "name": name,
        "localPath": name,
        "repository": None,
        "gitRef": None,
        "source": "local",
        "localDigest": digest,
        "localSizeBytes": 9,
        **extra,
    }


def test_an_unchanged_rescan_merges_back_to_the_stored_document() -> None:
    # Given
    stored = _definition(models=[_model()], nodes=[_node()])

    # When
    merged = merge_definition(stored, _definition(models=[_model()], nodes=[_node()]))

    # Then
    assert merged == stored
    assert diff_definitions(stored, merged).is_empty


def test_derived_and_unknown_keys_survive_a_rescan_that_left_the_bytes_alone() -> None:
    # Given: what `push` and `pull` leave behind on an entry the scan still sees.
    stored = _definition(
        models=[_model(blobId="blob-1", sourceUri="https://hf.co/x", futureKey="keep")],
        nodes=[_node(blobId="blob-2", futureKey="keep")],
    )

    # When
    merged = merge_definition(stored, _definition(models=[_model()], nodes=[_node()]))

    # Then
    assert merged["models"][0] == stored["models"][0]
    assert merged["customNodes"][0] == stored["customNodes"][0]
    assert diff_definitions(stored, merged).is_empty


def test_a_moved_model_digest_invalidates_the_blob_and_source_uri_it_backed() -> None:
    # Given
    stored = _definition(models=[_model(sha256="old", blobId="blob-1", sourceUri="https://hf.co/x", futureKey="keep")])

    # When
    merged = merge_definition(stored, _definition(models=[_model(sha256="new")]))

    # Then
    entry = merged["models"][0]
    assert entry["sha256"] == "new"
    assert "blobId" not in entry
    assert "sourceUri" not in entry
    assert entry["futureKey"] == "keep"


def test_a_moved_node_digest_invalidates_the_blob_it_backed() -> None:
    # Given
    stored = _definition(nodes=[_node(digest="old", blobId="blob-2", futureKey="keep")])

    # When
    merged = merge_definition(stored, _definition(nodes=[_node(digest="new")]))

    # Then
    entry = merged["customNodes"][0]
    assert entry["localDigest"] == "new"
    assert "blobId" not in entry
    assert entry["futureKey"] == "keep"


def test_a_source_the_scan_no_longer_reports_disappears() -> None:
    # Given: a pack that used to be a git checkout and is now bare bytes.
    stored = _definition(nodes=[{"name": "pack", "localPath": "pack", "repository": "https://g/x", "source": "git"}])

    # When
    merged = merge_definition(stored, _definition(nodes=[_node()]))

    # Then
    assert merged["customNodes"][0]["repository"] is None
    assert merged["customNodes"][0]["source"] == "local"


def test_an_unknown_definition_key_survives_the_rescan() -> None:
    # Given
    stored = _definition(models=[_model()], futureBlock={"kept": True})

    # When
    merged = merge_definition(stored, _definition(models=[_model()]))

    # Then
    assert merged["futureBlock"] == {"kept": True}
    assert diff_definitions(stored, merged).is_empty


def test_counts_and_entries_name_every_affected_row() -> None:
    # Given
    stored = _definition(models=[_model("gone.safetensors"), _model("kept.safetensors", sha256="old")])
    scanned = _definition(models=[_model("kept.safetensors", sha256="new"), _model("fresh.safetensors")])

    # When
    diff = diff_definitions(stored, merge_definition(stored, scanned))

    # Then
    models = diff.as_json()["models"]
    assert (models["added"], models["removed"], models["changed"]) == (1, 1, 1)
    assert {(e["change"], e["name"]) for e in models["entries"]} == {
        ("added", "vae/fresh.safetensors"),
        ("removed", "vae/gone.safetensors"),
        ("changed", "vae/kept.safetensors"),
    }
    changed = next(e for e in models["entries"] if e["change"] == "changed")
    assert changed["fields"] == ["sha256"]


def test_a_single_valued_category_reports_a_bare_status() -> None:
    # Given
    stored = _definition(baseComfyVersion="v0.3.0")

    # When
    diff = diff_definitions(stored, merge_definition(stored, _definition(baseComfyVersion="v0.4.0")))

    # Then
    payload = diff.as_json()
    assert payload["baseComfyVersion"] == "changed"
    assert payload["pipDependencies"] == "unchanged"
    assert not diff.is_empty


def test_two_entries_sharing_an_identity_pair_positionally() -> None:
    # Given: only a hand-edited spec gets here, and it must stay diffable.
    stored = _definition(models=[_model(sha256="a"), _model(sha256="b")])

    # When
    diff = diff_definitions(stored, merge_definition(stored, _definition(models=[_model(sha256="a")])))

    # Then
    models = diff.as_json()["models"]
    assert (models["added"], models["removed"], models["changed"]) == (0, 1, 0)


def test_the_summary_says_so_when_nothing_moved() -> None:
    # Given
    stored = _definition(models=[_model()])

    # When / Then
    assert summarize_definition_diff(diff_definitions(stored, stored)) == "no changes"
    drifted = diff_definitions(stored, _definition())
    assert summarize_definition_diff(drifted) == "models +0 -1 ~0, customNodes +0 -0 ~0"
