"""Parity with the frontend's pinned uuid11.1.1 validator and lowercase wire ID."""

import json
from pathlib import Path

import pytest

from comfy_cli.saved_file_identity import canonical_saved_file_id


@pytest.mark.parametrize("version", range(1, 9))
@pytest.mark.parametrize("variant", "89ab")
def test_rfc_versions_and_variants(version, variant):
    identity = f"11111111-1111-{version}111-{variant}111-111111111111"
    assert canonical_saved_file_id(identity) == identity


@pytest.mark.parametrize("identity", ["00000000-0000-0000-0000-000000000000", "ffffffff-ffff-ffff-ffff-ffffffffffff"])
def test_frontend_nil_and_max_special_cases(identity):
    assert canonical_saved_file_id(identity) == identity


@pytest.mark.parametrize(
    "identity",
    [
        None,
        True,
        123,
        "",
        "a" * 32,
        "11111111-1111-0111-8111-111111111111",
        "11111111-1111-9111-8111-111111111111",
        "11111111-1111-f111-8111-111111111111",
        "11111111-1111-4111-7111-111111111111",
        "11111111-1111-4111-c111-111111111111",
        "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF",
        "978E5F58-49DC-4A81-8235-A0B91C204A1C",
        " 978e5f58-49dc-4a81-8235-a0b91c204a1c",
        "978e5f58-49dc-4a81-8235-a0b91c204a1c\n",
    ],
)
def test_malformed_or_noncanonical_received_ids_are_not_advertised(identity):
    assert canonical_saved_file_id(identity) is None


_CONSUMER = json.loads((Path(__file__).parent / "fixtures/saved_file_uuid_consumer.json").read_text())


@pytest.mark.parametrize("case", _CONSUMER["cases"], ids=lambda case: case["id"])
def test_matches_actual_pinned_frontend_uuid_consumer(case):
    assert (canonical_saved_file_id(case["input"]) is not None) is case["accepted"]
