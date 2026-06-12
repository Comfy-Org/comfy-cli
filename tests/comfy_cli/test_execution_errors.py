"""Tests for execution-failure parsing/classification (comfy_cli.execution_errors).

Pins the two contracts the June 2026 agent field test surfaced:

1. The cloud API-node session-token expiry ("Unauthorized: Please login first
   to use this node") classifies as ``transient_auth`` with a resubmit hint —
   not as a generic ``execution_error``, and never as advice to re-login.
2. Envelope verdicts carry the one-line cause + a short traceback tail, never
   the full raw server traceback (which used to land in the envelope twice).
"""

from __future__ import annotations

import json

from comfy_cli import error_codes, execution_errors

_TRACEBACK = [
    '  File "/app/comfyui/execution.py", line 455, in execute\n    raise ex\n',
    '  File "/app/comfyui/comfy_api_nodes/util/client.py", line 163, in poll_op\n    raw = await poll_op_raw(\n',
    '  File "/app/comfyui/comfy_api_nodes/util/client.py", line 436, in poll_op_raw\n    raise Exception(...)\n',
]


def _cloud_error_message(exception_message: str) -> str:
    """Build the JSON-encoded error_message string the cloud status API returns."""
    return json.dumps(
        {
            "exception_message": exception_message,
            "exception_type": "Exception",
            "node_id": "1",
            "node_type": "KlingImage2VideoNode",
            "traceback": _TRACEBACK,
        }
    )


# --- parse_error_message ----------------------------------------------------


def test_parse_json_encoded_string():
    parsed = execution_errors.parse_error_message(_cloud_error_message("boom"))
    assert parsed["exception_message"] == "boom"
    assert parsed["node_type"] == "KlingImage2VideoNode"
    assert parsed["node_id"] == "1"
    assert parsed["exception_type"] == "Exception"


def test_parse_truncates_traceback_to_tail():
    parsed = execution_errors.parse_error_message(_cloud_error_message("boom"))
    assert parsed["traceback_tail"] == _TRACEBACK[-2:]


def test_parse_accepts_decoded_dict():
    parsed = execution_errors.parse_error_message({"exception_message": "x", "node_type": "T"})
    assert parsed["exception_message"] == "x"
    assert parsed["node_type"] == "T"


def test_parse_plain_text_and_none():
    assert execution_errors.parse_error_message("plain failure\n") == {"exception_message": "plain failure"}
    assert execution_errors.parse_error_message(None) == {"exception_message": ""}


def test_parse_non_dict_json_falls_back_to_text():
    assert execution_errors.parse_error_message('["a", "b"]') == {"exception_message": '["a", "b"]'}


# --- classify ---------------------------------------------------------------


def test_transient_auth_direct():
    raw = _cloud_error_message("Unauthorized: Please login first to use this node.\n")
    verdict = execution_errors.classify(raw)
    assert verdict["code"] == "transient_auth"
    assert "resubmit" in verdict["hint"]
    assert "cloud login" in verdict["hint"]


def test_transient_auth_wrapped_in_polling_abort():
    raw = _cloud_error_message("Polling aborted due to error: Unauthorized: Please login first to use this node.\n")
    verdict = execution_errors.classify(raw)
    assert verdict["code"] == "transient_auth"


def test_ordinary_failure_stays_execution_error():
    raw = _cloud_error_message("The 'grok-imagine-video-1.5' model requires an input image.")
    verdict = execution_errors.classify(raw)
    assert verdict["code"] == "execution_error"


def test_message_is_one_line_with_node_prefix():
    raw = _cloud_error_message("boom")
    verdict = execution_errors.classify(raw)
    assert verdict["message"] == "KlingImage2VideoNode (node 1): boom"


def test_verdict_never_carries_full_traceback():
    raw = _cloud_error_message("boom")
    verdict = execution_errors.classify(raw)
    assert "traceback" not in verdict["details"]
    assert verdict["details"]["traceback_tail"] == _TRACEBACK[-2:]
    # The frames above the tail are gone from the verdict entirely.
    assert _TRACEBACK[0] not in json.dumps(verdict)


def test_classify_handles_empty_input():
    verdict = execution_errors.classify(None)
    assert verdict["code"] == "execution_error"
    assert verdict["message"] == "ComfyUI reported an execution error."


def test_verdict_codes_are_registered():
    for raw in (None, _cloud_error_message("Unauthorized: Please login first to use this node.")):
        assert error_codes.is_registered(execution_errors.classify(raw)["code"])
