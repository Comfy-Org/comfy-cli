"""On-disk job state: round-trips + terminal handling."""

from __future__ import annotations

import json

from comfy_cli import jobs_state

# The autouse ``_isolate_jobs_state_dir`` fixture in ``tests/comfy_cli/conftest.py``
# pins ``jobs_state.state_dir`` to a per-test tmp path. No local fixture needed.


class TestRoundTrip:
    def test_new_and_read_returns_equal_state(self):
        s = jobs_state.new(
            prompt_id="abc-123",
            client_id="cid",
            workflow="/tmp/x.json",
            where="local",
            host="127.0.0.1",
            port=8188,
        )
        jobs_state.write(s)
        loaded = jobs_state.read("abc-123")
        assert loaded is not None
        assert loaded.prompt_id == "abc-123"
        assert loaded.workflow == "/tmp/x.json"
        assert loaded.status == "queued"
        assert loaded.host == "127.0.0.1"

    def test_read_missing_returns_none(self):
        assert jobs_state.read("nonexistent") is None

    def test_read_tolerates_unknown_keys(self):
        s = jobs_state.new(prompt_id="x", client_id=None, workflow="/w.json", where="local")
        jobs_state.write(s)
        path = jobs_state.state_path("x")
        data = json.loads(path.read_text())
        data["future_field_we_dont_know_about"] = "ignored"
        path.write_text(json.dumps(data))
        loaded = jobs_state.read("x")
        assert loaded is not None
        assert loaded.prompt_id == "x"

    def test_read_tolerates_corrupt_json(self):
        path = jobs_state.state_path("garbage")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not valid json")
        assert jobs_state.read("garbage") is None


class TestTerminal:
    def test_completed_marks_completed_at(self):
        s = jobs_state.new(prompt_id="x", client_id=None, workflow="/w.json", where="local")
        s.status = "completed"
        s.outputs = ["https://example/img.png"]
        jobs_state.write(s)
        loaded = jobs_state.read("x")
        assert loaded.completed_at is not None
        assert loaded.is_terminal is True

    def test_running_does_not_mark_completed_at(self):
        s = jobs_state.new(prompt_id="x", client_id=None, workflow="/w.json", where="local")
        s.status = "running"
        jobs_state.write(s)
        loaded = jobs_state.read("x")
        assert loaded.completed_at is None
        assert loaded.is_terminal is False

    def test_terminal_statuses_enum(self):
        # Pin the closed set so a typo in any caller is caught.
        assert jobs_state.TERMINAL_STATUSES == frozenset({"completed", "error", "cancelled"})


class TestPath:
    def test_prompt_id_with_slashes_is_safe(self):
        # If a backend ever returns a slashy id, we shouldn't escape the dir.
        s = jobs_state.new(prompt_id="weird/id", client_id=None, workflow="/w.json", where="local")
        jobs_state.write(s)
        loaded = jobs_state.read("weird/id")
        assert loaded is not None
        # File lives inside state_dir, no traversal.
        path = jobs_state.state_path("weird/id")
        assert path.parent == jobs_state.state_dir()


class TestAtomicity:
    def test_repeated_writes_replace_not_append(self):
        s = jobs_state.new(prompt_id="x", client_id=None, workflow="/w.json", where="local")
        jobs_state.write(s)
        s.status = "running"
        jobs_state.write(s)
        s.status = "completed"
        s.outputs = ["a", "b"]
        jobs_state.write(s)
        loaded = jobs_state.read("x")
        assert loaded.status == "completed"
        assert loaded.outputs == ["a", "b"]
