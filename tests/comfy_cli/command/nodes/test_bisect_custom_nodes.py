import json
from unittest.mock import patch

import pytest

from comfy_cli.command.custom_nodes.bisect_custom_nodes import BisectState


@pytest.fixture(scope="function")
def bisect_state():
    return BisectState(
        status="running",
        all=["node1", "node2", "node3"],
        range=["node1", "node2", "node3"],
        active=["node1", "node2"],
    )


def test_good():
    bisect_state = BisectState(
        status="running",
        all=["node1", "node2", "node3"],
        range=["node1", "node2", "node3"],
        active=["node1"],
    )
    new_state = bisect_state.good()
    assert new_state.status == "running"
    assert new_state.all == bisect_state.all
    assert set(new_state.range) == set(["node3", "node2"])
    assert len(new_state.active) == 1


def test_good_resolved(bisect_state: BisectState):
    new_state = bisect_state.good()
    assert new_state.status == "resolved"
    assert new_state.all == bisect_state.all
    assert new_state.range == ["node3"]
    assert new_state.active == []


def test_bad(bisect_state):
    new_state = bisect_state.bad()
    assert new_state.status == "running"
    assert new_state.all == bisect_state.all
    assert new_state.range == ["node1", "node2"]
    assert new_state.active == ["node2"]


def test_bad_resolved():
    bisect_state = BisectState(
        status="running",
        all=["node1", "node2", "node3"],
        range=["node1", "node2", "node3"],
        active=["node1"],
    )
    new_state = bisect_state.bad()
    assert new_state.status == "resolved"
    assert new_state.all == bisect_state.all
    assert new_state.range == ["node1"]
    assert new_state.active == []


@patch("comfy_cli.command.custom_nodes.bisect_custom_nodes.execute_cm_cli")
def test_save(mock_execute_cm_cli, bisect_state, tmp_path):
    state_file = tmp_path / "bisect_state.json"
    bisect_state.save(state_file)
    assert state_file.exists()
    assert mock_execute_cm_cli.call_count == 2
    with state_file.open() as f:
        saved_state = json.load(f)
    assert saved_state == bisect_state._asdict()


@patch("comfy_cli.command.custom_nodes.bisect_custom_nodes.execute_cm_cli")
def test_reset(mock_execute_cm_cli, bisect_state):
    new_state = bisect_state.reset()
    assert new_state.status == "idle"
    assert new_state.all == ["node1", "node2", "node3"]
    assert new_state.range == ["node1", "node2", "node3"]
    assert new_state.active == ["node1", "node2", "node3"]
    assert mock_execute_cm_cli.call_count == 1


def test_load_existing_state(tmp_path):
    state_file = tmp_path / "bisect_state.json"
    state_data = {
        "status": "running",
        "all": ["node1", "node2", "node3"],
        "range": ["node1", "node2", "node3"],
        "active": ["node1", "node2"],
    }
    with state_file.open("w") as f:
        json.dump(state_data, f)

    loaded_state = BisectState.load(state_file)
    assert loaded_state.status == state_data["status"]
    assert loaded_state.all == state_data["all"]
    assert loaded_state.range == state_data["range"]
    assert loaded_state.active == state_data["active"]


def test_load_nonexistent_state(tmp_path):
    state_file = tmp_path / "bisect_state.json"
    loaded_state = BisectState.load(state_file)
    assert loaded_state.status == "idle"
    assert loaded_state.all == []
    assert loaded_state.range == []
    assert loaded_state.active == []


@patch("comfy_cli.command.custom_nodes.bisect_custom_nodes.execute_cm_cli")
def test_set_custom_node_enabled_states(mock_execute_cm_cli, bisect_state):
    bisect_state.set_custom_node_enabled_states()
    assert mock_execute_cm_cli.call_count == 2


@patch("comfy_cli.command.custom_nodes.bisect_custom_nodes.execute_cm_cli")
def test_set_custom_node_enabled_states_no_active_nodes(mock_execute_cm_cli):
    bisect_state = BisectState(
        status="running",
        all=["node1", "node2", "node3"],
        range=["node1", "node2", "node3"],
        active=[],
    )
    bisect_state.set_custom_node_enabled_states()
    assert mock_execute_cm_cli.call_count == 1
