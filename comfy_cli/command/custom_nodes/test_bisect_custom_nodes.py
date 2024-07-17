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


def test_good(bisect_state):
    new_state = bisect_state.good()
    assert new_state.status == "running"
    assert new_state.all == bisect_state.all
    assert new_state.range == ["node2", "node3"]
    assert new_state.active == ["node2"]


def test_good_resolved(bisect_state):
    bisect_state.range = ["node2"]
    new_state = bisect_state.good()
    assert new_state.status == "resolved"
    assert new_state.all == bisect_state.all
    assert new_state.range == ["node2"]
    assert new_state.active == ["node2"]


def test_bad(bisect_state):
    new_state = bisect_state.bad()
    assert new_state.status == "running"
    assert new_state.all == bisect_state.all
    assert new_state.range == ["node1", "node2"]
    assert new_state.active == ["node1"]


def test_bad_resolved(bisect_state):
    bisect_state.range = ["node1"]
    new_state = bisect_state.bad()
    assert new_state.status == "resolved"
    assert new_state.all == bisect_state.all
    assert new_state.range == ["node1"]
    assert new_state.active == ["node1"]


def test_save(bisect_state, tmp_path):
    bisect_state.save()
    state_file = tmp_path / "bisect_state.json"
    assert state_file.exists()
    with state_file.open() as f:
        saved_state = json.load(f)
    assert saved_state == vars(bisect_state)


def test_reset(bisect_state):
    bisect_state.reset()
    assert bisect_state.status == "idle"
    assert bisect_state.range == bisect_state.all
    assert bisect_state.active == bisect_state.all


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
    loaded_state = BisectState.load()
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


def test_inactive_nodes(bisect_state):
    bisect_state.active = ["node1", "node2"]
    assert bisect_state.inactive_nodes == ["node3"]


@patch("comfy_cli.command.custom_nodes.bisect_custom_nodes.execute_cm_cli")
def test_set_custom_node_enabled_states(mock_execute_cm_cli, bisect_state):
    bisect_state.set_custom_node_enabled_states()
    mock_execute_cm_cli.assert_called_once_with(["enable", "node1", "node2"])


@patch("comfy_cli.command.custom_nodes.bisect_custom_nodes.execute_cm_cli")
def test_set_custom_node_enabled_states_no_active_nodes(
    mock_execute_cm_cli, bisect_state
):
    bisect_state.active = []
    bisect_state.set_custom_node_enabled_states()
    mock_execute_cm_cli.assert_called_once_with(["disable", "node1", "node2", "node3"])


def test_str(bisect_state, capsys):
    expected_output = """BisectState(status=running)
        bad nodes: 3
        test set nodes: 2
        --------------------------
        0. node1
        1. node2
        """
    print(bisect_state)
    captured = capsys.readouterr()
    assert captured.out.strip() == expected_output.strip()


if __name__ == "__main__":
    pytest.main()
