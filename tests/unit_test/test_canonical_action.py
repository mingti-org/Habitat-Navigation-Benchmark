from pathlib import Path
import importlib.util

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "internnav"
    / "evaluator"
    / "canonical_action.py"
)
SPEC = importlib.util.spec_from_file_location("habitat_canonical_action", MODULE_PATH)
canonical_action = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canonical_action)


def _response(chunk, execute_horizon=8, **extra):
    response = {
        "schema_version": 2,
        "control_mode": "continuous",
        "stop": False,
        "action_horizon": 16,
        "valid_action_dim": 4,
        "execution_semantics": "canonical_relative_v1",
        "action_format": "relative_delta_xyzyaw",
        "action_unit": {"xyz": "m", "yaw": "deg"},
        "continuous_action": chunk,
        "chunk_execute_horizon": execute_horizon,
    }
    response.update(extra)
    return response


def test_canonical_chunk_uses_previous_body_frame_yaw():
    chunk = np.zeros((16, 4), dtype=np.float32)
    chunk[0] = [1.0, 0.0, 0.0, 90.0]
    chunk[1] = [1.0, 0.0, 0.0, 0.0]

    trajectory = canonical_action.canonical_response_to_trajectory(
        _response(chunk, execute_horizon=2)
    )

    np.testing.assert_allclose(
        trajectory,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 90.0],
                [1.0, 1.0, 0.0, 90.0],
            ]
        ),
        atol=1e-6,
    )


def test_execute_horizon_limits_reconstruction_and_discrete_action_count():
    chunk = np.zeros((16, 4), dtype=np.float32)
    chunk[:4, 0] = 0.25

    response = _response(chunk, execute_horizon=1)

    trajectory = canonical_action.canonical_response_to_trajectory(response)
    actions = canonical_action.habitat_actions_from_response(response)

    assert trajectory.shape == (2, 4)
    assert actions == [1]


def test_short_and_zero_non_stop_plans_fall_back_to_forward():
    short_chunk = np.zeros((16, 4), dtype=np.float32)
    short_chunk[0, 0] = 0.2413

    assert canonical_action.habitat_actions_from_response(
        _response(short_chunk, execute_horizon=1)
    ) == [1]
    assert canonical_action.habitat_actions_from_response(
        _response(np.zeros((16, 4), dtype=np.float32), execute_horizon=1)
    ) == [1]


def test_pure_positive_yaw_uses_habitat_turn_left():
    chunk = np.zeros((16, 4), dtype=np.float32)
    chunk[0, 3] = 30.0

    actions = canonical_action.habitat_actions_from_response(
        _response(chunk, execute_horizon=2)
    )

    assert actions == [2, 2]


def test_stop_oracle_and_legacy_actions_keep_their_execution_semantics():
    chunk = np.zeros((16, 4), dtype=np.float32)

    assert canonical_action.habitat_actions_from_response({"actions": [3, 1]}) == [3, 1]
    assert canonical_action.habitat_actions_from_response({"actions": [], "stop": True}) == [0]
    assert canonical_action.habitat_actions_from_response(
        {"actions": [1], "stop": True}
    ) == [0]
    assert canonical_action.habitat_actions_from_response(
        _response(chunk, oracle_goal_gps=[1.0, -0.5])
    ) == []
    assert canonical_action.habitat_actions_from_response(
        {"actions": np.asarray([2, 1])}
    ) == [2, 1]


def test_client_capabilities_negotiate_chunk_with_legacy_fallback():
    assert canonical_action.CLIENT_CAPABILITIES == {
        "action_transports": ["chunk", "discrete"],
        "execution_semantics": ["canonical_relative_v1", "pose_events_v1"],
        "control_protocols": ["high_policy_replan_ack_v1"],
    }


def test_invalid_canonical_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        canonical_action.canonical_response_to_trajectory(
            _response(np.zeros((8, 4), dtype=np.float32))
        )


@pytest.mark.parametrize("schema_version", [None, 3])
def test_missing_or_wrong_schema_version_is_rejected(schema_version):
    response = _response(np.zeros((16, 4), dtype=np.float32))
    if schema_version is None:
        response.pop("schema_version")
    else:
        response["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version"):
        canonical_action.canonical_response_to_trajectory(response)
