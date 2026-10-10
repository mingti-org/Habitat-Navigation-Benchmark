"""Half-turn selection checked against real Habitat native turn primitives.

No renderer or dataset is required. The native SDK is required in the parent
Habitat CI job; minimal CPU environments may skip this integration module.
"""
from pathlib import Path
import importlib.util

import numpy as np
import pytest

habitat_sim = pytest.importorskip("habitat_sim")
magnum = pytest.importorskip("magnum")
ActuationSpec = habitat_sim.agent.controls.ActuationSpec

spec = importlib.util.spec_from_file_location(
    "habitat_native_heading_contract",
    Path(__file__).resolve().parents[2] / "internnav/evaluator/canonical_action.py",
)
canonical_action = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_action)


@pytest.mark.parametrize("direction,turn", [(1, 2), (-1, 3)])
@pytest.mark.parametrize("initial_deg", [0.0, 43.0, -120.0, 360.0, -360.0])
def test_half_turn_sign_and_endpoint_follow_native_control(direction, turn, initial_deg):
    graph = habitat_sim.SceneGraph()
    node = graph.get_root_node().create_child()
    node.rotation = magnum.Quaternion.rotation(magnum.Deg(initial_deg), magnum.Vector3.y_axis())
    agent = habitat_sim.Agent(node)
    trajectory = np.asarray([[0., 0., 0., 0.], [0., 0., 0., direction * 180.]])
    actions = canonical_action.trajectory_to_discrete_actions_close_to_goal(trajectory)
    assert actions == [turn] * 12
    headings = []
    for action in [None, *actions]:
        if action is not None:
            agent.controls.action(node, "turn_left" if action == 2 else "turn_right",
                                  ActuationSpec(amount=15.0), apply_filter=False)
        front = node.rotation.transform_vector(magnum.Vector3(0., 0., -1.))
        headings.append(np.arctan2(-front.x, -front.z))
    actual = np.rad2deg(np.unwrap(headings))
    np.testing.assert_allclose(np.diff(actual), direction * 15.0, atol=2e-5)
    assert actual[-1] - actual[0] == pytest.approx(direction * 180., abs=2e-5)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_native_half_turn_continues_forward_instead_of_reporting_no_progress(direction):
    graph = habitat_sim.SceneGraph()
    node = graph.get_root_node().create_child()
    agent = habitat_sim.Agent(node)
    trajectory = np.asarray([[0., 0., 0., 0.], [-1., 0., 0., direction * 180.]])
    actions = canonical_action.trajectory_to_discrete_actions_close_to_goal(trajectory)
    assert len(actions) == 16
    for action in actions:
        name = {1: "move_forward", 2: "turn_left", 3: "turn_right"}[action]
        agent.controls.action(node, name, ActuationSpec(amount=.25 if action == 1 else 15.),
                              apply_filter=False)
    np.testing.assert_allclose(np.asarray(node.translation), [0., 0., 1.], atol=1e-6)
