"""Physical camera regression using the real Evaluator and Habitat renderer.

Set HABITAT_CAMERA_TEST_CONFIG to a Habitat YAML with real scene/dataset assets.
No policy weights or HTTP service are needed. The only substitute is the policy
agent, whose environment/camera setters are enough for Evaluator construction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.slow


class _NoPolicy:
    def set_env(self, env):
        self.env = env

    def set_camera_params(self, params):
        self.camera_params = params


@pytest.fixture(params=[0, 2], ids=["level", "look_down_30_degrees"])
def evaluator(request, tmp_path):
    config_path = os.environ.get("HABITAT_CAMERA_TEST_CONFIG")
    if not config_path:
        pytest.skip("set HABITAT_CAMERA_TEST_CONFIG to run the real Habitat camera test")
    from internnav.evaluator.final_habitat_vln_evaluator import Evaluator

    args = SimpleNamespace(
        save_video=False,
        sim_gpu=int(os.environ.get("HABITAT_CAMERA_TEST_GPU", "0")),
        success_distance=None,
        init_look_down_steps=request.param,
    )
    instance = Evaluator(
        str(Path(config_path).resolve()),
        os.environ.get("HABITAT_CAMERA_TEST_SPLIT", "val_unseen"),
        str(tmp_path),
        args,
        _NoPolicy(),
        max_steps=64,
    )
    try:
        yield instance
    finally:
        instance.env.close()


def test_all_views_match_native_turns_and_reach_canonical_wire(evaluator, tmp_path):
    import json_numpy
    from habitat_sim.utils.common import quat_to_coeffs

    from enactive.simulator.canonical import CanonicalAdapter
    from internnav.evaluator.final_habitat_vln_evaluator import Action, build_traj_request
    from internnav.evaluator.HTTPTrajectoryClient import Gr00tTrajectoryClient

    env = evaluator.env
    episode = env.episodes[0]
    observations, initial_height, _distance = evaluator._init_episode(episode)
    view_sensors = {"front": "rgb", "left": "rgb_left", "right": "rgb_right", "rear": "rgb_rear"}
    baseline = {name: np.asarray(observations[sensor]).copy() for name, sensor in view_sensors.items()}
    initial_state = env.sim.get_agent_state()
    sensor_rotations = {
        name: np.asarray(quat_to_coeffs(initial_state.sensor_states[sensor].rotation)).copy()
        for name, sensor in view_sensors.items()
    }
    for image in baseline.values():
        assert not evaluator._is_corrupt_rgb(image), "renderer returned invalid RGB"
        assert image.std() > 5, "camera test needs visible, nonuniform scene geometry"

    observation = evaluator._build_observation(
        observations, 0, agent_height=initial_height, metrics=env.get_metrics()
    )
    camera_height = evaluator.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1]
    body = build_traj_request(observation, "Inspect all current views.", camera_height)
    client = Gr00tTrajectoryClient("http://unused.invalid/act", env_id="camera-direction-test")
    client.reset(body["instruction"], episode_id=str(episode.episode_id), scene_id=str(episode.scene_id))
    prepared = client._protocol_observation(client._prepare_observation_payload(body))
    packet = json.JSONDecoder().decode(json_numpy.dumps({"observation": prepared}))
    received = CanonicalAdapter(debug_dir_fallback=tmp_path).parse_request(packet)
    for name, image in baseline.items():
        np.testing.assert_array_equal(received.rgb_views[name], image)

    turn_angle = float(evaluator.config.habitat.simulator.turn_angle)
    count = round(90.0 / turn_angle)
    assert abs(count * turn_angle - 90.0) < 1e-6, "test config turn angle must divide 90 degrees"
    for view, action, turn_count in (
        ("left", Action.TURN_LEFT.value, count),
        ("right", Action.TURN_RIGHT.value, count),
        ("rear", Action.TURN_LEFT.value, 2 * count),
    ):
        evaluator._init_episode(episode)
        before = np.asarray(env.sim.get_agent_state().position).copy()
        for _ in range(turn_count):
            turned = env.step(action)
        after = env.sim.get_agent_state()
        np.testing.assert_array_equal(after.position, before)

        # Compare actual native sensor quaternions, not the configuration's yaw
        # constants. Quaternion signs are equivalent representations.
        front_rotation = np.asarray(quat_to_coeffs(after.sensor_states["rgb"].rotation))
        dot = abs(np.dot(sensor_rotations[view], front_rotation))
        np.testing.assert_allclose(dot, 1.0, atol=1e-6, rtol=0)

        front = np.asarray(turned["rgb"])
        assert not evaluator._is_corrupt_rgb(front), "renderer returned invalid RGB after turning"
        difference = np.abs(front.astype(np.float64) - baseline[view].astype(np.float64)).mean()
        assert difference < 0.05, f"native {view} turn disagrees with the image labeled {view}: MAE={difference}"
