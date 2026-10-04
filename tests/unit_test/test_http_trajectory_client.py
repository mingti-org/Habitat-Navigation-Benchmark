import importlib.util
from pathlib import Path
import sys
import types

import json_numpy
import numpy as np
import pytest
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_http_client_module(monkeypatch):
    internnav_package = types.ModuleType("internnav")
    internnav_package.__path__ = [str(PROJECT_ROOT / "internnav")]
    evaluator_package = types.ModuleType("internnav.evaluator")
    evaluator_package.__path__ = [str(PROJECT_ROOT / "internnav" / "evaluator")]
    evaluator_stub = types.ModuleType(
        "internnav.evaluator.final_habitat_vln_evaluator"
    )
    evaluator_stub.BaseTrajectoryClient = object
    monkeypatch.setitem(sys.modules, "internnav", internnav_package)
    monkeypatch.setitem(sys.modules, "internnav.evaluator", evaluator_package)
    monkeypatch.setitem(
        sys.modules,
        "internnav.evaluator.final_habitat_vln_evaluator",
        evaluator_stub,
    )

    canonical_name = "internnav.evaluator.canonical_action"
    canonical_spec = importlib.util.spec_from_file_location(
        canonical_name,
        PROJECT_ROOT / "internnav" / "evaluator" / "canonical_action.py",
    )
    canonical_module = importlib.util.module_from_spec(canonical_spec)
    monkeypatch.setitem(sys.modules, canonical_name, canonical_module)
    canonical_spec.loader.exec_module(canonical_module)

    client_spec = importlib.util.spec_from_file_location(
        "habitat_http_trajectory_client",
        PROJECT_ROOT / "internnav" / "evaluator" / "HTTPTrajectoryClient.py",
    )
    client_module = importlib.util.module_from_spec(client_spec)
    client_spec.loader.exec_module(client_module)
    return client_module


class _Response:
    def __init__(self, payload):
        self.text = json_numpy.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return json_numpy.loads(self.text)


def _observation(**values):
    observation = {
        "instruction": "go forward",
        "gps": [0.0, 0.0],
        "yaw": 0.0,
        "camera_height": 0.0,
    }
    observation.update(values)
    return observation


def _client(client_module):
    client = client_module.Gr00tTrajectoryClient(
        "http://policy/act",
        env_id="habitat-worker-0",
    )
    client.reset(
        "go forward",
        episode_id="episode-1",
        scene_id="data/scene-1/scene.glb",
    )
    return client


def test_prepare_observation_converts_habitat_coordinates(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    client = client_module.Gr00tTrajectoryClient(
        "http://policy/act",
        env_id="habitat-worker-0",
    )

    prepared = client._prepare_observation_payload(
        _observation(
            gps=np.array([1.25, -0.5], dtype=np.float32),
            yaw=np.float32(np.deg2rad(90.0)),
            camera_height=np.float32(0.75),
            reference_path_gps=[[0.0, 0.0], [2.0, 0.5]],
            metrics={
                "reference_path_gps": [[0.0, 0.0], [2.0, 0.5]],
                "goal_gps": [3.0, -1.0],
            },
        )
    )

    assert np.allclose(prepared["state_xyzyaw"], [1.25, 0.5, 0.75, 90.0])
    assert np.allclose(
        prepared["reference_path_gps"], [[0.0, 0.0], [2.0, -0.5]]
    )
    assert np.allclose(
        prepared["metrics"]["reference_path_gps"],
        [[0.0, 0.0], [2.0, -0.5]],
    )
    assert np.allclose(prepared["metrics"]["goal_gps"], [3.0, 1.0])


def test_prepare_observation_preserves_finite_depth(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    client = client_module.Gr00tTrajectoryClient(
        "http://policy/act",
        env_id="habitat-worker-0",
    )
    depth = np.array([[0.5, 1.0]], dtype=np.float32)

    prepared = client._prepare_observation_payload(_observation(depth=depth))

    assert prepared["depth"] is depth


def test_prepare_observation_omits_nonfinite_depth(monkeypatch, capsys):
    client_module = _load_http_client_module(monkeypatch)
    client = client_module.Gr00tTrajectoryClient(
        "http://policy/act",
        env_id="habitat-worker-0",
    )

    prepared = client._prepare_observation_payload(
        _observation(
            depth=np.array([[0.5, np.nan]], dtype=np.float32),
            episode_id="episode-7",
            step_id=23,
        )
    )

    assert "depth" not in prepared
    assert "episode=episode-7 step=23" in capsys.readouterr().out


def test_action_response_converts_oracle_goal_to_habitat_coordinates(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)

    response = client_module._action_response(
        {"oracle_goal_gps": [2.0, 1.5]},
        [],
        0,
    )

    assert np.allclose(response["oracle_goal_gps"], [2.0, -1.5])


def test_query_negotiates_chunk_and_reuses_feedback_during_replan(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    chunk = np.zeros((16, 4), dtype=np.float32)
    chunk[0, 0] = 0.25
    responses = iter(
        [
            {
                "schema_version": 2,
                "control_mode": "continuous",
                "stop": False,
                "replan_required": True,
                "control_event": {
                    "type": "high_policy_replan",
                    "token": "segment-7",
                },
            },
            {
                "schema_version": 2,
                "control_mode": "continuous",
                "stop": False,
                "action_horizon": 16,
                "valid_action_dim": 4,
                "execution_semantics": "canonical_relative_v1",
                "action_format": "relative_delta_xyzyaw",
                "action_unit": {"xyz": "m", "yaw": "deg"},
                "continuous_action": chunk,
                "chunk_execute_horizon": 1,
            },
        ]
    )
    requests = []

    def fake_post(url, data, headers, timeout):
        requests.append(json_numpy.loads(data))
        assert url == "http://policy/act"
        assert headers == {"Content-Type": "application/json"}
        assert timeout == 300.0
        return _Response(next(responses))

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = _client(client_module)

    result = client.query(_observation(executed_actions=[2, 1]))

    assert result["actions"] == [1]
    assert result["replan_rounds"] == 1
    assert result["action_transport"] == "chunk"
    assert result["schema_version"] == 2
    assert result["chunk_execute_horizon"] == 1
    assert len(requests) == 2
    assert [
        request["observation"]["request_sequence"] for request in requests
    ] == [0, 1]
    assert [request["observation"]["reset"] for request in requests] == [True, False]
    for request in requests:
        observation = request["observation"]
        assert observation["executed_actions"] == [2, 1]
        assert np.allclose(observation["state_xyzyaw"], [0.0, 0.0, 0.0, 0.0])
        assert observation["client_capabilities"] == {
            "action_transports": ["chunk", "discrete"],
            "execution_semantics": ["canonical_relative_v1"],
            "control_protocols": ["high_policy_replan_ack_v1"],
        }
    assert requests[1]["observation"]["control_event"] == {
        "type": "high_policy_replan_ack",
        "token": "segment-7",
    }


def test_query_rejects_legacy_http_200_error_response(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)

    def fake_post(*args, **kwargs):
        del args, kwargs
        return _Response({"error": "invalid observation", "stop": True})

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = _client(client_module)

    with pytest.raises(RuntimeError, match="invalid observation"):
        client.query(_observation())


def test_replan_control_requires_non_empty_token(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)

    with pytest.raises(ValueError, match="non-empty token"):
        client_module._replan_ack(
            {
                "replan_required": True,
                "control_event": {
                    "type": "high_policy_replan",
                    "token": "",
                },
            }
        )


def test_query_retries_same_ack_after_transport_failure(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    chunk = np.zeros((16, 4), dtype=np.float32)
    responses = iter(
        [
            _Response(
                {
                    "schema_version": 2,
                    "control_mode": "continuous",
                    "stop": False,
                    "replan_required": True,
                    "control_event": {
                        "type": "high_policy_replan",
                        "token": "segment-11",
                    },
                }
            ),
            requests.ConnectionError("response lost"),
            _Response(
                {
                    "schema_version": 2,
                    "control_mode": "continuous",
                    "stop": False,
                    "action_horizon": 16,
                    "valid_action_dim": 4,
                    "execution_semantics": "canonical_relative_v1",
                    "action_format": "relative_delta_xyzyaw",
                    "action_unit": {"xyz": "m", "yaw": "deg"},
                    "continuous_action": chunk,
                    "chunk_execute_horizon": 1,
                }
            ),
        ]
    )
    requests_seen = []

    def fake_post(url, data, headers, timeout):
        del url, headers, timeout
        requests_seen.append(json_numpy.loads(data))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    result = _client(client_module).query(_observation(step_id=17))

    assert result["actions"] == [1]
    assert len(requests_seen) == 3
    assert requests_seen[1] == requests_seen[2]
    assert requests_seen[0]["observation"]["step_id"] == 17
    assert requests_seen[0]["observation"]["request_sequence"] == 0
    assert requests_seen[1]["observation"]["step_id"] == 17
    assert requests_seen[1]["observation"]["request_sequence"] == 1
    assert requests_seen[1]["observation"]["control_event"]["token"] == "segment-11"


def test_episode_end_is_explicit_and_uses_stable_terminal_event(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    posted = []

    def fake_post(url, data, headers, timeout):
        del headers, timeout
        payload = json_numpy.loads(data)
        posted.append((url, payload))
        return _Response({"status": "success", "event_id": payload["event_id"]})

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = _client(client_module)
    result = client.end_episode(
        {
            "termination_kind": "model_stop",
            "steps": 9,
            "success": True,
            "client_trajectory_xyyaw_m": [[0.0, 0.0, 0.0]],
        }
    )

    assert posted[0][0] == "http://policy/episode_end"
    payload = posted[0][1]
    assert result == {"status": "success", "event_id": payload["event_id"]}
    assert payload["termination_kind"] == "model_stop"
    assert payload["steps"] == 9
    assert payload["success"] is True
    assert payload["event_id"].startswith("habitat-episode-end-")
    assert client._active_identity is None


def test_episode_end_retries_identical_payload_after_lost_response(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    posted = []

    def fake_post(url, data, headers, timeout):
        del headers, timeout
        payload = json_numpy.loads(data)
        posted.append((url, payload))
        if len(posted) == 1:
            raise client_module.requests.ConnectionError("terminal response lost")
        return _Response({"status": "success", "event_id": payload["event_id"]})

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = _client(client_module)

    client.end_episode(
        {
            "termination_kind": "environment_done",
            "steps": 4,
            "success": False,
        }
    )

    assert len(posted) == 2
    assert posted[0] == posted[1]
    assert client._active_identity is None


def test_dagger_abort_response_raises_only_episode_local_signal(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)

    monkeypatch.setattr(
        client_module.requests,
        "post",
        lambda *args, **kwargs: _Response(
            {
                "dagger_abort_episode": True,
                "dagger_failure_type": "habitat_replay_teacher_unavailable_target",
                "continuous_action": np.zeros((16, 4), dtype=np.float32),
            }
        ),
    )
    client = _client(client_module)

    with pytest.raises(
        client_module.DaggerEpisodeAbort,
        match="habitat_replay_teacher_unavailable_target",
    ):
        client.query(_observation())

    assert client._active_identity is not None


def test_unavailable_teacher_before_query_has_no_policy_session(monkeypatch):
    from unittest.mock import Mock

    client_module = _load_http_client_module(monkeypatch)
    post = Mock(side_effect=AssertionError("no policy session was opened"))
    monkeypatch.setattr(client_module.requests, "post", post)
    client = _client(client_module)
    result = client.end_episode({"termination_kind": "accepted_teacher_unavailable"})
    assert result == {"status": "not_started", "no_policy_session": True}
    assert client.episode_end_result == result
    assert client._active_identity is None
    post.assert_not_called()
    client.reset("next", episode_id="episode-2", scene_id="scene-2")
    assert client.episode_end_result is None


def test_scheduled_session_registers_before_first_act_and_keeps_terminal_receipt(monkeypatch):
    from enactive.eval.online.episode_scheduler_protocol import SchedulerClient

    client_module = _load_http_client_module(monkeypatch)
    events = []
    scheduler = types.SimpleNamespace(session=lambda task, identity: events.append(("session", task, identity)))
    monkeypatch.setattr(SchedulerClient, "from_environment", lambda: scheduler)
    receipt = {"index_path": "/fixture/lease/dagger_samples/all_samples.jsonl"}

    def post(url, data, **_kwargs):
        payload = json_numpy.loads(data)
        events.append((url, payload))
        if url.endswith("/act"):
            return _Response({"dagger_abort_episode": True,
                              "dagger_failure_type": "habitat_replay_teacher_unavailable_target"})
        return _Response({"status": "success", "event_id": payload["event_id"], "dagger_receipt": receipt})

    monkeypatch.setattr(client_module.requests, "post", post)
    client = client_module.Gr00tTrajectoryClient("http://policy/act", env_id="worker")
    client._scheduler_task = {"lease_id": "fixture-lease"}
    client.reset("follow", episode_id="episode-1", scene_id="scene-1")
    assert not events
    with pytest.raises(client_module.DaggerEpisodeAbort):
        client.query(_observation())
    assert [event[0] for event in events] == ["session", "http://policy/act"]
    assert events[0][1] == client._scheduler_task
    assert events[0][2] == {key: events[1][1]["observation"][key]
                           for key in ("assignment_id", "env_id", "episode_id")}
    result = client.end_episode({"termination_kind": "accepted_teacher_unavailable"})
    assert events[-1][0] == "http://policy/episode_end"
    assert result["dagger_receipt"] == receipt
    assert client.episode_end_result == result


@pytest.mark.parametrize(
    ("condition", "expects_action"),
    [("robot_demo", True), ("video_demo", False)],
)
def test_demo_lifecycle_materializes_only_robot_action_abi(
    monkeypatch,
    condition,
    expects_action,
):
    client_module = _load_http_client_module(monkeypatch)

    class _Entry:
        replay = types.SimpleNamespace(sha256="5" * 64)

        def result_metadata(self, **values):
            return {
                "evaluation_condition": values["condition"].value,
                "demo_role": values["condition"].demo_role,
                "demo_step_count": values["demo_step_count"],
                "demo_manifest_sha256": values["manifest_sha256"],
                "replay_sha256": self.replay.sha256,
                "provenance_sha256": "6" * 64,
                "source_split": "val_seen",
                "source_scene_id": "source-scene",
                "source_episode_id": "source-1",
                "source_instruction_sha256": "7" * 64,
                "target_split": "val_seen",
                "target_scene_id": "data/scene/scene.glb",
                "target_episode_id": "70",
                "target_instruction_sha256": "8" * 64,
                "demo_query_lifecycle_status": values["lifecycle_status"],
            }

    class _Manifest:
        sha256 = "4" * 64

        def entry_for(self, **values):
            assert values == {
                "split": "val_seen",
                "scene_id": "data/scene/scene.glb",
                "episode_id": "70",
                "instruction": "go forward",
            }
            return _Entry()

    class _Replay:
        step_count = 2
        entry = _Entry()

        @staticmethod
        def observation_at(index):
            return {
                "instruction": "source instruction",
                "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "rgb_views": {"front": np.zeros((2, 2, 3), dtype=np.uint8)},
                "state_xyzyaw": np.zeros(4, dtype=np.float32),
                "step_id": index,
                "frame_id": f"frame-{index}",
            }

        @staticmethod
        def continuous_action_at(index):
            assert index in (0, 1)
            return np.zeros((16, 4), dtype=np.float32) if expects_action else None

    monkeypatch.setattr(
        client_module,
        "load_habitat_demonstration_manifest",
        lambda *args, **kwargs: _Manifest(),
    )
    monkeypatch.setattr(
        client_module,
        "load_episode_replay",
        lambda *args, **kwargs: _Replay(),
    )
    posted = []
    chunk = np.zeros((16, 4), dtype=np.float32)

    def fake_post(url, data, headers, timeout):
        del headers, timeout
        payload = json_numpy.loads(data)
        posted.append((url, payload))
        if url.endswith("/demo"):
            from enactive.inference.navigation.demonstration import parse_demonstration_request
            parsed = parse_demonstration_request(payload)
            assert parsed.request_id == payload["request_id"]
            operation = payload["operation"]
            response = {"status": "success", "operation": operation}
            if operation == "ingest":
                response.update(
                    {
                        "demo_role": condition,
                        "demonstration_steps": sum(
                            item[1].get("operation") == "ingest" for item in posted
                        ),
                    }
                )
            return _Response(response)
        if url.endswith("/episode_end"):
            return _Response({"status": "success", "event_id": payload["event_id"]})
        return _Response(
            {
                "schema_version": 2,
                "control_mode": "continuous",
                "stop": False,
                "action_horizon": 16,
                "valid_action_dim": 4,
                "execution_semantics": "canonical_relative_v1",
                "action_format": "relative_delta_xyzyaw",
                "action_unit": {"xyz": "m", "yaw": "deg"},
                "continuous_action": chunk,
                "chunk_execute_horizon": 1,
            }
        )

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = client_module.Gr00tTrajectoryClient(
        "http://policy/act",
        env_id="habitat-worker-0",
        evaluation_condition=condition,
        demonstration_manifest="manifest.json",
        eval_split="val_seen",
    )
    client.reset(
        "go forward",
        episode_id="70",
        scene_id="data/scene/scene.glb",
    )
    result = client.query(_observation())
    client.end_episode(
        {
            "termination_kind": "model_stop",
            "steps": 1,
            "success": True,
        }
    )

    assert result["actions"] == [1]
    demo_payloads = [payload for url, payload in posted if url.endswith("/demo")]
    assert [payload["operation"] for payload in demo_payloads] == [
        "ingest",
        "ingest",
        "query_reset",
    ]
    assert [payload["request_sequence"] for payload in demo_payloads] == [0, 1, 2]
    assert all(
        ("continuous_action" in payload) is expects_action
        for payload in demo_payloads[:2]
    )
    if expects_action:
        assert all(payload["schema_version"] == 2 for payload in demo_payloads[:2])
    else:
        assert all("schema_version" not in payload for payload in demo_payloads[:2])
    assert client.evaluation_metadata()["demo_query_lifecycle_status"] == "complete"
    assert client.evaluation_metadata()["target_scene_id"] == "data/scene/scene.glb"


def test_partial_demo_failure_releases_client_identity(monkeypatch):
    client_module = _load_http_client_module(monkeypatch)
    client = client_module.Gr00tTrajectoryClient(
        "http://policy/act",
        env_id="habitat-worker-0",
    )
    client._condition = client_module.EvaluationCondition.VIDEO_DEMO
    client._eval_split = "val_seen"
    client._manifest = types.SimpleNamespace(entry_for=lambda **kwargs: object())
    replay = types.SimpleNamespace(
        step_count=2,
        entry=types.SimpleNamespace(replay=types.SimpleNamespace(sha256="5" * 64)),
        observation_at=lambda index: {
            "instruction": "source",
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "state_xyzyaw": np.zeros(4, dtype=np.float32),
            "step_id": index,
        },
        continuous_action_at=lambda index: None,
    )
    monkeypatch.setattr(client_module, "load_episode_replay", lambda *args: replay)

    posted = []

    def fake_post(url, data, **kwargs):
        payload = json_numpy.loads(data)
        posted.append(payload)
        if url.endswith("/episode_end"):
            assert payload["termination_kind"] == "demo_ingest_failed"
            return _Response({"status": "success", "event_id": payload["event_id"]})
        assert payload["operation"] == "ingest"
        if len(posted) == 2:
            raise RuntimeError("second demo step failed")
        return _Response({"status": "success", "operation": "ingest", "demo_role": "video_demo", "demonstration_steps": 1})

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="second demo step failed"):
        client.reset(
            "go forward",
            episode_id="70",
            scene_id="data/scene/scene.glb",
        )

    assert posted[-1]["termination_kind"] == "demo_ingest_failed"
    assert client._active_identity is None
