"""HTTP bridge plus client-side execution for Enactive Habitat evaluation.

The client advertises its response capabilities, reconstructs canonical SE(2)
chunks, and converts them to Habitat-native discrete actions. Legacy server
``actions`` responses remain supported during the migration window.
"""

import hashlib
import os
from typing import Optional
import uuid

import requests
from internnav.evaluator.final_habitat_vln_evaluator import BaseTrajectoryClient
import json_numpy
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image

from internnav.evaluator.canonical_action import (
    CLIENT_CAPABILITIES,
    habitat_actions_from_response,
)

json_numpy.patch()

_MAX_REPLAN_ROUNDS = 4
_MAX_REPLAN_ACK_POST_ATTEMPTS = 3
_MAX_EPISODE_END_POST_ATTEMPTS = 3

_RESPONSE_METADATA_KEYS = (
    "action_transport",
    "schema_version",
    "control_mode",
    "action_horizon",
    "chunk_execute_horizon",
    "oracle_goal_gps",
    "oracle_goal_progress_index",
    "oracle_goal_radius",
    "oracle_max_steps",
    "post_error_student_steps",
    "low_policy_mode",
    "eval_gt_low_policy",
    "eval_gt_low_policy_stop",
    "eval_gt_low_policy_phase",
    "eval_gt_final_dist_m",
    "eval_gt_progress_index",
    "eval_gt_low_policy_fallback",
    "dagger_failure_type",
    "dagger_correction_weight",
    "dagger_student_actions_discrete",
    "dagger_teacher_actions_discrete",
    "dagger_final_actions_discrete",
    "dagger_correction_applied",
)


def _flip_lateral_axis(value):
    converted = np.asarray(value, dtype=np.float32).copy()
    if converted.size:
        converted[..., 1] *= -1.0
    return converted.tolist()


def _replan_ack(result: dict) -> dict:
    event = result.get("control_event")
    if not isinstance(event, dict) or event.get("type") != "high_policy_replan":
        raise ValueError(
            "replan_required response must include control_event.type="
            "'high_policy_replan'"
        )
    token = str(event.get("token") or "").strip()
    if not token:
        raise ValueError("replan_required control_event must include a non-empty token")
    return {
        "type": "high_policy_replan_ack",
        "token": token,
    }


def _post_act(url: str, payload: dict, *, timeout: float):
    observation = payload.get("observation")
    control_event = observation.get("control_event") if isinstance(observation, dict) else None
    is_replan_ack = bool(
        isinstance(control_event, dict)
        and control_event.get("type") == "high_policy_replan_ack"
    )
    attempts = _MAX_REPLAN_ACK_POST_ATTEMPTS if is_replan_ack else 1
    body = json_numpy.dumps(payload)
    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status_code is None or int(status_code) >= 500
            if not is_replan_ack or not retryable or attempt + 1 >= attempts:
                raise
    raise RuntimeError("unreachable replan ACK retry state")


def _action_response(result: dict, actions: list[int], replan_rounds: int) -> dict:
    response = {
        "actions": actions,
        "pixel_goal": result.get("pixel_goal", None),
        "replan_rounds": replan_rounds,
        "action_transport": result.get(
            "action_transport",
            "chunk" if "continuous_action" in result else "discrete",
        ),
    }
    for key in _RESPONSE_METADATA_KEYS:
        if key in result:
            response[key] = result[key]
    if response.get("oracle_goal_gps") is not None:
        response["oracle_goal_gps"] = _flip_lateral_axis(
            response["oracle_goal_gps"]
        )
    return response

#更换模型只需要添加并调用一个新的类即可，以下面这个为例
class Gr00tTrajectoryClient(BaseTrajectoryClient):
    def __init__(self, url, *, env_id: str, debug_output_path=None):
        self.url = url
        self.timeout = float(os.getenv("HABITAT_CLIENT_TIMEOUT", "300"))
        self.debug_output_path = debug_output_path
        self._base_env_id = str(env_id).strip()
        if not self._base_env_id:
            raise ValueError("Habitat env_id is required")
        self._client_session_id = uuid.uuid4().hex
        self._active_identity = None
        self._next_request_sequence = 0

    def reset(self, instruction: str, **kwargs):
        episode_id = str(kwargs.get("episode_id") or "").strip()
        scene_id = str(kwargs.get("scene_id") or "").strip()
        if not episode_id or not scene_id:
            raise ValueError("Habitat reset requires episode_id and scene_id")
        if self._active_identity is not None:
            raise RuntimeError(
                "previous Habitat policy session was not ended before reset"
            )
        scene_digest = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:12]
        self._active_identity = (
            self._client_session_id,
            f"{self._base_env_id}:{scene_digest}",
            episode_id,
        )
        self._next_request_sequence = 0

    def _protocol_observation(
        self,
        observation: dict,
        *,
        control_event: Optional[dict] = None,
    ) -> dict:
        if self._active_identity is None:
            raise RuntimeError("Habitat trajectory client was queried before reset")
        assignment_id, env_id, episode_id = self._active_identity
        sequence = self._next_request_sequence
        request_digest = hashlib.sha256(
            "\0".join(
                (
                    assignment_id,
                    env_id,
                    episode_id,
                    str(sequence),
                )
            ).encode("utf-8")
        ).hexdigest()
        prepared = dict(observation)
        prepared.update(
            {
                "assignment_id": assignment_id,
                "env_id": env_id,
                "episode_id": episode_id,
                "request_id": f"habitat-act-{request_digest}",
                "request_sequence": sequence,
                "reset": sequence == 0,
            }
        )
        if control_event is not None:
            prepared["control_event"] = dict(control_event)
        else:
            prepared.pop("control_event", None)
        return prepared

    def end_episode(self, event: dict) -> dict:
        if self._active_identity is None:
            raise RuntimeError("Habitat trajectory client has no active episode")
        assignment_id, env_id, episode_id = self._active_identity
        event_digest = hashlib.sha256(
            "\0".join(self._active_identity).encode("utf-8")
        ).hexdigest()
        payload = {
            "assignment_id": assignment_id,
            "env_id": env_id,
            "episode_id": episode_id,
            "event_id": f"habitat-episode-end-{event_digest}",
            "termination_kind": str(
                event.get("termination_kind") or "environment_done"
            ),
            "termination_reason": str(
                event.get("termination_reason")
                or event.get("termination_kind")
                or "environment_done"
            ),
            "steps": int(event.get("steps", 0) or 0),
            "success": bool(event.get("success", False)),
            "client_trajectory_xyyaw_m": list(
                event.get("client_trajectory_xyyaw_m") or []
            ),
        }
        body = json_numpy.dumps(payload)
        endpoint = self.url.rsplit("/act", 1)[0] + "/episode_end"
        response = None
        for attempt in range(_MAX_EPISODE_END_POST_ATTEMPTS):
            try:
                response = requests.post(
                    endpoint,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                status_code = getattr(
                    getattr(exc, "response", None),
                    "status_code",
                    None,
                )
                retryable = status_code is None or int(status_code) >= 500
                if not retryable or attempt + 1 >= _MAX_EPISODE_END_POST_ATTEMPTS:
                    raise
        if response is None:
            raise RuntimeError("Habitat episode_end produced no response")
        try:
            result = response.json()
        except (TypeError, ValueError):
            result = {"status": "success"}
        self._active_identity = None
        self._next_request_sequence = 0
        return result if isinstance(result, dict) else {"status": "success"}

    def _save_client_observation(self, obs: dict):
        if not self.debug_output_path:
            return
        if os.getenv("HABITAT_SAVE_CLIENT_OBSERVATIONS", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        try:
            rgb = obs.get("rgb")
            if rgb is None:
                return
            arr = np.asarray(rgb)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.ndim != 3 or arr.shape[-1] != 3:
                return
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = np.ascontiguousarray(arr)

            instruction = str(obs.get("instruction", "unknown_instruction"))
            safe_instruction = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in instruction)
            safe_instruction = "_".join(part for part in safe_instruction.split("_") if part)[:60] or "unknown_instruction"
            step_id = obs.get("step_id", "na")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_dir = Path(self.debug_output_path) / "_client_observations" / safe_instruction
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / f"client_{timestamp}_step{step_id}.png"
            txt_path = out_dir / f"client_{timestamp}_step{step_id}.txt"
            Image.fromarray(arr).save(img_path)

            mean = arr.mean(axis=(0, 1))
            std = arr.std(axis=(0, 1))
            arr_i = arr.astype(np.int16)
            hdiff = float(np.abs(np.diff(arr_i, axis=1)).mean())
            vdiff = float(np.abs(np.diff(arr_i, axis=0)).mean())
            black_frac = float((arr < 5).all(axis=2).mean())
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"instruction: {instruction}\n")
                f.write(f"step_id: {step_id}\n")
                for key in ("gps", "yaw", "camera_height"):
                    if key in obs:
                        f.write(f"{key}: {self._json_safe(obs.get(key))}\n")
                metrics = obs.get("metrics")
                if isinstance(metrics, dict):
                    small_metrics = {
                        key: metrics.get(key)
                        for key in ("distance_to_goal", "success", "spl", "ndtw")
                        if key in metrics
                    }
                    f.write(f"metrics: {self._json_safe(small_metrics)}\n")
                f.write(f"shape: {arr.shape}\n")
                f.write(f"dtype: {arr.dtype}\n")
                f.write(f"min: {int(arr.min())}\n")
                f.write(f"max: {int(arr.max())}\n")
                f.write(f"mean: [{mean[0]}, {mean[1]}, {mean[2]}]\n")
                f.write(f"std: [{std[0]}, {std[1]}, {std[2]}]\n")
                f.write(f"hfdiff: {hdiff}\n")
                f.write(f"vfdiff: {vdiff}\n")
                f.write(f"black_frac: {black_frac}\n")
        except Exception:
            return

    def _prepare_observation_payload(self, obs: dict) -> dict:
        """Make Habitat observations JSON-safe before json_numpy serialization.

        Habitat RGB is often a non-contiguous RGB view over an RGBA render
        buffer, with strides like (W*4, 4, 1).  Some JSON numpy serializers do
        not preserve that view correctly, which shows up as colorful noise on
        the server side.  Send an explicit contiguous RGB copy.
        """
        obs_payload = dict(obs)
        depth = obs_payload.get("depth")
        if depth is None:
            obs_payload.pop("depth", None)
        else:
            try:
                depth_array = np.asarray(depth)
                valid_depth = (
                    depth_array.size > 0
                    and np.issubdtype(depth_array.dtype, np.number)
                    and bool(np.isfinite(depth_array).all())
                )
            except (TypeError, ValueError):
                valid_depth = False
            if not valid_depth:
                print(
                    "[HabitatClient] WARNING: omitting invalid depth from canonical "
                    f"request episode={obs_payload.get('episode_id', '')} "
                    f"step={obs_payload.get('step_id', '')}",
                    flush=True,
                )
                obs_payload.pop("depth", None)
        gps = np.asarray(obs_payload["gps"], dtype=np.float32).reshape(-1)
        yaw_rad = float(np.asarray(obs_payload["yaw"]).reshape(-1)[0])
        camera_height = float(np.asarray(obs_payload["camera_height"]).reshape(-1)[0])
        obs_payload["state_xyzyaw"] = [
            float(gps[0]),
            -float(gps[1]),
            camera_height,
            float(np.degrees(yaw_rad)),
        ]
        if obs_payload.get("reference_path_gps") is not None:
            obs_payload["reference_path_gps"] = _flip_lateral_axis(
                obs_payload["reference_path_gps"]
            )
        metrics = obs_payload.get("metrics")
        if isinstance(metrics, dict):
            metrics = dict(metrics)
            for key in ("reference_path_gps", "goal_gps"):
                if metrics.get(key) is not None:
                    metrics[key] = _flip_lateral_axis(metrics[key])
            obs_payload["metrics"] = metrics
        obs_payload["client_capabilities"] = {
            key: list(values) for key, values in CLIENT_CAPABILITIES.items()
        }
        rgb = obs_payload.get("rgb")
        if rgb is not None:
            arr = np.asarray(rgb)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.ndim == 3 and arr.shape[-1] == 3:
                if arr.dtype != np.uint8:
                    if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                obs_payload["rgb"] = np.ascontiguousarray(arr)
        return obs_payload

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): Gr00tTrajectoryClient._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Gr00tTrajectoryClient._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            if value.size <= 16:
                return value.tolist()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def query(self, obs: dict, **kwargs) -> list[int]:
        obs_payload = self._prepare_observation_payload(obs)
        metrics = obs_payload.get("metrics")
        if isinstance(metrics, dict):
            obs_payload["metrics"] = self._json_safe(metrics)
        if self.debug_output_path:
            self._save_client_observation(obs_payload)
        # 1. 使用 HTTP 发送；控制响应在同一 observation 上透明 ACK 后重试。
        original_obs_payload = dict(obs_payload)
        replan_rounds = 0
        control_event = None
        while True:
            obs_payload = self._protocol_observation(
                original_obs_payload,
                control_event=control_event,
            )
            payload = {"observation": obs_payload}
            if self.debug_output_path:
                payload["debug_output_path"] = self.debug_output_path
            resp = _post_act(
                self.url,
                payload,
                timeout=self.timeout,
            )
            self._next_request_sequence += 1

            # 2. 还原返回值
            result = json_numpy.loads(resp.text)
            if isinstance(result, str):
                result = json_numpy.loads(result)
            if not isinstance(result, dict):
                raise ValueError(f"unexpected /act response: {type(result)!r}")
            if result.get("error"):
                raise RuntimeError(f"policy server rejected observation: {result['error']}")
            if not bool(result.get("replan_required", False)):
                break
            if replan_rounds >= _MAX_REPLAN_ROUNDS:
                raise RuntimeError(
                    "policy server exceeded the maximum of "
                    f"{_MAX_REPLAN_ROUNDS} transparent replan rounds"
                )
            control_event = _replan_ack(result)
            replan_rounds += 1
            
        if (
            "actions" in result
            or "continuous_action" in result
            or bool(result.get("stop", False))
            or result.get("oracle_goal_gps") is not None
        ):
            actions = habitat_actions_from_response(result)
            return _action_response(result, actions, replan_rounds)
        
        # 3. 获取 delta poses 
        dp_actions = result["action"]
        
        # dp_actions = torch.from_numpy(dp_actions_np)

        # ！！！如果模型输出结果不匹配比如尺寸归一化等要处理的话改这个函数就行，没有就算了
        dp_actions = gr00t_output_to_dp_actions(dp_actions)

        # 4. 在此处进行 "轨迹 -> 离散动作" 的转换
        # 这样逻辑就回到了 Client 端
        actions = traj_to_actions_Gr00t(dp_actions)

        if not actions:
            # 如果actions_list 为空，正在追加默认动作 [1] 以维持运行。
            actions = [1]
        
        return _action_response(result, actions, replan_rounds)

def traj_to_actions_Gr00t(dp_actions,use_discrate_action=True):
    def reconstruct_xy_from_delta(delta_xyt):
        """
        Input:
            delta_xyt: [B, T, 3], dx, dy are position increments in global coordinates, dθ is heading difference (not used for position)
            start_xy: [B, 2] starting point
        Output:
            xy: [B, T+1, 2] reconstructed global trajectory
        """
        start_xy = np.zeros((len(delta_xyt), 2))
        delta_xy = delta_xyt[:, :, :2]  # Take dx, dy parts
        cumsum_xy = np.cumsum(delta_xy, axis=1)  # [B, T, 2]

        B = delta_xyt.shape[0]
        T = delta_xyt.shape[1]
        xy = np.zeros((B, T + 1, 2))
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy[:, None, :] + cumsum_xy

        return xy

    def trajectory_to_discrete_actions_close_to_goal(trajectory, step_size=0.25, turn_angle_deg=15, lookahead=4):
        actions = []
        yaw = 0.0
        pos = trajectory[0]
        turn_angle_rad = np.deg2rad(turn_angle_deg)
        traj = trajectory
        goal = trajectory[-1]

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        while np.linalg.norm(pos - goal) > 0.2:
            # Find the nearest trajectory point index to current position
            dists = np.linalg.norm(traj - pos, axis=1)
            nearest_idx = np.argmin(dists)
            # Look ahead a bit (not exceeding trajectory end)
            target_idx = min(nearest_idx + lookahead, len(traj) - 1)
            target = traj[target_idx]
            # Target direction
            target_dir = target - pos
            if np.linalg.norm(target_dir) < 1e-6:
                break
            target_yaw = np.arctan2(target_dir[1], target_dir[0])
            # Difference between current yaw and target yaw
            delta_yaw = normalize_angle(target_yaw - yaw)
            n_turns = int(round(delta_yaw / turn_angle_rad))
            if n_turns > 0:
                actions += [2] * n_turns
            elif n_turns < 0:
                actions += [3] * (-n_turns)
            yaw = normalize_angle(yaw + n_turns * turn_angle_rad)

            # Move forward one step
            next_pos = pos + step_size * np.array([np.cos(yaw), np.sin(yaw)])

            # If moving forward one step makes us farther from goal, stop
            if np.linalg.norm(next_pos - goal) > np.linalg.norm(pos - goal):
                break

            actions.append(1)
            pos = next_pos

        return actions

    # unnormalize
    dp_actions[:, :, :2] /= 4.0
    all_trajectory = reconstruct_xy_from_delta(dp_actions.float().cpu().numpy())
    trajectory = np.mean(all_trajectory, axis=0)
    if use_discrate_action:
        actions = trajectory_to_discrete_actions_close_to_goal(trajectory)
        return actions
    else:
        return trajectory
    
def gr00t_output_to_dp_actions(gr00t_out):
        """
        把 Gr00t 输出转换为 traj_to_actions_Gr00t 需要的格式。

        支持以下 gr00t_out 形式：
        - numpy array shape (T, 4)  # 单序列
        - numpy array shape (1, T, 4)  # batch=1
        - torch tensor 同上

        Gr00t 输出列 assumed: [dx, dy, dz, dyaw_degrees]
        返回: torch.Tensor shape (1, T, 3) dtype=float32, last dim = [dx, dy, dyaw_rad*12]
        """
        # 转 numpy / torch 兼容
        if isinstance(gr00t_out, torch.Tensor):
            arr = gr00t_out.detach().cpu().numpy()
        else:
            arr = np.asarray(gr00t_out)

        # 支持 (T,4) 或 (1,T,4) 或 (B,T,4)
        if arr.ndim == 2 and arr.shape[1] == 4:
            arr = arr[None, :, :]  # -> (1, T, 4)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pass
        else:
            raise ValueError(f"Unsupported gr00t_out shape: {arr.shape}, expected (T,4) or (1,T,4) or (B,T,4)")

        # 取 (dx, dy, dyaw)
        # 列索引假设： 0=dx, 1=dy, 2=dz (unused), 3=dyaw (单位：度)
        dx = arr[:, :, 0].astype(np.float32)
        dy = arr[:, :, 1].astype(np.float32)
        dyaw_deg = arr[:, :, 3].astype(np.float32)

        # deg -> rad
        dyaw_rad = np.deg2rad(dyaw_deg)

        # 根据之前讨论，把 yaw 放大（保持和 traj_to_actions_Gr00t 里相同的放大逻辑）
        dyaw_rad = dyaw_rad * 1.0  # base conversion
        # 注意：traj_to_actions_Gr00t 会再做 *=12 的处理（如果你在函数里保留那一行）
        # 此处不再重复乘 12，除非你在 traj_to_actions_Gr00t 中没有加那一行。

        dp = np.stack([dx, dy, dyaw_rad], axis=-1)  # (B, T, 3)

        return torch.from_numpy(dp).float()  # 返回 torch Tensor (B, T, 3)
