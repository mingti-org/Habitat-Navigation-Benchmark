"""Client-side execution helpers for Enactive canonical relative action chunks."""

from __future__ import annotations

from typing import Any

import numpy as np


CONTINUOUS_SCHEMA_VERSION = 2
CONTINUOUS_ACTION_HORIZON = 16
CONTINUOUS_VALID_ACTION_DIM = 4
CONTINUOUS_DEFAULT_EXECUTE_HORIZON = 8
CONTINUOUS_EXECUTION_SEMANTICS = "canonical_relative_v1"
CONTINUOUS_ACTION_FORMAT = "relative_delta_xyzyaw"
CONTINUOUS_ACTION_UNIT = {"xyz": "m", "yaw": "deg"}
REPLAN_ACK_CONTROL_PROTOCOL = "high_policy_replan_ack_v1"

CLIENT_CAPABILITIES = {
    # Advertised order is preference order during server negotiation. Keep the
    # legacy discrete transport as a rolling-upgrade fallback only.
    "action_transports": ["chunk", "discrete"],
    "execution_semantics": [CONTINUOUS_EXECUTION_SEMANTICS],
    "control_protocols": [REPLAN_ACK_CONTROL_PROTOCOL],
}


def _validate_response_header(response: dict[str, Any]) -> None:
    required_fields = {
        "schema_version": CONTINUOUS_SCHEMA_VERSION,
        "control_mode": "continuous",
        "action_horizon": CONTINUOUS_ACTION_HORIZON,
        "valid_action_dim": CONTINUOUS_VALID_ACTION_DIM,
        "execution_semantics": CONTINUOUS_EXECUTION_SEMANTICS,
        "action_format": CONTINUOUS_ACTION_FORMAT,
        "action_unit": CONTINUOUS_ACTION_UNIT,
    }
    for key, expected in required_fields.items():
        if response.get(key) != expected:
            raise ValueError(
                f"Habitat client requires {key}={expected!r}, got {response.get(key)!r}"
            )


def _validate_execute_horizon(value: Any) -> int:
    if value is None:
        return CONTINUOUS_DEFAULT_EXECUTE_HORIZON
    try:
        execute_horizon = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Habitat chunk_execute_horizon must be an integer") from exc
    if not 1 <= execute_horizon <= CONTINUOUS_ACTION_HORIZON:
        raise ValueError(
            "Habitat chunk_execute_horizon must be between 1 and "
            f"{CONTINUOUS_ACTION_HORIZON}, got {execute_horizon}"
        )
    return execute_horizon


def canonical_response_to_trajectory(response: dict[str, Any]) -> np.ndarray:
    """Reconstruct the selected canonical prefix as origin-relative SE(2) poses.

    The returned array includes the local origin as row zero. Each canonical
    delta is expressed in the previous pose's body frame, and yaw is in degrees.
    """

    if not isinstance(response, dict):
        raise ValueError("Habitat continuous response must be a dictionary")
    _validate_response_header(response)
    try:
        chunk = np.asarray(response.get("continuous_action"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Habitat continuous_action must be a numeric h16x4 array") from exc
    expected_shape = (CONTINUOUS_ACTION_HORIZON, CONTINUOUS_VALID_ACTION_DIM)
    if chunk.shape != expected_shape:
        raise ValueError(
            f"Habitat continuous_action must have shape {expected_shape}, got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("Habitat continuous_action must contain only finite values")

    execute_horizon = _validate_execute_horizon(response.get("chunk_execute_horizon"))
    trajectory = np.zeros((execute_horizon + 1, CONTINUOUS_VALID_ACTION_DIM), dtype=np.float64)
    for index, (dx, dy, dz, dyaw_deg) in enumerate(chunk[:execute_horizon], start=1):
        previous = trajectory[index - 1]
        yaw_rad = np.deg2rad(previous[3])
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        trajectory[index, 0] = previous[0] + cos_yaw * dx - sin_yaw * dy
        trajectory[index, 1] = previous[1] + sin_yaw * dx + cos_yaw * dy
        trajectory[index, 2] = previous[2] + dz
        trajectory[index, 3] = previous[3] + dyaw_deg
    return trajectory


def trajectory_to_discrete_actions_close_to_goal(
    trajectory,
    step_size=0.25,
    turn_angle_deg=15,
    lookahead=1,
    pos_tolerance=0.2,
    max_actions=64,
    positive_yaw_action=2,
):
    """Convert a local trajectory to Habitat forward/left/right actions."""

    traj = np.asarray(trajectory)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("trajectory must have shape (N, 2+) with at least x,y")
    if positive_yaw_action not in (2, 3):
        raise ValueError("positive_yaw_action must be Habitat action 2 or 3")

    traj_xy = traj[:, :2]
    goal_yaw = np.deg2rad(traj[-1, 3]) if traj.shape[1] >= 4 else None
    yaw = np.deg2rad(traj[0, 3]) if traj.shape[1] >= 4 else 0.0
    if len(traj_xy) < 2:
        return []

    actions = []
    pos = traj_xy[0].astype(np.float64)
    turn_angle_rad = np.deg2rad(turn_angle_deg)

    def resample_path(path_xy, min_spacing):
        if len(path_xy) <= 2:
            return path_xy.astype(np.float64), np.arange(len(path_xy))
        sampled = [path_xy[0].astype(np.float64)]
        indices = [0]
        accumulated = 0.0
        last = path_xy[0].astype(np.float64)
        for source_idx, current_value in enumerate(path_xy[1:], 1):
            current = current_value.astype(np.float64)
            accumulated += float(np.linalg.norm(current - last))
            if accumulated >= min_spacing:
                sampled.append(current)
                indices.append(source_idx)
                accumulated = 0.0
            last = current
        if np.linalg.norm(sampled[-1] - path_xy[-1]) > 1e-6:
            sampled.append(path_xy[-1].astype(np.float64))
            indices.append(len(path_xy) - 1)
        return np.asarray(sampled, dtype=np.float64), np.asarray(indices)

    track_xy, source_indices = resample_path(traj_xy, min_spacing=max(step_size * 0.8, 0.12))
    goal = track_xy[-1]
    total_xy_displacement = float(np.linalg.norm(track_xy[-1] - track_xy[0]))
    # Match the server's 0.25 m geometric STOP window without changing the
    # position tolerance or the long-distance XY path-following policy.
    allow_final_yaw_alignment = total_xy_displacement <= 0.25
    waypoint_idx = 1
    no_progress_steps = 0
    max_no_progress_steps = 12
    waypoint_reach_tolerance = min(pos_tolerance, max(0.05, step_size * 0.75))
    lookahead_distance = step_size * max(1.0, float(lookahead))

    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def signed_error(bearing, curr_yaw, pose_yaw):
        """Use a matched pose only when left/right native turn counts tie."""
        error = normalize_angle(bearing - curr_yaw)
        positive = (bearing - curr_yaw) % (2 * np.pi)
        negative = (curr_yaw - bearing) % (2 * np.pi)

        def turn_count(angle):
            return int(np.ceil(max(0.0, angle - turn_angle_rad * 0.5 - 1e-12) / turn_angle_rad))

        positive_count = turn_count(positive)
        if positive_count == 0 or positive_count != turn_count(negative) or pose_yaw is None:
            return error
        # A later corner's heading must not dictate how to face this XY target.
        if abs(normalize_angle(pose_yaw - bearing)) > turn_angle_rad * 0.5 + 1e-12:
            return error
        raw_hint = pose_yaw - curr_yaw
        epsilon = np.deg2rad(1e-3)
        if abs(raw_hint) > np.pi + epsilon or abs(raw_hint) < 1e-12:
            return error
        hint = normalize_angle(raw_hint)
        if abs(abs(raw_hint) - np.pi) <= epsilon:
            hint = np.copysign(np.pi, raw_hint)
        return np.copysign(abs(error), hint)

    def apply_action(curr_pos, curr_yaw, action):
        if action == 1:
            next_pos = curr_pos + step_size * np.array([np.cos(curr_yaw), np.sin(curr_yaw)])
            return next_pos, curr_yaw
        if action == 2:
            sign = 1.0 if positive_yaw_action == 2 else -1.0
            return curr_pos, curr_yaw + sign * turn_angle_rad
        sign = 1.0 if positive_yaw_action == 3 else -1.0
        return curr_pos, curr_yaw + sign * turn_angle_rad

    def turn_action_for_yaw_sign(sign):
        if sign > 0:
            return 2 if positive_yaw_action == 2 else 3
        return 3 if positive_yaw_action == 2 else 2

    def append_final_turns(curr_yaw, remaining_budget):
        if goal_yaw is None or remaining_budget <= 0:
            return [], curr_yaw
        yaw_error = signed_error(goal_yaw, curr_yaw, goal_yaw)
        turn_count = int(round(yaw_error / turn_angle_rad))
        if turn_count == 0:
            return [], curr_yaw
        turn_count = int(np.clip(turn_count, -remaining_budget, remaining_budget))
        if turn_count > 0:
            extra_actions = [turn_action_for_yaw_sign(1)] * turn_count
        else:
            extra_actions = [turn_action_for_yaw_sign(-1)] * (-turn_count)
        final_yaw = curr_yaw + turn_count * turn_angle_rad
        return extra_actions, final_yaw

    while len(actions) < max_actions:
        if waypoint_idx >= len(track_xy):
            break
        if np.linalg.norm(pos - goal) <= pos_tolerance:
            if goal_yaw is None or not allow_final_yaw_alignment:
                break
            yaw_error = signed_error(goal_yaw, yaw, goal_yaw)
            if abs(yaw_error) <= turn_angle_rad * 0.5:
                break
            action = turn_action_for_yaw_sign(yaw_error)
            actions.append(action)
            pos, yaw = apply_action(pos, yaw, action)
            continue

        while (
            waypoint_idx < len(track_xy)
            and np.linalg.norm(pos - track_xy[waypoint_idx]) <= waypoint_reach_tolerance
        ):
            waypoint_idx += 1
            no_progress_steps = 0
        if waypoint_idx >= len(track_xy):
            break

        target_idx = waypoint_idx
        covered_length = 0.0
        while target_idx < len(track_xy) - 1 and covered_length < lookahead_distance:
            covered_length += float(np.linalg.norm(track_xy[target_idx + 1] - track_xy[target_idx]))
            target_idx += 1
        target = track_xy[target_idx]
        to_target = target - pos
        target_dist = float(np.linalg.norm(to_target))
        if target_dist < 1e-8:
            waypoint_idx = min(waypoint_idx + 1, len(track_xy) - 1)
            continue

        target_yaw = np.arctan2(to_target[1], to_target[0])
        yaw_error = signed_error(
            target_yaw, yaw,
            np.deg2rad(traj[source_indices[target_idx], 3]) if traj.shape[1] >= 4 else None,
        )
        turn_threshold = turn_angle_rad * 0.5
        if yaw_error > turn_threshold:
            action = turn_action_for_yaw_sign(1)
            next_pos, next_yaw = apply_action(pos, yaw, action)
        elif yaw_error < -turn_threshold:
            action = turn_action_for_yaw_sign(-1)
            next_pos, next_yaw = apply_action(pos, yaw, action)
        else:
            forward_pos, forward_yaw = apply_action(pos, yaw, 1)
            if float(np.linalg.norm(target - forward_pos)) <= target_dist + 1e-3:
                action = 1
                next_pos, next_yaw = forward_pos, forward_yaw
            else:
                negative_action = turn_action_for_yaw_sign(-1)
                positive_action = turn_action_for_yaw_sign(1)
                negative_pos, negative_yaw = apply_action(pos, yaw, negative_action)
                positive_pos, positive_yaw = apply_action(pos, yaw, positive_action)
                negative_delta = target - negative_pos
                positive_delta = target - positive_pos
                negative_error = abs(
                    normalize_angle(np.arctan2(negative_delta[1], negative_delta[0]) - negative_yaw)
                )
                positive_error = abs(
                    normalize_angle(np.arctan2(positive_delta[1], positive_delta[0]) - positive_yaw)
                )
                if negative_error <= positive_error:
                    action, next_pos, next_yaw = negative_action, negative_pos, negative_yaw
                else:
                    action, next_pos, next_yaw = positive_action, positive_pos, positive_yaw

        actions.append(action)
        previous_distance = np.linalg.norm(pos - track_xy[waypoint_idx])
        pos, yaw = next_pos, next_yaw
        new_distance = np.linalg.norm(pos - track_xy[waypoint_idx])
        if new_distance <= previous_distance - 1e-4:
            no_progress_steps = 0
        else:
            no_progress_steps += 1
        if no_progress_steps >= max_no_progress_steps:
            break

    if allow_final_yaw_alignment and len(actions) < max_actions:
        extra_actions, yaw = append_final_turns(yaw, max_actions - len(actions))
        actions.extend(extra_actions)

    return actions


def habitat_actions_from_response(
    response: dict[str, Any], *, diagnostics: dict[str, Any] | None = None,
) -> list[int]:
    """Resolve legacy discrete or canonical continuous responses for Habitat."""

    if diagnostics is not None:
        diagnostics.update(stop=bool(response.get("stop", False)), forward_fallback=False)
    if bool(response.get("stop", False)):
        if diagnostics is not None:
            diagnostics["native_actions"] = [0]
        return [0]
    if response.get("oracle_goal_gps") is not None:
        return []
    if "continuous_action" not in response and "actions" in response:
        raw_actions = response.get("actions")
        actions = (
            []
            if raw_actions is None
            else [int(action) for action in np.asarray(raw_actions).reshape(-1)]
        )
        return actions
    if "continuous_action" not in response:
        raise ValueError("response has neither actions nor continuous_action")

    trajectory = canonical_response_to_trajectory(response)
    max_actions = _validate_execute_horizon(response.get("chunk_execute_horizon"))
    actions = trajectory_to_discrete_actions_close_to_goal(
        trajectory,
        max_actions=max_actions,
        positive_yaw_action=2,
    )
    if diagnostics is not None:
        diagnostics.update(
            prefix_endpoint=trajectory[-1].tolist(),
            native_actions=list(actions),
            forward_fallback=not bool(actions),
        )
    # STOP is explicit in schema v2. A non-stop chunk that cannot be discretized
    # must keep the rollout alive and match the legacy Habitat forward fallback.
    return actions or [1]
