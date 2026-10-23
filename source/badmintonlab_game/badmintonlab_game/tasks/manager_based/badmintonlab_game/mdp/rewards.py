# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_height_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalize horizontal foot motion while the feet are close to the ground."""
    robot: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    foot_height = foot_pos_w[..., 2]
    foot_speed_xy = torch.linalg.norm(foot_vel_w[..., :2], dim=-1)
    foot_speed_xy = torch.clamp(foot_speed_xy, min=0.0, max=5.0)
    near_ground = (foot_height < foot_height_threshold).float()

    return torch.sum(foot_speed_xy * near_ground, dim=1)


def racket_ground_support_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 3.0,
    force_scale: float = 10.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Penalize sustained, high-force racket contact with the ground.

    This reads the contact sensor's filtered force matrix, so contacts with
    the shuttlecock or other robot bodies are excluded. The contact must stay
    above ``force_threshold`` throughout the configured sensor history before
    the penalty is activated.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids]
    force_history = torch.linalg.norm(force_history, dim=-1).amax(dim=-1)

    sustained_contact = torch.all(force_history > max(force_threshold, 0.0), dim=1).any(dim=-1)
    current_force = force_history[:, 0].amax(dim=-1)
    penalty = torch.clamp(current_force / max(force_scale, 1e-6), min=0.0, max=max_penalty)
    return sustained_contact.float() * penalty


def _event_scale(env: ManagerBasedRLEnv) -> float:
    """Cancel RewardManager's dt scaling for one-step events."""
    return 1.0 / max(float(env.step_dt), 1e-6)


def _hit_budget_mask(env: ManagerBasedRLEnv, side: int, hit_cap: int) -> torch.Tensor:
    """Per-rally hit-budget mask: zero once this player's hit count exceeds the cap."""
    if hit_cap <= 0:
        hit_cap = int(getattr(env.cfg, "process_reward_cap", 0) or 0)
    if hit_cap <= 0:
        return torch.ones(env.num_envs, device=env.device)
    return (env.hit_count[:, side] <= hit_cap).float()


def agent_root_height_violation_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.55,
    height_sigma: float = 10.0,
) -> torch.Tensor:
    """Zero-at-valid, bounded cost for dropping the robot root too low."""
    robot: Articulation = env.scene[asset_cfg.name]
    deficit = torch.clamp(minimum_height - robot.data.root_pos_w[:, 2], min=0.0)
    return 1.0 - torch.exp(-deficit * height_sigma)


def agent_root_orientation_violation_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    side: int,
    max_rp_angle_deg: float = 45.0,
    max_yaw_angle_deg: float = 90.0,
    rad_sigma: float = 10.0,
) -> torch.Tensor:
    """Zero-at-valid, bounded canonical-frame orientation violation cost."""
    robot: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = wrap_to_pi(yaw - side * torch.pi)
    rp_limit = torch.deg2rad(torch.tensor(max_rp_angle_deg, device=roll.device, dtype=roll.dtype))
    yaw_limit = torch.deg2rad(torch.tensor(max_yaw_angle_deg, device=yaw.device, dtype=yaw.dtype))
    excess = (
        torch.clamp(roll.abs() - rp_limit, min=0.0)
        + torch.clamp(pitch.abs() - rp_limit, min=0.0)
        + torch.clamp(yaw.abs() - yaw_limit, min=0.0)
    )
    return 1.0 - torch.exp(-excess * rad_sigma)


def agent_action_rate_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Action-rate penalty for one action term instead of the full joint action."""
    action_term = env.action_manager.get_term(action_name)
    offset = 0
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if name == action_name:
            previous = env.action_manager.prev_action[:, offset : offset + term.action_dim]
            return torch.sum(torch.square(action_term.raw_actions - previous), dim=-1)
        offset += term.action_dim
    raise ValueError(f"Unknown action term: {action_name}")


def agent_interval_approach_cost(
    env: ManagerBasedRLEnv,
    side: int,
    min_height: float = 0.5,
    max_height: float = 2.0,
    full_reward_radius: float = 0.1,
    distance_sigma: float = 0.45,
    hit_cap: int = 0,
) -> torch.Tensor:
    """One-shot receive-position error cost, zero for an ideal approach.

    The environment emits the event both when the player hits and when the
    incoming shuttle leaves the receive interval as a miss.  A policy therefore
    cannot avoid this cost merely by refusing to swing.
    """
    del min_height, max_height
    distance = env.approach_event_distance[:, side]
    valid = env.approach_reward_event[:, side]
    outside = torch.clamp(distance - max(full_reward_radius, 0.0), min=0.0)
    proximity = torch.exp(-(outside / max(distance_sigma, 1e-6)))
    return valid.float() * (1.0 - proximity) * _hit_budget_mask(env, side, hit_cap) * _event_scale(env)


def agent_outgoing_shot_quality_cost(
    env: ManagerBasedRLEnv,
    shuttle_cfg: SceneEntityCfg,
    side: int,
    target_speed: float = 10.0,
    minimum_net_clearance_height: float = 1.55,
    direction_sigma: float = 0.15,
    eps: float = 1e-6,
    hit_cap: int = 0,
) -> torch.Tensor:
    """One-shot outgoing speed/direction error cost, zero for a quality shot."""
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    sign = 1.0 if side == 0 else -1.0
    velocity = shuttle.data.root_lin_vel_w.clone()
    velocity[:, :2] *= sign
    speed = torch.linalg.norm(velocity, dim=-1)
    speed_quality = torch.clamp(speed / max(target_speed, eps), 0.0, 1.0)

    shuttle_pos = shuttle.data.root_pos_w - env.scene.env_origins
    shuttle_pos = shuttle_pos.clone()
    shuttle_pos[:, :2] *= sign
    direction = velocity / torch.clamp(speed.unsqueeze(-1), min=eps)
    court = env.cfg.court
    net_distance = torch.clamp(court.net_x - shuttle_pos[:, 0], min=eps)
    half_width = court.singles_half_width if court.use_singles_bounds else court.doubles_half_width
    zeros = torch.zeros_like(net_distance)
    normals = torch.stack(
        (
            torch.stack((-torch.ones_like(net_distance), zeros, zeros), dim=-1),
            torch.stack((shuttle_pos[:, 1] - half_width, net_distance, zeros), dim=-1),
            torch.stack((-shuttle_pos[:, 1] - half_width, -net_distance, zeros), dim=-1),
            torch.stack((minimum_net_clearance_height - shuttle_pos[:, 2], zeros, -net_distance), dim=-1),
        ),
        dim=1,
    )
    signed_violations = torch.sum(normals * direction.unsqueeze(1), dim=-1)
    normalized_violations = torch.relu(signed_violations) / torch.clamp(torch.linalg.norm(normals, dim=-1), min=eps)
    direction_error = torch.linalg.norm(normalized_violations, dim=-1)
    direction_quality = torch.exp(-(direction_error / max(direction_sigma, eps)))
    event = env.outgoing_eval_step[:, side] == env.episode_length_buf
    return (
        event.float()
        * (1.0 - speed_quality * direction_quality)
        * _hit_budget_mask(env, side, hit_cap)
        * _event_scale(env)
    )


def agent_rally_result_reward(env: ManagerBasedRLEnv, side: int) -> torch.Tensor:
    """Reward rally results only after a winner has made a valid hit."""
    won = env.rally_just_resolved & (env.rally_loser == 1 - side) & (env.hit_count[:, side] > 0)
    lost = env.rally_just_resolved & (env.rally_loser == side)
    return (won.float() - lost.float()) * _event_scale(env)
