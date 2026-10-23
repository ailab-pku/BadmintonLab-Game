from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_heading_court(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot heading as a unit vector in the court/world xy frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    forward_b = torch.tensor((1.0, 0.0, 0.0), device=robot.data.root_quat_w.device, dtype=robot.data.root_quat_w.dtype)
    forward_b = forward_b.expand(robot.data.root_quat_w.shape[0], 3)
    forward_w = quat_apply(robot.data.root_quat_w, forward_b)
    heading_xy = forward_w[:, :2]
    return heading_xy / torch.clamp(torch.linalg.norm(heading_xy, dim=-1, keepdim=True), min=1e-6)


def robot_pos_court(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot root position in court-local coordinates."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w - env.scene.env_origins


def canonical_robot_heading_court(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    side: int,
) -> torch.Tensor:
    """Robot heading rotated into the common negative-x-side court frame."""
    sign = 1.0 if side == 0 else -1.0
    return sign * robot_heading_court(env, asset_cfg)


def canonical_robot_pos_court(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    side: int,
) -> torch.Tensor:
    """Robot position rotated into the common negative-x-side court frame."""
    sign = 1.0 if side == 0 else -1.0
    root_local = robot_pos_court(env, asset_cfg)
    return torch.cat((sign * root_local[:, :2], root_local[:, 2:3]), dim=-1)


def shuttle_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shuttle position expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    rel_w = shuttle.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_w)


def shuttle_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shuttle linear velocity expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    rel_vel_w = shuttle.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)
