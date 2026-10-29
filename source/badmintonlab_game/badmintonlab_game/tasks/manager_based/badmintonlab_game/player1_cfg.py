from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
    reset_joints_by_scale,
    reset_root_state_uniform,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from badmintonlab_game.assets.court import BadmintonCourtCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_COURT_CFG = BadmintonCourtCfg()
_ROBOT_RESET_Y_RANGE = (-0.15 * _COURT_CFG.singles_half_width, 0.15 * _COURT_CFG.singles_half_width)


def canonical_robot_heading_court(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    side: int,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    forward_b = torch.zeros_like(robot.data.root_pos_w)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(robot.data.root_quat_w, forward_b)
    heading_xy = forward_w[:, :2]
    heading = heading_xy / torch.clamp(torch.linalg.norm(heading_xy, dim=-1, keepdim=True), min=1e-6)
    return heading if side == 0 else -heading


def canonical_robot_pos_court(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    side: int,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    root_local = robot.data.root_pos_w - env.scene.env_origins
    if side == 0:
        return root_local
    return torch.cat((-root_local[:, :2], root_local[:, 2:3]), dim=-1)


def shuttle_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    rel_w = shuttle.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_w)


def shuttle_vel_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    shuttle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    shuttle: RigidObject = env.scene[shuttle_cfg.name]
    rel_vel_w = shuttle.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)


@configclass
class EventsCfg:
    reset_player_1_root = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1"),
            "pose_range": {
                "x": (0.5 * _COURT_CFG.half_length - 0.35, 0.5 * _COURT_CFG.half_length + 0.35),
                "y": _ROBOT_RESET_Y_RANGE,
                "z": (-0.01, 0.01),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (math.pi - 0.12, math.pi + 0.12),
            },
            "velocity_range": {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_player_1_joints = EventTerm(
        func=reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", joint_names=[".*"]),
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class ObsCfg(ObsGroup):
    robot_heading_court = ObsTerm(
        func=canonical_robot_heading_court, params={"asset_cfg": SceneEntityCfg("robot_1"), "side": 1}
    )
    robot_pos_court = ObsTerm(
        func=canonical_robot_pos_court, params={"asset_cfg": SceneEntityCfg("robot_1"), "side": 1}
    )
    shuttle_pos_b = ObsTerm(
        func=shuttle_pos_b,
        params={"asset_cfg": SceneEntityCfg("robot_1"), "shuttle_cfg": SceneEntityCfg("shuttle")},
    )
    shuttle_vel_b = ObsTerm(
        func=shuttle_vel_b,
        params={"asset_cfg": SceneEntityCfg("robot_1"), "shuttle_cfg": SceneEntityCfg("shuttle")},
    )
    opponent_heading_court = ObsTerm(
        func=canonical_robot_heading_court, params={"asset_cfg": SceneEntityCfg("robot_0"), "side": 1}
    )
    opponent_pos_court = ObsTerm(
        func=canonical_robot_pos_court, params={"asset_cfg": SceneEntityCfg("robot_0"), "side": 1}
    )
    opponent_projected_gravity = ObsTerm(func=projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")})
    opponent_joint_pos = ObsTerm(func=joint_pos_rel, scale=1.0, params={"asset_cfg": SceneEntityCfg("robot_0")})
    base_lin_vel = ObsTerm(func=base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
    base_ang_vel = ObsTerm(func=base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
    projected_gravity = ObsTerm(func=projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")})
    joint_pos = ObsTerm(func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
    joint_vel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
    last_action = ObsTerm(func=last_action, params={"action_name": "player_1"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
