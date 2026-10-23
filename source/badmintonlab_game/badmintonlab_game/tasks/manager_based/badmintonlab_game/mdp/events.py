from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_shuttle_drop_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    canonical_position: tuple[float, float, float],
) -> None:
    """Drop the shuttle from a fixed mirrored point on a random player's side.

    ``canonical_position`` is specified for player 0 (the negative-x half).
    Player 1 resets mirror both horizontal coordinates, matching the canonical
    court transform used by the observations.  Linear and angular velocity are
    zero, so the first player must create the opening shot rather than receive
    a synthetic serve from the environment.
    """

    shuttle: RigidObject = env.scene[asset_cfg.name]
    device = shuttle.data.root_pos_w.device
    dtype = shuttle.data.root_pos_w.dtype

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    else:
        env_ids = env_ids.to(device=device)

    num_resets = len(env_ids)
    initial_hitter = torch.randint(0, 2, (num_resets,), device=device)
    canonical = torch.tensor(canonical_position, device=device, dtype=dtype).expand(num_resets, 3)
    mirror = torch.where(initial_hitter == 0, 1.0, -1.0).to(dtype=dtype).unsqueeze(-1)
    start_local = canonical.clone()
    start_local[:, :2] *= mirror

    root_state = torch.zeros((num_resets, 13), device=device, dtype=dtype)
    root_state[:, :3] = start_local + env.scene.env_origins[env_ids]
    root_state[:, 3] = 1.0
    shuttle.write_root_state_to_sim(root_state, env_ids=env_ids)

    env.next_hitter[env_ids] = initial_hitter
