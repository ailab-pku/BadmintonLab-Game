from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
from isaaclab.sim.utils import bind_physics_material
from isaacsim.core.utils.stage import get_current_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def spawn_urdf_with_racket_material(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the source robot and bind its racket material before replication.

    InteractiveScene replicates the source environment after all entities have
    been spawned.  Binding here therefore authors the material on editable
    source collision prims, which is inherited by every replicated instance.
    """
    prim = spawn_from_urdf(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    material_name = prim.GetPath().pathString.rsplit("/", 1)[-1].lower()
    material_path = f"/World/PhysicsMaterials/{material_name}_racket"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.2,
        dynamic_friction=0.15,
        restitution=0.75,
        friction_combine_mode="average",
        restitution_combine_mode="max",
    )
    material_cfg.func(material_path, material_cfg)
    racket_prim = get_current_stage().GetPrimAtPath(f"{prim.GetPath().pathString}/racket_link")
    pending = [racket_prim]
    while pending:
        child = pending.pop()
        if child.IsInstance():
            child.SetInstanceable(False)
        pending.extend(child.GetChildren())
    bind_physics_material(
        f"{prim.GetPath().pathString}/racket_link",
        material_path,
        stage=get_current_stage(),
    )
    return prim


def reset_shuttle_drop_state(
    env: ManagerBasedRLEnv,
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

    env.initial_hitter[env_ids] = initial_hitter
