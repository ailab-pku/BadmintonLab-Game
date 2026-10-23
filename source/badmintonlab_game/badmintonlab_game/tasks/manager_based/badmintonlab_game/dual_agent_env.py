from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from badmintonlab_game.assets.court import is_net_fault, is_out_of_bounds


class DualAgentBadmintonEnv(ManagerBasedEnv):
    """Two-player badminton playback environment."""

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        n = self.num_envs

        self.just_hit = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self.next_hitter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.last_hitter = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.rally_resolved = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.shuttle_landed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.prev_shuttle_pos_w = torch.zeros((n, 3), device=self.device)
        self.curr_shuttle_pos_w = torch.zeros((n, 3), device=self.device)

        self._shuttle: RigidObject = self.scene["shuttle"]
        self._contacts: tuple[ContactSensor, ContactSensor] = (
            self.scene.sensors["racket_contact_0"],
            self.scene.sensors["racket_contact_1"],
        )

        self._drag_k = float(cfg.shuttle_drag_k)
        self._mass = float(cfg.shuttle_mass)
        self._g_vec = torch.tensor((0.0, 0.0, float(cfg.gravity_z)), device=self.device)

    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self._apply_shuttle_forces()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self._detect_landing()

        self._update_hits()
        self._update_rally_result()
        reset_ids = self.rally_resolved.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.extras

    def _update_hits(self):
        self.just_hit.zero_()
        expected_hitter = self.next_hitter.clone()
        for side, contact in enumerate(self._contacts):
            contact_force = (
                torch.linalg.norm(contact.data.force_matrix_w_history, dim=-1)
                .amax(dim=1)
                .amax(dim=-1)
                .amax(dim=-1)
            )
            self.just_hit[:, side] = (
                (contact_force > self.cfg.hit_contact_threshold)
                & (expected_hitter == side)
                & ~self.rally_resolved
            )
        hit_ids = self.just_hit.nonzero(as_tuple=False)
        if hit_ids.numel() > 0:
            env_ids, sides = hit_ids.unbind(dim=1)
            self.next_hitter[env_ids] = 1 - sides
            self.last_hitter[env_ids] = sides

    def _update_rally_result(self):
        self.prev_shuttle_pos_w.copy_(self.curr_shuttle_pos_w)
        self.curr_shuttle_pos_w.copy_(self._shuttle.data.root_pos_w)
        local = self.curr_shuttle_pos_w - self.scene.env_origins
        prev_local = self.prev_shuttle_pos_w - self.scene.env_origins
        net_fault = is_net_fault(prev_local, local, self.cfg.court)
        out_fault = is_out_of_bounds(local, self.cfg.court) & (self.last_hitter >= 0)
        self.rally_resolved |= net_fault | out_fault | self.shuttle_landed

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.just_hit[env_ids] = False
        self.last_hitter[env_ids] = -1
        self.rally_resolved[env_ids] = False
        self.shuttle_landed[env_ids] = False
        self.prev_shuttle_pos_w[env_ids] = self._shuttle.data.root_pos_w[env_ids]
        self.curr_shuttle_pos_w[env_ids] = self._shuttle.data.root_pos_w[env_ids]

    def _apply_shuttle_forces(self):
        velocity = self._shuttle.data.root_lin_vel_w
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        force = -self._drag_k * speed * velocity + self._mass * self._g_vec
        force_b = quat_apply_inverse(self._shuttle.data.root_quat_w, force)
        self._shuttle.set_external_force_and_torque(
            force_b.unsqueeze(1), torch.zeros_like(force_b).unsqueeze(1), is_global=False
        )

    def _detect_landing(self):
        local_z = self._shuttle.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.shuttle_landed.copy_(local_z <= self.cfg.shuttle_landing_detection_height)
