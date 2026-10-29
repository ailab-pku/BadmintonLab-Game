from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from badmintonlab_game.assets.court import is_net_fault, is_out_of_bounds


class DualAgentBadmintonEnv(ManagerBasedEnv, gym.Env):
    """Two-player badminton playback environment."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg)
        n = self.num_envs

        self.just_hit = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self.next_hitter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.last_hitter = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.rally_resolved = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._in_contact = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self.serve_height_fault = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.double_hit_fault = torch.zeros(n, dtype=torch.bool, device=self.device)

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
        self.just_hit.zero_()
        self.recorder_manager.record_pre_step()
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

        # Preserve the just-finished rally's flags before reset clears its state.
        self.extras["rally_resolved"] = self.rally_resolved.clone()
        self.extras["serve_height_fault"] = self.serve_height_fault.clone()
        self.extras["double_hit_fault"] = self.double_hit_fault.clone()
        reset_ids = self.rally_resolved.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager.record_post_step()
        return self.obs_buf, self.extras

    def _update_hits(self):
        # Read each physics step: history maxima lose ordering and can count old
        # contacts again. A sustained contact is one hit until it separates.
        contacts = torch.stack(
            [
                torch.linalg.norm(contact.data.force_matrix_w, dim=-1).flatten(1).amax(dim=1)
                > self.cfg.hit_contact_threshold
                for contact in self._contacts
            ],
            dim=1,
        )
        new_contacts = contacts & ~self._in_contact
        self._in_contact.copy_(contacts)
        height = self._shuttle.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        for side in range(2):
            hit = new_contacts[:, side] & ~self.rally_resolved
            wrong_turn = hit & (self.next_hitter != side)
            high_serve = hit & (self.last_hitter < 0) & (height > self.cfg.max_serve_height)
            self.double_hit_fault |= wrong_turn
            self.serve_height_fault |= high_serve
            self.rally_resolved |= wrong_turn | high_serve
            valid = hit & ~self.rally_resolved
            self.just_hit[:, side] |= valid
            self.next_hitter[valid] = 1 - side
            self.last_hitter[valid] = side

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
        self._in_contact[env_ids] = False
        self.serve_height_fault[env_ids] = False
        self.double_hit_fault[env_ids] = False
        self.prev_shuttle_pos_w[env_ids] = self._shuttle.data.root_pos_w[env_ids]
        self.curr_shuttle_pos_w[env_ids] = self._shuttle.data.root_pos_w[env_ids]

    def _apply_shuttle_forces(self):
        velocity = self._shuttle.data.root_lin_vel_w
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        force = -self._drag_k * speed * velocity + self._mass * self._g_vec
        force_b = quat_apply_inverse(self._shuttle.data.root_quat_w, force)
        self._shuttle.permanent_wrench_composer.set_forces_and_torques(
            forces=force_b.unsqueeze(1), torques=torch.zeros_like(force_b).unsqueeze(1), is_global=False
        )

    def _detect_landing(self):
        local_z = self._shuttle.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.shuttle_landed |= local_z <= self.cfg.shuttle_landing_detection_height
