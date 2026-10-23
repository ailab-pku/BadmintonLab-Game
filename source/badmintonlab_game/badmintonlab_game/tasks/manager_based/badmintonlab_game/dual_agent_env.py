from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from badmintonlab_game.assets.court import is_net_fault, is_out_of_bounds


class DualAgentBadmintonEnv(ManagerBasedRLEnv):
    """Two-player court whose learning interface is provided by SharedPolicyVecEnvWrapper."""

    READY = 0
    RECOVER = 1

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        n = self.num_envs

        self.single_agent_action_dim = self.action_manager.get_term("player_0").action_dim
        self.agent_reward_buf = torch.zeros((n, 2), device=self.device)
        self.just_hit = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self.initial_hitter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.next_hitter = torch.zeros(n, dtype=torch.long, device=self.device)
        self.hit_count = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        self.last_hitter = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.rally_resolved = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.rally_just_resolved = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.rally_loser = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.shuttle_landed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.prev_shuttle_pos_w = torch.zeros((n, 3), device=self.device)
        self.curr_shuttle_pos_w = torch.zeros((n, 3), device=self.device)
        self.net_fault = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.out_fault = torch.zeros(n, dtype=torch.bool, device=self.device)

        self.outgoing_eval_step = torch.full((n, 2), -1, dtype=torch.long, device=self.device)

        self.approach_min_distance = torch.full((n, 2), torch.inf, device=self.device)
        self.approach_event_distance = torch.zeros((n, 2), device=self.device)
        self.approach_reward_event = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self.approach_reward_pending = torch.zeros((n, 2), dtype=torch.bool, device=self.device)

        self.return_history_window = max(int(cfg.return_history_window), 1)
        self.return_history = torch.zeros((n, 2, self.return_history_window), dtype=torch.bool, device=self.device)
        self.return_history_next = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        self.return_opportunity_slot = torch.full((n, 2), -1, dtype=torch.long, device=self.device)
        self.return_opportunity_pending = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self.return_hit_pending = torch.zeros((n, 2), dtype=torch.bool, device=self.device)

        self.agent_state = torch.full((n, 2), self.READY, dtype=torch.long, device=self.device)

        self._shuttle: RigidObject = self.scene["shuttle"]
        self._robots: tuple[Articulation, Articulation] = (self.scene["robot_0"], self.scene["robot_1"])
        self._sweet_spot_ids = tuple(robot.find_bodies("racket_sweet_spot")[0][0] for robot in self._robots)
        self._contacts: tuple[ContactSensor, ContactSensor] = (
            self.scene.sensors["racket_contact_0"],
            self.scene.sensors["racket_contact_1"],
        )
        self.player_1_reward_manager = RewardManager(cfg.player_1_rewards, self)
        self._drag_k = float(cfg.shuttle_drag_k)
        self._mass = float(cfg.shuttle_mass)
        self._g_vec = torch.tensor((0.0, 0.0, float(cfg.gravity_z)), device=self.device)
        for sensor_name in ("racket_contact_0", "racket_contact_1"):
            history_len = int(getattr(self.cfg.scene, sensor_name).history_length)
            if history_len != int(self.cfg.decimation):
                raise ValueError(
                    f"{sensor_name}.history_length ({history_len}) must equal decimation "
                    f"({self.cfg.decimation}): the hit detector reduces the whole history "
                    "window, which must cover exactly one control step's substeps."
                )

    def step(self, action: torch.Tensor):
        """Advance physics, update event state, compute rewards, and handle resets."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self._apply_shuttle_forces()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._detect_landing()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._update_hits()
        self._update_rally_result()
        self._update_approach_state()
        self._update_return_history()
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        reward_0 = self.reward_manager.compute(dt=self.step_dt)
        reward_1 = self.player_1_reward_manager.compute(dt=self.step_dt)
        self.approach_reward_event.zero_()
        self.agent_reward_buf = torch.stack((reward_0, reward_1), dim=1)
        self.reward_buf = self.agent_reward_buf.mean(dim=1)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            self.recorder_manager.record_post_reset(reset_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _apply_shuttle_forces(self):
        """Apply aerodynamic drag and gravity to the shuttlecock."""
        velocity = self._shuttle.data.root_lin_vel_w
        force = -self._drag_k * torch.linalg.norm(velocity, dim=-1, keepdim=True) * velocity + self._mass * self._g_vec
        force_b = quat_apply_inverse(self._shuttle.data.root_quat_w, force)
        self._shuttle.set_external_force_and_torque(
            force_b.unsqueeze(1), torch.zeros_like(force_b).unsqueeze(1), is_global=False
        )

    def _detect_landing(self):
        """Update the instantaneous landed flag from the shuttlecock height."""
        local_z = self._shuttle.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.shuttle_landed.copy_(local_z <= self.cfg.shuttle_landing_detection_height)

    def _update_hits(self):
        """Detect valid alternating hits and advance the two-agent state machine."""
        self.just_hit.zero_()
        expected_hitter = self.next_hitter.clone()
        for side, contact in enumerate(self._contacts):
            contact_force = (
                torch.linalg.norm(contact.data.force_matrix_w_history, dim=-1).amax(dim=1).amax(dim=-1).amax(dim=-1)
            )
            hit = (contact_force > self.cfg.hit_contact_threshold) & (expected_hitter == side) & (~self.rally_resolved)
            self.just_hit[:, side] = hit
            self.hit_count[:, side] += hit.long()
            self.last_hitter[hit] = side
            self.outgoing_eval_step[hit, side] = self.episode_length_buf[hit] + 1
        hit_ids = self.just_hit.nonzero(as_tuple=False)
        if hit_ids.numel() > 0:
            self.next_hitter[hit_ids[:, 0]] = 1 - hit_ids[:, 1]

        self.agent_state[self.just_hit[:, 1], 0] = self.READY
        self.agent_state[self.just_hit[:, 0], 1] = self.READY
        self.agent_state[self.just_hit] = self.RECOVER

        for side in range(2):
            ready_ids = self.just_hit[:, 1 - side].nonzero(as_tuple=False).flatten()
            if ready_ids.numel() > 0:
                self.approach_reward_pending[ready_ids, side] = True
                self.approach_min_distance[ready_ids, side] = torch.inf
                slots = self.return_history_next[ready_ids, side]
                self.return_history[ready_ids, side, slots] = False
                self.return_history_next[ready_ids, side] = (slots + 1) % self.return_history_window
                self.return_opportunity_slot[ready_ids, side] = slots
                self.return_opportunity_pending[ready_ids, side] = True
                self.return_hit_pending[ready_ids, side] = False

    def _update_rally_result(self):
        """Resolve a rally exclusively from shuttlecock events.

        A robot falling or tilting is deliberately not a scoring event.  The
        rally continues until the shuttle lands, goes out, hits the net, or
        the episode times out, so a policy cannot win merely by inducing an
        opponent posture that makes the opponent fall.
        """
        self.rally_just_resolved.zero_()
        self.prev_shuttle_pos_w.copy_(self.curr_shuttle_pos_w)
        self.curr_shuttle_pos_w.copy_(self._shuttle.data.root_pos_w)
        local = self.curr_shuttle_pos_w - self.scene.env_origins
        prev_local = self.prev_shuttle_pos_w - self.scene.env_origins
        self.net_fault = is_net_fault(prev_local, local, self.cfg.court) & ~self.rally_resolved
        self.out_fault = is_out_of_bounds(local, self.cfg.court) & ~self.rally_resolved & (self.last_hitter >= 0)
        self.rally_just_resolved = ~self.rally_resolved & (self.net_fault | self.out_fault | self.shuttle_landed)
        fault_with_hitter = (self.net_fault | self.out_fault) & (self.last_hitter >= 0)
        landing_loser = (local[:, 0] >= 0.0).long()
        loser = torch.where(fault_with_hitter, self.last_hitter, landing_loser)
        self.rally_loser[self.rally_just_resolved] = loser[self.rally_just_resolved]
        self.rally_resolved |= self.rally_just_resolved

    def _update_approach_state(self) -> None:
        """Track minimum racket distance while a READY agent receives the shuttlecock."""
        local = self.curr_shuttle_pos_w - self.scene.env_origins
        previous_local = self.prev_shuttle_pos_w - self.scene.env_origins
        for side, robot in enumerate(self._robots):
            reward_cfg = self.cfg.rewards if side == 0 else self.cfg.player_1_rewards
            params = reward_cfg.interval_approach.params
            sign = 1.0 if side == 0 else -1.0
            min_height = float(params["min_height"])
            max_height = float(params["max_height"])
            canonical_x = sign * local[:, 0]
            in_window = (local[:, 2] >= min_height) & (local[:, 2] <= max_height)
            tracking = (
                (self.agent_state[:, side] == self.READY)
                & self.approach_reward_pending[:, side]
                & in_window
                & (canonical_x < 0.0)
            )
            sweet_pos = robot.data.body_pos_w[:, self._sweet_spot_ids[side]]
            distance = torch.linalg.norm(self._shuttle.data.root_pos_w - sweet_pos, dim=-1)
            self.approach_min_distance[:, side] = torch.where(
                tracking,
                torch.minimum(self.approach_min_distance[:, side], distance),
                self.approach_min_distance[:, side],
            )

            hit = self.just_hit[:, side]
            crossed_interval = (previous_local[:, 2] >= max_height) & (local[:, 2] < min_height)
            missed = (
                (self.agent_state[:, side] == self.READY)
                & self.approach_reward_pending[:, side]
                & (previous_local[:, 2] >= min_height)
                & (local[:, 2] < min_height)
                & ((self.approach_min_distance[:, side] < torch.inf) | crossed_interval)
                & (canonical_x < 0.0)
            )
            completed = self.approach_reward_pending[:, side] & (hit | missed)
            event_distance = torch.where(
                self.approach_min_distance[:, side] < torch.inf,
                self.approach_min_distance[:, side],
                distance,
            )
            self.approach_event_distance[completed, side] = event_distance[completed]
            self.approach_reward_event[completed, side] = True
            self.approach_reward_pending[completed, side] = False
            self.approach_min_distance[completed, side] = torch.inf

    def _update_return_history(self) -> None:
        """Record whether each receive opportunity produces a successful return."""
        local = self.curr_shuttle_pos_w - self.scene.env_origins
        previous_local = self.prev_shuttle_pos_w - self.scene.env_origins
        for side in range(2):
            self.return_hit_pending[self.just_hit[:, side], side] = True
            if side == 0:
                crossed = (previous_local[:, 0] <= 0.0) & (local[:, 0] > 0.0)
            else:
                crossed = (previous_local[:, 0] >= 0.0) & (local[:, 0] < 0.0)
            successful_cross = crossed & (~self.net_fault) & (self.last_hitter == side)
            success = successful_cross & self.return_opportunity_pending[:, side] & self.return_hit_pending[:, side]
            success_ids = success.nonzero(as_tuple=False).squeeze(-1)
            if len(success_ids):
                slots = self.return_opportunity_slot[success_ids, side]
                self.return_history[success_ids, side, slots] = True
                self.return_opportunity_pending[success_ids, side] = False
                self.return_hit_pending[success_ids, side] = False
                self.return_opportunity_slot[success_ids, side] = -1

            resolved = self.rally_just_resolved & self.return_opportunity_pending[:, side]
            resolved_ids = resolved.nonzero(as_tuple=False).squeeze(-1)
            if len(resolved_ids):
                self.return_opportunity_pending[resolved_ids, side] = False
                self.return_hit_pending[resolved_ids, side] = False
                self.return_opportunity_slot[resolved_ids, side] = -1

    def _reset_idx(self, env_ids: Sequence[int]):
        """Log episode statistics and reset all per-environment event state."""
        player_0_hits = self.hit_count[env_ids, 0].float().mean()
        player_1_hits = self.hit_count[env_ids, 1].float().mean()
        player_0_opening_rate = (self.initial_hitter[env_ids] == 0).float().mean()
        player_0_return_rate = self.return_history[env_ids, 0].float().mean()
        player_1_return_rate = self.return_history[env_ids, 1].float().mean()
        super()._reset_idx(env_ids)
        for key in list(self.extras["log"]):
            if key.startswith("Episode_Reward/"):
                self.extras["log"][f"Player0/{key}"] = self.extras["log"].pop(key)
        player_1_info = self.player_1_reward_manager.reset(env_ids)
        self.extras["log"].update({f"Player1/{key}": value for key, value in player_1_info.items()})
        self.extras["log"]["Metrics/Rally/player_0_mean_hits"] = player_0_hits
        self.extras["log"]["Metrics/Rally/player_1_mean_hits"] = player_1_hits
        self.extras["log"]["Metrics/Rally/player_0_opening_rate"] = player_0_opening_rate
        self.extras["log"]["Metrics/Rally/player_0_return_rate"] = player_0_return_rate
        self.extras["log"]["Metrics/Rally/player_1_return_rate"] = player_1_return_rate
        opening_hitter = self.initial_hitter[env_ids]
        self.just_hit[env_ids] = False
        self.next_hitter[env_ids] = opening_hitter
        self.hit_count[env_ids] = 0
        self.last_hitter[env_ids] = -1
        self.rally_resolved[env_ids] = False
        self.rally_just_resolved[env_ids] = False
        self.rally_loser[env_ids] = -1
        self.shuttle_landed[env_ids] = False
        self.net_fault[env_ids] = False
        self.out_fault[env_ids] = False
        self.outgoing_eval_step[env_ids] = -1
        self.approach_min_distance[env_ids] = torch.inf
        self.approach_event_distance[env_ids] = 0.0
        self.approach_reward_event[env_ids] = False
        self.approach_reward_pending[env_ids] = False
        self.approach_reward_pending[env_ids, opening_hitter] = True
        self.return_opportunity_slot[env_ids] = -1
        self.return_opportunity_pending[env_ids] = False
        self.return_hit_pending[env_ids] = False
        opening_slots = self.return_history_next[env_ids, opening_hitter]
        self.return_history[env_ids, opening_hitter, opening_slots] = False
        self.return_history_next[env_ids, opening_hitter] = (opening_slots + 1) % self.return_history_window
        self.return_opportunity_slot[env_ids, opening_hitter] = opening_slots
        self.return_opportunity_pending[env_ids, opening_hitter] = True
        self.agent_state[env_ids] = self.READY
        new_pos = self._shuttle.data.root_pos_w[env_ids]
        self.prev_shuttle_pos_w[env_ids] = new_pos
        self.curr_shuttle_pos_w[env_ids] = new_pos
