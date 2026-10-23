"""Expose both players of each court as samples for one shared policy."""

from __future__ import annotations

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class SharedPolicyVecEnvWrapper(VecEnv):
    """Expose N two-player courts as 2N independent samples to RSL-RL."""

    def __init__(self, env: gym.Env, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.device = self.unwrapped.device
        self.court_count = self.unwrapped.num_envs
        self.num_envs = 2 * self.court_count
        self.num_actions = self.unwrapped.single_agent_action_dim
        self.max_episode_length = self.unwrapped.max_episode_length
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf.repeat(2)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value[: self.court_count]

    def seed(self, seed: int = -1):
        return self.unwrapped.seed(seed)

    def _agent_obs(self, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
        policy = torch.cat((obs_dict["player_0"], obs_dict["player_1"]), dim=0)
        return TensorDict({"policy": policy}, batch_size=[self.num_envs])

    def get_observations(self):
        return self._agent_obs(self.unwrapped.observation_manager.compute())

    def reset(self):
        obs, extras = self.env.reset()
        return self._agent_obs(obs), extras

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected agent actions {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}")
        env_actions = torch.cat((actions[: self.court_count], actions[self.court_count :]), dim=-1)
        obs, _, terminated, truncated, extras = self.env.step(env_actions)
        rewards = self.unwrapped.agent_reward_buf.transpose(0, 1).reshape(-1)
        dones = (terminated | truncated).long().repeat(2)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.repeat(2)
        return self._agent_obs(obs), rewards, dones, extras

    def close(self):
        return self.env.close()
