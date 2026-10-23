# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play badminton with a separate TorchScript policy for each player."""

import argparse
import random
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--player0_checkpoint", required=True, help="Player 0 TorchScript model including observation normalization."
)
parser.add_argument(
    "--player1_checkpoint", required=True, help="Player 1 TorchScript model including observation normalization."
)
parser.add_argument("--task", default="BadmintonLab-Game", help="Registered badminton task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of two-player courts.")
parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 samples a seed).")
parser.add_argument("--clip_actions", type=float, default=None, help="Optional symmetric action clipping limit.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import badmintonlab_game.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedEnvCfg, _agent_cfg):
    """Run each player's deterministic policy on that player's observations."""
    if args_cli.clip_actions is not None and args_cli.clip_actions <= 0:
        raise ValueError("--clip_actions must be positive.")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    player0_checkpoint_path = retrieve_file_path(args_cli.player0_checkpoint)
    player1_checkpoint_path = retrieve_file_path(args_cli.player1_checkpoint)
    print(f"[INFO] Player 0 TorchScript model: {player0_checkpoint_path}")
    print(f"[INFO] Player 1 TorchScript model: {player1_checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        player0_policy = torch.jit.load(player0_checkpoint_path, map_location=env.unwrapped.device).eval()
        player1_policy = torch.jit.load(player1_checkpoint_path, map_location=env.unwrapped.device).eval()
        while simulation_app.is_running():
            with torch.inference_mode():
                player0_action = player0_policy(obs["player_0"])
                player1_action = player1_policy(obs["player_1"])
                if args_cli.clip_actions is not None:
                    player0_action = player0_action.clamp(-args_cli.clip_actions, args_cli.clip_actions)
                    player1_action = player1_action.clamp(-args_cli.clip_actions, args_cli.clip_actions)
                obs, _ = env.step(torch.cat((player0_action, player1_action), dim=-1))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
