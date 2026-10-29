# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play badminton with a separate end-to-end TorchScript policy for each player."""

import argparse
import random
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--player0_checkpoint",
    required=True,
    help="Player 0 compiled TorchScript model including observation normalization.",
)
parser.add_argument(
    "--player1_checkpoint",
    required=True,
    help="Player 1 compiled TorchScript model including observation normalization.",
)
parser.add_argument("--task", default="BadmintonLab-Game-Play", help="Registered badminton task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of two-player courts.")
parser.add_argument("--num_rallies", type=int, default=100, help="Rallies per court before comparing total scores.")
parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 samples a seed).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_rallies < 1:
    parser.error("--num_rallies must be positive")
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import badmintonlab_game.tasks  # noqa: F401
from match_score import MatchScore


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedEnvCfg, _agent_cfg):
    """Run each player's deterministic policy on that player's observations."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    player0_checkpoint_path = retrieve_file_path(args_cli.player0_checkpoint)
    player1_checkpoint_path = retrieve_file_path(args_cli.player1_checkpoint)
    print(f"[INFO] Player 0 model: {player0_checkpoint_path}")
    print(f"[INFO] Player 1 model: {player1_checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        device = env.unwrapped.device
        score = MatchScore(env.unwrapped.num_envs, device, args_cli.num_rallies)
        player0_policy = torch.jit.load(player0_checkpoint_path, map_location=device).eval()
        player1_policy = torch.jit.load(player1_checkpoint_path, map_location=device).eval()
        while simulation_app.is_running():
            with torch.inference_mode():
                player0_action = player0_policy(obs["player_0"]).to(device=device, dtype=torch.float32)
                player1_action = player1_policy(obs["player_1"]).to(device=device, dtype=torch.float32)
                obs, info = env.step(torch.cat((player0_action, player1_action), dim=-1))
            score.update(info)
            if score.all_finished:
                break
        score.summary()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
