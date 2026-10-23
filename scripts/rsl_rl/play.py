# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a shared-policy badminton checkpoint."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Booster-Badminton-Dual-v1", help="Registered badminton task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of two-player courts.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (-1 samples a seed).")
parser.add_argument("--real-time", action="store_true", help="Pace playback to simulation time.")
parser.add_argument("--video", action="store_true", help="Record one playback video, then exit.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in control steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import badmintonlab_game.tasks  # noqa: F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.shared_policy_wrapper import SharedPolicyVecEnvWrapper

from common import load_weights, prepare_agent_cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Control both players with the same deterministic policy."""
    agent_cfg = prepare_agent_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    log_dir = os.path.dirname(checkpoint)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Playing checkpoint: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "play"),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = SharedPolicyVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        load_weights(runner, checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        obs = env.get_observations()
        timestep = 0
        while simulation_app.is_running():
            start_time = time.perf_counter()
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
            if args_cli.real_time:
                time.sleep(max(env.unwrapped.step_dt - (time.perf_counter() - start_time), 0.0))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
