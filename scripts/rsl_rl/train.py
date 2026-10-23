# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one PPO policy controlling both players on each badminton court."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Booster-Badminton-Dual-v1", help="Registered badminton task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of two-player courts.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (-1 samples a seed).")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of PPO iterations.")
parser.add_argument("--distributed", action="store_true", help="Train using multiple GPUs.")
parser.add_argument("--video", action="store_true", help="Record training videos.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in control steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Control steps between videos.")
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
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import badmintonlab_game.tasks  # noqa: F401
from badmintonlab_game.tasks.manager_based.badmintonlab_game.shared_policy_wrapper import SharedPolicyVecEnvWrapper

from common import prepare_agent_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect both players' experience and update their shared policy."""
    agent_cfg = prepare_agent_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args_cli))
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.distributed:
        if "cpu" in env_cfg.sim.device:
            raise ValueError("Distributed training requires a CUDA device.")
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = env_cfg.sim.device
        agent_cfg.seed += app_launcher.local_rank
    env_cfg.seed = agent_cfg.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = None
    if agent_cfg.resume:
        resume_path = (
            retrieve_file_path(args_cli.checkpoint)
            if args_cli.checkpoint
            else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = SharedPolicyVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] Self-play: {env.court_count} courts, {env.num_envs} player samples per step.")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "..", ".git")):
            runner.add_git_repo_to_log(__file__)
        if resume_path:
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
