# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a two-player TorchScript match to MP4 without opening a window."""

import argparse
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--player0_checkpoint", required=True, help="Player 0 compiled TorchScript model.")
parser.add_argument("--player1_checkpoint", required=True, help="Player 1 compiled TorchScript model.")
parser.add_argument("--task", default="BadmintonLab-Game-Play", help="Registered badminton task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 samples a seed).")
parser.add_argument("--num_rallies", type=int, default=100, help="Rallies per court before comparing total scores.")
parser.add_argument("--output", type=Path, default=Path("outputs/videos/match.mp4"), help="Output MP4 path.")
parser.add_argument(
    "--video_length", type=int, default=1000, help="Maximum frames (1000 = 20 s); match may finish earlier."
)
parser.add_argument("--width", type=int, default=1280, help="Video width in pixels.")
parser.add_argument("--height", type=int, default=720, help="Video height in pixels.")
parser.add_argument("--shuttle_scale", type=float, default=1.5, help="Visual-only shuttle mesh scale.")
parser.add_argument("--camera_eye", type=float, nargs=3, default=(8.5, -11.0, 7.0), help="Camera position in meters.")
parser.add_argument("--camera_target", type=float, nargs=3, default=(0.0, 0.0, 0.2), help="Camera look-at position.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_rallies < 1:
    parser.error("--num_rallies must be positive")
args_cli.headless = True
args_cli.enable_cameras = True
args_cli.livestream = 0
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import omni.replicator.core as rep
import torch

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import badmintonlab_game.tasks  # noqa: F401
from render_scene import MatchVisuals
from match_score import MatchScore


@hydra_task_config(args_cli.task, "")
def main(env_cfg: ManagerBasedEnvCfg, _agent_cfg):
    """Run one court and capture each completed control step from a fixed camera."""
    env_cfg.scene.num_envs = 1
    env_cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.render.antialiasing_mode = "DLAA"
    env_cfg.sim.render.enable_shadows = True
    env_cfg.sim.render.enable_ambient_occlusion = True
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args_cli.camera_eye)
    env_cfg.viewer.lookat = tuple(args_cli.camera_target)
    env_cfg.viewer.resolution = (args_cli.width, args_cli.height)

    player0_path = retrieve_file_path(args_cli.player0_checkpoint)
    player1_path = retrieve_file_path(args_cli.player1_checkpoint)
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        raw = env.unwrapped
        device = raw.device
        obs, _ = env.reset()
        score = MatchScore(raw.num_envs, device, args_cli.num_rallies)
        visuals = MatchVisuals(raw, args_cli.shuttle_scale)
        player0_policy = torch.jit.load(player0_path, map_location=device).eval()
        player1_policy = torch.jit.load(player1_path, map_location=device).eval()
        fps = round(1.0 / raw.step_dt)
        render_product = rep.create.render_product(env_cfg.viewer.cam_prim_path, env_cfg.viewer.resolution)
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([render_product])
        frames = 0
        try:
            # Warm up the renderer without advancing physics or the policies.
            for _ in range(10):
                raw.sim.render()
            print(f"[INFO] Recording {args_cli.video_length} frames at {fps} FPS to {output}", flush=True)
            with imageio.get_writer(
                output, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2
            ) as writer:
                for _ in range(args_cli.video_length):
                    if not simulation_app.is_running():
                        break
                    with torch.inference_mode():
                        action0 = player0_policy(obs["player_0"]).to(device=device, dtype=torch.float32)
                        action1 = player1_policy(obs["player_1"]).to(device=device, dtype=torch.float32)
                        obs, info = env.step(torch.cat((action0, action1), dim=-1))
                    score.update(info)
                    visuals.update()
                    raw.sim.render()
                    writer.append_data(rgb.get_data()[:, :, :3])
                    frames += 1
                    if frames % 100 == 0:
                        print(f"[INFO] Recorded {frames}/{args_cli.video_length} frames", flush=True)
                    if score.all_finished:
                        break
        finally:
            rgb.detach()
            render_product.destroy()
        print(f"[INFO] Saved {frames} frames ({frames / fps:.2f} s): {output}", flush=True)
        score.summary()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
