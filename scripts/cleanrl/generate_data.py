# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from CleanRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import pickle as pkl
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with CleanRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--buffer_size",
    type=int,
    default=2**16,
    help="Number of possible elements in the buffer.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append CleanRL cli arguments
cli_args.add_clean_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import go2_locomotion.tasks  # noqa: F401
from go2_locomotion.tasks.utils.cleanrl.ppo import Agent  # noqa: F401
from go2_locomotion.tasks.utils.cleanrl.replay_buffer import SeqReplayBuffer  # noqa: F401


def main():
    """Play with CleanRL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    data_collector_cfg = cli_args.parse_clean_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "clean_rl", data_collector_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if data_collector_cfg.load_run is None:
        print("[ERROR] argument `--load_run` is empty, no model to load!")
        return

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, data_collector_cfg.load_run, data_collector_cfg.load_checkpoint)
    print(f"[INFO] Loading model: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    if args_cli.buffer_size < env_cfg.scene.num_envs:
        print(
            "[ERROR] argument `--buffer_size` should be greater "
            f"or equal number of environments: {env_cfg.scene.num_envs}."
        )
        return

    if args_cli.buffer_size % env_cfg.scene.num_envs != 0:
        print(
            f"[ERROR] argument `--buffer_size` should be multiple of number of environments: {env_cfg.scene.num_envs}."
        )
        return

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    actor_sd = torch.load(resume_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_collector_cfg.privileged_actor is not None:
        actor = data_collector_cfg.privileged_actor(env).to(device)
    else:
        actor = Agent(env).to(device)
    actor.load_state_dict(actor_sd)
    actor.eval()

    obs = env.reset()[0]

    single_action_space = env.unwrapped.single_action_space.shape
    dummy_action_scale = torch.zeros(single_action_space)

    target_actor = data_collector_cfg.target_actor(env, dummy_action_scale, dummy_action_scale)
    dummy_hidden = torch.zeros(
        (
            env.unwrapped.num_envs,
            target_actor.memory.hidden_size,
        ),
        device=device,
    )

    with torch.no_grad():
        actions, _, _, _ = actor.get_action_and_value(
            actor.obs_rms(obs["privileged"], update=False), deterministic=True
        )
        actions_min, _ = actions.min(dim=0)
        actions_max, _ = actions.max(dim=0)

    rb_expert = SeqReplayBuffer(
        args_cli.buffer_size,
        (
            sum(
                [
                    np.prod(v.shape)
                    for k, v in env.unwrapped.single_observation_space["student"].items()
                    if k != "depth_image"
                ]
            ),
        ),
        env.unwrapped.single_observation_space["privileged"].shape,
        # Remove last dimension of the image, which in fact is 1
        env.unwrapped.single_observation_space["student"]["depth_image"].shape[:-1],
        single_action_space,
        "cpu",
        "cpu",
        env.unwrapped.num_envs,
        target_actor.memory.hidden_size,
    )
    del target_actor

    for _ in tqdm(range(args_cli.buffer_size // env.unwrapped.num_envs), mininterval=1.0):
        with torch.no_grad():
            actions, _, _, _ = actor.get_action_and_value(
                actor.obs_rms(obs["privileged"], update=False), deterministic=True
            )
            actions_min = torch.min(actions_min, actions.min(dim=0)[0])
            actions_max = torch.max(actions_max, actions.max(dim=0)[0])

        next_obs, rewards, next_done, _, _ = env.step(actions)

        def obs_remove_image(obs: dict) -> torch.Tensor:
            return torch.cat([val for name, val in obs.items() if name != "depth_image"], axis=1)

        rb_expert.add(
            obs_remove_image(obs["student"]),
            obs["privileged"],
            obs["student"]["depth_image"].squeeze(-1),
            obs_remove_image(next_obs["student"]),
            next_obs["privileged"],
            next_obs["student"]["depth_image"].squeeze(-1),
            actions,
            rewards,
            next_done,
            dummy_hidden,
            dummy_hidden,
        )

        obs = next_obs

    save_data = {
        "buffer": rb_expert,
        "actions": {"min": actions_min, "max": actions_max},
    }

    with open(f"{log_dir}/rb_expert.pkl", "wb") as handle:
        pkl.dump(save_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print("Dataset saved")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
