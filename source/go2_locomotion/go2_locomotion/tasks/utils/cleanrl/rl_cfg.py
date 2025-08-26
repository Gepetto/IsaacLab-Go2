# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch.nn as nn
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .ppo import Agent


@configclass
class CleanRlPpoActorCriticCfg:
    seed: int = 42

    agent: Agent = None

    save_interval: int = MISSING

    learning_rate: float = MISSING
    num_steps: int = MISSING
    num_iterations: int = MISSING
    gamma: float = MISSING
    gae_lambda: float = MISSING
    updates_epochs: int = MISSING
    minibatch_size: int = MISSING
    clip_coef: float = MISSING
    ent_coef: float = MISSING
    vf_coef: float = MISSING
    max_grad_norm: float = MISSING
    norm_adv: bool = MISSING
    clip_vloss: bool = MISSING
    anneal_lr: bool = MISSING

    experiment_name: str = MISSING
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = MISSING

    load_run: str = MISSING
    load_checkpoint: str = MISSING


@configclass
class CleanRlDDPGCfg:
    seed: int = 42
    save_interval: int = MISSING

    privileged_actor: nn.Module = None
    target_actor: nn.Module = None
    critic: nn.Module = None
    vision_nn: nn.Module = None

    torch_deterministic: bool = True
    critic_learning_rate: float = MISSING
    actor_learning_rate: float = MISSING
    buffer_size: int = MISSING
    num_iterations: int = MISSING
    exploration_noise: float = MISSING
    batch_size: int = MISSING
    gamma: float = MISSING
    policy_frequency: int = MISSING
    tau: float = MISSING

    learning_starts: int = MISSING
    seq_len: int = MISSING
    nb_critics: int = MISSING
    local_steps: int = MISSING
    image_decimation: int = MISSING
    noise_clip: float = MISSING
    policy_noise: float = MISSING

    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    experiment_name: str = MISSING
    wandb_project: str = MISSING

    load_run: str = MISSING
    load_checkpoint: str = MISSING


@configclass
class CleanRlDDPGDataCollectorCfg:
    agent_cfg: CleanRlPpoActorCriticCfg = MISSING
    ddpg_algo_cfg: CleanRlDDPGCfg = MISSING
