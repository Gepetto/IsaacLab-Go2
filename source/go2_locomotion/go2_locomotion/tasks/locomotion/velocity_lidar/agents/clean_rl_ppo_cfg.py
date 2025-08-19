# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from torch import nn

from isaaclab.utils import configclass

from go2_locomotion.tasks.utils.cleanrl.ppo import Agent, layer_init
from go2_locomotion.tasks.utils.cleanrl.rl_cfg import CleanRlPpoActorCriticCfg


class PrivilegedAgent(Agent):
    def __init__(self, envs):
        super().__init__(envs)
        SINGLE_OBSERVATION_SPACE = envs.unwrapped.single_observation_space["policy"].shape
        SINGLE_ACTION_SPACE = envs.unwrapped.single_action_space.shape
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(SINGLE_ACTION_SPACE)), std=0.01),
        )


@configclass
class Go2PrivilegedPPORunnerCfg(CleanRlPpoActorCriticCfg):
    save_interval = 1000

    agent = PrivilegedAgent

    learning_rate = 1.0e-3
    num_steps = 24
    num_iterations = 300
    gamma = 0.99
    gae_lambda = 0.95
    updates_epochs = 5
    minibatch_size = 16384
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 2.0
    max_grad_norm = 1.0
    norm_adv = True
    clip_vloss = True
    anneal_lr = True

    experiment_name = "go2_privileged"
    logger = "tensorboard"
    wandb_project = "go2_privileged"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"
