# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from isaaclab.utils import configclass

from go2_locomotion.tasks.utils.cleanrl.ppo import Agent, layer_init
from go2_locomotion.tasks.utils.cleanrl.rl_cfg import CleanRlDDPGCfg, CleanRlPpoActorCriticCfg

VISION_HIDDEN_SIZE = 128


class PrivilegedAgent(Agent):
    def __init__(self, env):
        super().__init__(env, "privileged")
        single_observation_space = np.prod(env.unwrapped.single_observation_space["privileged"].shape)
        single_action_space = env.unwrapped.single_action_space.shape
        self.critic = nn.Sequential(
            layer_init(nn.Linear(single_observation_space, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(single_observation_space, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(single_action_space)), std=0.01),
        )


class Actor(nn.Module):
    def __init__(self, env, action_max: torch.Tensor, action_min: torch.Tensor):
        super().__init__()
        single_observation_space = sum(
            [
                np.prod(v.shape)
                for k, v in env.unwrapped.single_observation_space["student"].items()
                if k != "depth_image"
            ]
        )
        single_action_space = np.prod(env.unwrapped.single_action_space.shape)
        self.memory = nn.GRU(
            single_observation_space + VISION_HIDDEN_SIZE,
            hidden_size=256,
            batch_first=True,
        )
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, single_action_space)

        # action rescaling
        self.register_buffer("action_scale", (action_max - action_min) / 2.0)
        self.register_buffer("action_bias", (action_max - action_min) / 2.0)

    def forward(self, x, vision_latent, hidden_in):
        if hidden_in is None:
            raise NotImplementedError

        x = torch.cat([x, vision_latent], -1).unsqueeze(1)
        time_latent, hidden_out = self.memory(x, hidden_in)
        x = F.elu(self.fc1(time_latent))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = torch.tanh(self.fc_mu(x))

        return x.squeeze(1) * self.action_scale + self.action_bias, hidden_out.squeeze(0)


class DepthOnlyFCBackbone48x48(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.image_compression = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
            nn.Flatten(),
            # Why 800?
            nn.Linear(800, VISION_HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE),
        )

        self.output_activation = nn.ELU()

    def forward(self, vobs):
        bs, w, h = vobs.size()

        vobs = vobs.view(-1, 1, w, h)

        vision_latent = self.image_compression(vobs)
        vision_latent = self.output_activation(vision_latent)
        vision_latent = vision_latent.view(bs, VISION_HIDDEN_SIZE)

        # if hist:
        #     vision_latent = vision_latent.repeat_interleave(5, axis=1)

        return vision_latent


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        single_observation_space = np.prod(env.unwrapped.single_observation_space["privileged"].shape)
        single_action_space = np.prod(env.unwrapped.single_action_space.shape)
        self.memory = nn.GRU(
            single_observation_space + single_action_space,
            hidden_size=256,
            batch_first=True,
        )  # dummy memory for compatibility
        self.fc1 = nn.Linear(
            single_observation_space + single_action_space,
            512,
        )
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], -1)
        x = F.elu(self.ln1(self.fc1(x)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x, None


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


@configclass
class Go2PrivilegedDDPGRunnerCfg(CleanRlDDPGCfg):
    save_interval = 500

    privileged_actor = PrivilegedAgent
    target_actor = Actor
    critic = QNetwork
    vision_nn = DepthOnlyFCBackbone48x48

    torch_deterministic = True

    critic_learning_rate = 3.0e-4
    actor_learning_rate = 5.0e-4

    buffer_size = 1024
    num_iterations = 500000
    exploration_noise = 0.1
    batch_size = 128
    gamma = 0.99
    policy_frequency = 2
    tau = 0.05

    learning_starts = 50
    seq_len = 1
    nb_critics = 10
    local_steps = 8
    image_decimation = 5
    noise_clip = 0.2
    policy_noise = 0.8

    experiment_name = "go2_privileged"
    logger = "tensorboard"
    wandb_project = "go2_privileged"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"
