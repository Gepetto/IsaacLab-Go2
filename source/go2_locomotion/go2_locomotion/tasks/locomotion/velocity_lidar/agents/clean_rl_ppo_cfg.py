# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from go2_locomotion.tasks.utils.cleanrl.rl_cfg import CleanRlPpoActorCriticCfg


@configclass
class Go2FlatPPORunnerCfg(CleanRlPpoActorCriticCfg):
    save_interval = 1000

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

    experiment_name = "go2_flat"
    logger = "tensorboard"
    wandb_project = "go2_flat"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"
