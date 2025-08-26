# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import itertools
import numpy as np
import os
import pickle as pkl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .replay_buffer import SeqReplayBuffer, SeqReplayBufferSamples


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.memory = nn.GRU(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            hidden_size=256,
            batch_first=True,
        )  # dummy memory for compatibility
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
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


class Actor(nn.Module):
    def __init__(self, env, vision_latent_size: int):
        super().__init__()
        single_observation_space_size = env.unwrapped.single_observation_space["privileged"].shape
        self.memory = nn.GRU(
            single_observation_space_size,
            hidden_size=256,
            batch_first=True,
        )
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, np.prod(env.unwrapped.single_action_space.shape))

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, vision_latent, hidden_in):
        hidden_in = torch.swapaxes(hidden_in, 0, 1)

        x = torch.cat([x, vision_latent], -1)
        time_latent, hidden_out = self.memory(x, hidden_in)
        x = F.elu(self.fc1(time_latent))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = torch.tanh(self.fc_mu(x))

        return x * self.action_scale + self.action_bias, hidden_out, None


class DepthOnlyFCBackbone48x48(nn.Module):
    def __init__(self, scandots_output_dim):
        super().__init__()

        activation = nn.ELU()
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
            nn.Linear(1568, scandots_output_dim),  # 48x48
            nn.LeakyReLU(),
            nn.Linear(128, scandots_output_dim),
        )

        self.output_activation = activation

    def forward(self, vobs):
        bs, seql, w, h = vobs.size()

        vobs = vobs.view(-1, 1, w, h)

        vision_latent = self.output_activation(self.image_compression(vobs))
        vision_latent = vision_latent.view(bs, seql, 128)

        # if hist:
        #     vision_latent = vision_latent.repeat_interleave(5, axis=1)

        return vision_latent


def obs_remove_image(obs: dict) -> torch.Tensor:
    return torch.cat([val for name, val in obs.items() if name != "depth_image"], axis=1)


def DDPG(envs, ddpg_cfg, run_path, expert_run_path):
    if ddpg_cfg.logger == "wandb":
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter

        writer = WandbSummaryWriter(log_dir=run_path, flush_secs=10, cfg=ddpg_cfg.to_dict())
    elif ddpg_cfg.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

        writer = TensorboardSummaryWriter(log_dir=run_path)
    else:
        raise AssertionError("logger type not found")

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    # TRY NOT TO MODIFY: seeding
    random.seed(ddpg_cfg.seed)
    np.random.seed(ddpg_cfg.seed)
    torch.manual_seed(ddpg_cfg.seed)
    torch.backends.cudnn.deterministic = ddpg_cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{expert_run_path}/rb_expert.pkl", "rb") as handle:
        save_data = pkl.load(handle)

    actions_min = save_data["actions"]["min"]
    actions_max = save_data["actions"]["max"]

    actor = ddpg_cfg.target_actor(envs, actions_min, actions_max).to(device)
    vision_nn = ddpg_cfg.vision_nn(envs).to(device)

    qfs = [ddpg_cfg.critic(envs).to(device) for _ in range(ddpg_cfg.nb_critics)]
    qf_targets = [ddpg_cfg.critic(envs).to(device) for _ in range(ddpg_cfg.nb_critics)]
    for i in range(ddpg_cfg.nb_critics):
        qf_targets[i].load_state_dict(qfs[i].state_dict())

    q_optimizer = optim.Adam(
        itertools.chain(*([q.parameters() for q in qfs])),
        lr=ddpg_cfg.critic_learning_rate,
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()) + list(vision_nn.parameters()),
        lr=ddpg_cfg.actor_learning_rate,
    )

    single_action_space_size = envs.unwrapped.single_action_space.shape

    rb_expert = save_data["buffer"]
    rb = SeqReplayBuffer(
        ddpg_cfg.buffer_size,
        (
            sum(
                [
                    np.prod(v.shape)
                    for k, v in envs.unwrapped.single_observation_space["student"].items()
                    if k != "depth_image"
                ]
            ),
        ),
        envs.unwrapped.single_observation_space["privileged"].shape,
        # Remove last dimension of the image, which in fact is 1
        envs.unwrapped.single_observation_space["student"]["depth_image"].shape[:-1],
        single_action_space_size,
        device,
        "cpu",
        envs.unwrapped.num_envs,
        actor.memory.hidden_size,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=ddpg_cfg.seed)
    vision_latent = None

    gru_p_hidden_in = torch.zeros(
        (envs.unwrapped.num_envs, actor.memory.hidden_size),
        device=device,
    )
    gru_p_hidden_out = gru_p_hidden_in.clone()

    for global_step in range(ddpg_cfg.num_iterations):
        # ALGO LOGIC: put action logic here
        if global_step % ddpg_cfg.image_decimation == 0:
            with torch.no_grad():
                vision_latent = vision_nn(obs["student"]["depth_image"].squeeze(-1))
        if global_step < ddpg_cfg.learning_starts:
            actions = torch.zeros((envs.unwrapped.num_envs, *single_action_space_size), device=device).uniform_(-1, 1)
        else:
            with torch.no_grad():
                actions, gru_p_hidden_out = actor(
                    obs_remove_image(obs["student"]),
                    vision_latent,
                    gru_p_hidden_in.unsqueeze(0),
                )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, true_dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        if true_dones.any():
            true_dones_idx = torch.argwhere(true_dones).squeeze()
            gru_p_hidden_out[:, true_dones_idx] = 0

        rb.add(
            obs_remove_image(obs["student"]),
            obs["privileged"],
            obs["student"]["depth_image"].squeeze(-1),
            obs_remove_image(next_obs["student"]),
            next_obs["privileged"],
            next_obs["student"]["depth_image"].squeeze(-1),
            actions,
            rewards,
            terminations,
            gru_p_hidden_in,
            gru_p_hidden_out,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # It is either squeezed and unsqueezed here or the buffer has to be more complex...
        gru_p_hidden_in = gru_p_hidden_out.squeeze(0)

        actor_training_step = 0

        # ALGO LOGIC: training.
        if global_step > ddpg_cfg.learning_starts:
            for local_steps in range(ddpg_cfg.local_steps):
                data_agent = rb.sample(ddpg_cfg.batch_size // 2, ddpg_cfg.seq_len)
                data_expert = rb_expert.sample(ddpg_cfg.batch_size // 2, ddpg_cfg.seq_len)
                # Create new buffer sample, that is a 50/50 mix of agent and expert
                data = SeqReplayBufferSamples(
                    **{
                        arg: torch.cat([getattr(data_agent, arg), getattr(data_expert, arg)], dim=0).to(device)
                        # Iterate over elements of the named tuple
                        for arg in SeqReplayBufferSamples.__match_args__
                    }
                )

                with torch.no_grad():
                    vlatent = vision_nn(data.vision_next_observations.squeeze(1))
                    next_state_actions, _ = actor(
                        data.next_observations.squeeze(1),
                        vlatent,
                        data.p_ini_hidden_out.swapaxes(0, 1),
                    )

                    noise = torch.randn_like(next_state_actions) * ddpg_cfg.policy_noise
                    clipped_noise = noise.clamp(-ddpg_cfg.noise_clip, ddpg_cfg.noise_clip)
                    next_state_actions = (next_state_actions + clipped_noise).clamp(actions_min, actions_max)

                    targets_selected = torch.randperm(ddpg_cfg.nb_critics)[:2]
                    qf_next_targets = torch.stack(
                        [
                            qf_targets[i](
                                data.privileged_next_observations.squeeze(1),
                                next_state_actions,
                            )[0]
                            for i in targets_selected
                        ]
                    )
                    min_qf_next_target = qf_next_targets.min(dim=0).values
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * ddpg_cfg.gamma * (
                        min_qf_next_target
                    ).view(-1)

                true_samples_nb = data.mask.sum()
                qf_a_values = torch.stack([qf(data.privileged_observations, data.actions)[0].view(-1) for qf in qfs])

                squared_loss = (qf_a_values - next_q_value.unsqueeze(0)) ** 2
                qf_loss = (squared_loss * data.mask.view(-1).unsqueeze(0)).sum() / (
                    true_samples_nb * ddpg_cfg.nb_critics
                )

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # Update critics
                if local_steps % ddpg_cfg.policy_frequency == 0:
                    for qf, qf_target in zip(qfs, qf_targets):
                        for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                            target_param.data.copy_(ddpg_cfg.tau * param.data + (1 - ddpg_cfg.tau) * target_param.data)

                if local_steps == ddpg_cfg.local_steps - 1:
                    actor_training_step += 1

                    vlatent = vision_nn(data.vision_observations.squeeze(1))
                    diff_actions, _ = actor(
                        data.observations.squeeze(1),
                        vlatent,
                        data.p_ini_hidden_in.swapaxes(0, 1),
                    )

                    qs = torch.stack([qf(data.privileged_observations, diff_actions.unsqueeze(1))[0] for qf in qfs])
                    actor_loss = -(qs.squeeze(-1) * data.mask.unsqueeze(0)).sum() / (
                        true_samples_nb * ddpg_cfg.nb_critics
                    )

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optimizer.step()

                    writer.add_scalar("loss/actor_loss", actor_loss.item(), actor_training_step)
                    writer.add_scalar("infos/Q_max", qs.max().item(), actor_training_step)

        if (global_step + 1) % ddpg_cfg.save_interval == 0:
            model_path = f"{run_path}/model_{global_step}.pt"
            torch.save(actor.state_dict(), model_path)
            print("Saved model")
