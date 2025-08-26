from __future__ import annotations

import torch
from typing import NamedTuple


class SeqReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    privileged_observations: torch.Tensor
    vision_observations: torch.Tensor
    next_observations: torch.Tensor
    privileged_next_observations: torch.Tensor
    vision_next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    p_ini_hidden_in: torch.Tensor
    p_ini_hidden_out: torch.Tensor
    mask: torch.Tensor


class SeqReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        privileged_observation_space,
        vision_space,
        action_space,
        training_device="cpu",
        storing_device="cpu",
        num_envs: int = 1,
        gru_hidden_size: int = 256,
    ):
        buffer_args = {"dtype": torch.float32, "device": storing_device}

        self.overflow = False
        self.pos = 0
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.storing_device = storing_device
        self.training_device = training_device

        self.observations = torch.zeros((self.buffer_size, *observation_space), **buffer_args)
        self.next_observations = torch.zeros_like(self.observations)

        self.privileged_observations = torch.zeros((self.buffer_size, *privileged_observation_space), **buffer_args)
        self.privileged_next_observations = torch.zeros_like(self.privileged_observations)

        self.vision_observations = torch.zeros(
            (self.buffer_size, *vision_space),
            **buffer_args,
        )
        self.vision_next_observations = torch.zeros_like(self.vision_observations)

        self.actions = torch.zeros(self.buffer_size, *action_space, **buffer_args)
        self.rewards = torch.zeros((self.buffer_size,), **buffer_args)
        self.dones = torch.zeros_like(self.rewards)

        self.p_ini_hidden_in = torch.zeros((self.buffer_size, gru_hidden_size), **buffer_args)
        self.p_ini_hidden_out = torch.zeros_like(self.p_ini_hidden_in)

        # For the current episodes that started being added to the replay buffer
        # but aren't done yet. We want to still sample from them, however the masking
        # needs a termination point to not overlap to the next episode when full
        # or even to the empty part of the buffer when not full.
        self.markers = torch.zeros_like(self.rewards, dtype=torch.bool)

    def add(
        self,
        obs,
        privi_obs,
        vobs,
        next_obs,
        next_privi_obs,
        next_vobs,
        action,
        reward,
        done,
        p_ini_hidden_in,
        p_ini_hidden_out,
    ) -> None:
        start_idx = self.pos
        stop_idx = min(self.pos + obs.shape[0], self.buffer_size)
        b_max_idx = stop_idx - start_idx

        # Current episodes last transition marker
        self.markers[start_idx:stop_idx] = 1
        # We need to unmark previous transitions as last
        # but only if it is not the first add to the replay buffer
        if self.pos > 0:
            self.markers[self.prev_start_idx : self.prev_stop_idx] = 0
            if self.prev_overflow:
                self.markers[: self.prev_overflow_size] = 0

        self.overflow = False
        overflow_size = 0
        if self.pos + obs.shape[0] > self.buffer_size:
            self.overflow = True
            overflow_size = self.pos + obs.shape[0] - self.buffer_size

        assert start_idx % self.num_envs == 0, f"start_idx is not a multiple of {self.num_envs}"
        assert stop_idx % self.num_envs == 0, f"stop_idx is not a multiple of {self.num_envs}"
        assert b_max_idx == 0 or b_max_idx == self.num_envs, f"b_max_idx is not either 0 or {self.num_envs}"

        for target, source in (
            (self.observations, obs),
            (self.vision_observations, vobs),
            (self.next_observations, next_obs),
            (self.vision_next_observations, next_vobs),
            (self.privileged_observations, privi_obs),
            (self.privileged_next_observations, next_privi_obs),
            (self.actions, action),
            (self.rewards, reward),
            (self.dones, done),
            (self.p_ini_hidden_in, p_ini_hidden_in),
            (self.p_ini_hidden_out, p_ini_hidden_out),
        ):
            # Copy to avoid modification by reference
            target[start_idx:stop_idx] = source[:b_max_idx].clone().to(self.storing_device)

        self.prev_start_idx = start_idx
        self.prev_stop_idx = stop_idx
        self.prev_overflow = self.overflow
        self.prev_overflow_size = overflow_size

        assert overflow_size == 0 or overflow_size == self.num_envs, f"overflow_size is not either 0 or {self.num_envs}"
        if self.overflow:
            for target, source in (
                (self.observations, obs),
                (self.vision_observations, vobs),
                (self.next_observations, next_obs),
                (self.vision_next_observations, next_vobs),
                (self.privileged_observations, privi_obs),
                (self.privileged_next_observations, next_privi_obs),
                (self.actions, action),
                (self.rewards, reward),
                (self.dones, done),
                (self.p_ini_hidden_in, p_ini_hidden_in),
                (self.p_ini_hidden_out, p_ini_hidden_out),
            ):
                # Copy to avoid modification by reference
                target[start_idx:stop_idx] = source[:b_max_idx].clone().to(self.storing_device)

            self.pos = overflow_size
        else:
            self.pos += obs.shape[0]

    def sample(self, batch_size: int, sequence_len: int = 5) -> SeqReplayBufferSamples:
        upper_bound = self.buffer_size if self.overflow else self.pos
        batch_inds = torch.randint(0, upper_bound, size=(batch_size,), device=self.storing_device)
        return self._get_samples(batch_inds, sequence_len)

    def _get_samples(self, batch_inds, sequence_len: int = 5) -> SeqReplayBufferSamples:
        # Using modular arithmetic we get the indices of all the transitions of the episode starting from batch_inds
        # we get "episodes" of length sequence_len, but their true length may be less, they can have ended before that
        # we'll deal with that using a mask
        # Using flat indexing we can actually slice through a tensor using
        # different starting points for each dimension of an axis
        # as long as the slice size remains constant

        batch_size = batch_inds.shape[0]
        # [1, 2, 3].repeat(3) -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
        batch_inds_increase = torch.arange(sequence_len, device=self.storing_device).repeat(batch_size) * self.num_envs
        # [1, 2, 3].repeat_interleave(3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        inds_flat = (batch_inds.repeat_interleave(sequence_len) + batch_inds_increase) % self.buffer_size

        def _get_reshaped_bath(arg: str):
            buffer = getattr(self, arg)
            return buffer[inds_flat].reshape((batch_size, sequence_len, *buffer.shape[1:]))

        args = {
            arg: _get_reshaped_bath(arg)
            for arg in (
                "observations",
                "privileged_observations",
                "next_observations",
                "privileged_next_observations",
                "actions",
                "rewards",
                "dones",
                "p_ini_hidden_in",
                "p_ini_hidden_out",
                "vision_observations",
                "vision_next_observations",
            )
        }

        gathered_markers = self.markers[inds_flat].reshape((batch_size, sequence_len))
        args["mask"] = torch.cat(
            [
                torch.ones((batch_size, 1), device=self.storing_device),
                (1 - ((args["dones"] >= 1.0) | gathered_markers).float()).cumprod(dim=1)[:, 1:],
            ],
            dim=1,
        )
        return SeqReplayBufferSamples(**args)
