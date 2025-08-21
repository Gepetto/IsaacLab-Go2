# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from go2_locomotion.tasks.utils.rl.torchrl import RLOptPPOConfig


# Convenience configurations for different scenarios
@configclass
class Go2RLOptPPOConfig(RLOptPPOConfig):
    """RLOpt PPO configuration for Digit V3."""

    def __post_init__(self):
        """Post-initialization setup."""
        # Set mini_batch_size to frames_per_batch if not specified
        self.loss.mini_batch_size = int(self.collector.frames_per_batch / self.loss.epochs)


@configclass
class Go2TestRLOptPPOConfig(Go2RLOptPPOConfig):
    """RLOpt PPO configuration for Digit V3 on flat terrain."""

    def __post_init__(self):
        """Post-initialization setup for flat terrain."""
        super().__post_init__()

        # Adjust configurations for flat terrain (typically easier)
        self.policy.num_cells = [256, 256, 128]
        self.value_net.num_cells = [256, 256, 128]
        self.collector.total_frames = 300_000_000  # Fewer frames for flat terrain
