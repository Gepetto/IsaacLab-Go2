# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass

from . import patterns


@configclass
class LivoxPatternCfg(PatternBaseCfg):
    """Configuration for Livox LiDAR pattern for ray-casting.

    Supports various Livox sensor types including Avia, Horizon, HAP, MID360, MID40, MID70, and Tele.
    Each sensor has predefined scan patterns that can be loaded from data files.
    """

    func: Callable = patterns.livox_pattern

    sensor_type: str = MISSING
    """Type of Livox sensor. Supported: 'avia', 'horizon', 'HAP', 'mid360', 'mid40', 'mid70', 'tele'."""

    samples: int = 24000
    """Number of ray samples per scan frame. Defaults to 24000."""

    downsample: int = 1
    """Downsampling factor for ray patterns. Defaults to 1 (no downsampling)."""

    use_simple_grid: bool = False
    """Whether to use simple grid pattern instead of real scan pattern. Defaults to False."""

    # Simple grid parameters (used when use_simple_grid=True)
    horizontal_line_num: int = 80
    """Number of horizontal lines for simple grid. Defaults to 80."""

    vertical_line_num: int = 50
    """Number of vertical lines for simple grid. Defaults to 50."""

    horizontal_fov_deg_min: float = -180
    """Minimum horizontal FOV in degrees for simple grid. Defaults to -180."""

    horizontal_fov_deg_max: float = 180
    """Maximum horizontal FOV in degrees for simple grid. Defaults to 180."""

    vertical_fov_deg_min: float = -2
    """Minimum vertical FOV in degrees for simple grid. Defaults to -2."""

    vertical_fov_deg_max: float = 57
    """Maximum vertical FOV in degrees for simple grid. Defaults to 57."""

    rolling_window_start: int = 0
    """Starting index for rolling window sampling from pattern files. Defaults to 0."""
