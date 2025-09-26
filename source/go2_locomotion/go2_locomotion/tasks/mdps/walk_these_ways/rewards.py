from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward tracking of linear velocity commands (xy axes)
    in the gravity aligned robot frame
    using exponential kernel.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def reward_roll_pitch_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, :2])


def reward_foot_slip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Get contact indices

    # Multiply by foot velocity
    pass


def reward_foot_slip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalize foot slip: non-zero horizontal velocity while foot is in contact.
    Returns a tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    foot_ids = [asset.data.body_names.index(name) for name in asset_cfg.foot_links]

    # --- Get velocities of those links in world frame ---
    body_vels = asset.data.body_link_lin_vel_w  # (num_envs, num_bodies, 3)
    foot_vels = body_vels[:, foot_ids, :2]  # (num_envs, num_feet, 2) → only x,y

    # --- Compute horizontal slip speed (L2 norm) ---
    slip_speed = torch.linalg.norm(foot_vels, dim=-1)  # (num_envs, num_feet)

    # --- Get contact states for those links ---
    contacts = asset.data.body_contact_w[:, foot_ids]  # (num_envs, num_feet), bool tensor

    # --- Reward: penalize slip when in contact ---
    slip_penalty = slip_speed * contacts.float()  # (num_envs, num_feet)
    reward = -slip_penalty.mean(dim=-1)  # average across feet

    return reward


def reward_calf_thigh(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    pass


def reward_joint_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    pass


def reward_joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]


def reward_joint_velocities(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    pass


def reward_joint_acceleration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    pass


def reward_action_smoothness_1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    pass


def reward_action_smoothness_2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ """
    asset: RigidObject = env.scene[asset_cfg.name]
    pass


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)
