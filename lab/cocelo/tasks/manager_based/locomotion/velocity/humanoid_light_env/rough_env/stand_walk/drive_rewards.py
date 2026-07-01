from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of xy velocity commands in the yaw-aligned robot frame."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of world-frame yaw angular velocity commands."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def reward_keep_balance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def feet_crossing_lateral_penalty(
    env: ManagerBasedRLEnv,
    min_width: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet that cross or become narrower than a lateral width threshold."""
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    lateral_distance = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    return torch.clamp(min_width - lateral_distance, min=0.0)
