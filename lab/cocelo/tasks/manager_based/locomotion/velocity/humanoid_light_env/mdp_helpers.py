from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def coupled_joint_pos_motor_space(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
    coupled_pairs: tuple[tuple[str, str, float, float, str], ...],
) -> torch.Tensor:
    """Return joint positions in the configured sim-to-sim order."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def coupled_joint_vel_motor_space(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
    coupled_pairs: tuple[tuple[str, str, float, float, str], ...],
) -> torch.Tensor:
    """Return joint velocities in the configured sim-to-sim order."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def body_ang_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Body angular velocity expressed in each selected body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    return math_utils.quat_apply_inverse(body_quat, body_ang_vel_w).reshape(env.num_envs, -1)


def body_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity direction projected into each selected body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    gravity_dir = asset.data.GRAVITY_VEC_W.unsqueeze(1)
    return math_utils.quat_apply_inverse(body_quat, gravity_dir).reshape(env.num_envs, -1)
