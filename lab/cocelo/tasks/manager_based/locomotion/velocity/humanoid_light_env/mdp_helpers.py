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


def coupled_joint_pos_motor_space(
    env: ManagerBasedEnv,
    joint_names: Sequence[str],
    coupled_pairs: Sequence[tuple[str, str, float, float, str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint positions with coupled pitch/roll slots replaced by motor-space values."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return _replace_coupled_pairs_with_motor_values(pos, joint_names, coupled_pairs)


def coupled_joint_vel_motor_space(
    env: ManagerBasedEnv,
    joint_names: Sequence[str],
    coupled_pairs: Sequence[tuple[str, str, float, float, str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint velocities with coupled pitch/roll slots replaced by motor-space values."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return _replace_coupled_pairs_with_motor_values(vel, joint_names, coupled_pairs)


def _replace_coupled_pairs_with_motor_values(
    joint_value: torch.Tensor,
    joint_names: Sequence[str],
    coupled_pairs: Sequence[tuple[str, str, float, float, str]],
) -> torch.Tensor:
    motor_value = joint_value.clone()
    joint_name_to_index = {name: index for index, name in enumerate(joint_names)}

    for pitch_name, roll_name, gear_ratio_1, gear_ratio_2, coupling in coupled_pairs:
        pitch_id = joint_name_to_index[pitch_name]
        roll_id = joint_name_to_index[roll_name]

        pitch = joint_value[:, pitch_id]
        roll = joint_value[:, roll_id]

        if coupling == "roll_sum":
            if pitch_name.startswith("left_ankle_"):
                motor_1 = -float(gear_ratio_1) * (roll - pitch)
                motor_2 = -float(gear_ratio_2) * (roll + pitch)
            elif pitch_name.startswith("right_ankle_"):
                motor_1 = -float(gear_ratio_1) * (roll + pitch)
                motor_2 = -float(gear_ratio_2) * (roll - pitch)
            elif pitch_name.startswith("torso_"):
                motor_1 = float(gear_ratio_1) * (roll - pitch)
                motor_2 = -float(gear_ratio_2) * (roll + pitch)
            else:
                raise RuntimeError(f"Unsupported coupled observation pair: {pitch_name}, {roll_name}")
        else:
            raise RuntimeError(f"Unsupported coupled observation mapping: {coupling}")

        motor_value[:, pitch_id] = motor_1
        motor_value[:, roll_id] = motor_2

    return motor_value


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
