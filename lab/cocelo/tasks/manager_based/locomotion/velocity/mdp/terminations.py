# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor

def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  
    elif env.scene.cfg.terrain.terrain_type == "generator":
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        asset: RigidObject = env.scene[asset_cfg.name]

        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def time_illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    time_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    return torch.any(contact_time >= time_threshold, dim=1)


def specific_joint_lower_limit_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
    threshold: float = -0.72,
) -> torch.Tensor:
    """Terminate when any specified joint goes below the given threshold."""
    asset: RigidObject = env.scene[asset_cfg.name]

    if joint_names is None:
        joint_names = ["left_shoulder_joint", "right_shoulder_joint"]

    # joint index 찾기
    joint_ids = [asset.find_joints(name)[0][0] for name in joint_names]

    # 해당 조인트 위치 추출
    joint_pos = asset.data.joint_pos[:, joint_ids]

    # 하나라도 threshold보다 작으면 terminate
    return (joint_pos < threshold).any(dim=1)


def joint_pos_limit_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_soft_limits: bool = False,
    margin: float = 0.0,
) -> torch.Tensor:
    """Terminate when any selected joint position exceeds its configured limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_limits = asset.data.soft_joint_pos_limits if use_soft_limits else asset.data.joint_pos_limits
    joint_pos_limits = joint_pos_limits[:, asset_cfg.joint_ids]

    lower_limits = joint_pos_limits[..., 0] + margin
    upper_limits = joint_pos_limits[..., 1] - margin
    return torch.logical_or(joint_pos < lower_limits, joint_pos > upper_limits).any(dim=1)
