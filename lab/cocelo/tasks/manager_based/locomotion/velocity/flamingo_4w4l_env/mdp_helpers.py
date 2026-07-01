from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def measure_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return contact-force magnitudes for the selected sensor bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.norm(forces, dim=-1)


def height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.0) -> torch.Tensor:
    """Return ray-hit terrain heights relative to the raycaster origin."""
    sensor = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
