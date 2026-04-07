
from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_ang_vel_z_link_exp(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_y = torch.abs(env.command_manager.get_command(command_name)[:, 1])
    # compute the error
    ang_vel = torch.square(asset.data.root_link_ang_vel_b[:, 2]) * lin_vel_y
    return ang_vel

def reward_feet_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_feet_distance: float = 0.4885,
    max_feet_distance: float = 0.4885,
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    # foot positions in world frame
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]  # [N,2,2]
    dist = torch.norm(foot_pos[:,0,:] - foot_pos[:,1,:], dim=-1)
    penalize_min = torch.clip(min_feet_distance - dist, 0.0, 1.0)
    penalize_max = torch.clip(dist - max_feet_distance, 0.0, 1.0)
    return penalize_min + penalize_max

def reward_nominal_foot_position_adaptive(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg_left: SceneEntityCfg | None = None,
    sensor_cfg_right: SceneEntityCfg | None = None,
    command_name: str = "base_velocity",
    base_height_target: float = 0.36288,
    foot_radius: float = 0.127,
    temperature: float = 200.0,
    sigma_wrt_v: float = 0.5,
) -> torch.Tensor:
    """
    Reward foot height tracking relative to a dynamic target height per foot.
    Each foot uses its corresponding height sensor to adapt the target height in real time:
    nominal_height = foot_radius - base_height_target + delta,
    where delta is the max detected terrain step for that foot.

    Args:
        env: ManagerBasedRLEnv
        asset_cfg: SceneEntityCfg for the robot asset (contains foot body_ids)
        sensor_cfg_left: SceneEntityCfg for left foot height sensor (RayCaster)
        sensor_cfg_right: SceneEntityCfg for right foot height sensor (RayCaster)
        command_name: name of the command to retrieve velocity commands
        base_height_target: static base height target (m)
        foot_radius: radius/offset of the foot link (m)
        sigma: Gaussian width for height error
        sigma_wrt_v: Gaussian width for velocity attenuation

    Returns:
        Tensor of shape [num_envs] with reward values
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmds = env.command_manager.get_command(command_name)
    num_envs = env.num_envs
    device = env.device

    # Build per-foot dynamic target heights [num_envs, 2]
    target_height = torch.full((num_envs, len(asset_cfg.body_ids)), base_height_target, device=device)
    # Left foot
    if sensor_cfg_left is not None:
        sl: RayCaster = env.scene[sensor_cfg_left.name]
        sensor_z_l = sl.data.pos_w[:, 2]                       # [N]
        hit_z_l    = torch.max(sl.data.ray_hits_w[..., 2], dim=1).values  # [N]
        delta_l    = (sensor_z_l - hit_z_l) + 0.05             # [N], +여유마진
        target_height[:, 0] = base_height_target - delta_l

    # Right foot
    if sensor_cfg_right is not None:
        sr: RayCaster = env.scene[sensor_cfg_right.name]
        sensor_z_r = sr.data.pos_w[:, 2]
        hit_z_r    = torch.max(sr.data.ray_hits_w[..., 2], dim=1).values
        delta_r    = (sensor_z_r - hit_z_r) + 0.05
        target_height[:, 1] = base_height_target - delta_r

    # Compute nominal foot height relative to base origin [N,2]
    # nominal_height = foot_radius - (base_height_target - delta)
    nominal_height = foot_radius - target_height  # [N,2]

    # World->base translation + rotation
    base_pos  = asset.data.root_link_pos_w   # [N,3]
    base_quat = asset.data.root_link_quat_w  # [N,4]
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [N,2,3]
    foot_base  = foot_world - base_pos.unsqueeze(1)                  # [N,2,3]

    # Calculate reward per foot
    reward = torch.zeros(num_envs, device=device)
    for i in range(len(asset_cfg.body_ids)):
        fb = quat_apply_inverse(base_quat, foot_base[:, i, :])  # [N,3]
        err = nominal_height[:, i] - fb[:, 2]                    # [N]
        reward += torch.exp(-err.square() * temperature)

    # Average across feet and apply velocity attenuation
    vel_norm = torch.norm(cmds[:, :3], dim=1)                   # [N]
    reward = (reward / len(asset_cfg.body_ids)) * torch.exp(-vel_norm.square() / sigma_wrt_v)

    return reward

def reward_nominal_foot_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    command_name: str = "base_velocity",
    base_height_target: float = 0.36288,
    foot_radius: float = 0.127,
    sigma: float = 0.005,
    sigma_wrt_v: float = 0.5
) -> torch.Tensor:
    """
    Reward foot height tracking relative to nominal base height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmds = env.command_manager.get_command(command_name)
    # base frame data
    base_pos = asset.data.root_link_pos_w  # [N,3]
    base_quat = asset.data.root_link_quat_w
    # body-frame foot positions
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [N,2,3]
    foot_base = foot_world - base_pos.unsqueeze(1)
    reward = torch.zeros(env.num_envs, device=env.device)
    nominal_height = -(base_height_target - foot_radius)
    for i in range(len(asset_cfg.body_ids)):
        fb = quat_apply_inverse(base_quat, foot_base[:,i,:])  # [N,3]
        err = nominal_height - fb[:,2]
        reward += torch.exp(-(torch.square(err))/sigma)
    vel_norm = torch.norm(cmds[:, :3], dim=1)
    reward = (reward/len(asset_cfg.body_ids)) * torch.exp(-(vel_norm**2)/sigma_wrt_v)
    return reward


def reward_leg_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    temperature = 50.0 # 0.001,
) -> torch.Tensor:
    """
    Encourage symmetry in Y direction between two feet.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_apply_inverse(base_quat, foot_base[:,i,:])
    err = (foot_base[:,0,1].abs() - foot_base[:,1,1].abs())
    return torch.exp(-(temperature * torch.square(err)))


def reward_same_foot_x_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize X-axis displacement difference of two feet in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_apply_inverse(base_quat, foot_base[:,i,:])
    dx = foot_base[:,0,0] - foot_base[:,1,0]
    return torch.abs(dx)

def reward_same_foot_y_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Encourage symmetry in Y direction between two feet.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_apply_inverse(base_quat, foot_base[:,i,:])
    dy = (foot_base[:,0,1].abs() - foot_base[:,1,1].abs())
    return torch.abs(dy)


def backward_falloff_on_step(
    env: ManagerBasedRLEnv,
    height_diff_threshold: float = 0.04,
    backward_vel_threshold: float = 0.02,
    vel_scale: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize moving backward when one wheel is already higher than the other.

    Returns a normalized penalty in roughly [0, 1] so it stays comparable to other task rewards.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    wheel_pos_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    height_diff = torch.abs(wheel_pos_z[:, 0] - wheel_pos_z[:, 1])
    one_wheel_up = height_diff > height_diff_threshold

    cmd = env.command_manager.get_command(command_name)
    forward_cmd = cmd[:, 0] > 0.1
    backward_vel = torch.clamp(-asset.data.root_link_lin_vel_b[:, 0] - backward_vel_threshold, min=0.0)
    normalized_penalty = torch.tanh(backward_vel / max(vel_scale, 1.0e-6))
    return normalized_penalty * one_wheel_up.float() * forward_cmd.float()


def stair_ascent_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
    height_threshold: float = 0.05,
    forward_vel_threshold: float = 0.05,
    height_scale: float = 0.12,
    vel_scale: float = 0.4,
    completion_bonus: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward actual stair ascent with a normalized output.

    The reward only activates on stairs, while moving forward, and stays in a bounded range so it does not
    dominate the rest of the reward terms.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    stair_mask = _stair_mask(env, sensor_cfg_left, sensor_cfg_right)

    cmd = env.command_manager.get_command(command_name)
    forward_cmd = (cmd[:, 0] > 0.1).float()
    forward_vel = torch.clamp(asset.data.root_link_lin_vel_b[:, 0] - forward_vel_threshold, min=0.0)

    wheel_pos_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    max_wheel_z = torch.max(wheel_pos_z, dim=1).values
    min_wheel_z = torch.min(wheel_pos_z, dim=1).values

    ascent_height = torch.clamp(max_wheel_z - height_threshold, min=0.0)
    both_wheels_up = (min_wheel_z > height_threshold).float()

    normalized_height = torch.tanh(ascent_height / max(height_scale, 1.0e-6))
    normalized_forward = torch.tanh(forward_vel / max(vel_scale, 1.0e-6))
    progress_reward = normalized_height * normalized_forward
    completion_reward = completion_bonus * both_wheels_up
    return stair_mask * forward_cmd * (progress_reward + completion_reward)

def reward_same_foot_z_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize difference in Z positions of two feet in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_apply_inverse(base_quat, foot_base[:,i,:])
    dz = foot_base[:,0,2] - foot_base[:,1,2]
    return torch.square(dz)

# def reward_action_smooth(
#     env
# ) -> torch.Tensor:
#     """
#     Penalize second order action changes.
#     """
#     a = env.action_manager.action
#     pa = env.action_manager.prev_action
#     p2 = env.action_manager.prev2_action
#     return torch.sum((a - 2*pa + p2)**2, dim=1)


def reward_keep_balance(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Constant reward for being alive.
    """
    return torch.ones(env.num_envs, device=env.device)


def _stair_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
) -> torch.Tensor:
    left_mask = env.scene.sensors[sensor_cfg_left.name].data.mask.bool().view(-1)
    right_mask = env.scene.sensors[sensor_cfg_right.name].data.mask.bool().view(-1)
    return torch.logical_or(left_mask, right_mask).float()


def yaw_tracking_error_l2_stair(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    lin_vel_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    stair_mask = _stair_mask(env, sensor_cfg_left, sensor_cfg_right)
    command = env.command_manager.get_command(command_name)
    moving_mask = (torch.abs(command[:, 0]) > lin_vel_threshold).float()
    yaw_error = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    return torch.square(yaw_error) * stair_mask * moving_mask


def yaw_rate_l2_stair(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
    lin_vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize actual yaw rotation on stairs regardless of commanded yaw."""
    asset: RigidObject = env.scene[asset_cfg.name]
    stair_mask = _stair_mask(env, sensor_cfg_left, sensor_cfg_right)
    command = env.command_manager.get_command(command_name)
    moving_mask = (torch.abs(command[:, 0]) > lin_vel_threshold).float()
    return torch.square(asset.data.root_link_ang_vel_b[:, 2]) * stair_mask * moving_mask


def joint_deviation_zero_l1_stair(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    stair_mask = _stair_mask(env, sensor_cfg_left, sensor_cfg_right)
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * stair_mask


def leg_reference_trajectory_stair(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    forward_threshold: float = 0.1,
    amplitude: float = 0.3,
    frequency: float = 1.0,
    phase_offset: float = math.pi,
    temperature: float = 20.0,
    shoulder_amplitude: float | None = None,
    leg_amplitude: float | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    stair_mask = _stair_mask(env, sensor_cfg_left, sensor_cfg_right)
    command = env.command_manager.get_command(command_name)
    moving_mask = (torch.abs(command[:, 0]) > forward_threshold).float()

    shoulder_amp = amplitude if shoulder_amplitude is None else shoulder_amplitude
    leg_amp = (-amplitude) if leg_amplitude is None else leg_amplitude

    phase = env.episode_length_buf.float() * env.step_dt * (2.0 * math.pi * frequency)
    left_profile = 1.0 - torch.cos(phase)
    right_profile = 1.0 - torch.cos(phase + phase_offset)

    joint_count = len(asset_cfg.joint_ids)
    if joint_count == 4:
        targets = torch.stack(
            (
                shoulder_amp * left_profile,
                shoulder_amp * right_profile,
                leg_amp * left_profile,
                leg_amp * right_profile,
            ),
            dim=1,
        )
    elif joint_count == 2:
        targets = torch.stack(
            (
                leg_amp * left_profile,
                leg_amp * right_profile,
            ),
            dim=1,
        )
    else:
        raise ValueError(f"Expected 2 or 4 joints for stair reference tracking, got {joint_count}.")

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    tracking_error = torch.mean(torch.square(joint_pos - targets), dim=1)
    return torch.exp(-temperature * tracking_error) * stair_mask * moving_mask


def wheel_action_zero_event(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    event_command_name: str = "event",
    wheel_action_name: str = "wheel_vel",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    cmd = env.command_manager.get_command(command_name)         # [B, D]
    event_command = env.command_manager.get_command(event_command_name)  # [B, 2]
    
    wheel_action = env.action_manager.get_term(wheel_action_name).processed_actions
    wheel_action_l2 = torch.sum(torch.square(wheel_action), dim=1)  # [B]

    no_cmd_mask = torch.norm(cmd, dim=-1) < 1e-3

    return wheel_action_l2 * no_cmd_mask * event_command[:, 0]