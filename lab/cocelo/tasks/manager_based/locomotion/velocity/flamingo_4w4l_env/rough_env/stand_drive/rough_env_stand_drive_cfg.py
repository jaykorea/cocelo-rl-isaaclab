# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.cocelo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.cocelo.tasks.manager_based.locomotion.velocity.flamingo_4w4l_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from lab.cocelo.assets.flamingo.flamingo_4w4l_v2 import FLAMINGO4W4L_CFG  # isort: skip


@configclass
class Flamingo4w4lRewardsCfg:
    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.1),
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.1),
        },
    )
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    dof_pos_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names="(?!wheel_joint).*",
            ),
        },
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint"],
            ),
        },
    )

    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_joint"],
            ),
        },
    )

    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_leg_joint"],
            ),
        },
    )

    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=".*_joint",
            ),
        },
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "base_link",
                    ".*_shoulder_link",
                    ".*_hip_link",
                    ".*_leg_link",
                ],
            ),
            "threshold": 1.0,
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-25.0,
        params={
            "target_height": 0.47957,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="base_link",
            ),
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

@configclass
class Flamingo4w4lRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Flamingo4w4lRewardsCfg = Flamingo4w4lRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # scene
        self.scene.robot = FLAMINGO4W4L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # events: reset joints
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.5, 0.5)

        # events: push robot
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
            },
        }

        # events: base mass
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # events: physics material
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]

        # events: reset base
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]

        self.terminations.bad_orientation.params["limit_angle"] = 0.8


@configclass
class Flamingo4w4lRoughEnvCfg_PLAY(Flamingo4w4lRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 20.0

        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5

        # spawn the robot randomly in the grid
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # scene
        self.scene.robot = FLAMINGO4W4L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # disable randomization for play
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False

        # play: disable actuator gain randomization
        self.events.randomize_joint_actuator_gains = None

        # play: disable push
        self.events.push_robot = None

        # play: reset joints
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.5, 0.5)

        # play: keep env.yaml-style base mass unless you intentionally want wider randomization
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # play: physics material
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]

        # play: no initial velocity perturbation
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]

        self.terminations.bad_orientation.params["limit_angle"] = 0.8
