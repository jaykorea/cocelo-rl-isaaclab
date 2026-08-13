# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.humanoid_light_env.rough_env.stand_walk.drive_rewards as mdp_walk
from lab.flamingo.tasks.manager_based.locomotion.velocity.humanoid_light_env.velocity_env_cfg import (
    CurriculumCfg,
    LocomotionVelocityRoughEnvCfg,
)
from lab.flamingo.assets.flamingo.humanoid_light_rev1_0_2 import HUMANOID_LIGHT_CFG  # isort: skip


# -----------------------------------------------------------------------------
# Robot-specific naming (URDF)
# -----------------------------------------------------------------------------
FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
BASE_CONTACT_BODY_NAMES = ["pelvis_link", "torso_roll_link"]
BASE_MASS_BODY_NAMES = ["pelvis_link"]


@configclass
class HumanoidRewardsCfg:
    # -- task tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp_walk.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp_walk.track_ang_vel_z_world_exp,
    #     weight=0.5,
    #     params={"command_name": "base_velocity", "std": 0.25},
    # )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    keep_balance = RewTerm(func=mdp_walk.reward_keep_balance, weight=1.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # legs
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                    # torso
                    "torso_.*_joint",
                ],
            )
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_joint",  
                ],
            )
        },
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_.*_joint", "head_joint"])},
    )

    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])},
    )

    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "torso_.*_joint",
                ],
            )
        },
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"])},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.075)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "threshold": 0.6,
        },
    )


@configclass
class HumanoidFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HumanoidRewardsCfg = HumanoidRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = HUMANOID_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---------------- Observations (play-style: no corruption, no height scanner) ----------------
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        self.scene.height_scanner = None
        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_policy.height_scan = None

        # ---------------- Events ----------------
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)

        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_MASS_BODY_NAMES
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 3.0)

        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        #! ****************** Terrain setup **************** !#
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        
        # ---------------- Commands ----------------
        self.commands.base_velocity.resampling_time_range = (3.0, 13.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # ---------------- Terminations ----------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_CONTACT_BODY_NAMES


@configclass
class HumanoidFlatEnvCfg_PLAY(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True

        self.scene.robot = HUMANOID_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Observations (no corruption)
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        #! ****************** Terrain setup **************** !#
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_CONTACT_BODY_NAMES

        self.commands.base_velocity.resampling_time_range = (3.0, 13.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)