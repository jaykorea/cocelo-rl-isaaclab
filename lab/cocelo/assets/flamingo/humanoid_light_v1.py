# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
)
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from lab.cocelo.assets.flamingo import FLAMINGO_ASSETS_DATA_DIR


@configclass
class CoupledDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Delayed PD actuator config that keeps coupled-gear metadata for sim-to-sim exports."""

    gear_ratio_1: float = 1.0
    gear_ratio_2: float = 1.0
    gamma: float = 1.0


ROBSTRIDE_MOTOR_SPECS = {
    "O0": {"rated_voltage": 48.0, "reduction_ratio": 10.0, "torque_constant_nm_per_arms": 1.48},
    "O2": {"rated_voltage": 48.0, "reduction_ratio": 7.75, "torque_constant_nm_per_arms": 1.22},
    "O3": {"rated_voltage": 48.0, "reduction_ratio": 9.0, "torque_constant_nm_per_arms": 2.36},
    "O4": {"rated_voltage": 48.0, "reduction_ratio": 9.0, "torque_constant_nm_per_arms": 2.10},
    "O5": {"rated_voltage": 48.0, "reduction_ratio": 7.75, "torque_constant_nm_per_arms": 0.94}, # 25, 0.5
    "O6": {"rated_voltage": 48.0, "reduction_ratio": 9.0, "torque_constant_nm_per_arms": 1.10},
}


HUMANOID_LIGHT_JOINT_MOTOR_TYPES = {
    "head_joint": "O5",
    "torso_yaw_joint": "O3",
    "torso_roll_joint": "O6",
    "torso_pitch_joint": "O6",
    "_hip_pitch_joint": "O4",
    "_hip_roll_joint": "O3",
    "_hip_yaw_joint": "O3",
    "_knee_joint": "O4",
    "_ankle_pitch_joint": "O0",
    "_ankle_roll_joint": "O0",
    "_shoulder_pitch_joint": "O3", 
    "_shoulder_roll_joint": "O3", 
    "_shoulder_yaw_joint": "O2",
    "_elbow_joint": "O6", 
    "_wrist_joint": "O0", 
}


def get_humanoid_light_joint_motor_type(joint_name: str) -> str:
    if joint_name in HUMANOID_LIGHT_JOINT_MOTOR_TYPES:
        return HUMANOID_LIGHT_JOINT_MOTOR_TYPES[joint_name]
    for joint_suffix, motor_type in HUMANOID_LIGHT_JOINT_MOTOR_TYPES.items():
        if joint_suffix.startswith("_") and joint_name.endswith(joint_suffix):
            return motor_type
    raise KeyError(f"No motor type configured for joint: {joint_name}")


HUMANOID_LIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{FLAMINGO_ASSETS_DATA_DIR}/Robots/Flamingo/humanoid_light_rev_1_0_10_45/humanoid_light_rev_1_0_10_45.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # head
            "head_joint": 0.0,

            # legs
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,

            # body
            "torso_yaw_joint": 0.0,
            "torso_roll_joint": 0.0,
            "torso_pitch_joint": 0.0,

            # arms
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
            ".*_wrist_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # -----------------------------
        # HEAD
        # -----------------------------
        "heads": DelayedPDActuatorCfg(
            joint_names_expr=[
                "head_joint",
            ],
            effort_limit={
                "head_joint": 5.5,
            },
            velocity_limit={
                "head_joint": 20.0,
            },
            stiffness={
                "head_joint": 20.0,
            },
            damping={
                "head_joint": 0.25,
            },
            armature={
                "head_joint": 0.01,
            },
        ),

        # -----------------------------
        # LEGS: hips + knees
        # -----------------------------
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                "left_hip_.*_joint",
                "right_hip_.*_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                # left leg
                "left_hip_pitch_joint": 120.0,
                "left_hip_roll_joint": 60.0,
                "left_hip_yaw_joint": 60.0,
                "left_knee_joint": 120.0,

                # right leg
                "right_hip_pitch_joint": 120.0,
                "right_hip_roll_joint": 60.0,
                "right_hip_yaw_joint": 60.0,
                "right_knee_joint": 120.0,
            },
            velocity_limit={
                # left leg
                "left_hip_pitch_joint": 80.0,
                "left_hip_roll_joint": 20.0,
                "left_hip_yaw_joint": 20.0,
                "left_knee_joint": 80.0,

                # right leg
                "right_hip_pitch_joint": 80.0,
                "right_hip_roll_joint": 20.0,
                "right_hip_yaw_joint": 20.0,
                "right_knee_joint": 80.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_yaw_joint": 100.0,
                ".*_knee_joint": 100.0,
            },
            damping={
                ".*_hip_pitch_joint": 1.0,
                ".*_hip_roll_joint": 1.0,
                ".*_hip_yaw_joint": 1.0,
                ".*_knee_joint": 1.0,
            },
            armature={
                ".*_hip_.*_joint": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),

        # -----------------------------
        # ANKLES
        # -----------------------------
        "ankles": CoupledDelayedPDActuatorCfg(
            joint_names_expr=[
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],

            # external gear ratio
            gear_ratio_1=-2.0,
            gear_ratio_2=-2.0,

            gamma=1.0,

            effort_limit={
                "left_ankle_pitch_joint": 14.0,
                "left_ankle_roll_joint": 14.0,
                "right_ankle_pitch_joint": 14.0,
                "right_ankle_roll_joint": 14.0,
            },
            velocity_limit={
                "left_ankle_pitch_joint": 36.0,
                "left_ankle_roll_joint": 36.0,
                "right_ankle_pitch_joint": 36.0,
                "right_ankle_roll_joint": 36.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.25,
                ".*_ankle_roll_joint": 0.25,
            },
            armature={
                ".*_ankle_.*_joint": 0.01,
            },
        ),

        # -----------------------------
        # BODY: torso yaw
        # -----------------------------
        "body": DelayedPDActuatorCfg(
            joint_names_expr=[
                "torso_yaw_joint",
            ],
            effort_limit={
                "torso_yaw_joint": 60.0,
            },
            velocity_limit={
                "torso_yaw_joint": 20.0,
            },
            stiffness={
                "torso_yaw_joint": 100.0,
            },
            damping={
                "torso_yaw_joint": 1.0,
            },
            armature={
                "torso_yaw_joint": 0.01,
            },
        ),

        # -----------------------------
        # BODY: torso pitch + roll
        # -----------------------------
        "torso_pitch_roll": CoupledDelayedPDActuatorCfg(
            joint_names_expr=[
                "torso_pitch_joint",
                "torso_roll_joint",
            ],

            gear_ratio_1=1.0,
            gear_ratio_2=1.0,

            gamma=1.0,

            effort_limit={
                "torso_pitch_joint": 36.0,
                "torso_roll_joint": 36.0,
            },
            velocity_limit={
                "torso_pitch_joint": 53.0,
                "torso_roll_joint": 53.0,
            },
            stiffness={
                "torso_pitch_joint": 100.0,
                "torso_roll_joint": 100.0,
            },
            damping={
                "torso_pitch_joint": 1.0,
                "torso_roll_joint": 1.0,
            },
            armature={
                "torso_pitch_joint": 0.01,
                "torso_roll_joint": 0.01,
            },
        ),

        # -----------------------------
        # ARMS
        # -----------------------------
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_joint",
            ],
            effort_limit={
                # shoulders
                "left_shoulder_pitch_joint": 60.0,
                "left_shoulder_roll_joint": 60.0,
                "left_shoulder_yaw_joint": 17.0,
                "right_shoulder_pitch_joint": 60.0,
                "right_shoulder_roll_joint": 60.0,
                "right_shoulder_yaw_joint": 17.0,

                # elbows
                "left_elbow_joint": 36.0,
                "right_elbow_joint": 36.0,

                # wrists
                "left_wrist_joint": 14.0,
                "right_wrist_joint": 14.0,
            },
            velocity_limit={
                # shoulders
                "left_shoulder_pitch_joint": 20.0,
                "left_shoulder_roll_joint": 20.0,
                "left_shoulder_yaw_joint": 40.0,
                "right_shoulder_pitch_joint": 20.0,
                "right_shoulder_roll_joint": 20.0,
                "right_shoulder_yaw_joint": 40.0,

                # elbows
                "left_elbow_joint": 53.0,
                "right_elbow_joint": 53.0,

                # wrists
                "left_wrist_joint": 36.0,
                "right_wrist_joint": 36.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 1.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 1.0,
                ".*_elbow_joint": 1.0,
                ".*_wrist_joint": 1.0,
            },
            armature={
                ".*_shoulder_.*_joint": 0.01,
                ".*_elbow_joint": 0.01,
                ".*_wrist_joint": 0.01,
            },
        ),
    },
)
