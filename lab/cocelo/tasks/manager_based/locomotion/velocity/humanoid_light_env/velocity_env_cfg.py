# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import lab.cocelo.tasks.manager_based.locomotion.velocity.mdp as mdp

# -----------------------------------------------------------------------------
# Sim-to-sim contract
# -----------------------------------------------------------------------------

BASE_LINK_NAME = "pelvis_link"
TORSO_LINK_NAME = "torso_roll_link"
FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
BASE_CONTACT_BODY_NAMES = [BASE_LINK_NAME, TORSO_LINK_NAME]

# This is the joint order currently expected by envs/humanoid_light_v1:
# dof_pos, dof_vel, raw action, policy action, and PD target all use this order.
SIM2SIM_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_hip_roll_joint",
    "left_ankle_pitch_joint",
    "left_hip_yaw_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_hip_roll_joint",
    "right_ankle_pitch_joint",
    "right_hip_yaw_joint",
    "right_ankle_roll_joint",
                                                            
    "torso_yaw_joint",
    "head_joint",
    "torso_pitch_joint",
    "torso_roll_joint",

    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_joint",
    "right_wrist_joint",
]

SIM2SIM_JOINTS_CFG = SceneEntityCfg(
    "robot",
    joint_names=SIM2SIM_JOINT_NAMES,
    preserve_order=True,
)

COUPLED_MOTOR_OBS_PAIRS = (
    ("left_ankle_pitch_joint", "left_ankle_roll_joint", -2.0, -2.0, "roll_sum"),
    ("right_ankle_pitch_joint", "right_ankle_roll_joint", -2.0, -2.0, "roll_sum"),
    ("torso_pitch_joint", "torso_roll_joint", 1.0, 1.0, "roll_sum"),
)

COUPLED_MOTOR_OBS_PARAMS = {
    "asset_cfg": SIM2SIM_JOINTS_CFG,
    "joint_names": SIM2SIM_JOINT_NAMES,
    "coupled_pairs": COUPLED_MOTOR_OBS_PAIRS,
}

# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene definition fixed for the current MuJoCo sim-to-sim contract."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    # Current sim-to-sim policy input has no height map / height scan.
    height_scanner = None

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=4000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.53, 0.81, 0.98), intensity=1500.0),
    )


# -----------------------------------------------------------------------------
# MDP settings
# -----------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Command specification matching config/env_table.yaml humanoid_light_v1."""

    base_velocity = mdp.UniformVelocityWithZCommandCfg(
        asset_name="robot",
        resampling_time_range=(9.0, 13.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityWithZCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.3, 0.3),
            pos_z=(0.0, 0.0),
        ),
        initial_phase_time=2.0,
    )


@configclass
class ActionsCfg:
    """Action order and scale fixed to the MuJoCo sim-to-sim implementation."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=SIM2SIM_JOINT_NAMES,
        scale=0.5,
        offset=0.0,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class StackPolicyCfg(ObsGroup):
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos,
        #     params={"asset_cfg": SIM2SIM_JOINTS_CFG},
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        #     scale=1.0,
        # )
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel,
        #     params={"asset_cfg": SIM2SIM_JOINTS_CFG},
        #     noise=Unoise(n_min=-1.5, n_max=1.5),
        #     scale=0.15,
        # )
        joint_pos = ObsTerm(func=mdp.coupled_joint_pos_motor_space, params=COUPLED_MOTOR_OBS_PARAMS, noise=Unoise(n_min=-0.05, n_max=0.05), scale=1.0)
        joint_vel = ObsTerm(func=mdp.coupled_joint_vel_motor_space, params=COUPLED_MOTOR_OBS_PARAMS, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)

        lower_ang_vel = ObsTerm(
            func=mdp.body_ang_vel_link,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="lower_imu")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
            scale=0.25,
        )
        upper_ang_vel = ObsTerm(
            func=mdp.body_ang_vel_link,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="upper_imu"),},
            noise=Unoise(n_min=-0.15, n_max=0.15),
            scale=0.25,
        )
        lower_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="lower_imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        upper_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="upper_imu"),},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoneStackPolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_scaled_commands,
            params={"command_name": "base_velocity", "scale": (2.0, 2.0, 1.0)},
        )

        # Force-disable height scan. Current sim-to-sim policy input has no height map.
        height_scan = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StackCriticCfg(ObsGroup):
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos,
        #     params={"asset_cfg": SIM2SIM_JOINTS_CFG},
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        #     scale=1.0,
        # )
        # joint_vel = ObsTerm(
        #     func=mdp.joint_vel,
        #     params={"asset_cfg": SIM2SIM_JOINTS_CFG},
        #     noise=Unoise(n_min=-1.5, n_max=1.5),
        #     scale=0.15,
        # )
        joint_pos = ObsTerm(func=mdp.coupled_joint_pos_motor_space, params=COUPLED_MOTOR_OBS_PARAMS, scale=1.0)
        joint_vel = ObsTerm(func=mdp.coupled_joint_vel_motor_space, params=COUPLED_MOTOR_OBS_PARAMS, scale=0.15)
        lower_ang_vel = ObsTerm(
            func=mdp.body_ang_vel_link,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="lower_imu")},
            scale=0.25,
        )
        upper_ang_vel = ObsTerm(
            func=mdp.body_ang_vel_link,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="upper_imu"),},
            scale=0.25,
        )
        lower_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="lower_imu")},
        )
        upper_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="upper_imu"),},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class NoneStackCriticCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_scaled_commands,
            params={"command_name": "base_velocity", "scale": (2.0, 2.0, 1.0)},
        )
        height_scan = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class InfoCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SIM2SIM_JOINTS_CFG}, scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SIM2SIM_JOINTS_CFG}, scale=1.0)
        joint_torque = ObsTerm(func=mdp.joint_torques, params={"asset_cfg": SIM2SIM_JOINTS_CFG}, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_link, scale=1.0)
        lower_ang_vel = ObsTerm(
            func=mdp.body_ang_vel_link,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="lower_imu")},
            scale=0.25,
        )
        upper_ang_vel = ObsTerm(
            func=mdp.body_ang_vel_link,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="upper_imu")},
            scale=0.25,
        )
        lower_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="lower_imu")},
        )
        upper_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="upper_imu")},
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_scaled_commands,
            params={"command_name": "base_velocity", "scale": (1.0, 1.0, 1.0)},
        )
        base_lin_vel_z = ObsTerm(func=mdp.base_lin_vel_z_link, scale=1.0)
        base_lin_vel_y = ObsTerm(func=mdp.base_lin_vel_y_link, scale=1.0)
        base_lin_vel_x = ObsTerm(func=mdp.base_lin_vel_x_link, scale=1.0)

        def __post_init__(self):                                                                                                                                                                        
            self.enable_corruption = False
            self.concatenate_terms = True

    stack_policy: StackPolicyCfg = StackPolicyCfg()
    none_stack_policy: NoneStackPolicyCfg = NoneStackPolicyCfg()
    stack_critic: StackCriticCfg = StackCriticCfg()
    none_stack_critic: NoneStackCriticCfg = NoneStackCriticCfg()
    obs_info: InfoCfg = InfoCfg()


@configclass
class EventCfg:
    """Domain randomization events kept as-is for robust training."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_joint_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_com_positions,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "com_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
            },
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_CONTACT_BODY_NAMES),
            "threshold": 1.0,
        },
    )

    # Flat sim-to-sim has no rough terrain boundary. Kept disabled for both cfgs below.
    terrain_out_of_bounds = None


@configclass
class CurriculumCfg:
    """Curriculum disabled to keep the exported policy contract fixed."""

    terrain_levels = None


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Rough class kept for import compatibility, but fixed to sim-to-sim observations."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        self.observations.none_stack_policy.height_scan = None
        self.observations.none_stack_critic.height_scan = None

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class LocomotionVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Flat environment fixed to the current MuJoCo sim-to-sim contract."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        self.observations.none_stack_policy.height_scan = None
        self.observations.none_stack_critic.height_scan = None

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
