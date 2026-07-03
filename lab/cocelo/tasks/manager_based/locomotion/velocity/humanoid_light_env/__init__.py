# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
    rough_env,
)

##
# Register Gym environments.
##


#########################################RSL-RL#################################################
################################################################################################
gym.register(
    id="Isaac-Velocity-Flat-Humanoid-Light-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_walk_cfg.HumanoidFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidLightFlatPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Humanoid-Light-v1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_walk_cfg.HumanoidFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidLightFlatPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-Light-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_walk_cfg.HumanoidRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidLightRoughPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-Light-v1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_walk_cfg.HumanoidRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.co_rl_cfg.HumanoidLightRoughPPORunnerCfg_Stand_Walk,
    },
)
