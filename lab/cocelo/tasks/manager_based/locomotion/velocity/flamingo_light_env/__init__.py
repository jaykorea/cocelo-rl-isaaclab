# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import (
    agents,
    flat_env,
)

#########################################RSL_RL##########################################################
#######################################################################################################

###########################################Track Velocity##############################################
gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FlamingoLightFlatPPORunnerCfg_Stand_Drive,
    },
)
gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Light-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.FlamingoLightFlatPPORunnerCfg_Stand_Drive,
    },
)
###########################################Track Velocity##############################################


######################################################################################################
############################################RSL_RL######################################################