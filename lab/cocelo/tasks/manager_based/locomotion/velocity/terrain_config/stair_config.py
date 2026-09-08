# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import (
    MeshInvertedPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(10.0, 10.0),
    border_width=7.5,
    num_rows=20,
    num_cols=10,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.05, 0.9),
    use_cache=True,
    sub_terrains={ 
        "flat": MeshPlaneTerrainCfg(proportion=0.2),
        "pyramid_stairs_easy": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.025, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_medium": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.1, 0.2),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_hard": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.1, 0.3),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
"""Rough terrains configuration."""
