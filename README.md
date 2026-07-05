<p align="center">
  <img src="assets/readme/f4-light.webp" alt="FlaminGO Light in motion" width="100%">
</p>

<p align="center">
  <img src="assets/readme/robots.jpg" alt="COCELO robot platforms" width="100%">
</p>

<h1 align="center">Physical AI for Legged-Wheeled Robots</h1>

<p align="center">
  <strong>Intelligence in Motion, New Possibility.</strong><br>
  Reinforcement learning environments for COCELO robots in Isaac Lab.
</p>

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/Lab-2.0.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Physical AI Robot Learning
This repository develops reinforcement learning environments for legged-wheeled robots, focusing on embodied
intelligence, locomotion control, and scalable robot policy training in Isaac Lab.

## Setup
- This repo is tested on Ubuntu 20.04, and I recommend you to install 'local install'
### 1. Install Isaac Sim
  ```
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html
  ```
### 2. Install Isaac Lab
  ```
  https://github.com/isaac-sim/IsaacLab
  ```

### 3. Install lab.cocelo package
i. clone repository
   ```
   git clone https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot
   ```
ii. install lab.cocelo pip package by running below command
   - run it on 'lab.cocelo' root path
   ```
   conda activate env_isaaclab # change to you conda env
   pip install -e .
   ```
iii. Unzip assets(usd asset) on folder
   - Since git does not correctly upload '.usd' file, you should manually unzip the usd files on assests folder
   ```
    path example: lab/cocelo/assets/data/Robots/Flamingo/flamingo_light_v01_2_2/assets.zip
   ```

## Launch script
  - run it on 'lab.cocelo' root path

### Train
#### Flamingo Light
  ```
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Flamingo-Light-v1 --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
  ```

#### Flamingo Pro
  ```
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Flamingo-Pro-v3 --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
  ```

#### Flamingo 4W4L
  ```
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Flamingo4w4l-v1 --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
  ```

#### Humanoid Light
  ```
    python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Humanoid-Light-v1 --num_envs 4096 --headless --num_policy_stacks 2 --num_critic_stacks 2
  ```

### Play
#### Flamingo Light
  ```
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Flamingo-Light-Play-v1 --num_envs 64 --num_policy_stacks 2 --num_critic_stacks 2 --load_run {folder name}
  ```

#### Flamingo Pro
  ```
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Flamingo-Pro-Play-v3 --num_envs 64 --num_policy_stacks 2 --num_critic_stacks 2 --load_run {folder name}
  ```

#### Flamingo 4W4L
  ```
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Flamingo4w4l-v1-Play --num_envs 64 --num_policy_stacks 2 --num_critic_stacks 2 --load_run {folder name}
  ```

#### Humanoid Light
  ```
    python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Humanoid-Light-v1-Play --num_envs 64 --num_policy_stacks 2 --num_critic_stacks 2 --load_run {folder name}
  ```
