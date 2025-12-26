from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from .utils.state_handler import StateHandler


class RslRlVecEnvWrapperWithStateHandler(RslRlVecEnvWrapper):

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        *,
        num_policy_stacks: int,
        num_critic_stacks: int,
        clip_actions: float | None = None,
        update_obs_dims_in_manager: bool = True,
    ):
        # 1) base wrapper가 __init__에서 "policy"를 읽기 전에, registry 정보를 만들어 줘야 함
        if hasattr(env.unwrapped, "observation_manager"):
            gdim = env.unwrapped.observation_manager.group_obs_dim

            # policy를 만들 수 있는 조건일 때만 생성
            if "policy" not in gdim and "stack_policy" in gdim and "none_stack_policy" in gdim:
                stack_policy_dim = int(gdim["stack_policy"][0])
                nonstack_policy_dim = int(gdim["none_stack_policy"][0])
                tmp_policy = StateHandler(num_policy_stacks + 1, stack_policy_dim, nonstack_policy_dim)
                gdim["policy"] = (tmp_policy.num_obs,)

            # critic도 동일
            if "critic" not in gdim and "stack_critic" in gdim and "none_stack_critic" in gdim:
                stack_critic_dim = int(gdim["stack_critic"][0])
                nonstack_critic_dim = int(gdim["none_stack_critic"][0])
                tmp_critic = StateHandler(num_critic_stacks + 1, stack_critic_dim, nonstack_critic_dim)
                gdim["critic"] = (tmp_critic.num_obs,)

        # 2) 이제 base wrapper init 호출 (여기서 policy KeyError가 나면 안 됨)
        super().__init__(env, clip_actions=clip_actions)

        # 3) 실제 handler 생성
        self.policy_state_handler = None
        self.critic_state_handler = None

        if hasattr(self.unwrapped, "observation_manager"):
            gdim = self.unwrapped.observation_manager.group_obs_dim

            if "stack_policy" in gdim and "none_stack_policy" in gdim:
                stack_policy_dim = int(gdim["stack_policy"][0])
                nonstack_policy_dim = int(gdim["none_stack_policy"][0])
                self.policy_state_handler = StateHandler(
                    num_policy_stacks + 1, stack_policy_dim, nonstack_policy_dim
                )
                self.num_obs = self.policy_state_handler.num_obs

            if "stack_critic" in gdim and "none_stack_critic" in gdim:
                stack_critic_dim = int(gdim["stack_critic"][0])
                nonstack_critic_dim = int(gdim["none_stack_critic"][0])
                self.critic_state_handler = StateHandler(
                    num_critic_stacks + 1, stack_critic_dim, nonstack_critic_dim
                )
                self.num_privileged_obs = self.critic_state_handler.num_obs

            # base wrapper가 이미 읽은 값과 일관되게 유지하고 싶으면 켬
            if update_obs_dims_in_manager:
                if self.policy_state_handler is not None:
                    self.unwrapped.observation_manager.group_obs_dim["policy"] = (self.num_obs,)
                if self.critic_state_handler is not None:
                    self.unwrapped.observation_manager.group_obs_dim["critic"] = (self.num_privileged_obs,)

        # base wrapper와 동일하게 초기 reset
        self.env.reset()

    def get_observations(self):
        # base wrapper와 동일한 경로로 obs_dict 확보
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        # policy 합성
        if self.policy_state_handler is not None and "stack_policy" in obs_dict and "none_stack_policy" in obs_dict:
            if self.policy_state_handler.stack_buffer is None:
                obs_dict["policy"] = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
            else:
                obs_dict["policy"] = self.policy_state_handler.update(obs_dict["stack_policy"], obs_dict["none_stack_policy"])

        # critic 합성
        if self.critic_state_handler is not None and "stack_critic" in obs_dict and "none_stack_critic" in obs_dict:
            if self.critic_state_handler.stack_buffer is None:
                obs_dict["critic"] = self.critic_state_handler.reset(obs_dict["stack_critic"], obs_dict["none_stack_critic"])
            else:
                obs_dict["critic"] = self.critic_state_handler.update(obs_dict["stack_critic"], obs_dict["none_stack_critic"])

        return obs_dict["policy"], {"observations": obs_dict}

    def reset(self):
        obs_dict, _ = self.env.reset()

        if self.policy_state_handler is not None and "stack_policy" in obs_dict and "none_stack_policy" in obs_dict:
            obs_dict["policy"] = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])

        if self.critic_state_handler is not None and "stack_critic" in obs_dict and "none_stack_critic" in obs_dict:
            obs_dict["critic"] = self.critic_state_handler.reset(obs_dict["stack_critic"], obs_dict["none_stack_critic"])

        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor):
        # base wrapper의 contract 유지 (clip + done long)
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        if self.policy_state_handler is not None and "stack_policy" in obs_dict and "none_stack_policy" in obs_dict:
            obs_dict["policy"] = self.policy_state_handler.update(obs_dict["stack_policy"], obs_dict["none_stack_policy"])

        if self.critic_state_handler is not None and "stack_critic" in obs_dict and "none_stack_critic" in obs_dict:
            obs_dict["critic"] = self.critic_state_handler.update(obs_dict["stack_critic"], obs_dict["none_stack_critic"])

        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return obs_dict["policy"], rew, dones, extras
