from typing import Any

import numpy as np
from gymnasium import Wrapper

from course_package.environment.environment import EnergySystem


class MARLEnv(Wrapper):
    def __init__(self, env_config: dict[str, Any]):
        super().__init__(EnergySystem(env_config))
        self.env: EnergySystem
        self.num_agents = self.env.num_agents
        self.step_count = 0
        self._obs_keys = tuple(self.env.active_observations)

        self.obs_low = self.env.observation_space.low.astype(np.float32, copy=False)
        self.obs_high = self.env.observation_space.high.astype(np.float32, copy=False)
        self._obs_scale = 2.0 / (self.obs_high - self.obs_low + 1e-8)

    def _vectorize_obs(self, obs_dict):
        if isinstance(obs_dict, np.ndarray):
            return obs_dict.astype(np.float32, copy=False)

        out = np.empty_like(self.obs_low, dtype=np.float32)
        col = 0

        for key in self._obs_keys:
            value = np.asarray(obs_dict[key], dtype=np.float32)

            if value.ndim == 0 or value.ndim == 1:
                out[:, col] = value
                col += 1
            else:
                width = int(np.prod(value.shape[1:]))
                out[:, col : col + width] = value.reshape(self.num_agents, width)
                col += width

        return out

    def _normalize(self, obs):
        return np.clip((obs - self.obs_low) * self._obs_scale - 1.0, -1.0, 1.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._normalize(self._vectorize_obs(obs)), info

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.env.action_space.shape:
            raise ValueError(
                f"actions must have shape {self.env.action_space.shape}, got {actions.shape}"
            )
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = np.clip(
            actions, self.env.action_space.low, self.env.action_space.high
        )

        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.step_count += 1
        return (
            self._normalize(self._vectorize_obs(obs)),
            rewards,
            terminated,
            truncated,
            info,
        )

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "env"), name)
