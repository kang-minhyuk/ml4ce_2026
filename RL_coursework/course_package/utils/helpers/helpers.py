"""Common helper functions and classes for RL algorithms."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from course_package import Expert, MARLEnv
from course_package.policies.expert import PIDParams, initialize_mpc_param


@dataclass
class BaseAlgorithmConfig:
    """Base class for algorithm configuration."""

    gamma: float = 0.99
    max_steps: int = 50_000
    max_time: float = 5.0 * 60.0
    num_eval_episodes: int = 3
    group_identifier: str = "GXXX"


def compute_returns(rewards: NDArray, gamma: float = 1.0) -> NDArray:
    """
    Compute discounted returns G_t = r_t + γ r_{t+1} + ...

    Parameters
    ----------
    rewards : NDArray
        Array of rewards for each timestep in the episode.
    gamma : float, optional
        Discount factor for future rewards, by default 1.0 (undiscounted).
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    discounted_return = 0.0
    for t in reversed(range(len(rewards))):
        discounted_return = rewards[t] + gamma * discounted_return
        returns[t] = discounted_return
    return returns


def evaluate_marl_policy(
    env: MARLEnv,
    policy,
    num_episodes: int = 10,
    seed: int = 1234,
) -> NDArray:
    """
    Evaluate a shared policy (used by every agent) in the environment.

    The policy's deterministic ``act`` is applied to each agent's local
    observation separately, as in grading.

    Parameters
    ----------
    env : MARLEnv
        The multi-agent reinforcement learning environment.
    policy
        The single policy shared by all agents. It must define
        ``act(local_obs) -> action`` mapping one agent's observation
        ``(obs_dim,)`` to that agent's action ``(action_dim,)``.
    num_episodes : int, optional
        The number of episodes to evaluate the policy, by default 10.
    seed : int, optional
        The random seed for reproducibility, by default 1234.

    Returns
    -------
    NDArray
        Array of shape (num_episodes, env.num_agents)of episode returns.
    """
    returns_array = np.zeros((num_episodes, env.num_agents), dtype=np.float32)
    for i in range(num_episodes):
        episode_returns = _rollout(env, policy, seed=seed + i)
        returns_array[i, :] = episode_returns
    return returns_array


def make_expert(env: MARLEnv, gain_dir: str | Path | None = None) -> Expert:
    """Create the model-based expert (battery MPC + thermal PID), one MPC per agent."""
    mpc_params = [
        initialize_mpc_param(env.env, agent_idx=i) for i in range(env.num_agents)
    ]
    return Expert(mpc_params, PIDParams(), gain_dir=gain_dir)


def evaluate_expert(
    env: MARLEnv,
    expert: Expert,
    num_episodes: int = 10,
    seed: int = 1234,
) -> NDArray:
    """
    Evaluate the expert controller on the same episodes as ``evaluate_marl_policy``.

    Returns
    -------
    NDArray
        Array of shape (num_episodes, env.num_agents) of episode returns.
    """
    returns_array = np.zeros((num_episodes, env.num_agents), dtype=np.float32)
    for i in range(num_episodes):
        env.reset(seed=seed + i)
        expert.reset()
        done = False
        while not done:
            _, rewards, done, _, _ = env.step(expert(env))
            returns_array[i, :] += rewards
    return returns_array


def _shared_policy_actions(env: MARLEnv, policy, obs: NDArray) -> NDArray:
    """
    Apply the shared policy to each agent's local observation, one agent at a time.

    Parameters
    ----------
    env : MARLEnv
        The multi-agent reinforcement learning environment.
    policy
        The single policy shared by all agents (see ``evaluate_marl_policy``).
    obs : NDArray
        Observations ``(num_agents, obs_dim)``; row ``i`` is agent ``i``'s local observation.

    Returns
    -------
    NDArray
        Clipped actions ``(num_agents, action_dim)``; row ``i`` is agent ``i``'s action.
    """
    actions = np.stack(
        [np.asarray(policy.act(local_obs), dtype=np.float32) for local_obs in obs]
    )
    if actions.shape != env.action_space.shape:
        raise ValueError(
            f"policy.act must return one action of shape {env.action_space.shape[1:]} "
            f"per agent, got {actions.shape[1:]}"
        )
    # MARLEnv maps NaN actions to 0, so they would otherwise go unnoticed.
    if not np.all(np.isfinite(actions)):
        raise ValueError("policy.act returned non-finite actions")
    return np.clip(actions, env.action_space.low, env.action_space.high)


def _rollout(env: MARLEnv, policy, seed: int | None = None) -> NDArray:
    """
    Rollout a single episode with one shared policy acting for every agent.

    Parameters
    ----------
    env : MARLEnv
        The multi-agent reinforcement learning environment.
    policy
        The single policy shared by all agents (see ``evaluate_marl_policy``).
    seed : int, optional
        Random seed for reproducibility. If None, no seed is set.

    Returns
    -------
    NDArray
        The episode return (sum of rewards) for each agent.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    episode_returns = np.zeros(env.num_agents, dtype=np.float32)
    while not done:
        actions = _shared_policy_actions(env, policy, obs)
        (obs, rewards, done, _, _) = env.step(actions)
        episode_returns += rewards
    return episode_returns
