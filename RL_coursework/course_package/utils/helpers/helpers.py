"""Common helper functions and classes for RL algorithms."""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from course_package import MARLEnv
from course_package.policies.rl.networks import PolicyNetwork


@dataclass
class BaseAlgorithmConfig:
    """Base class for algorithm configuration."""

    gamma: float = 0.99
    max_steps: int = 50_000
    max_time: float = 5.0 * 60.0
    num_eval_episodes: int = 3
    group_identifier: str = "GXXX"


def choose_action(
    state: NDArray, policy_net: PolicyNetwork
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Sample actions from a diagonal Gaussian policy.

    Works for a single observation ``(obs_dim,)`` or a batch with one row per
    agent ``(num_agents, obs_dim)``. Log-probabilities and entropies are summed
    over the action dimension only, so each agent keeps its own value.

    Parameters
    ----------
    state : NDArray
        The current observation(s).
    policy_net : PolicyNetwork
        The policy network that outputs the mean and log standard deviation of the action distribution.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Sampled actions ``(..., action_dim)``, log-probabilities ``(...)`` and entropies ``(...)``.
    """
    state_t = torch.from_numpy(state).float()
    mean, log_std = policy_net(state_t)
    log_std = torch.clamp(log_std, min=-0.0)
    std = torch.exp(log_std)

    dist = torch.distributions.Normal(mean, std)
    action = dist.sample()
    logprob = dist.log_prob(action).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)

    return action, logprob, entropy


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
    policy: PolicyNetwork,
    num_episodes: int = 10,
    seed: int = 1234,
) -> NDArray:
    """
    Evaluate a shared policy (used by every agent) in the environment.

    Parameters
    ----------
    env : MARLEnv
        The multi-agent reinforcement learning environment.
    policy : PolicyNetwork
        The single policy network shared by all agents.
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


def make_shared_policy(obs_dim: int, action_dim: int) -> PolicyNetwork:
    """Create the single policy network shared by all agents."""
    return PolicyNetwork(obs_dim, action_dim)


def _rollout(env: MARLEnv, policy: PolicyNetwork, seed: int | None = None) -> NDArray:
    """
    Rollout a single episode with one shared policy acting for every agent.

    Parameters
    ----------
    env : MARLEnv
        The multi-agent reinforcement learning environment.
    policy : PolicyNetwork
        The shared policy network.
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
        action, _, _ = choose_action(obs, policy)
        actions = np.clip(
            action.detach().numpy(), env.action_space.low, env.action_space.high
        )
        (obs, rewards, done, _, _) = env.step(actions)
        episode_returns += rewards
    return episode_returns
