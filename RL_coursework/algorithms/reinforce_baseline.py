"""REINFORCE with a value-function baseline for a single shared policy."""

from copy import deepcopy
from dataclasses import dataclass
from time import time

import numpy as np
import torch
from tqdm import tqdm

from course_package import MARLEnv
from course_package.utils.helpers import (
    BaseAlgorithmConfig,
    CSVLogger,
    compute_returns,
    evaluate_marl_policy,
    save_training_plots,
    set_seed,
    setup_model_saving,
)


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """Hyperparameters for REINFORCE with a learned state-value baseline."""

    entropy_weight: float = 0.01
    gradient_clip: float = 1.0
    lr_policy: float = 1e-3
    lr_value: float = 1e-3
    train_batch_size: int = 1176
    group_identifier: str = "REINFORCE"


class PolicyNetwork(torch.nn.Module):
    """
    Single policy shared by all agents, with Gaussian outputs.

    It maps ONE agent's local observation to that agent's action; the same
    weights are used for every agent. The layers act row-wise, so a batch of
    local observations (one row per agent) is processed with the same weights.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, action_dim * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        return mean, log_std

    @torch.no_grad()
    def act(self, local_obs: np.ndarray) -> np.ndarray:
        """Deterministic (mean) action ``(action_dim,)`` for one agent's observation ``(obs_dim,)``."""
        mean, _ = self(torch.as_tensor(local_obs, dtype=torch.float32))
        return mean.numpy()


class ValueNetwork(torch.nn.Module):
    """State-value baseline evaluated on one agent's local observation."""

    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def choose_action(
    state: np.ndarray, policy_net: PolicyNetwork
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample actions from the diagonal Gaussian shared policy.

    Works for a single local observation ``(obs_dim,)`` or a batch with one row
    per agent ``(num_agents, obs_dim)``. Each agent gets an independent sample
    from the same policy. Log-probabilities and entropies are summed over the
    action dimension only, so each agent keeps its own value.

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


def train(
    env: MARLEnv,
    config: AlgorithmConfig,
    seed: int = 0,
) -> tuple[PolicyNetwork, dict[str, str]]:
    """
    Train ONE shared policy (used by every agent) with REINFORCE + baseline.

    MARLEnv exposes observations with shape ``(num_agents, obs_dim)``, where row
    ``i`` is agent ``i``'s local observation. During training the shared policy
    processes all rows in a single batched forward pass (same weights for every
    row), and the experience of every agent is pooled into one training batch
    for a single gradient update per iteration.
    """
    set_seed(seed)
    agent_policy = PolicyNetwork(env.observation_space.shape[1], env.action_space.shape[1])
    pbar = tqdm(
        desc=f"Running {config.group_identifier} RL training loop",
        total=config.max_steps,
    )
    save_folder = setup_model_saving(config.group_identifier)

    value_net = ValueNetwork(env.observation_space.shape[1])
    policy_opt = torch.optim.Adam(agent_policy.parameters(), lr=config.lr_policy)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=config.lr_value)

    save_paths = {"shared_policy": save_folder / "shared_policy.pt"}
    logger = CSVLogger(save_folder / "shared_policy.csv")

    best_eval_return = -np.inf
    best_state_dict = deepcopy(agent_policy.state_dict())

    start_time = time()
    step_count = 0
    batch_idx = 0

    def budget_remaining() -> bool:
        pbar.update(step_count - pbar.n)
        return (
            step_count < config.max_steps and time() - start_time < config.max_time
        )

    while budget_remaining():
        logprobs, entropies, values, returns = [], [], [], []
        episode_returns = []

        batch_size = 0
        while batch_size < config.train_batch_size and budget_remaining():
            obs, _ = env.reset(seed=seed + batch_idx)
            done = False
            episode_rewards = []

            while not done and budget_remaining():
                actions, logprob, entropy = choose_action(obs, agent_policy)
                value = value_net(torch.from_numpy(obs).float()).squeeze(-1)

                logprobs.append(logprob)
                entropies.append(entropy)
                values.append(value)

                obs, rewards, done, _, _ = env.step(actions.detach().numpy())
                episode_rewards.append(np.asarray(rewards, dtype=np.float32))

                batch_size += 1
                step_count += 1

            rewards = np.stack(episode_rewards)  # (steps, num_agents)
            returns.append(
                np.stack(
                    [
                        compute_returns(rewards[:, a], gamma=config.gamma)
                        for a in range(rewards.shape[1])
                    ],
                    axis=1,
                )
            )
            episode_returns.append(float(rewards.sum()))

        returns = torch.as_tensor(np.concatenate(returns), dtype=torch.float32)
        returns = returns.flatten()
        values = torch.stack(values).flatten()
        logprobs = torch.stack(logprobs).flatten()
        entropies = torch.stack(entropies).flatten()

        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        policy_loss = (
            -(advantages * logprobs).mean()
            - config.entropy_weight * entropies.mean()
        )
        value_loss = torch.mean((values - returns) ** 2)

        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            agent_policy.parameters(), config.gradient_clip
        )
        policy_opt.step()

        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        eval_returns = evaluate_marl_policy(
            env,
            agent_policy,
            num_episodes=config.num_eval_episodes,
            seed=seed + 10_000 + batch_idx,
        ).mean(axis=0)
        eval_return = float(eval_returns.sum())
        train_return = float(np.mean(episode_returns))
        step_count += env.episode_len * config.num_eval_episodes


        if eval_return > best_eval_return:
            best_eval_return = eval_return
            best_state_dict = deepcopy(agent_policy.state_dict())
            torch.save(best_state_dict, save_paths["shared_policy"])

        logger.log(
            step=step_count,
            episode=batch_idx,
            agent_id="shared_policy",
            train_return=train_return,
            eval_return=eval_return,
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropies.mean().detach()),
        )
        batch_idx += 1

    logger.close()
    agent_policy.load_state_dict(best_state_dict)
    pbar.close()
    save_training_plots(save_folder, title=config.group_identifier)
    return agent_policy, save_paths
