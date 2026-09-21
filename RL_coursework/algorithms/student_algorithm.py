"""Group XX: <member names>. Implementation of my own RL algorithm."""

from dataclasses import dataclass

from course_package import MARLEnv
from course_package.policies.rl.networks import PolicyNetwork
from course_package.utils.helpers import (
    BaseAlgorithmConfig,
    set_seed,
)


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """Hyperparameter specification for Custom Algorithm."""

    hyperparameter_1: float = 0.5
    hyperparameter_2: int = 10
    group_identifier: str = "GROUP_NAME"


def train(
    env: MARLEnv,
    agent_policy: PolicyNetwork,
    config: AlgorithmConfig,
    seed: int = 0,
) -> tuple[PolicyNetwork, dict[str, str]]:
    """
    Train the single policy shared by all agents.
    `env` returns observations of shape (num_agents, num_features), which
    `agent_policy` can process in one batched call.
    """
    set_seed(seed)

    # YOUR CODE GOES HERE

    return agent_policy, {}
