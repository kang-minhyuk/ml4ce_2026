"""Group XX: <member names>. Implementation of my own RL algorithm."""

from dataclasses import dataclass

import numpy as np

from course_package import MARLEnv
from course_package.utils.helpers import (
    BaseAlgorithmConfig,
    set_seed,
)


class SharedPolicy:
    """
    The single policy shared by all agents.

    It maps ONE agent's local observation to that agent's action. The same
    parameters are used for every agent: ``act`` is called once per agent, so
    agents behave differently only because their observations differ.
    """

    def __init__(self, obs_dim: int, action_dim: int):
        # YOUR POLICY PARAMETERS GO HERE (e.g. numpy arrays or a torch.nn.Module)
        self.weights = np.zeros((action_dim, obs_dim), dtype=np.float32)

    def act(self, local_obs: np.ndarray) -> np.ndarray:
        """
        Deterministic action used for evaluation and grading.

        ``local_obs`` has shape (obs_dim,). Return an array of shape
        (action_dim,) with values in [-1, 1].
        """
        return np.tanh(self.weights @ local_obs)


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """Hyperparameter specification for Custom Algorithm."""

    hyperparameter_1: float = 0.5
    hyperparameter_2: int = 10
    group_identifier: str = "GROUP_NAME"


def train(
    env: MARLEnv,
    config: AlgorithmConfig,
    seed: int = 0,
) -> tuple[SharedPolicy, dict[str, str]]:
    """
    Create and train the single policy shared by all agents.

    `env` returns observations of shape (num_agents, num_features): row i is
    the local observation of agent i. Apply the same policy to every row and
    update its parameters with the experience of all agents.
    """
    set_seed(seed)
    policy = SharedPolicy(env.observation_space.shape[1], env.action_space.shape[1])

    # YOUR CODE GOES HERE

    return policy, {}
