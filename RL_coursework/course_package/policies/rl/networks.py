import torch
from torch import Tensor


class PolicyNetwork(torch.nn.Module):
    """Policy network for continuous action spaces with Gaussian outputs."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, action_dim * 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        return mean, log_std


class ValueNetwork(torch.nn.Module):
    """Value network for estimating state values."""

    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
