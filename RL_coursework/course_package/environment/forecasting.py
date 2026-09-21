"""Offline RNN forecasting utilities for DER environment observations."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class _GRUModel(nn.Module):
    """Base GRU model to forecast a single target."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        """Initialize a GRU branch.

        Parameters
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Number of forecast outputs.
        hidden_size : int
            Number of hidden units in the GRU.
        num_layers : int
            Number of recurrent layers.
        """
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Return branch forecasts from the final recurrent state."""
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class RNNForecaster(nn.Module):
    """Parallel GRU forecaster for temperature, price, and demand.

    Parameters
    ----------
    input_groups : dict[str, list[int]]
        Input column indices for ``temp``, ``price``, and ``demand`` branches.
    n_agents : int
        Number of agent demand series.
    hidden_size : int, default=64
        Number of hidden units per GRU branch.
    num_layers : int, default=1
        Number of recurrent layers per branch.

    Attributes
    ----------
    input_groups : dict[str, list[int]]
        Input column indices for ``temp``, ``price``, and ``demand`` branches.
    temp_net : _GRUModel
        GRU network for temperature forecasting.
    price_net : _GRUModel
        GRU network for price forecasting.
    demand_net : _GRUModel
        GRU network for demand forecasting.
    """

    def __init__(
        self,
        input_groups: dict[str, list[int]],
        n_agents: int,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        """Initialize the parallel GRU forecaster.

        Parameters
        ----------
        input_groups : dict[str, list[int]]
            Input column indices for ``temp``, ``price``, and ``demand`` branches.
        n_agents : int
            Number of agent demand series.
        hidden_size : int, default=64
            Number of hidden units per GRU branch.
        num_layers : int, default=1
            Number of recurrent layers per branch.
        """
        super().__init__()
        self.input_groups = input_groups
        self.temp_net = _GRUModel(len(input_groups["temp"]), 2, hidden_size, num_layers)
        self.price_net = _GRUModel(
            len(input_groups["price"]), 2, hidden_size, num_layers
        )
        self.demand_net = _GRUModel(
            len(input_groups["demand"]), 2 * n_agents, hidden_size, num_layers
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forecast each target group with its corresponding GRU module.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, window_width, n_features).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Forecasts for temperature, price, and demand.
        """
        temp = self.temp_net(x[..., self.input_groups["temp"]])
        price = self.price_net(x[..., self.input_groups["price"]])
        demand = self.demand_net(x[..., self.input_groups["demand"]])
        return temp, price, demand

    def predict_concat(self, x: Tensor) -> Tensor:
        """Return forecasts concatenated in the environment's expected order.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, window_width, n_features).

        Returns
        -------
        Tensor
            Concatenated forecasts.
        """
        return torch.cat(self.forward(x), dim=-1)


def load_training_data(
    dataset_dir: str | Path,
) -> tuple[pd.DatetimeIndex, NDArray, dict[str, list[int]], int]:
    """Load target variables, cyclic time marks, and branch input groups.

    Parameters
    ----------
    dataset_dir : str | Path
        Directory containing ``weather.csv``, ``grid_cost_coefficient.csv``, and
        ``agent_*.csv`` files.

    Returns
    -------
    tuple[pd.DatetimeIndex, NDArray, dict[str, list[int]], int]
        Datetime index, feature matrix, branch input indices, and agent count.
    """
    dataset_dir = Path(dataset_dir)
    weather = pd.read_csv(dataset_dir / "weather.csv")
    price = pd.read_csv(dataset_dir / "grid_cost_coefficient.csv")

    weather["datetime"] = pd.to_datetime(weather["datetime"])
    price["datetime"] = pd.to_datetime(price["datetime"])
    df = weather.merge(price[["datetime", "grid_cost_coeff"]], on="datetime")

    domestic_cols = []
    for path in sorted(
        dataset_dir.glob("agent_*.csv"), key=lambda p: int(p.stem.split("_")[1])
    ):
        agent = pd.read_csv(path, usecols=["datetime", "domestic_load"])
        agent["datetime"] = pd.to_datetime(agent["datetime"])
        col = f"domestic_load_{len(domestic_cols)}"
        df = df.merge(agent.rename(columns={"domestic_load": col}), on="datetime")
        domestic_cols.append(col)

    cyclic_cols = [
        col
        for col in [
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "dow_sin",
            "dow_cos",
        ]
        if col in df.columns
    ]

    feature_cols = []
    for col in ["external_temp", "grid_cost_coeff", *domestic_cols, *cyclic_cols]:
        if col not in feature_cols:
            feature_cols.append(col)

    input_groups = {
        "temp": [feature_cols.index(col) for col in ["external_temp", *cyclic_cols]],
        "price": [feature_cols.index(col) for col in ["grid_cost_coeff", *cyclic_cols]],
        "demand": [feature_cols.index(col) for col in [*domestic_cols, *cyclic_cols]],
    }

    df = df.sort_values("datetime")
    return (
        pd.DatetimeIndex(df["datetime"]),
        df[feature_cols].to_numpy(np.float32),
        input_groups,
        len(domestic_cols),
    )


def make_training_windows(
    data: NDArray,
    input_groups: dict[str, list[int]],
    n_agents: int,
    window_width: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Build historical input windows and raw target arrays.

    Parameters
    ----------
    data : NDArray
        Feature matrix with shape (n_timesteps, n_features).
    input_groups : dict[str, list[int]]
        Input column indices for ``temp``, ``price``, and ``demand`` branches.
    n_agents : int
        Number of agents.
    window_width : int
        Number of historical timesteps to include in each input window.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        Input windows and temperature, price, and demand targets.
    """
    external_temp_idx = input_groups["temp"][0]
    grid_cost_idx = input_groups["price"][0]
    domestic_idx = input_groups["demand"][:n_agents]
    x, y_temp, y_price, y_demand = [], [], [], []

    for t in range(window_width - 1, data.shape[0] - 12):
        x.append(data[t - window_width + 1 : t + 1, :])
        y_temp.append([data[t + 6, external_temp_idx], data[t + 12, external_temp_idx]])
        y_price.append([data[t + 6, grid_cost_idx], data[t + 12, grid_cost_idx]])
        y_demand.append([*data[t + 1, domestic_idx], *data[t + 4, domestic_idx]])

    return (
        np.array(x, dtype=np.float32),
        np.array(y_temp, dtype=np.float32),
        np.array(y_price, dtype=np.float32),
        np.array(y_demand, dtype=np.float32),
    )


def scale_targets(y: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Apply z-score normalization to targets.

    Parameters
    ----------
    y : NDArray
        Target array.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Normalized targets, mean, and standard deviation.
    """
    mean = y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (y - mean) / std, mean, std


def plot_learning_curves(history: dict[str, list[float]], output: str | Path) -> Figure:
    """Plot learning curves for each forecasted variable.

    Parameters
    ----------
    history : dict[str, list[float]]
        Training and validation loss history.
    output : str | Path
        Output figure path.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for ax, name in zip(axes, ["temp", "price", "demand"]):
        ax.plot(history[f"{name}_train"], label="Train")
        ax.plot(history[f"{name}_val"], label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(output, dpi=300)
    return fig


def forecast_all(
    model: RNNForecaster,
    data: NDArray,
    input_mean: NDArray,
    input_std: NDArray,
    target_stats: tuple[
        tuple[NDArray, NDArray], tuple[NDArray, NDArray], tuple[NDArray, NDArray]
    ],
    window_width: int,
    batch_size: int,
    device: torch.device,
) -> NDArray:
    """Forecast all timesteps and inverse-transform branch outputs.

    Parameters
    ----------
    model : RNNForecaster
        Trained forecasting model.
    data : NDArray
        Raw feature matrix.
    input_mean : NDArray
        Input normalization mean.
    input_std : NDArray
        Input normalization standard deviation.
    target_stats : tuple
        Per-branch target means and standard deviations.
    window_width : int
        Number of timesteps per input window.
    batch_size : int
        Batch size for prediction.
    device : torch.device
        Torch device used for prediction.

    Returns
    -------
    NDArray
        Raw-scale forecast array.
    """
    padded = np.concatenate(
        [np.repeat(data[:1, :], window_width - 1, axis=0), data], axis=0
    )
    windows = np.array(
        [padded[t : t + window_width, :] for t in range(data.shape[0])],
        dtype=np.float32,
    )
    windows = (windows - input_mean) / input_std

    preds = []
    model.eval()
    with torch.no_grad():
        for xb in DataLoader(torch.as_tensor(windows), batch_size=batch_size):
            pred = model.predict_concat(xb.to(device)).cpu().numpy()
            preds.append(pred)

    predictions = np.concatenate(preds, axis=0)
    temp_mean, temp_std = target_stats[0]
    price_mean, price_std = target_stats[1]
    demand_mean, demand_std = target_stats[2]
    n_demand = demand_mean.shape[-1]
    predictions[:, :2] = predictions[:, :2] * temp_std + temp_mean
    predictions[:, 2:4] = predictions[:, 2:4] * price_std + price_mean
    predictions[:, 4 : 4 + n_demand] = (
        predictions[:, 4 : 4 + n_demand] * demand_std + demand_mean
    )
    return predictions.astype(np.float32)


def save_forecasts(
    output: str | Path,
    datetime: pd.DatetimeIndex,
    predictions: NDArray,
) -> None:
    """Save aligned raw forecast arrays for the environment.

    Parameters
    ----------
    output : str | Path
        Output ``.npz`` file path.
    datetime : pd.DatetimeIndex
        Datetime index aligned with predictions.
    predictions : NDArray
        Raw forecast array.
    """
    n_agents = (predictions.shape[1] - 4) // 2
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        datetime=np.asarray(datetime.astype(str), dtype="U64"),
        external_temp_pred1=predictions[:, 0],
        external_temp_pred2=predictions[:, 1],
        grid_cost_coeff_pred1=predictions[:, 2],
        grid_cost_coeff_pred2=predictions[:, 3],
        domestic_load_pred1=predictions[:, 4 : 4 + n_agents].T,
        domestic_load_pred2=predictions[:, 4 + n_agents :].T,
    )


def train_forecaster(args: Namespace) -> None:
    """Train parallel GRU branches and save raw forecasts.

    Parameters
    ----------
    args : Namespace
        Training arguments.
    """
    datetime, data, input_groups, n_agents = load_training_data(args.data_dir)
    x, y_temp, y_price, y_demand = make_training_windows(
        data, input_groups, n_agents, args.window_width
    )

    input_mean = x.mean(axis=(0, 1), keepdims=True)
    input_std = x.std(axis=(0, 1), keepdims=True)
    input_std = np.where(input_std == 0.0, 1.0, input_std)
    x = (x - input_mean) / input_std

    y_temp, temp_mean, temp_std = scale_targets(y_temp)
    y_price, price_mean, price_std = scale_targets(y_price)
    y_demand, demand_mean, demand_std = scale_targets(y_demand)

    split = max(int((1.0 - args.val_ratio) * x.shape[0]), 1)
    split = min(split, x.shape[0] - 1)
    device = torch.device(args.device)
    model = RNNForecaster(
        input_groups=input_groups,
        n_agents=n_agents,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_set = TensorDataset(
        torch.as_tensor(x[:split]),
        torch.as_tensor(y_temp[:split]),
        torch.as_tensor(y_price[:split]),
        torch.as_tensor(y_demand[:split]),
    )
    val_tensors = [
        torch.as_tensor(arr).to(device)
        for arr in [x[split:], y_temp[split:], y_price[split:], y_demand[split:]]
    ]
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    history = {
        key: []
        for key in [
            "temp_train",
            "temp_val",
            "price_train",
            "price_val",
            "demand_train",
            "demand_val",
        ]
    }

    for epoch in tqdm(range(args.epochs), desc="Training loop"):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        n_samples = 0
        for xb, ytb, ypb, ydb in loader:
            xb = xb.to(device)
            ytb = ytb.to(device)
            ypb = ypb.to(device)
            ydb = ydb.to(device)

            optimizer.zero_grad()
            temp_pred, price_pred, demand_pred = model(xb)
            temp_loss = loss_fn(temp_pred, ytb)
            price_loss = loss_fn(price_pred, ypb)
            demand_loss = loss_fn(demand_pred, ydb)
            loss = temp_loss + price_loss + demand_loss
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            totals += batch_size * np.array([
                temp_loss.item(),
                price_loss.item(),
                demand_loss.item(),
            ])
            n_samples += batch_size

        model.eval()
        with torch.no_grad():
            temp_pred, price_pred, demand_pred = model(val_tensors[0])
            val_losses = [
                loss_fn(temp_pred, val_tensors[1]).item(),
                loss_fn(price_pred, val_tensors[2]).item(),
                loss_fn(demand_pred, val_tensors[3]).item(),
            ]

        for name, train_loss, val_loss in zip(
            ["temp", "price", "demand"], totals / n_samples, val_losses
        ):
            history[f"{name}_train"].append(float(train_loss))
            history[f"{name}_val"].append(float(val_loss))

    fig = plot_learning_curves(history, args.learning_curve_path)
    predictions = forecast_all(
        model=model,
        data=data,
        input_mean=input_mean,
        input_std=input_std,
        target_stats=(
            (temp_mean, temp_std),
            (price_mean, price_std),
            (demand_mean, demand_std),
        ),
        window_width=args.window_width,
        batch_size=args.batch_size,
        device=device,
    )
    save_forecasts(args.output, datetime, predictions)

    plt.close(fig)
