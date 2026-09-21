from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    apply_publication_style,
)

apply_publication_style()


def _log_folder(log_folder: str | Path | Mapping[str, str | Path]) -> Path:
    """Normalize a log directory or the save-path mapping returned by train()."""
    if isinstance(log_folder, Mapping):
        if not log_folder:
            raise ValueError("The model path mapping is empty.")
        return Path(next(iter(log_folder.values()))).parent
    return Path(log_folder)


def _extract_csv_data(
    log_folder: str | Path | Mapping[str, str | Path],
) -> dict[str, pd.DataFrame]:
    """Extract training and evaluation returns from a CSV log file.

    Usage hint
    ----------
    python plot_logs.py path/to/log.csv"""

    folder = _log_folder(log_folder)
    data = {
        csv_path.stem: pd.read_csv(csv_path)
        for csv_path in sorted(folder.glob("*.csv"))
    }
    if not data:
        raise FileNotFoundError(f"No CSV training logs found in {folder}")
    return data


def _format_axes(ax: plt.Axes, title: str | None = None):
    ax.set_xlabel("Steps")
    ax.set_ylabel("Return")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(False)
    ax.figure.tight_layout()


def _plot_single_learning_curve(
    steps: pd.Series,
    train_return: pd.Series,
    eval_return: pd.Series | None,
    ax: plt.Axes,
    agent: str | None = None,
):
    if agent is None:
        label_train = "Train"
        label_eval = "Validation"
    else:
        label_train = f"{agent} (train)"
        label_eval = f"{agent} (valid)"
    ax.plot(steps, train_return, label=label_train)

    if eval_return is not None:
        color = ax.get_lines()[-1].get_color()
        ax.plot(steps, eval_return, color=color, linestyle="--", label=label_eval)


def plot_learning_curves(
    log_folder: str | Path | Mapping[str, str | Path],
    title: str | None = None,
    *,
    show: bool = True,
) -> plt.Figure:
    """Plot the training and evaluation returns of the shared policy.

    The return is summed over all agents, so a single curve describes the run."""
    data = _extract_csv_data(log_folder)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in data.items():
        _plot_single_learning_curve(
            steps=df["step"],
            train_return=df["train_return"],
            eval_return=df["eval_return"] if "eval_return" in df.columns else None,
            ax=ax,
            agent=name if len(data) > 1 else None,
        )

    _format_axes(ax, title)
    if show:
        plt.show()
    return fig


def plot_reward_distribution(rewards, labels=None) -> plt.Figure:
    """
    Arguments:
        reward_arr : must be a [n_episodes, n_data_sets] array
    """
    if isinstance(rewards, list) or rewards.ndim == 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.grid(False)
        ax.hist(rewards, density=True, edgecolor="black", alpha=0.5)
        ax.set_xlabel("Global episode return.")
        ax.set_ylabel("Probability density")
        plt.ticklabel_format(style="sci", axis="x", scilimits=(-4, 4))
        if labels is not None:
            ax.legend(labels)

    else:
        fig, axs = plt.subplots(1, rewards.shape[1], figsize=(5 * rewards.shape[1], 5))
        axs = np.atleast_1d(axs)
        if np.std(rewards.mean(axis=0)) > 100000:
            for i in range(rewards.shape[1]):
                axs[i].hist(rewards[:, i], density=True, edgecolor="black", alpha=0.5)
                if labels is not None:
                    axs[i].legend([labels[i]])
        else:
            for i in range(rewards.shape[1]):
                axs[i].hist(rewards[:, i], density=True, edgecolor="black", alpha=0.5)
                if labels is not None:
                    axs[i].set_title(labels[i])
        for i in range(rewards.shape[1]):
            axs[i].set_xlabel("Global episode return.")
            axs[i].set_ylabel("Probability density")
            axs[i].ticklabel_format(style="sci", axis="x", scilimits=(-4, 4))
            axs[i].grid(False)

    fig.tight_layout()
    return fig
