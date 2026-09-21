from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .style import apply_publication_style

apply_publication_style()

if TYPE_CHECKING:
    from course_package import EnergySystem


def plot_battery(env: "EnergySystem", agent_idx: int = 0) -> Figure:
    """Plot battery state, trip demand, and net battery flow.

    Parameters
    ----------
    env : EnergySystem
        Environment with recorded rollout data.
    agent_idx : int, default=0
        agent index.
    """
    start = env.current_episode_start
    T = env.episode_len
    t_datetime = env.datetime[start : start + T]

    ev_soc = env.ev_soc[agent_idx, :T]
    ev_trip_demand = env.ev_trip_demand[agent_idx, start : start + T]
    b_in = env.b_in[agent_idx, :T]
    b_out = env.b_out[agent_idx, :T]
    ev_available = env.ev_availability[agent_idx, start : start + T]
    unavailable_mask = ev_available == 0

    fig, ax1 = plt.subplots(figsize=(12, 4))

    for i in range(len(t_datetime) - 1):
        if unavailable_mask[i]:
            ax1.axvspan(
                t_datetime[i],
                t_datetime[i + 1],
                alpha=0.15,
                facecolor="grey",
                zorder=0,
                edgecolor="none",
            )

    ax1.step(
        t_datetime, ev_soc, where="post", label="EV SoC", linewidth=2, color="tab:blue"
    )
    ax1.step(
        t_datetime,
        ev_trip_demand,
        where="post",
        label="EV trip demand",
        linewidth=2,
        color="tab:red",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("EV State / Demand")

    ax2 = ax1.twinx()
    ax2.step(
        t_datetime,
        b_in - b_out,
        where="post",
        label="b_in - b_out",
        linewidth=2,
        color="tab:green",
    )
    ax2.set_ylabel("Battery flows (kWh)")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
    return fig


def plot_comfort(env: "EnergySystem", agent_idx: int = 0) -> Figure:
    """Plot indoor temperature, comfort band, and thermal load.

    Parameters
    ----------
    env : EnergySystem
        Environment with recorded rollout data.
    agent_idx : int, default=0
        agent index.
    """
    start = env.current_episode_start
    T = env.episode_len
    t_datetime = env.datetime[start : start + T]

    temp_in = env.indoor_temp[agent_idx, :T]
    temp_sp = env.indoor_temp_setpoint[agent_idx, start : start + T]
    heat = env.heating_load[agent_idx, :T]
    cool = env.cooling_load[agent_idx, :T]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t_datetime, temp_in, label="Indoor temperature", linewidth=2)
    ax1.plot(
        t_datetime,
        temp_sp,
        label="Set point",
        linewidth=0.75,
        linestyle="--",
        color="tab:red",
    )
    ax1.fill_between(
        t_datetime,
        temp_sp - env.delta_temp,
        temp_sp + env.delta_temp,
        alpha=0.1,
        label="Comfort interval",
        color="tab:red",
    )
    ax1.set_ylabel("Temperature")

    ax2 = ax1.twinx()
    ax2.step(
        t_datetime,
        heat - cool,
        where="post",
        label="Net thermal load",
        color="tab:green",
        linewidth=1.0,
    )
    ax2.set_ylabel("Thermal load [kWh]")
    ax2.spines["right"].set_color("tab:green")
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", labelcolor="tab:green")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()

    return fig


def plot_electricity_import(env: "EnergySystem", agent_idx: int = 0) -> Figure:
    """Plot per-step grid import and electricity price.

    Parameters
    ----------
    env : EnergySystem
        Environment with recorded rollout data.
    agent_idx : int, default=0
        agent index.
    """
    start = env.current_episode_start
    T = env.episode_len
    t_datetime = env.datetime[start : start + T]

    net_grid = env.pros_import[agent_idx, :T]
    price = env.grid_cost_coeff[start : start + T]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.step(t_datetime, net_grid, where="post", label="Grid import", linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy flow")
    ax2 = ax1.twinx()
    ax2.plot(
        t_datetime, price, color="tab:red", linestyle="--", label="Electricity price"
    )
    ax2.set_ylabel("Price")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.spines["right"].set_color("tab:red")
    ax2.spines["right"].set_visible(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
    return fig


def plot_normalized_actions_battery(
    env: "EnergySystem", actions_history: NDArray, agent_idx: int = 0
) -> Figure:
    """Plot battery actions against net battery flow.

    Parameters
    ----------
    env : EnergySystem
        Environment with recorded rollout data.
    actions_history : NDArray
        Normalized action history.
    agent_idx : int, default=0
        agent index.
    """
    start = env.current_episode_start
    T = env.episode_len
    t_datetime = env.datetime[start : start + T]

    b_in = env.b_in[agent_idx, :T]
    b_out = env.b_out[agent_idx, :T]
    a_battery = actions_history[:, 1]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.step(
        t_datetime,
        b_in - b_out,
        where="post",
        label="Battery net flow",
        linewidth=2,
        color="tab:blue",
    )
    ax1.set_ylabel("Battery net flow [kWh]")

    ax2 = ax1.twinx()
    ax2.step(
        t_datetime,
        a_battery,
        where="post",
        label="Battery action",
        linewidth=2,
        color="tab:orange",
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.spines["right"].set_color("tab:orange")
    ax2.spines["right"].set_visible(True)
    ax2.set_ylabel("Battery action (normalized)")

    h3, l3 = ax1.get_legend_handles_labels()
    h4, l4 = ax2.get_legend_handles_labels()
    fig.legend(h3 + h4, l3 + l4, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
    return fig


def plot_normalized_actions_thermal(
    env: "EnergySystem", actions_history: NDArray, agent_idx: int = 0
) -> Figure:
    """Plot thermal actions against net thermal load.

    Parameters
    ----------
    env : EnergySystem
        Environment with recorded rollout data.
    actions_history : NDArray
        Normalized action history.
    agent_idx : int, default=0
        agent index.
    """
    start = env.current_episode_start
    T = env.episode_len
    t_datetime = env.datetime[start : start + T]

    heat = env.heating_load[agent_idx, :T]
    cool = env.cooling_load[agent_idx, :T]
    a_thermal = actions_history[:, 0]

    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.step(
        t_datetime,
        heat - cool,
        where="post",
        label="Net thermal load",
        linewidth=2,
        color="tab:red",
    )
    ax1.set_ylabel("Thermal loads [kWh]")
    ax2 = ax1.twinx()
    ax2.step(
        t_datetime,
        a_thermal,
        where="post",
        label="Thermal action",
        linewidth=2,
        color="tab:green",
    )
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.spines["right"].set_color("tab:green")
    ax2.spines["right"].set_visible(True)
    ax2.set_ylabel("Thermal action (normalized)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
    return fig
