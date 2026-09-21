"""Implementation of a PID controller for temperature regulation in a DER environment."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import sample
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

if TYPE_CHECKING:
    from course_package.environment import EnergySystem, MARLEnv

# pylint: disable=redefined-outer-name


@dataclass
class PIDParams:
    """PID configuration parameters.

    Attributes
    ----------
    n_episodes_avg : int
        Number of episodes used when tuning gains.
    memory : int
        Error-history length.
    w_magnitude : float
        Weight for action magnitude in the tuning objective.
    w_change : float
        Weight for action change in the tuning objective.
    """

    n_episodes_avg: int = 10
    memory: int = 12
    w_magnitude: float = 0.01
    w_change: float = 0.01


def random_search(
    objective_function: Callable,
    bounds: NDArray,
    mode: Literal["normal", "log"] = "normal",
    iter: int = 100,
) -> tuple[np.ndarray, float]:
    """Run random search over bounded parameters.

    Parameters
    ----------
    objective_function : Callable
        Objective function to minimize.
    bounds : NDArray
        Parameter bounds shaped ``(n_params, 2)``.
    mode : Literal["normal", "log"], default="normal"
        Sampling distribution type.
    iter : int, default=100
        Number of random samples.

    Returns
    -------
    tuple[np.ndarray, float]
        Best parameter vector and objective value.
    """
    x_dim = bounds.shape[0]
    sample_param = np.zeros((iter, x_dim))
    obj_values = np.zeros(iter)

    if mode == "log":
        sample_param = 10 ** np.random.uniform(
            np.log10(bounds[:, 0]), np.log10(bounds[:, 1]), size=(iter, bounds.shape[0])
        )
    elif mode == "normal":
        sample_param = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(iter, bounds.shape[0])
        )
    obj_values = list(map(objective_function, sample_param))

    idx_best_obj = np.argmin(obj_values)
    return (
        sample_param[idx_best_obj, :],
        obj_values[idx_best_obj],
    )


class TemperaturePID:
    """PID controller for temperature regulation.

    Parameters
    ----------
    param : PIDParams
        PID configuration values.
    log : bool, default=False
        Whether to write optimization logs.
    gain_dir : Optional[str], default=None
        Optional path to pre-tuned gains.
    information : Literal["perfect", "forecast"], default="perfect"
        Information mode used by the expert interface.

    Attributes
    ----------
    Ks : np.ndarray
        PID gains.
    n_episodes_avg : int
        Number of episodes to average for objective evaluation.
    error : deque
        Error history for integral and derivative calculations.
    w_magnitude : float
        Weight for magnitude term in objective function.
    w_change : float
        Weight for change term in objective function.
    information : str
        Information mode used by the expert interface.
    _log : bool
        Whether to write optimization logs.
    """

    def __init__(
        self,
        param: PIDParams,
        log: bool = False,
        gain_dir: str | None = None,
        information: Literal["perfect", "forecast"] = "perfect",
    ) -> None:
        """Initialize PID controller state and optional gain loading.

        Parameters
        ----------
        param : PIDParams
            PID configuration values.
        log : bool, default=False
            Whether to write optimization logs.
        gain_dir : Optional[str], default=None
            Optional path to pre-tuned gains.
        information : Literal["perfect", "forecast"], default="perfect"
            Information mode used by the expert interface.
        """
        self.n_episodes_avg = param.n_episodes_avg
        self.memory = param.memory
        self.errors: dict[int, deque[float]] = {}
        self.w_magnitude = param.w_magnitude
        self.w_change = param.w_change
        self.information = information
        self._log = log
        if gain_dir is None:
            self.Ks = np.array([0.1, 0.01, 0.001])
        else:
            self.load_gains(gain_dir)

    def __call__(self, env: "EnergySystem | MARLEnv", agent_idx: int = 0) -> float:
        """Return the computed thermal action.

        Parameters
        ----------
        env : EnergySystem | MARLEnv
            Environment containing current thermal context.
        agent_idx : int, default=0
            Agent index.

        Returns
        -------
        float
            PID action value.
        """
        return self.compute_action(env, agent_idx=agent_idx)

    def compute_action(
        self, env: "EnergySystem | MARLEnv", agent_idx: int = 0
    ) -> float:
        """Compute normalized PID thermal action.

        Parameters
        ----------
        env : EnergySystem | MARLEnv
            Environment containing current thermal context.
        agent_idx : int, default=0
            Agent index.

        Returns
        -------
        float
            PID action value.
        """
        Kp, Ki, Kd = self.Ks
        error = self._agent_error(agent_idx)
        T = env.indoor_temp[agent_idx, env.current_step]
        Tsp = env.indoor_temp_setpoint[
            agent_idx,
            min(env.current_step_global + 1, env.indoor_temp_setpoint.shape[1] - 1),
        ]
        error.append(Tsp - T)
        a = np.float32(Kp * error[-1] + Ki * sum(error) + Kd * (error[-1] - error[-2]))
        return a

    def _agent_error(self, agent_idx: int) -> deque[float]:
        """Return the PID error memory for one independent agent."""
        if agent_idx not in self.errors:
            self.errors[agent_idx] = deque([0.0] * self.memory, maxlen=self.memory)
        return self.errors[agent_idx]

    def load_gains(self, path: str) -> None:
        """Load PID gains from a file.

        Parameters
        ----------
        path : str
            Path to ``.npz`` gains file.
        """
        Ks = np.load(path)
        self.Ks = np.array([Ks["Kp"], Ks["Ki"], Ks["Kd"]])

    @property
    def log(self) -> bool:
        """Flag to write optimization logs.

        Returns
        -------
        bool
            Whether logging is enabled.
        """
        return self._log

    @log.setter
    def log(self, value: bool) -> None:
        """Enable or disable objective logging to file."""
        self._log = value
        if value:
            self.log_path = Path.cwd() / "PID_log.log"
            self.log_f = open(self.log_path, "w+", encoding="utf-8")
            self.log_f.write("error,umag,ucha,obj_fcn\n")

    def objective_function(
        self,
        Ks: NDArray,
        env: "EnergySystem | MARLEnv",
        episode_starts: list,
        agent_idx: int = 0,
    ) -> float:
        """Evaluate PID gains on sampled episodes.

        Parameters
        ----------
        Ks : NDArray
            Candidate gains ``[Kp, Ki, Kd]``.
        env : EnergySystem | MARLEnv
            Environment used for evaluation.
        episode_starts : list
            Episode start indices for rollouts.
        agent_idx : int, default=0
            Agent for which the gains are tuned.

        Returns
        -------
        float
            Scalar objective to minimize.
        """
        self.Ks = Ks
        error_list = []
        action_list = []
        for t_start in episode_starts:
            self.reset_error()
            env.reset(options={"episode_start": t_start})
            done = False
            while not done:
                a = self.compute_action(env, agent_idx=agent_idx)
                actions = np.zeros(env.action_space.shape, dtype=np.float32)
                actions[agent_idx, 0] = a
                _, _, done, _, _ = env.step(actions)

                action_list.append(a)
                error_list.append(
                    (
                        env.indoor_temp[agent_idx, env.current_step]
                        - env.indoor_temp_setpoint[agent_idx, env.current_step_global]
                    )
                    ** 2
                )
        error = np.mean(error_list)
        u_mag = np.mean(np.abs(action_list))
        u_change = np.mean(np.abs(np.diff(action_list)))
        obj = error + self.w_magnitude * u_mag + self.w_change * u_change
        if self.log:
            self.log_f.write(f"{error},{u_mag},{u_change},{obj}\n")

        return obj

    def reset_error(self) -> None:
        """Reset the error history."""
        self.errors.clear()

    def update_temperature(
        self,
        env: "EnergySystem | MARLEnv",
        heat_load: float,
        cool_load: float,
        agent_idx: int = 0,
    ) -> NDArray:
        """Return indoor temperature after applying heat and cool loads.

        Parameters
        ----------
        env : EnergySystem | MARLEnv
            Environment containing current thermal context.
        heat_load : float
            Heating load.
        cool_load : float
            Cooling load.
        agent_idx : int, default=0
            Agent index.

        Returns
        -------
        NDArray
            Updated indoor temperature value.
        """
        T = env.indoor_temp[agent_idx, env.current_step]
        Text = env.external_temp[env.current_step_global]
        solar_ghi = env.solar_ghi[env.current_step_global]

        temp_change = (
            heat_load
            - cool_load
            + (Text - T) / env.building_thermal_resistance
            + solar_ghi * env.building_solar_gain
        ) / env.building_thermal_capacitance

        return T + temp_change


def tune_gains(
    param: PIDParams,
    env: "EnergySystem | MARLEnv",
    bounds: NDArray,
    agent_idx: int = 0,
) -> NDArray:
    """Tune PID gains using random search followed by local optimization.

    Parameters
    ----------
    param : PIDParams
        PID tuning parameters.
    env : EnergySystem | MARLEnv
        Environment used for evaluation rollouts.
    bounds : NDArray
        Gain bounds shaped ``(3, 2)``.
    agent_idx : int, default=0
        Agent for which the gains are tuned.

    Returns
    -------
    NDArray
        Optimized gains ``[Kp, Ki, Kd]``.

    Raises
    ------
    ValueError
        If local optimization fails.
    """

    pid = TemperaturePID(param)
    episode_starts = sample(list(env.valid_episode_starts), param.n_episodes_avg)
    Ks_rs, _ = random_search(
        lambda p: pid.objective_function(p, env, episode_starts, agent_idx=agent_idx),
        bounds,
        mode="log",
        iter=500,
    )
    print(f"Random search best gains: Kp={Ks_rs[0]}, Ki={Ks_rs[1]}, Kd={Ks_rs[2]}")

    pid.log = True
    bounds[:, 0] = 0.5 * Ks_rs
    bounds[:, 1] = 1.5 * Ks_rs
    result = minimize(
        pid.objective_function,
        args=(env, episode_starts, agent_idx),
        x0=Ks_rs,
        bounds=bounds,
        method="Powell",
    )
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)


if __name__ == "__main__":
    import json
    import os

    from course_package.environment import MARLEnv
    from course_package.utils.plots.trajectories import plot_comfort

    path = os.path.join(os.getcwd(), r"data\datasets\UK_IC\env_config")
    with open(path, "r", encoding="utf-8") as f:
        env_config: dict = json.load(f)

    env = MARLEnv(env_config)
    pid_config = PIDParams(n_episodes_avg=100, memory=12, w_magnitude=1.0, w_change=1.0)
    bounds = np.array([[1e-7, 1.0], [1e-7, 1.0], [1e-7, 1.0]])  # Bounds for Kp, Ki, Kd
    Ks_opt = tune_gains(pid_config, env, bounds)
    np.savez_compressed("pid_gains.npz", Kp=Ks_opt[0], Ki=Ks_opt[1], Kd=Ks_opt[2])
    print(f"Optimized PID gains: Kp={Ks_opt[0]}, Ki={Ks_opt[1]}, Kd={Ks_opt[2]}")

    pid = TemperaturePID(pid_config)
    pid.load_gains("pid_gains.npz")
    env.reset()
    done = False
    while not done:
        a = pid.compute_action(env)
        actions = np.zeros(env.action_space.shape, dtype=np.float32)
        actions[0, 0] = a
        _, _, done, _, _ = env.step(actions)
    plot_comfort(env.env)
