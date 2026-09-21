"""MPC implementation for battery control in DER environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyomo.environ as pyo
from numpy.typing import NDArray

if TYPE_CHECKING:
    from course_package.environment import EnergySystem, MARLEnv


# pylint: disable=redefined-outer-name


@dataclass
class MPCParameters:
    """MPC optimization parameters for battery control.

    Attributes
    ----------
    N : int
        Forecast horizon.
    b_in_max : NDArray
        Maximum charging power.
    b_out_max : NDArray
        Maximum discharging power.
    SoC_min : float
        Minimum state of charge.
    SoC_max : float
        Maximum state of charge.
    ch_efficiency : float
        Battery charging efficiency.
    disch_efficiency : float
        Battery discharging efficiency.
    battery_depreciation_cost : float
        Battery throughput cost coefficient.
    mobility_reserve : float
        Additional departure-energy reserve.
    ev_demand_error_scale : float
        Initial relative standard deviation for EV demand forecast error.
    ev_demand_error_growth : float
        Linear forecast-error growth factor.
    ev_demand_error_clip : float
        Symmetric clipping bound for sampled relative demand errors.
    seed : int
        Random seed for forecast-error sampling.
    """

    N: int  # Forecast horizon
    b_in_max: NDArray  # min charging power [N]
    b_out_max: NDArray  # max discharging power [N]
    SoC_min: float  # minimum SoC
    SoC_max: float  # maximum SoC
    ch_efficiency: float  # battery charging efficiency
    disch_efficiency: float  # battery discharging efficiency
    battery_depreciation_cost: float  # battery throughput cost coefficient
    mobility_reserve: float = 2.0
    ev_demand_error_scale: float = 0.10
    ev_demand_error_growth: float = 1.0
    ev_demand_error_clip: float = 0.50
    seed: int = 42

    def ev_forecast(
        self,
        env: EnergySystem | MARLEnv,
        rng: np.random.Generator,
        information: Literal["perfect", "forecast"] = "forecast",
        agent_idx: int = 0,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Return availability, trip-demand, and mobility-requirement trajectories.

        Parameters
        ----------
        env : EnergySystem | MARLEnv
            Environment state.
        rng : np.random.Generator
            Random number generator for forecast errors.
        information : Literal["perfect", "forecast"], default="forecast"
            Information mode.
        agent_idx : int, default=0
            Agent index.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            Availability, trip demand, and required SoC trajectories.
        """
        if information == "perfect":
            start = env.current_step_global
            indices = np.minimum(
                start + np.arange(self.N + 1),
                env.ev_availability.shape[-1] - 1,
            )
            availability = env.ev_availability[agent_idx, indices[: self.N]].astype(
                np.float32
            )
            trip_demand = env.ev_trip_demand[agent_idx, indices[: self.N]].astype(
                np.float32
            )
            required_soc = (
                env.ev_required_soc_departure[agent_idx, indices]
                * env.ev_availability[agent_idx, indices]
            ).astype(np.float32)
            return availability, trip_demand, required_soc

        availability = np.zeros(self.N, dtype=np.float32)
        trip_demand = np.zeros(self.N, dtype=np.float32)
        required_soc = np.zeros(self.N + 1, dtype=np.float32)

        idx = min(env.current_step_global, env.ev_availability.shape[-1] - 1)
        if env.ev_availability[agent_idx, idx] <= 0.0:
            return availability, trip_demand, required_soc

        departure_step = max(
            0, int(round(float(env.ev_time_to_departure[agent_idx, idx])))
        )
        availability[: min(departure_step, self.N)] = 1.0
        if departure_step > self.N:
            return availability, trip_demand, required_soc

        nominal_required_soc = float(env.ev_required_soc_departure[agent_idx, idx])
        nominal_demand = max(nominal_required_soc - self.SoC_min, 0.0)
        sigma = self.ev_demand_error_scale * (
            1.0 + self.ev_demand_error_growth * departure_step / self.N
        )
        relative_error = float(
            np.clip(
                rng.normal(0.0, sigma),
                -self.ev_demand_error_clip,
                self.ev_demand_error_clip,
            )
        )
        forecast_demand = nominal_demand * (1.0 + relative_error)
        forecast_requirement = min(
            self.SoC_min + forecast_demand + self.mobility_reserve,
            self.SoC_max,
        )

        required_soc[departure_step] = forecast_requirement
        if departure_step < self.N:
            trip_demand[departure_step] = forecast_demand
        return availability, trip_demand, required_soc

    def price(
        self,
        env: EnergySystem | MARLEnv,
        information: Literal["perfect", "forecast"] = "perfect",
    ) -> NDArray:
        """Return electricity price over the MPC horizon.

        Parameters
        ----------
        env : EnergySystem | MARLEnv
            Environment state.
        information : Literal["perfect", "forecast"], default="perfect"
            Information mode.

        Returns
        -------
        NDArray
            Price trajectory.
        """
        if information == "forecast":
            idx = min(env.current_step_global, env.grid_cost_coeff.shape[-1] - 1)
            if not all(
                hasattr(env, name)
                for name in ["grid_cost_coeff_pred1", "grid_cost_coeff_pred2"]
            ):
                raise AttributeError(
                    "Forecast MPC requires grid_cost_coeff_pred1 and "
                    "grid_cost_coeff_pred2 in the environment."
                )
            anchors = np.array([0, 6, 12], dtype=np.float32)
            values = np.array(
                [
                    env.grid_cost_coeff[idx],
                    env.grid_cost_coeff_pred1[idx],
                    env.grid_cost_coeff_pred2[idx],
                ],
                dtype=np.float32,
            )
            return np.interp(np.arange(self.N), anchors, values).astype(np.float32)

        return env.grid_cost_coeff[
            env.current_step_global : env.current_step_global + self.N
        ]

    def SoC0(self, env: EnergySystem | MARLEnv, agent_idx: int = 0) -> float:
        """Return initial state of charge from environment.

        Parameters
        ----------
        env : EnergySystem | MARLEnv
            Environment state.
        agent_idx : int, default=0
            Agent index.

        Returns
        -------
        float
            Current EV state of charge.
        """
        return env.ev_soc[agent_idx, env.current_step]


class BatteryMPC:
    """MPC controller for normalized battery charging/discharging actions.

    Parameters
    ----------
    param : MPCParameters
        MPC parameters.
    information : Literal["perfect", "forecast"], default="forecast"
        Information mode.
    """

    def __init__(
        self,
        param: MPCParameters,
        information: Literal["perfect", "forecast"] = "forecast",
    ) -> None:
        """Initialize MPC model with battery parameters and optimization constraints."""
        self.N = param.N
        self.dt = 1.0
        self.information = information

        self.lambda_mob = 1000.0
        self.lambda_throughput = 0.02
        self.lambda_ramp = 0.05
        self.lambda_deficit = 1000.0

        self.param = param
        self.rng = np.random.default_rng(param.seed)

        self.model = pyo.ConcreteModel()

        self.model.t = pyo.RangeSet(0, self.N - 1)

        self.model.b_in = pyo.Var(self.model.t, domain=pyo.NonNegativeReals)
        self.model.b_out = pyo.Var(self.model.t, domain=pyo.NonNegativeReals)
        self.model.SoC = pyo.Var(
            self.model.t | {self.N}, bounds=(param.SoC_min, param.SoC_max)
        )
        self.model.epsilon = pyo.Var(
            self.model.t | {self.N}, domain=pyo.NonNegativeReals
        )
        self.model.beta = pyo.Var(self.model.t, domain=pyo.NonNegativeReals)

        self.model.y = pyo.Var(self.model.t, domain=pyo.Binary)

        self.model.SoC0 = pyo.Param(initialize=0.0, mutable=True)
        self.model.SoC_req = pyo.Param(
            self.model.t | {self.N}, initialize=0.0, mutable=True
        )
        self.model.mu = pyo.Param(self.model.t, initialize=0.0, mutable=True)
        self.model.b_in_max = pyo.Param(initialize=param.b_in_max, mutable=True)
        self.model.b_out_max = pyo.Param(initialize=param.b_out_max, mutable=True)
        self.model.SoC_min = pyo.Param(initialize=param.SoC_min, mutable=True)
        self.model.SoC_max = pyo.Param(initialize=param.SoC_max, mutable=True)
        self.model.d_EV = pyo.Param(self.model.t, initialize=0.0, mutable=True)
        self.model.price = pyo.Param(self.model.t, initialize=0.0, mutable=True)
        self.model.previous_power = pyo.Param(initialize=0.0, mutable=True)

        def dynamics_rule(model: pyo.ConcreteModel, t: int) -> pyo.Expression:
            """Define battery SoC dynamics constraint."""
            return (
                model.SoC[t + 1]
                == model.SoC[t]
                + model.b_in[t] * self.dt
                - model.b_out[t] * self.dt
                - model.d_EV[t]
                + model.beta[t]
            )

        self.model.dynamics = pyo.Constraint(self.model.t, rule=dynamics_rule)

        def mobility_rule(model: pyo.ConcreteModel, t: int) -> pyo.Expression:
            """Define mobility SoC requirement constraint."""
            return model.SoC[t] + model.epsilon[t] >= model.SoC_req[t]

        self.model.mobility = pyo.Constraint(
            self.model.t | {self.N}, rule=mobility_rule
        )

        self.model.cons_b_in_ub = pyo.ConstraintList()
        for t in self.model.t:
            self.model.cons_b_in_ub.add(
                expr=self.model.b_in[t]
                <= (self.model.SoC_max - self.model.SoC[t])
                * self.model.mu[t]
                * self.model.y[t]
            )
            self.model.cons_b_in_ub.add(
                expr=self.model.b_in[t]
                <= self.model.b_in_max * self.model.mu[t] * self.model.y[t]
            )

        self.model.cons_b_out_ub = pyo.ConstraintList()
        for t in self.model.t:
            self.model.cons_b_out_ub.add(
                expr=self.model.b_out[t]
                <= self.model.b_out_max * self.model.mu[t] * (1 - self.model.y[t])
            )
            self.model.cons_b_out_ub.add(
                expr=self.model.b_out[t]
                <= (self.model.SoC[t] - self.model.SoC_min)
                * self.model.mu[t]
                * (1 - self.model.y[t])
            )

        def objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
            """Define MPC stage and ramp objective."""
            stage_cost = sum(
                model.price[t]
                * (
                    model.b_in[t] / self.param.ch_efficiency
                    - model.b_out[t] * self.param.disch_efficiency
                )
                + self.param.battery_depreciation_cost
                * (model.b_in[t] + model.b_out[t])
                + self.lambda_throughput * (model.b_in[t] + model.b_out[t]) ** 2
                + self.lambda_mob * model.epsilon[t] ** 2
                + self.lambda_deficit * model.beta[t] ** 2
                for t in model.t
            )
            terminal_cost = self.lambda_mob * model.epsilon[self.N] ** 2
            ramp_cost = self.lambda_ramp * sum(
                (
                    model.b_in[t]
                    - model.b_out[t]
                    - (
                        model.previous_power
                        if t == 0
                        else model.b_in[t - 1] - model.b_out[t - 1]
                    )
                )
                ** 2
                for t in model.t
            )
            return stage_cost + terminal_cost + ramp_cost

        self.model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def __call__(self, env: EnergySystem, agent_idx: int = 0) -> float:
        """Solve MPC and return the normalized action.

        Parameters
        ----------
        env : EnergySystem
            Environment state.
        agent_idx : int, default=0
            Agent index.

        Returns
        -------
        float
            Normalized action.
        """
        return self.solve(env, agent_idx=agent_idx)[-1]

    def reset_rng(self) -> None:
        """Reset forecast-error sampling to the configured seed."""
        self.rng = np.random.default_rng(self.param.seed)

    def solve(
        self, env: EnergySystem, agent_idx: int = 0
    ) -> tuple[float, float, float]:
        """Solve MPC optimization problem and compute control action.

        Parameters
        ----------
        param : MPCParameters
            Current MPC parameters from environment.
        env : EnergySystem
            Environment state.
        agent_idx : int
            Index of the agent for which to solve the MPC problem.

        Returns
        -------
        tuple of (float, float, float)
            Charging power, discharging power, and normalized action.
        """

        self.update(env, agent_idx=agent_idx)

        if pyo.value(self.model.mu[0]) == 0.0:
            return 0.0, 0.0, np.float32(0.0)
        else:
            self.model.SoC[0].fix(self.model.SoC0.value)
            solver = pyo.SolverFactory("gurobi")
            results = solver.solve(self.model, tee=False)

            assert np.isclose(
                self.model.b_in[0].value * self.model.b_out[0].value, 0.0, atol=1e-4
            ), "b_in and b_out cannot be both positive at the same time."
            if results["Solver"][0]["Termination condition"] == "optimal":
                a = self._compute_normalized_action()
                return (
                    pyo.value(self.model.b_in[0]),
                    pyo.value(self.model.b_out[0]),
                    a,
                )
            else:
                print("MPC optimization failed to find optimal solution.")
                return 0.0, 0.0, np.float32(0.0)

    def update(self, env: EnergySystem, agent_idx: int = 0) -> None:
        """Update MPC model parameters from current environment state.

        Parameters
        ----------
        env : EnergySystem
            Environment state.
        agent_idx : int, default=0
            Agent index.
        """
        self.model.SoC0.set_value(self.param.SoC0(env, agent_idx=agent_idx))
        mu_values, d_ev_values, soc_req_values = self.param.ev_forecast(
            env,
            self.rng,
            information=self.information,
            agent_idx=agent_idx,
        )
        for t in self.model.t | {self.N}:
            self.model.SoC_req[t].set_value(soc_req_values[t])
        for t in self.model.t:
            self.model.mu[t].set_value(mu_values[t])
        for t in self.model.t:
            self.model.d_EV[t].set_value(d_ev_values[t])
        price_values = self.param.price(env, information=self.information)
        for t in self.model.t:
            self.model.price[t].set_value(price_values[t])
        previous_power = 0.0
        if env.current_step > 0:
            previous_power = (
                env.b_in[agent_idx, env.current_step - 1]
                - env.b_out[agent_idx, env.current_step - 1]
            )
        self.model.previous_power.set_value(previous_power)

    def _compute_normalized_action(self) -> float:
        """Compute normalized action from charging/discharging powers.

        Returns
        -------
        float
            Normalized action in [-1, 1] range.
        """
        SoC0 = self.model.SoC0.value
        if self.model.y[0].value == 1:
            ub = min(self.model.b_in_max.value, self.model.SoC_max.value - SoC0)
            a = self.model.b_in[0].value / ub
        else:
            ub = min(self.model.b_out_max.value, SoC0 - self.model.SoC_min.value)
            a = -self.model.b_out[0].value / ub

        return a


def initialize_mpc_param(
    env: EnergySystem | MARLEnv,
    N: int = 12,
    agent_idx: int = 0,
    mobility_reserve: float = 2.0,
    ev_demand_error_scale: float = 0.10,
    ev_demand_error_growth: float = 1.0,
    ev_demand_error_clip: float = 0.50,
    seed: int = 42,
) -> MPCParameters:
    """Initialize MPC parameters from environment.

    Parameters
    ----------
    env : EnergySystem | MARLEnv
        Environment containing battery parameters.
    N : int, optional
        Forecast horizon, by default 12.
    agent_idx : int, default=0
        Agent whose battery parameters are used.
    mobility_reserve : float, default=2.0
        Additional departure-energy reserve in kWh.
    ev_demand_error_scale : float, default=0.10
        Initial relative standard deviation of the EV-demand forecast.
    ev_demand_error_growth : float, default=1.0
        Linear forecast-error growth across one prediction horizon.
    ev_demand_error_clip : float, default=0.50
        Symmetric bound applied to sampled relative demand errors.
    seed : int, default=42
        Seed for reproducible forecast-error sampling.

    Returns
    -------
    MPCParameters
        Configured MPC parameters.
    """
    return MPCParameters(
        N=N,
        SoC_max=env.battery_capacity[agent_idx],
        SoC_min=env.battery_soc_min[agent_idx],
        b_in_max=env.b_in_max[agent_idx],
        b_out_max=env.b_out_max[agent_idx],
        ch_efficiency=env.ch_efficiency[agent_idx],
        disch_efficiency=env.disch_efficiency[agent_idx],
        battery_depreciation_cost=env.battery_depreciation_cost,
        mobility_reserve=mobility_reserve,
        ev_demand_error_scale=ev_demand_error_scale,
        ev_demand_error_growth=ev_demand_error_growth,
        ev_demand_error_clip=ev_demand_error_clip,
        seed=seed,
    )


if __name__ == "__main__":
    import json
    import os

    from course_package.environment import MARLEnv

    path = os.path.join(os.getcwd(), r"data\datasets\UK_IC\env_config")
    with open(path, "r", encoding="utf-8") as f:
        env_config: dict = json.load(f)

    env = MARLEnv(env_config)
    param = initialize_mpc_param(env, N=12)
    mpc = BatteryMPC(param)
    a_list: list[float] = []
    for _ in range(1):
        env.reset()
        done = False
        while not done:
            b_in, b_out, a = mpc.solve(env)
            actions = np.zeros(env.action_space.shape, dtype=np.float32)
            actions[0, 1] = a
            _, _, done, _, _ = env.step(actions)
            print(f"b_in: {b_in}, b_out: {b_out}, a: {a}")
            a_list.append(a)

    from course_package.utils.plots.trajectories import plot_battery

    plot_battery(env.env)
