"""Definition of reward functions for the DER environment."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from course_package.environment import EnergySystem


def local_grid_consumption_fcn(env: "EnergySystem") -> NDArray:
    """Compute per-agent local grid consumption reward term.

    Parameters
    ----------
    env : EnergySystem
        Environment containing current import and price signals.

    Returns
    -------
    NDArray
        Per-agent grid consumption reward component.
    """
    p = env.pros_import[:, env.current_step]
    # return -1 * np.maximum(0, p) * env.grid_cost_coeff[env.current_step_global]
    return -1 * p * env.grid_cost_coeff[env.current_step_global]


def district_grid_consumption(env: "EnergySystem") -> NDArray:
    """Compute the district grid consumption reward term.

    Parameters
    ----------
    env : EnergySystem
        Environment containing import and grid parameters.

    Returns
    -------
    NDArray
        Per-agent district grid consumption reward component.
    """
    p = env.pros_import[:, env.current_step]
    g = env.district_import[env.current_step]
    grid_losses = env.grid_resistance / (env.grid_voltage**2) * g**2
    return (p**2) * max(0.0, g + grid_losses) * np.sign(-1 * p)


def marginal_contribution(env: "EnergySystem") -> NDArray:
    """Compute the marginal contribution reward term.

    Parameters
    ----------
    env : EnergySystem
        Environment containing current imports and cost parameters.

    Returns
    -------
    NDArray
        Per-agent marginal contribution reward component.
    """
    p = env.pros_import[:, env.current_step]
    g = env.district_import[env.current_step]

    price = env.grid_cost_coeff[env.current_step_global]
    C_d = env.distribution_charge
    C_s = env.battery_depreciation_cost
    K = env.grid_resistance / (env.grid_voltage**2)

    marginal_grid_cost = -price * (p + 2 * K * g * p - K * p**2)
    export_cost = -C_d * np.minimum(0.0, p)
    battery_cost = -C_s * (
        env.b_in[:, env.current_step] - env.b_out[:, env.current_step]
    )

    return marginal_grid_cost + export_cost + battery_cost


def comfort_penalty_fcn(env: "EnergySystem") -> NDArray:
    """Compute the indoor temperature comfort penalty term.

    Parameters
    ----------
    env : EnergySystem
        Environment containing indoor temperature and setpoint state.

    Returns
    -------
    NDArray
        Per-agent comfort penalty.
    """
    indoor_temp = env.indoor_temp[:, env.current_step + 1]
    setpoint = env.indoor_temp_setpoint[
        :, min(env.current_step_global + 1, env.indoor_temp_setpoint.shape[-1] - 1)
    ]
    temp_diff = np.abs(indoor_temp - setpoint)

    return -1 * (temp_diff / env.delta_temp) ** 2  # <=0


def battery_penalty_fcn(
    env: "EnergySystem", critical_time: int = 3, slope: float = 1.5
) -> NDArray:
    """Compute the EV battery state-of-charge penalty term.

    Parameters
    ----------
    env : EnergySystem
        Environment containing EV state and departure requirements.
    critical_time : int, default=3
        Time-to-departure value used in the urgency function.
    slope : float, default=1.5
        Slope used in the urgency function.

    Returns
    -------
    NDArray
        Per-agent battery penalty.
    """
    required_soc = env._get_required_ev_soc(env.current_step + 1)
    ev_soc = env.ev_soc[:, env.current_step + 1]
    time_to_departure_idx = min(
        env.current_step_global + 1, env.ev_time_to_departure.shape[-1] - 1
    )
    time_to_departure = env.ev_time_to_departure[:, time_to_departure_idx]

    urgency = 1 / (
        1 + np.exp(slope * (time_to_departure - critical_time))
    )  # urgency \in [0, 1]
    deficit = np.maximum(0.0, required_soc - ev_soc)

    penalty = np.divide(
        deficit, required_soc, out=np.zeros_like(deficit), where=required_soc > 0.0
    )
    return -1 * penalty * urgency  # <= 0


def thermal_smoothness_penalty_fcn(env: "EnergySystem") -> NDArray:
    """Compute the thermal load smoothness penalty term.

    Parameters
    ----------
    env : EnergySystem
        Environment containing thermal load history.

    Returns
    -------
    NDArray
        Per-agent thermal smoothness penalty.
    """
    if env.current_step == 0:
        return np.zeros(env.num_agents, dtype=np.float32)

    current_thermal_load = env.thermal_load[:, env.current_step]
    previous_thermal_load = env.thermal_load[:, env.current_step - 1]
    load_diff = np.abs(current_thermal_load - previous_thermal_load) / (env.rated_power)

    return -1 * load_diff
