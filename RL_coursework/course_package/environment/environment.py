"""
MARL environment for distributed energy resource (DER) optimization
in households with HVAC and EV charging systems.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box
from numpy.typing import NDArray

from course_package.environment.rewards import (
    battery_penalty_fcn,
    comfort_penalty_fcn,
    local_grid_consumption_fcn,
    thermal_smoothness_penalty_fcn,
)

# pylint: disable=no-member


class EnergySystem(gym.Env):
    """Multi-agent DER optimization environment.

    Parameters
    ----------
    env_config : Dict[str, Any]
        Environment configuration.
    seed : Optional[int], default=None
        Seed for reproducible episode sampling.

    Attributes
    ----------
    action_space : Box
        Joint action space for all agents.
    observation_space : Box
        Joint observation space for all agents.
    """

    SAFETY_FACTOR = 5.0

    def __init__(
        self,
        env_config: dict[str, Any],
        seed: int | None = None,
    ) -> None:
        """Initialize the multi-agent DER environment.

        Parameters
        ----------
        env_config : Dict[str, Any]
            Environment configuration.
        seed : Optional[int], default=None
            Seed for reproducible episode sampling.
        """
        super().__init__()

        self.episode_len = env_config["episode_len"]
        self.current_step = 0
        self.current_episode_start = 0

        self._parse_common_data(env_config)
        self._parse_agent_data(env_config)
        self._load_forecasting_data(env_config.get("forecasting_data"))

        self._setup_episode_boundaries()

        self.action_space = Box(
            low=np.array([[-1.0, -1.0]] * self.num_agents, dtype=np.float32),
            high=np.array([[1.0, 1.0]] * self.num_agents, dtype=np.float32),
            dtype=np.float32,
        )
        self.num_actions = self.action_space.shape[-1]

        self.action_space.seed(seed)
        self.reset(seed=seed)

        bounds = self._compute_observation_bounds(
            getattr(self, "prosumer_feature_bounds", {})
        )
        bounds = np.array(list(bounds.values()), dtype=np.float32).T
        self.observation_space = Box(
            low=bounds[:, 0, :],
            high=bounds[:, 1, :],
            dtype=np.float32,
        )
        self.obs_mapping = {
            key: idx for idx, key in enumerate(self.active_observations)
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the environment and sample a new episode boundary.

        Parameters
        ----------
        seed : Optional[int], default=None
            Optional reset seed.
        options : Optional[Dict], default=None
            Optional reset options, including ``episode_start``.

        Returns
        -------
        Tuple[Dict, Dict]
            Initial observations and info dictionary.
        """
        super().reset(seed=seed)

        episode_start = (options or {}).get("episode_start")

        if episode_start is None:
            episode_start = int(self.np_random.choice(self.valid_episode_starts))
        else:
            episode_start = int(episode_start)
            assert episode_start in self.valid_episode_starts, (
                f"Invalid episode_start {episode_start}."
            )

        self.current_episode_start = episode_start
        self.current_step = 0
        self.current_step_global = self.current_episode_start + self.current_step
        self.terminateds = {"__all__": False}

        # Actions
        self.actions_norm = np.zeros(
            (self.num_agents, self.episode_len, self.num_actions),
            dtype=np.float32,
        )
        self.pros_import = np.zeros(
            (self.num_agents, self.episode_len), dtype=np.float32
        )
        self.district_import = np.zeros((self.episode_len), dtype=np.float32)
        self.thermal_load = np.zeros_like(self.pros_import)
        self.heating_load = np.zeros_like(self.pros_import)
        self.cooling_load = np.zeros_like(self.pros_import)
        self.b_in = np.zeros_like(self.pros_import)
        self.b_out = np.zeros_like(self.pros_import)

        # States: battery
        self.ev_soc = np.zeros(
            (self.num_agents, self.episode_len + 1), dtype=np.float32
        )
        self.ev_soc[:, 0] = self.battery_capacity * 0.5
        self.soc_infeas_flag = np.zeros_like(self.pros_import).astype(bool)
        self.soc_violation = np.zeros_like(self.pros_import)

        # States: temperature
        self.indoor_temp = np.zeros(
            (self.num_agents, self.episode_len + 1), dtype=np.float32
        )
        self.setpoint_difference = np.zeros_like(self.pros_import)
        self.indoor_temp[:, 0] = np.array(
            [
                self.np_random.uniform(
                    low=self.indoor_temp_setpoint[i, self.current_episode_start]
                    - self.delta_temp,
                    high=self.indoor_temp_setpoint[i, self.current_episode_start]
                    + self.delta_temp,
                )
                for i in range(self.num_agents)
            ],
            dtype=np.float32,
        )
        self.temp_infeas_flag = np.zeros_like(self.soc_infeas_flag)
        self.hvac_mode = np.zeros_like(self.pros_import).astype(np.int8)

        # Costs
        self.grid_cost = np.zeros_like(self.pros_import)
        self.ev_penalty = np.zeros_like(self.pros_import)
        self.temp_penalty = np.zeros_like(self.pros_import)
        self.smooth_thermal = np.zeros_like(self.pros_import)
        self.smooth_battery = np.zeros_like(self.pros_import)

        self._update_temp_context()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: NDArray) -> tuple[dict, NDArray, bool, bool, dict]:
        """
        Execute one step of the environment.

        Parameters
        ----------
        action : NDArray
            Joint action array for all agents.

        Returns
        -------
        Tuple[Dict, NDArray, bool, bool, Dict]
            Next observations, reward vector, termination flag, truncation flag,
            and info.
        """
        action = self._validate_action(action)
        self._translate_action(action)

        self._compute_energy_balance()
        self._update_temperature()
        self._update_battery_soc()

        reward = self._compute_reward()

        self.current_step += 1
        self.current_step_global = self.current_episode_start + self.current_step
        terminated = self.current_step >= self.episode_len
        if not terminated:
            self._update_temp_context()
        obs = self._get_obs()

        truncated = False
        info = {"episode_start_idx": self.current_episode_start}

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment.

        Raises
        ------
        NotImplementedError
            Always raised because rendering is not implemented.
        """
        raise (NotImplementedError("Render method not implemented for EnergySystem."))

    def _check_comfort(self) -> np.ndarray:
        """Check whether current indoor temperatures remain within comfort bounds."""
        current_indoor_temp = self.indoor_temp[:, self.current_step]
        current_setpoint = self.indoor_temp_setpoint[:, self.current_step_global]

        lower_bound = current_setpoint - self.delta_temp
        upper_bound = current_setpoint + self.delta_temp

        in_comfort = (current_indoor_temp >= lower_bound) & (
            current_indoor_temp <= upper_bound
        )
        return in_comfort

    def _check_ev_soc(self) -> tuple[np.ndarray, np.ndarray]:
        """Check whether current EV state of charge can satisfy required demand."""
        current_soc = self.ev_soc[:, self.current_step]

        if hasattr(self, "ev_required_soc_departure"):
            required_soc = self._get_required_ev_soc(self.current_step)
            if hasattr(self, "ev_time_to_departure"):
                time_to_departure = self.ev_time_to_departure[
                    :, self.current_step_global
                ]
                max_charge_by_departure = self.b_in_max * time_to_departure
                soc_sufficient = current_soc + max_charge_by_departure >= required_soc
            else:
                soc_sufficient = current_soc >= required_soc
        else:
            soc_sufficient = np.ones(self.num_agents, dtype=bool)
            required_soc = np.zeros(self.num_agents, dtype=np.float32)

        return soc_sufficient, required_soc

    def _compute_ev_requirement(self) -> NDArray:
        """Compute minimum charging required to satisfy EV constraints."""
        soc_sufficient_flag, required_soc = self._check_ev_soc()
        current_soc = self.ev_soc[:, self.current_step]
        req_b_in = np.clip(
            required_soc - current_soc,
            0.0,
            self.b_in_max * self.ev_availability[:, self.current_step_global],
        )
        req_b_in = np.where(soc_sufficient_flag, 0.0, req_b_in)
        return req_b_in

    def _compute_thermal_load_bounds(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        current_indoor_temp, next_setpoint = self._get_temp_context(self.current_step)
        external_temp = self.external_temp[self.current_step_global]
        solar_ghi = self.solar_ghi[self.current_step_global]

        lower_bound = next_setpoint - self.delta_temp
        upper_bound = next_setpoint + self.delta_temp

        too_cold = current_indoor_temp < lower_bound
        too_hot = current_indoor_temp > upper_bound
        comfort = ~(too_cold | too_hot)

        def load_to_target(target_temp):
            return (
                self.building_thermal_capacitance * (target_temp - current_indoor_temp)
                - (external_temp - current_indoor_temp)
                / self.building_thermal_resistance
                - solar_ghi * self.building_solar_gain
            )

        q_lower = load_to_target(lower_bound)
        q_upper = load_to_target(upper_bound)

        heating_load_lb = np.where(too_cold, q_lower, 0.0)
        heating_load_ub = np.where(too_cold | comfort, q_upper, 0.0)

        cooling_load_lb = np.where(too_hot, q_upper, 0.0)
        cooling_load_ub = np.where(too_hot | comfort, q_lower, 0.0)

        heating_load_lb = np.clip(heating_load_lb, 0.0, self.rated_power)
        heating_load_ub = np.clip(heating_load_ub, 0.0, self.rated_power)
        cooling_load_lb = np.clip(cooling_load_lb, -self.rated_power, 0.0)
        cooling_load_ub = np.clip(cooling_load_ub, -self.rated_power, 0.0)

        return heating_load_lb, cooling_load_lb, heating_load_ub, cooling_load_ub

    def _compute_energy_balance(self) -> None:
        """Update per-agent and district electricity import balances."""
        self.pros_import[:, self.current_step] = (
            self.heating_load[:, self.current_step] / self.cop
            + self.cooling_load[:, self.current_step] / self.cop
            + (self.b_in[:, self.current_step] / self.ch_efficiency)
            - (self.b_out[:, self.current_step] * self.disch_efficiency)
            - self.pv_generation[:, self.current_step_global]
            + self.domestic_load[:, self.current_step_global]
        )
        self.district_import[self.current_step] = np.sum(
            self.pros_import[:, self.current_step]
        )

    def _compute_observation_bounds(
        self, manual_bounds: dict[str, tuple[float, float]] | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Compute observation bounds for active observations.

        ``manual_bounds`` maps a feature name to fixed ``(low, high)`` reference
        values (used for static prosumer features such as equipment capacities).
        """
        manual_bounds = manual_bounds or {}
        bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for key in self.active_observations:
            if not hasattr(self, key):
                continue
            value = getattr(self, key)
            if key in manual_bounds:
                low, high = manual_bounds[key]
            elif (
                isinstance(value, np.ndarray) and value.shape[-1] == self.history_len
            ):  # Get bounds from time-series data
                if value.ndim == 1:
                    low = np.nanmin(value)
                    high = np.nanmax(value)
                else:
                    low = np.nanmin(value, axis=-1)
                    high = np.nanmax(value, axis=-1)

            else:  # Handle observations that are not directly backed by time-series data
                if key == "ev_soc":
                    low = np.zeros_like(self.battery_capacity)
                    high = self.battery_capacity
                elif key == "setpoint_difference":
                    low = (
                        -self.SAFETY_FACTOR
                        * self.delta_temp
                        * np.ones(self.num_agents, dtype=np.float32)
                    )
                    high = (
                        self.SAFETY_FACTOR
                        * self.delta_temp
                        * np.ones(self.num_agents, dtype=np.float32)
                    )
                elif key == "indoor_temp":
                    low = np.minimum(
                        bounds["external_temp"][0],
                        np.min(self.indoor_temp_setpoint, axis=-1)
                        - self.SAFETY_FACTOR * self.delta_temp,
                    )
                    high = np.maximum(
                        bounds["external_temp"][1],
                        np.max(self.indoor_temp_setpoint, axis=-1)
                        + self.SAFETY_FACTOR * self.delta_temp,
                    )
                elif key == "hvac_mode":
                    low = -1 * np.ones(self.num_agents, dtype=np.float32)
                    high = np.ones(self.num_agents, dtype=np.float32)

            bounds[key] = (
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
            )

        for key, (low, high) in bounds.items():
            if low.shape == ():
                low = np.full((self.num_agents,), low.item(), dtype=np.float32)
            if high.shape == ():
                high = np.full((self.num_agents,), high.item(), dtype=np.float32)
            bounds[key] = [low, high]

        missing = [key for key in self.active_observations if key not in bounds]
        assert not missing, "Missing bounds for active observations: " + ", ".join(
            missing
        )

        return bounds

    def _compute_reward(self) -> NDArray:
        """Compute the weighted per-agent reward."""
        reward = np.zeros(self.num_agents, dtype=np.float32)

        self.grid_cost[:, self.current_step] = local_grid_consumption_fcn(self)
        self.temp_penalty[:, self.current_step] = comfort_penalty_fcn(self)
        self.ev_penalty[:, self.current_step] = battery_penalty_fcn(self)
        self.smooth_thermal[:, self.current_step] = thermal_smoothness_penalty_fcn(self)
        self.smooth_battery[:, self.current_step] = 0.0

        reward = (
            self.reward_weights["r_eco"] * self.grid_cost[:, self.current_step]
            + self.reward_weights["r_comfort"] * self.temp_penalty[:, self.current_step]
            + self.reward_weights["r_battery"] * self.ev_penalty[:, self.current_step]
            + self.reward_weights["r_smooth_thermal"]
            * self.smooth_thermal[:, self.current_step]
            + self.reward_weights["r_smooth_battery"]
            * self.smooth_battery[:, self.current_step]
        )

        return reward

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build observation dictionary for current episode timestep."""
        global_idx = min(
            self.current_episode_start + self.current_step, self.history_len - 1
        )
        episode_idx = min(self.current_step, self.episode_len - 1)

        obs = dict.fromkeys(self.active_observations)
        for key in obs:
            if hasattr(self, key):
                value = self.__getattribute__(key)
                if value.shape[-1] == self.history_len:
                    if key in ["indoor_temp_setpoint"]:
                        next_global_idx = min(global_idx + 1, self.history_len - 1)
                        value = value[:, next_global_idx]
                    else:
                        value = value[..., global_idx]
                elif value.shape[-1] == self.episode_len:
                    value = value[..., episode_idx]
                elif value.shape[-1] == self.episode_len + 1:
                    value = value[..., self.current_step]
            else:
                assert False, (
                    f"Observation '{key}' not found in environment attributes."
                )
            obs[key] = value
        return obs

    def _get_required_ev_soc(self, timestep: int) -> np.ndarray:
        """Return required EV SoC, enforcing terminal SoC consistency."""
        idx = min(
            self.current_episode_start + timestep,
            self.ev_required_soc_departure.shape[-1] - 1,
        )
        required_soc = self.ev_required_soc_departure[:, idx]
        required_soc = np.where(
            timestep < self.episode_len - 1, required_soc, self.ev_soc[:, 0]
        )
        required_soc *= self.ev_availability[:, self.current_step_global]
        return required_soc

    def _get_temp_context(self, timestep: int) -> tuple[np.ndarray, np.ndarray]:
        """Return current indoor temperature and next setpoint context."""
        current_indoor_temp = self.indoor_temp[:, timestep]
        next_timestep_global = min(
            self.current_episode_start + timestep + 1,
            self.indoor_temp_setpoint.shape[-1] - 1,
        )
        next_setpoint = self.indoor_temp_setpoint[:, next_timestep_global]
        return current_indoor_temp, next_setpoint

    def _parse_common_data(self, env_config: dict[str, Any]) -> None:
        """Parse and load common pricing/weather data and observation flags."""
        for key, value in env_config.items():
            if key not in ["agents", "weather_data", "pricing_data", "observations"]:
                self.__setattr__(key, value)

        data_dir = Path(self.data_dir)
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        self.data_dir = data_dir.resolve()

        if not hasattr(self, "reward_weights"):
            self.reward_weights = {
                k: 1.0
                for k in [
                    "r_eco",
                    "r_comfort",
                    "r_battery",
                    "r_smooth_thermal",
                    "r_smooth_battery",
                ]
            }

        self.observation_names = []
        self.active_observations = []
        for key, flag in env_config["observations"].items():
            self.observation_names.append(key)
            if flag["active"]:
                self.active_observations.append(key)

        pricing_path = self.data_dir / env_config.get("pricing_data", None)
        pricing = pd.read_csv(pricing_path)
        pricing["datetime"] = pd.to_datetime(pricing["datetime"])
        self.__setattr__("datetime", pricing["datetime"].values)
        pricing.set_index("datetime", inplace=True)
        pricing = pricing.loc[self.start_date : self.end_date]
        for col in pricing.columns:
            if col in self.observation_names:
                data = np.array(pricing[col].values, dtype=np.float32)
                self.__setattr__(col, data)

        weather_path = self.data_dir / env_config.get("weather_data", None)
        weather = pd.read_csv(weather_path)
        weather["datetime"] = pd.to_datetime(weather["datetime"])
        weather.set_index("datetime", inplace=True)
        weather = weather.loc[self.start_date : self.end_date]
        for col in weather.columns:
            if col in self.observation_names:
                data = np.array(weather[col].values, dtype=np.float32)
                self.__setattr__(col, data)

    def _parse_agent_data(self, env_config: dict[str, Any]) -> None:
        """Parse agent parameters and time-series arrays from configuration."""
        agents_config = env_config.get("agents", {})
        self.agent_ids = list(agents_config.keys())
        self.num_agents = len(self.agent_ids)

        param_values = {}
        for agent_id in self.agent_ids:
            for item_key, item_value in agents_config[agent_id].items():
                if item_key == "data_file":
                    continue
                if isinstance(item_value, dict):
                    for nested_key, nested_value in item_value.items():
                        if nested_key not in param_values:
                            param_values[nested_key] = []
                        param_values[nested_key].append(nested_value)
                elif isinstance(item_value, (int, float)):
                    if item_key not in param_values:
                        param_values[item_key] = []
                    param_values[item_key].append(item_value)

        for param_name, values in param_values.items():
            setattr(self, param_name, np.array(values, dtype=np.float32))

        data_file_arrays = {}
        for agent_id in self.agent_ids:
            data_file_path = agents_config[agent_id]["data_file"]
            data_file_path = self.data_dir / data_file_path
            df = pd.read_csv(data_file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df = df.loc[self.start_date : self.end_date]
            for column_name in df.columns:
                if column_name in self.observation_names:
                    if column_name not in data_file_arrays:
                        data_file_arrays[column_name] = []
                    data_file_arrays[column_name].append(df[column_name].values)

        for column_name, agent_data_list in data_file_arrays.items():
            try:
                setattr(self, column_name, np.array(agent_data_list, dtype=np.float32))
            except ValueError as e:
                raise ValueError(
                    f"Error converting data for column '{column_name}' to numpy array. "
                    f"agent data list shapes: {[data.shape for data in agent_data_list]}. "
                    "Check that all agent data files have the same number of timesteps and that the column exists in all files."
                ) from e

    def _load_forecasting_data(self, forecasting_data_dir: str | Path | None) -> None:
        """Load optional RNN forecast arrays for observation fields.

        Parameters
        ----------
        forecasting_data_dir : str | Path
            Directory containing the forecasting data files.
        """
        if forecasting_data_dir is None:
            return

        forecasting_path = Path(forecasting_data_dir)
        if not forecasting_path.is_absolute():
            forecasting_path = self.data_dir / forecasting_path
        if not forecasting_path.exists():
            raise FileNotFoundError(
                f"Forecasting data file not found: {forecasting_path}. "
                "Generate it with course_package.environment.forecasting first."
            )

        forecast_keys = [
            "external_temp_pred1",
            "external_temp_pred2",
            "grid_cost_coeff_pred1",
            "grid_cost_coeff_pred2",
            "domestic_load_pred1",
            "domestic_load_pred2",
        ]
        with np.load(forecasting_path, allow_pickle=False) as forecast_data:
            datetime = pd.Series(pd.to_datetime(forecast_data["datetime"]))
            start = pd.Timestamp(self.start_date)
            end = pd.Timestamp(self.end_date)
            if datetime.dt.tz is not None and start.tzinfo is None:
                start = start.tz_localize(datetime.dt.tz)
                end = end.tz_localize(datetime.dt.tz)
            mask = (datetime >= start) & (datetime <= end)
            idx = np.flatnonzero(mask.to_numpy())

            expected_len = len(self.grid_cost_coeff)
            for key in forecast_keys:
                if key not in forecast_data:
                    continue
                data = np.array(forecast_data[key], dtype=np.float32)

                if data.ndim == 1:
                    data = data[idx]
                elif data.shape[-1] == len(datetime):
                    data = data[..., idx]
                elif data.shape[0] == len(datetime):
                    data = data[idx, ...].T

                if data.ndim == 2 and key.startswith("domestic_load"):
                    data = data[: self.num_agents, :]
                    assert data.shape[0] == self.num_agents, (
                        f"Forecast '{key}' has {data.shape[0]} agents, "
                        f"expected {self.num_agents}."
                    )
                assert data.shape[-1] == expected_len, (
                    f"Forecast '{key}' has {data.shape[-1]} timesteps, "
                    f"expected {expected_len}."
                )
                setattr(self, key, data)

    def _setup_episode_boundaries(self) -> None:
        """Pre-compute non-overlapping fixed-length episode start indices."""
        self.history_len = len(self.grid_cost_coeff)

        assert self.history_len >= self.episode_len, (
            f"Data length ({self.history_len} timesteps) must be >= "
            f"episode_len ({self.episode_len} timesteps). "
            f"Please expand date range in env_config or reduce episode_len."
        )
        valid_episode_starts = np.arange(
            0,
            self.history_len - self.episode_len + 1,
            self.episode_len,
            dtype=np.int64,
        )
        if hasattr(self, "episode_starts"):
            requested_starts = np.array(self.episode_starts, dtype=np.int64)
            if requested_starts.ndim != 1 or requested_starts.size == 0:
                raise ValueError("episode_starts must be a non-empty 1D sequence.")
            if len(set(requested_starts.tolist())) != requested_starts.size:
                raise ValueError("episode_starts must not contain duplicates.")

            invalid_starts = sorted(
                set(requested_starts.tolist()) - set(valid_episode_starts.tolist())
            )
            if invalid_starts:
                raise ValueError(
                    "Configured episode_starts must be valid non-overlapping "
                    f"episode boundaries. Invalid starts: {invalid_starts[:10]}"
                )
            valid_episode_starts = requested_starts

        self.valid_episode_starts = valid_episode_starts
        self.num_possible_episodes = len(self.valid_episode_starts)

    def _translate_action(self, action: NDArray) -> None:
        """Translate normalized actions into thermal and battery control variables."""
        thermal_action = action[:, 0]
        battery_action = action[:, 1]

        heating_load, cooling_load = self._translate_action_thermal_v2(thermal_action)
        b_in, b_out = self._translate_action_battery(battery_action)

        self.heating_load[:, self.current_step] = heating_load
        self.cooling_load[:, self.current_step] = cooling_load
        self.thermal_load[:, self.current_step] = heating_load - cooling_load
        self.b_in[:, self.current_step] = b_in
        self.b_out[:, self.current_step] = b_out

    def _translate_action_battery(self, action: NDArray) -> tuple[NDArray, NDArray]:
        """Map normalized battery actions to charge and discharge powers."""
        charge_room = np.maximum(
            0.0, self.battery_capacity - self.ev_soc[:, self.current_step]
        )
        discharge_room = np.maximum(
            0.0, self.ev_soc[:, self.current_step] - self.battery_soc_min
        )

        b_in_lb = self._compute_ev_requirement()
        b_in_ub = np.minimum(
            self.b_in_max * self.ev_availability[:, self.current_step_global],
            charge_room,
        )
        b_in_ub = np.where(b_in_lb > b_in_ub, b_in_lb, b_in_ub)

        b_out_lb = np.zeros_like(b_in_lb)
        b_out_ub = np.minimum(
            self.b_out_max * self.ev_availability[:, self.current_step_global],
            discharge_room,
        )
        b_out_ub = np.where(b_in_lb > 0, 0.0, b_out_ub)

        b_in = np.maximum(action * (b_in_ub - b_in_lb), 0.0) + b_in_lb
        b_out = np.maximum(action * (b_out_ub - b_out_lb) * (-1.0), 0.0) + b_out_lb
        return b_in, b_out

    def _translate_action_thermal(self, action: NDArray) -> tuple[NDArray, NDArray]:
        """Map normalized thermal actions to feasible heating and cooling loads."""
        heating_load_lb, cooling_load_lb, heating_load_ub, cooling_load_ub = (
            self._compute_thermal_load_bounds()
        )
        heating_load = np.maximum(action * (heating_load_ub - heating_load_lb), 0.0)
        cooling_load = np.maximum(action * (cooling_load_ub - cooling_load_lb), 0.0)
        heating_load += heating_load_lb
        cooling_load += np.abs(cooling_load_lb)

        return heating_load, cooling_load

    def _translate_action_thermal_v2(self, action: NDArray) -> tuple[NDArray, NDArray]:
        """Map normalized thermal actions using HVAC mode gating constraints."""
        _, _, hub, cub = self._compute_thermal_load_bounds()

        u_heat = np.maximum(action, 0.0)
        u_cool = np.maximum(-1 * action, 0.0)
        u_heat = np.where(self.hvac_mode[:, self.current_step] == -1, 0.0, u_heat)
        u_cool = np.where(self.hvac_mode[:, self.current_step] == +1, 0.0, u_cool)

        heating_load = u_heat * self.rated_power * self.cop
        cooling_load = u_cool * self.rated_power * self.cop

        heating_load = np.minimum(heating_load, hub)
        cooling_load = np.minimum(cooling_load, -1 * cub)

        return heating_load, cooling_load

    def _update_battery_soc(self) -> None:
        """Update EV state of charge and infeasibility penalty terms."""
        next_soc = (
            self.ev_soc[:, self.current_step]
            + self.b_in[:, self.current_step]
            - self.b_out[:, self.current_step]
            - self.ev_trip_demand[:, self.current_step_global]
        )
        self.ev_soc[:, self.current_step + 1] = np.clip(
            next_soc, self.battery_soc_min, self.battery_capacity
        )
        self.soc_violation[:, self.current_step] = np.maximum(0.0, -next_soc)

    def _update_temperature(self) -> None:
        """Update indoor temperatures using thermal dynamics."""
        current_indoor_temp = self.indoor_temp[:, self.current_step]
        external_temp = self.external_temp[self.current_step_global]
        solar_ghi = self.solar_ghi[self.current_step_global]

        temp_change = (
            self.heating_load[:, self.current_step]
            - self.cooling_load[:, self.current_step]
            + (external_temp - current_indoor_temp) / self.building_thermal_resistance
            + solar_ghi * self.building_solar_gain
        ) / self.building_thermal_capacitance

        self.indoor_temp[:, self.current_step + 1] = current_indoor_temp + temp_change

    def _update_temp_context(self) -> None:
        """Update HVAC mode and setpoint deviation for current timestep."""
        current_indoor_temp, next_setpoint = self._get_temp_context(self.current_step)

        hvac_mode = np.zeros(self.num_agents, dtype=np.int8)
        hvac_mode = np.where(
            current_indoor_temp < next_setpoint - self.delta_temp, +1, hvac_mode
        )
        hvac_mode = np.where(
            current_indoor_temp > next_setpoint + self.delta_temp, -1, hvac_mode
        )

        self.hvac_mode[:, self.current_step] = hvac_mode
        self.setpoint_difference[:, self.current_step] = (
            current_indoor_temp - next_setpoint
        )

    def _validate_action(self, action: NDArray) -> None:
        if self.current_step >= self.episode_len:
            raise RuntimeError("Episode has terminated. Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, got {action.shape}"
            )
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action, self.action_space.low, self.action_space.high)
