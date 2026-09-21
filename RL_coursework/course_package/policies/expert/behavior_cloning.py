"""Definition of the expert controller."""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .mpc import BatteryMPC
from .pid import TemperaturePID


class Expert:
    """
    Expert controller for behavior cloning.

    Attributes
    ----------
    mpc : BatteryMPC
        Model predictive control (MPC) expert for battery actions.
    pid : TemperaturePID
        Temperature PID expert for thermal actions.
    """

    def __init__(
        self,
        mpc_config: object | list[object] | tuple[object, ...],
        pid_config: object,
        gain_dir: str | Path,
        information: Literal["perfect", "forecast"] = "forecast",
    ) -> None:
        """Compose MPC and PID expert controllers.

        Parameters
        ----------
        mpc_config : object or sequence of objects
            MPC configuration object. A sequence may be provided to use one
            battery MPC per agent.
        pid_config : object
            PID configuration object.
        gain_dir : str | Path
            Path to PID gains file.
        information : Literal["perfect", "forecast"], default="forecast"
            Information mode for expert forecasts.
        """
        if isinstance(mpc_config, (list, tuple)):
            self.mpcs = [
                BatteryMPC(config, information=information) for config in mpc_config
            ]
        else:
            self.mpcs = [BatteryMPC(mpc_config, information=information)]
        self.mpc = self.mpcs[0]
        self.pid = TemperaturePID(
            pid_config, gain_dir=gain_dir, information=information
        )

    def __call__(self, env: object, agent_idx: int | None = None) -> NDArray:
        """Return expert actions for MARLEnv.

        Parameters
        ----------
        env : object
            Environment instance.
        agent_idx : int, optional
            Agent index. If omitted, return the joint action matrix for all agents.

        Returns
        -------
        NDArray
            Joint action matrix ``(num_agents, 2)`` or one action vector
            ``[thermal, battery]`` when ``agent_idx`` is provided.
        """
        return self.act(env, agent_idx=agent_idx)

    def act(self, env: object, agent_idx: int | None = None) -> NDArray:
        """Compute clipped expert thermal and battery actions for MARLEnv.

        Parameters
        ----------
        env : object
            Environment instance.
        agent_idx : int, optional
            Agent index. If omitted, actions are computed for every agent.

        Returns
        -------
        NDArray
            Joint action matrix ``(num_agents, 2)`` or one action vector
            ``[thermal, battery]`` when ``agent_idx`` is provided.
        """
        if agent_idx is None:
            return self.act_all(env)

        return self.act_agent(env, agent_idx=agent_idx)

    def act_agent(self, env: object, agent_idx: int = 0) -> NDArray:
        """Compute the expert action for one agent."""
        a_battery = self._mpc_for(agent_idx)(env, agent_idx=agent_idx)
        a_thermal = self.pid(env, agent_idx=agent_idx)
        return np.clip(np.array([a_thermal, a_battery], dtype=np.float32), -1.0, 1.0)

    def act_all(self, env: object) -> NDArray:
        """Compute the joint action matrix expected by ``MARLEnv.step``."""
        return np.asarray(
            [self.act_agent(env, agent_idx=i) for i in range(env.num_agents)],
            dtype=np.float32,
        )

    def reset(self) -> None:
        """Reset expert controller internal state."""
        self.pid.reset_error()
        for mpc in self.mpcs:
            mpc.reset_rng()

    def _mpc_for(self, agent_idx: int) -> BatteryMPC:
        """Return the MPC assigned to one agent."""
        if len(self.mpcs) == 1:
            return self.mpcs[0]
        if agent_idx >= len(self.mpcs):
            raise ValueError(
                f"Expected an MPC configuration for agent {agent_idx}, "
                f"but only {len(self.mpcs)} were provided."
            )
        return self.mpcs[agent_idx]
