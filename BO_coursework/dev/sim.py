"""Analysis-only helpers: noise-free simulator for a scenario (never imported by algorithms)."""
import dataclasses
import json
from pathlib import Path

import numpy as np
from dbiolab.mammalian_cellculture_simulation import (
    CellType, FeedingEvent, MammalianRecipe, NoiseParameters, ReactorType,
)
from dbiolab.mammalian_cellculture_simulation.data_models import MammalianDataset

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from digital_lab.lab import _ShiftingReactor, _concentrations  # noqa: E402

SUITE = ROOT / "scenario_suite"
PRACTICE_DIR = ROOT / "src" / "digital_lab" / "lab_config"


OFFICIAL = Path(__file__).resolve().parent / "scen_official"


def scenario_path(sid):
    if sid >= 10000:
        return OFFICIAL / f"scenario_{sid - 10000:04d}.json"
    return SUITE / "scenarios" / f"scenario_{sid:04d}.json"


def load_scenario(sid):
    return json.loads(scenario_path(sid).read_text())


def practice_scenario():
    """The practice lab expressed as a scenario dict."""
    from digital_lab import constants as C
    clones = [json.loads((PRACTICE_DIR / f"{n}.json").read_text()) for n in C.CELLTYPES]
    reactors = {r: json.loads((PRACTICE_DIR / f"{r}.json").read_text()) for r in C.REACTOR_TYPES}
    return {"id": -1, "clones": clones, "reactors": reactors, "bounds": [list(b) for b in C.CONTINUOUS_BOUNDS],
            "noise": json.loads((PRACTICE_DIR / "noise_parameters.json").read_text()),
            "lab": {"budget": C.BUDGET, "max_experiments": C.MAX_EXPERIMENTS, "reactor_costs": C.REACTOR_COSTS,
                    "batch_sizes": C.BATCH_SIZES, "initial_design": C.INITIAL_DESIGN}}


class Sim:
    """Noise-free titer f(reactor, clone_index, x12)."""

    def __init__(self, sc):
        self.sc = sc
        self.clones = [CellType(**{k: v for k, v in c.items()}) for c in sc["clones"]]
        self.names = [c["name"] for c in sc["clones"]]
        self.reactors = {k: ReactorType(**v) for k, v in sc["reactors"].items()}
        self.noise = NoiseParameters()
        self.bounds = np.array(sc["bounds"], float)

    def __call__(self, reactor, ci, x):
        x = [float(v) for v in x]
        init = _concentrations(x[2], x[3]); init.x_total = 8e5; init.x_viable = 8e5
        recipe = MammalianRecipe(
            name="r", celltype=self.clones[ci], temperature=x[0], ph_value=x[1], initial_conditions=init,
            feeding_strategy=[FeedingEvent(time_point=120.0, feed_concentrations=_concentrations(x[4], x[5])),
                              FeedingEvent(time_point=240.0, feed_concentrations=_concentrations(x[6], x[7]))],
            runtime=336.0, measurement_intervals=12.0, noise_parameters=self.noise)
        reactor_obj = _ShiftingReactor(self.reactors[reactor], [(x[0], x[1]), (x[8], x[9]), (x[10], x[11])])
        return float(reactor_obj.run_dataset(MammalianDataset([recipe]), seed=None).data[0].c_product[-1])
