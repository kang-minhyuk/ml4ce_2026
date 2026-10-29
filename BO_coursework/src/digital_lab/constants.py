"""Default settings of the coursework problem.

The benchmark in `assessment/benchmark/config.py` can override the lab
parameters (config dir, budget, costs, time limit, seeds), so algorithms must
read these from the `lab` object they are given, never hard-code them.
"""

from pathlib import Path

DEFAULT_LAB_CONFIG_DIR = Path(__file__).resolve().parent / "lab_config"
N_CELLTYPES = 30  # clones
CELLTYPES = [f"celltype_{i}" for i in range(1, N_CELLTYPES + 1)]
LAB_CONFIG_FILENAMES = {
    "MTP": "MTP.json",
    "MBR": "MBR.json",
    "Pilot": "Pilot.json",
    "noise_parameters": "noise_parameters.json",
    **{name: f"{name}.json" for name in CELLTYPES},
}

# Every lab(x) call runs one batch at one scale, with exactly this many rows:
# a 24-well plate, 4 parallel mini bioreactors, or 1 pilot reactor.
BATCH_SIZES = {"MTP": 24, "MBR": 4, "Pilot": 1}
# The plate (screening) scale is constrained: every well shares temperature, pH and
# the starting medium, there are no feeds and no temperature/pH shifts. Only the clone
# differs per well.
SCREENING_REACTOR = "MTP"
PLATE_SHARED_COLUMNS = (2, 3, 4, 5)  # temperature, pH, initial glucose, initial glutamine
FEED_COLUMNS = (6, 7, 8, 9)
# Temperature and pH after each feed (the shift at that feed time): columns 10-13.
SHIFT_COLUMNS = {10: 2, 11: 3, 12: 2, 13: 3}  # shift column -> the initial column a plate must repeat
FEED_TIMES = (120.0, 240.0)  # h, day 5 and day 10
# Initial design run (and paid for) before the algorithm starts: number of batches per scale.
INITIAL_DESIGN = {"MTP": 1, "MBR": 1, "Pilot": 0}
N_FEATURES = 14
SEED = 42
N_SEEDS = 5
SEEDS = [SEED + i for i in range(N_SEEDS)]
BUDGET = 25000  # $
# Maximum number of experiments per run, including the initial design. The lab
# stops (like when the budget runs out) once this many have been run.
MAX_EXPERIMENTS = 300
# Wall-clock limit for one run of an algorithm (one seed), in seconds. At the
# limit the algorithm is stopped and graded on what it found so far.
TIME_LIMIT_SECONDS = 5 * 60
# Score of one run: the best Pilot titer found once 33 %, 66 % and 100 % of the budget
# is spent, summed. A checkpoint with no Pilot run yet contributes 0.
SCORE_CHECKPOINTS = (0.33, 0.66, 1.0)

REACTOR_TYPES = ["MTP", "MBR", "Pilot"]

REACTOR_COSTS = {
    "MTP": 10,  # $ per well / reactor
    "MBR": 500,
    "Pilot": 2000,
}

# Multi-fidelity setup: MTP/MBR are cheap, lower-fidelity exploratory reactors.
# Only titer measured in this reactor counts toward the objective.
OBJECTIVE_REACTOR = "Pilot"

# TODO: these are placeholder ranges — replace with the real experimental bounds.
BOUNDS = {
    "temperature": (30.0, 40.0),  # deg C
    "ph_value": (6.0, 8.0),
    "glc": (0.0, 100.0),  # mM, used for initial conditions and both feeds
    "gln": (0.0, 50.0),  # mM, used for initial conditions and both feeds
}

# Bounds for the 12 continuous input columns, in the order every input row uses
# after the (reactor, celltype) columns: temperature, pH, initial/feed-1/feed-2
# glucose+glutamine pairs, then temperature and pH from feed 1 on and from feed 2 on.
# A plate repeats the starting temperature and pH in the last four (no shifts).
CONTINUOUS_BOUNDS = [
    BOUNDS["temperature"],
    BOUNDS["ph_value"],
    BOUNDS["glc"],  # initial glc
    BOUNDS["gln"],  # initial gln
    BOUNDS["glc"],  # feed 1 glc
    BOUNDS["gln"],  # feed 1 gln
    BOUNDS["glc"],  # feed 2 glc
    BOUNDS["gln"],  # feed 2 gln
    BOUNDS["temperature"],  # temperature after feed 1
    BOUNDS["ph_value"],  # pH after feed 1
    BOUNDS["temperature"],  # temperature after feed 2
    BOUNDS["ph_value"],  # pH after feed 2
]

# Only these top-level imports are allowed in a submitted my_algorithm.py.
ALLOWED_IMPORTS = {"numpy", "scipy", "random", "sobol_seq", "time", "datetime", "typing"}
