"""The digital lab: a simulated mammalian cell culture experiment budgeted by cost and time."""

from digital_lab.constants import (
    BATCH_SIZES,
    BUDGET,
    INITIAL_DESIGN,
    MAX_EXPERIMENTS,
    N_CELLTYPES,
    REACTOR_COSTS,
    SCORE_CHECKPOINTS,
    SEEDS,
    TIME_LIMIT_SECONDS,
)
from digital_lab.lab import DigitalLab
from digital_lab.notebook import (
    check_my_algorithm,
    list_versions,
    run_comparison,
    save_version,
    show_results,
    show_scale_choices,
)

__all__ = [
    "BATCH_SIZES",
    "BUDGET",
    "INITIAL_DESIGN",
    "MAX_EXPERIMENTS",
    "N_CELLTYPES",
    "REACTOR_COSTS",
    "SCORE_CHECKPOINTS",
    "SEEDS",
    "TIME_LIMIT_SECONDS",
    "DigitalLab",
    "check_my_algorithm",
    "list_versions",
    "run_comparison",
    "save_version",
    "show_results",
    "show_scale_choices",
]
