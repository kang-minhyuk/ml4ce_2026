"""Digital lab: run simulated mammalian cell culture experiments against a cost budget and a time limit."""

import dataclasses
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dbiolab.mammalian_cellculture_simulation import (
    CellType,
    FeedConcentrations,
    FeedingEvent,
    MammalianCellcultureReactor,
    MammalianRecipe,
    NoiseParameters,
    ReactorType,
)
from dbiolab.mammalian_cellculture_simulation.data_models import MammalianDataset

from digital_lab.constants import (
    BATCH_SIZES,
    BUDGET,
    FEED_COLUMNS,
    FEED_TIMES,
    INITIAL_DESIGN,
    MAX_EXPERIMENTS,
    PLATE_SHARED_COLUMNS,
    SCREENING_REACTOR,
    CELLTYPES,
    CONTINUOUS_BOUNDS,
    DEFAULT_LAB_CONFIG_DIR,
    LAB_CONFIG_FILENAMES,
    N_FEATURES,
    OBJECTIVE_REACTOR,
    REACTOR_COSTS,
    REACTOR_TYPES,
    SCORE_CHECKPOINTS,
    SHIFT_COLUMNS,
    TIME_LIMIT_SECONDS,
)


class TimeLimitReached(BaseException):
    """Raised inside the algorithm when its time limit is up.

    A `BaseException` (not `Exception`) so an algorithm's own `except Exception`
    can't swallow it. The harness catches it and scores the run as it stands.
    """


def _concentrations(glc: float, gln: float) -> FeedConcentrations:
    """Build a concentration record with only glc/gln set (everything else zero).

    Parameters
    ----------
    glc : float
        Glucose concentration, mM.
    gln : float
        Glutamine concentration, mM.

    Returns
    -------
    FeedConcentrations
        Concentration record with glc/gln set and all other fields zero.
    """
    return FeedConcentrations(
        c_product=0.0,
        x_total=0.0,
        x_viable=0.0,
        x_dead=0.0,
        c_glc=glc,
        c_gln=gln,
        c_lac=0.0,
        c_ammonia=0.0,
        c_aggregates=0.0,
    )


class _ShiftingReactor(MammalianCellcultureReactor):
    """Reactor whose temperature and pH can change at each feed.

    dbiolab keeps one temperature and pH for the whole run. Here every segment
    between feeds is integrated with its own (temperature, pH).

    Parameters
    ----------
    reactor_type : ReactorType
        Reactor parameters.
    setpoints : list[tuple[float, float]]
        (temperature, pH) for each segment: before feed 1, after feed 1, after feed 2, ...
    """

    def __init__(self, reactor_type: ReactorType, setpoints: list[tuple[float, float]]):
        super().__init__(reactor_type)
        self.setpoints = setpoints

    def _solve_odes(self, recipe: MammalianRecipe):
        segment_recipes = [
            dataclasses.replace(recipe, temperature=temperature, ph_value=ph) for temperature, ph in self.setpoints
        ]
        current_time = 0.0
        current_y = recipe.initial_conditions.as_list()
        solutions = None
        for segment, feeding_event in enumerate(recipe.feeding_strategy):
            partial, current_time = self._calculate_partial_solution(
                current_time, feeding_event.time_point, current_y, segment_recipes[segment]
            )
            if solutions is None:
                solutions = partial
            else:
                solutions.extend(partial)
            current_y = self._update_conditions_with_feed(partial, feeding_event)
        partial, _ = self._calculate_partial_solution(
            current_time, recipe.runtime, current_y, segment_recipes[len(recipe.feeding_strategy)]
        )
        if solutions is None:
            return partial
        solutions.extend(partial)
        return solutions


class DigitalLab:
    """A budgeted, time-limited simulated lab with three scales.

    Call it with one batch at one scale: `titers, spent = lab(x)`. Every row of
    `x` must use the same reactor, and `x` must have exactly
    `lab.batch_sizes[reactor]` rows (24 wells on a plate, 4 mini bioreactors,
    1 pilot reactor). A plate has extra rules: all wells share temperature, pH
    and starting medium, and there are no feeds or temperature/pH shifts (see `plate()`).
    In the bioreactors, temperature and pH can be changed at each feed.

    A batch runs completely or not at all. If the remaining budget or
    experiment allowance can't cover it, nothing runs and an empty array is
    returned; check `can_run(reactor)` or `affordable_reactors()` first. Once
    the time limit is up the run is stopped (`TimeLimitReached`) and scored on
    what it found so far.

    The read-only attributes (`budget`, `batch_sizes`, `reactor_costs`,
    `bounds`, ...) describe the problem. Algorithms should read them from
    here rather than hard-coding them, since the benchmark may change them.

    Parameters
    ----------
    seed : int
        Seed for the simulation noise.
    config_dir : str or pathlib.Path, optional
        Directory holding the reactor, cell-type and noise JSON files.
    budget : int, optional
        Total budget, by default `BUDGET`.
    max_experiments : int, optional
        Maximum number of experiments (wells/reactors) per run, including the initial design.
    initial_design : dict[str, int] or None, optional
        Number of batches per scale the harness runs before the algorithm starts.
    batch_sizes : dict[str, int] or None, optional
        Exact number of rows per call for each reactor, by default `BATCH_SIZES`.
    time_limit : float, optional
        Wall-clock limit in seconds, counted from `start_clock()`.
    reactor_costs : dict[str, int] or None, optional
        Cost per experiment (well/reactor) for each reactor, by default `REACTOR_COSTS`.
    """

    def __init__(
        self,
        seed: int,
        *,
        config_dir: str | Path = DEFAULT_LAB_CONFIG_DIR,
        budget: int = BUDGET,
        max_experiments: int = MAX_EXPERIMENTS,
        initial_design: dict[str, int] | None = None,
        batch_sizes: dict[str, int] | None = None,
        time_limit: float = TIME_LIMIT_SECONDS,
        reactor_costs: dict[str, int] | None = None,
    ) -> None:
        config_dir = Path(config_dir)
        self.seed = seed
        self.budget = budget
        self.max_experiments = max_experiments
        self.initial_design = dict(initial_design or INITIAL_DESIGN)
        self.batch_sizes = dict(batch_sizes or BATCH_SIZES)
        self.time_limit = time_limit
        self.reactor_costs = dict(reactor_costs or REACTOR_COSTS)
        self.reactor_types = list(REACTOR_TYPES)
        self.celltypes = list(CELLTYPES)
        self.bounds = list(CONTINUOUS_BOUNDS)
        self.objective_reactor = OBJECTIVE_REACTOR
        self.screening_reactor = SCREENING_REACTOR
        self.feed_times = list(FEED_TIMES)
        # Money spent (in $) at which the run is scored: best Pilot titer so far at each, summed.
        self.score_checkpoints = [fraction * budget for fraction in SCORE_CHECKPOINTS]

        self._reactors = {
            name: ReactorType.from_json(str(config_dir / LAB_CONFIG_FILENAMES[name])) for name in REACTOR_TYPES
        }
        self._celltypes = {name: CellType.from_json(str(config_dir / LAB_CONFIG_FILENAMES[name])) for name in CELLTYPES}
        self._noise = NoiseParameters.from_json(str(config_dir / LAB_CONFIG_FILENAMES["noise_parameters"]))

        self.spent = 0
        self.n_experiments = 0
        self.n_batches = 0  # lab(x) calls that ran; the initial design is batch 0
        self.history: list[dict] = []
        self.time_limit_hit = False
        self._start_time = time.time()

    @property
    def remaining_budget(self) -> int:
        """Budget left to spend."""
        return self.budget - self.spent

    @property
    def remaining_experiments(self) -> int:
        """Experiments left before the lab stops."""
        return self.max_experiments - self.n_experiments

    @property
    def elapsed(self) -> float:
        """Seconds since `start_clock()` (or construction)."""
        return time.time() - self._start_time

    @property
    def time_remaining(self) -> float:
        """Seconds left before the run is stopped."""
        return max(0.0, self.time_limit - self.elapsed)

    def batch_cost(self, reactor: str) -> int:
        """Cost of one full batch in `reactor`.

        Parameters
        ----------
        reactor : str
            Reactor name.

        Returns
        -------
        int
            `batch_sizes[reactor] * reactor_costs[reactor]`.
        """
        return self.batch_sizes[reactor] * self.reactor_costs[reactor]

    def can_run(self, reactor: str) -> bool:
        """Whether the remaining budget and experiment allowance cover one more batch in `reactor`.

        Parameters
        ----------
        reactor : str
            Reactor name.

        Returns
        -------
        bool
            True if `lab(x)` with a full batch in `reactor` would run.
        """
        return (
            self.spent + self.batch_cost(reactor) <= self.budget
            and self.n_experiments + self.batch_sizes[reactor] <= self.max_experiments
        )

    def affordable_reactors(self) -> list[str]:
        """Reactors in which one more batch can still run; empty when the run is over.

        Returns
        -------
        list[str]
            Reactor names, cheapest first.
        """
        return sorted((r for r in self.reactor_types if self.can_run(r)), key=lambda r: self.reactor_costs[r])

    def plate(
        self, celltypes: list[str], temperature: float, ph_value: float, glucose: float, glutamine: float
    ) -> NDArray:
        """Build a valid plate batch: one clone per well, shared conditions, no feeds, no shifts.

        Parameters
        ----------
        celltypes : list[str]
            One clone per well; must have `batch_sizes[screening_reactor]` entries (repeats allowed).
        temperature : float
            Temperature for the whole plate.
        ph_value : float
            pH for the whole plate.
        glucose : float
            Initial glucose for the whole plate, mM.
        glutamine : float
            Initial glutamine for the whole plate, mM.

        Returns
        -------
        numpy.typing.NDArray
            `(batch_sizes[screening_reactor], 14)` object array, ready for `lab(x)`.
        """
        x = np.empty((len(celltypes), N_FEATURES), dtype=object)
        x[:, 0] = self.screening_reactor
        x[:, 1] = list(celltypes)
        x[:, 2:6] = [float(temperature), float(ph_value), float(glucose), float(glutamine)]
        x[:, 6:10] = 0.0
        for shift_col, initial_col in SHIFT_COLUMNS.items():
            x[:, shift_col] = x[:, initial_col]
        return x

    def start_clock(self) -> None:
        """Restart the wall-clock timer; called right before the algorithm is handed the lab."""
        self._start_time = time.time()

    def __call__(self, x: NDArray) -> tuple[NDArray, int]:
        """Run one full batch at one scale.

        Parameters
        ----------
        x : numpy.typing.NDArray
            `(batch_sizes[reactor], 14)` object array, every row with the same reactor;
            see the notebook for the column layout.

        Returns
        -------
        tuple[numpy.typing.NDArray, int]
            Titers for the batch (empty if the budget or experiment allowance
            can't cover the whole batch), and the cumulative budget spent so far.
        """
        self._check_input(x)
        self._check_batch(x)
        if not self.can_run(x[0, 0]):
            return np.array([]), self.spent
        self.n_batches += 1
        return self._run(x)

    def _check_batch(self, x: NDArray) -> None:
        """Check the one-scale, batch-size and plate rules.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Candidate batch.

        Raises
        ------
        ValueError
            If the batch breaks a rule.
        """
        reactors = set(x[:, 0])
        if len(reactors) != 1:
            raise ValueError(f"A batch must use one reactor only. Got {sorted(reactors)}.")
        reactor = x[0, 0]
        if len(x) != self.batch_sizes[reactor]:
            raise ValueError(
                f"A {reactor} batch must have exactly {self.batch_sizes[reactor]} rows. Got {len(x)} rows. "
                f"Batch sizes: {self.batch_sizes}."
            )
        if reactor == self.screening_reactor:
            for col in PLATE_SHARED_COLUMNS:
                if len({float(v) for v in x[:, col]}) != 1:
                    raise ValueError(
                        f"On a {reactor} plate all wells share temperature, pH and initial glucose/glutamine "
                        f"(columns {list(PLATE_SHARED_COLUMNS)}); column {col} differs. Use lab.plate(...)."
                    )
            if any(float(v) != 0.0 for v in x[:, list(FEED_COLUMNS)].flatten()):
                raise ValueError(
                    f"A {reactor} plate can't be fed: feed columns {list(FEED_COLUMNS)} must be 0. Use lab.plate(...)."
                )
            for shift_col, initial_col in SHIFT_COLUMNS.items():
                if any(float(row[shift_col]) != float(row[initial_col]) for row in x):
                    raise ValueError(
                        f"A {reactor} plate can't change temperature or pH: column {shift_col} must equal column "
                        f"{initial_col}. Use lab.plate(...)."
                    )

    def _run(self, x: NDArray) -> tuple[NDArray, int]:
        """Run rows in order until the budget or experiment limit runs out (no batch-size check).

        Parameters
        ----------
        x : numpy.typing.NDArray
            Input rows.

        Returns
        -------
        tuple[numpy.typing.NDArray, int]
            Titers for the rows that ran, and the cumulative budget spent.

        Raises
        ------
        TimeLimitReached
            If the time limit is up.
        """
        y = []
        for row in x:
            if self.elapsed >= self.time_limit:
                self.time_limit_hit = True
                raise TimeLimitReached
            cost = self.reactor_costs[row[0]]
            if self.spent + cost > self.budget or self.n_experiments >= self.max_experiments:
                break
            # Simulate before charging, so a run interrupted mid-experiment isn't billed for it.
            titer = float(self._simulate(row))
            self.spent += cost
            self.n_experiments += 1
            y.append(titer)
            self.history.append({
                "n_experiments": self.n_experiments,
                "batch": self.n_batches,
                "spent": self.spent,
                "time": self.elapsed,
                "titer": titer,
                "reactor": row[0],
            })
        return np.array(y), self.spent

    def _simulate(self, row: NDArray) -> float:
        """Simulate one input row and return its final titer.

        Parameters
        ----------
        row : numpy.typing.NDArray
            One input row.

        Returns
        -------
        float
            Final product concentration.
        """
        recipe = MammalianRecipe(
            name=f"Recipe_{self.n_experiments}",
            celltype=self._celltypes[row[1]],
            temperature=row[2],
            ph_value=row[3],
            initial_conditions=self._initial_conditions(row[4], row[5]),
            feeding_strategy=[
                FeedingEvent(time_point=FEED_TIMES[0], feed_concentrations=_concentrations(row[6], row[7])),
                FeedingEvent(time_point=FEED_TIMES[1], feed_concentrations=_concentrations(row[8], row[9])),
            ],
            runtime=336.0,
            measurement_intervals=12.0,
            noise_parameters=self._noise,
        )
        setpoints = [(row[2], row[3]), (row[10], row[11]), (row[12], row[13])]
        reactor = _ShiftingReactor(self._reactors[row[0]], setpoints)
        result = reactor.run_dataset(MammalianDataset([recipe]), seed=self.seed)
        return result.data[0].c_product[-1]

    @staticmethod
    def _initial_conditions(glc: float, gln: float) -> FeedConcentrations:
        """Build the reactor's initial conditions: given glc/gln plus a fixed starting cell count.

        Parameters
        ----------
        glc : float
            Initial glucose concentration, mM.
        gln : float
            Initial glutamine concentration, mM.

        Returns
        -------
        FeedConcentrations
            Initial conditions.
        """
        conditions = _concentrations(glc, gln)
        conditions.x_total = 800000.0
        conditions.x_viable = 800000.0
        return conditions

    def _check_input(self, x: NDArray) -> None:
        """Validate that x has the expected shape, per-column types and categorical values.

        Parameters
        ----------
        x : numpy.typing.NDArray
            Candidate input array, one row per experiment.

        Raises
        ------
        TypeError
            If x is not a 2-D numpy array, or a column has the wrong type.
        ValueError
            If x does not have `N_FEATURES` columns or names an unknown reactor/cell type.
        """
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise TypeError(f"Input (x) must be a 2-D numpy array. Got {type(x)} instead.")
        if x.shape[1] != N_FEATURES:
            raise ValueError(f"Input (x) must have {N_FEATURES} columns. Got {x.shape[1]} instead.")
        if not all(isinstance(val, str) for val in x[:, 0].tolist() + x[:, 1].tolist()):
            raise TypeError("Columns 0 and 1 (reactor and celltype) must be of type str. Use dtype=object.")
        if not all(isinstance(val, (float, int)) for val in x[:, 2:].flatten().tolist()):
            raise TypeError(f"Columns 2 to {N_FEATURES - 1} must be numbers.")
        unknown_reactors = set(x[:, 0]) - set(self.reactor_costs)
        if unknown_reactors:
            raise ValueError(f"Unknown reactor(s) {sorted(unknown_reactors)}. Must be one of {self.reactor_types}.")
        unknown_celltypes = set(x[:, 1]) - set(self.celltypes)
        if unknown_celltypes:
            raise ValueError(f"Unknown cell type(s) {sorted(unknown_celltypes)}. Must be one of {self.celltypes}.")


# The documented interface an algorithm may use. Everything else on the lab is internal.
PUBLIC_LAB_API = frozenset({
    "affordable_reactors", "batch_cost", "batch_sizes", "bounds", "budget", "can_run", "celltypes", "elapsed",
    "feed_times", "initial_design", "max_experiments", "n_batches", "n_experiments", "objective_reactor", "plate",
    "reactor_costs", "reactor_types", "remaining_budget", "remaining_experiments", "score_checkpoints",
    "screening_reactor", "seed", "spent", "time_limit", "time_remaining",
})
_HARMLESS_DUNDERS = frozenset({"__class__", "__doc__"})


def public_view(lab: DigitalLab, access_log: list[str]):
    """Wrap a lab so an algorithm can only use its documented, public interface.

    Reading any other attribute (private ones like `_simulate`, internal data like
    `history`, or dunders like `__dict__`) is recorded in `access_log` and raises
    AttributeError; so is any attempt to set an attribute. Dicts and lists are returned
    as copies, so an algorithm can't change the lab's costs, batch sizes, bounds or clones.

    Parameters
    ----------
    lab : DigitalLab
        The real lab.
    access_log : list[str]
        Receives one entry per blocked access, e.g. "get _simulate" or "set budget".

    Returns
    -------
    object
        A callable, read-only stand-in for the lab.
    """

    class Lab:  # pylint: disable=too-few-public-methods
        """Read-only public view of the digital lab (see `PUBLIC_LAB_API`)."""

        __slots__ = ()

        def __getattribute__(self, name):
            if name in PUBLIC_LAB_API:
                value = getattr(lab, name)
                return value.copy() if isinstance(value, (dict, list)) else value
            if name in _HARMLESS_DUNDERS:
                return object.__getattribute__(self, name)
            access_log.append(f"get {name}")
            raise AttributeError(f"The lab has no public attribute {name!r}. Use only: {sorted(PUBLIC_LAB_API)}.")

        def __setattr__(self, name, value):
            access_log.append(f"set {name}")
            raise AttributeError("The lab is read-only.")

        def __call__(self, x):
            return lab(x)

        def __repr__(self):
            return f"DigitalLab(spent={lab.spent}, budget={lab.budget}, n_experiments={lab.n_experiments})"

    return Lab()
