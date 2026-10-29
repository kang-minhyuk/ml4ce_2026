"""Run algorithms end-to-end against the digital lab and score them.

Shared by the student notebook and the assessment benchmark, so both score
algorithms identically.
"""

import contextlib
import signal
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np
import polars as pl
from scipy.stats.qmc import LatinHypercube

from digital_lab.constants import N_FEATURES, OBJECTIVE_REACTOR
from digital_lab.lab import DigitalLab, TimeLimitReached, public_view

AlgorithmFn = Callable[[np.ndarray, np.ndarray, DigitalLab, int], Any]


def initial_design(lab: DigitalLab, seed: int) -> np.ndarray:
    """Build the initial design: `lab.initial_design[reactor]` valid batches per scale.

    Plates get one random set of shared conditions each and distinct random
    clones (as far as the plate size allows). Bioreactor batches are a Latin
    hypercube over the continuous inputs with random clones. Everything depends
    on `seed`, so every seed starts from a different design.

    Parameters
    ----------
    lab : DigitalLab
        The lab, for the batch sizes, clones and bounds.
    seed : int
        Seed for the design.

    Returns
    -------
    numpy.ndarray
        The design rows, plates first.
    """
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in lab.bounds])
    high = np.array([b[1] for b in lab.bounds])
    batches = []
    for reactor in lab.reactor_types:
        size = lab.batch_sizes[reactor]
        for _ in range(lab.initial_design.get(reactor, 0)):
            replace = size > len(lab.celltypes)
            celltypes = rng.choice(lab.celltypes, size=size, replace=replace).tolist()
            if reactor == lab.screening_reactor:
                shared = low[:4] + rng.random(4) * (high[:4] - low[:4])
                batches.append(lab.plate(celltypes, *shared))
                continue
            unit = LatinHypercube(d=len(lab.bounds), seed=rng).random(n=size)
            x = np.empty((size, N_FEATURES), dtype=object)
            x[:, 0] = reactor
            x[:, 1] = celltypes
            x[:, 2:] = (low + unit * (high - low)).tolist()
            batches.append(x)
    return np.vstack(batches) if batches else np.empty((0, N_FEATURES), dtype=object)


def pilot_best_so_far(history: pl.DataFrame) -> list:
    """Compute the cumulative max of titer, only advancing on OBJECTIVE_REACTOR experiments.

    Rows before the first OBJECTIVE_REACTOR experiment get None (undefined
    objective yet) rather than being contaminated by cheaper-fidelity titer.

    Parameters
    ----------
    history : polars.DataFrame
        Experiment history with "reactor" and "titer" columns.

    Returns
    -------
    list
        One entry per row of history: the best OBJECTIVE_REACTOR titer
        observed up to and including that row, or None if none observed yet.
    """
    reactors = history["reactor"].to_list()
    titers = history["titer"].to_list()

    best = []
    running_best = None
    for reactor, titer in zip(reactors, titers):
        if reactor == OBJECTIVE_REACTOR:
            running_best = titer if running_best is None else max(running_best, titer)
        best.append(running_best)
    return best


def checkpoint_bests(history: pl.DataFrame, checkpoints: list) -> list:
    """Best OBJECTIVE_REACTOR titer found by the time each checkpoint amount of money was spent.

    Parameters
    ----------
    history : polars.DataFrame
        Experiment history with "reactor", "titer" and "spent" columns.
    checkpoints : list[float]
        Money spent at which to read off the best titer (e.g. 33 %, 66 % and 100 % of the budget).

    Returns
    -------
    list
        One value per checkpoint: the best Pilot titer among experiments paid for up to
        that amount, or None if there was no Pilot run by then.
    """
    pilot = history.filter(pl.col("reactor") == OBJECTIVE_REACTOR)
    bests = []
    for checkpoint in checkpoints:
        reached = pilot.filter(pl.col("spent") <= checkpoint)["titer"]
        bests.append(float(reached.max()) if reached.len() else None)  # type: ignore[arg-type]
    return bests


def score(bests: list) -> float:
    """Score of one run: the checkpoint bests summed, a missing one counting 0.

    Parameters
    ----------
    bests : list
        From `checkpoint_bests`.

    Returns
    -------
    float
        The score.
    """
    return float(sum(b or 0.0 for b in bests))


@contextlib.contextmanager
def _time_limit(seconds: float):
    """Raise `TimeLimitReached` in the running code after `seconds`, even mid-computation.

    Uses SIGALRM, so it only works on macOS/Linux in the main thread. Elsewhere
    (e.g. Windows) the run is only stopped at the next `lab(x)` call.

    Parameters
    ----------
    seconds : float
        Time limit.

    Yields
    ------
    None
    """
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable or not np.isfinite(seconds):
        yield
        return

    def _handler(signum, frame):
        raise TimeLimitReached

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-3))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _end_reason(lab: DigitalLab, error: str | None) -> str:
    """Say why a run ended, as a short label for plots.

    One of "time limit", "error", "budget", "experiment limit", "budget + experiment
    limit" (each remaining scale was blocked by one of the two), or "stopped early
    (could still run ...)" when the algorithm returned while some scale was still
    affordable.

    Parameters
    ----------
    lab : DigitalLab
        The lab after the run.
    error : str or None
        Traceback if the algorithm raised.

    Returns
    -------
    str
        The reason.
    """
    if lab.time_limit_hit:
        return "time limit"
    if error:
        return "error"
    if lab.affordable_reactors():
        return f"stopped early (could still run {'/'.join(lab.affordable_reactors())})"
    by_money = {r for r in lab.reactor_types if lab.spent + lab.batch_cost(r) > lab.budget}
    by_experiments = {r for r in lab.reactor_types if lab.n_experiments + lab.batch_sizes[r] > lab.max_experiments}
    if by_money == set(lab.reactor_types):
        return "budget"
    if by_experiments == set(lab.reactor_types):
        return "experiment limit"
    return "budget + experiment limit"


def run_algorithm(algorithm: AlgorithmFn, name: str, seed: int, lab_kwargs: dict | None = None) -> tuple:
    """Run one algorithm once, on a fresh lab with the given seed.

    `seed` controls the initial LHS design, the lab's simulation noise, and is
    handed to the algorithm so it can seed its own randomness. The initial
    design is run (and paid for) before the clock starts. At the time limit the
    algorithm is interrupted and the run is scored on the experiments done so
    far. An exception raised by the algorithm is caught and reported the same way.

    Parameters
    ----------
    algorithm : AlgorithmFn
        `(initial_x, initial_y, lab, seed) -> None`.
    name : str
        Label for the run's history ("algorithm" column).
    seed : int
        Seed for the initial design, the lab noise and the algorithm.
    lab_kwargs : dict or None, optional
        Extra keyword arguments for `DigitalLab` (config_dir, budget, initial_design, ...).

    Returns
    -------
    history : polars.DataFrame
        One row per experiment, with a cumulative "best" column (best
        OBJECTIVE_REACTOR titer so far, None until the first one).
    info : dict
        "end_reason" (why the run ended, see `_end_reason`), "wall_time" (s),
        "time_limit" (s), "time_limit_hit" (the algorithm was stopped at the
        limit), "error" (traceback string or None), "access_warnings"
        (blocked attempts to use non-public parts of the lab, e.g. "get _simulate"),
        "score_checkpoints" (money spent at each checkpoint), "checkpoint_bests"
        (best Pilot titer at each, None if no Pilot run yet) and "score" (their sum).
    """
    lab = DigitalLab(seed=seed, **(lab_kwargs or {}))
    initial_x = initial_design(lab, seed)
    initial_y, _ = lab._run(initial_x)  # pylint: disable=protected-access

    error = None
    access_log: list[str] = []
    lab.start_clock()
    start = time.time()
    try:
        with _time_limit(lab.time_limit):
            algorithm(initial_x, initial_y, public_view(lab, access_log), seed)
    except TimeLimitReached:
        lab.time_limit_hit = True
    except Exception:  # pylint: disable=broad-exception-caught
        error = traceback.format_exc()
    wall_time = time.time() - start

    history = pl.DataFrame(
        lab.history,
        schema={
            "n_experiments": pl.Int64, "batch": pl.Int64, "spent": pl.Int64, "time": pl.Float64,
            "titer": pl.Float64, "reactor": pl.String,
        },
    )
    history = history.with_columns(
        pl.Series("best", pilot_best_so_far(history), dtype=pl.Float64),
        pl.lit(name).alias("algorithm"),
        pl.lit(seed).alias("seed"),
    )
    bests = checkpoint_bests(history, lab.score_checkpoints)
    info = {
        "end_reason": _end_reason(lab, error),
        "wall_time": wall_time,
        "time_limit": lab.time_limit,
        "time_limit_hit": lab.time_limit_hit,
        "error": error,
        "access_warnings": sorted(set(access_log)),
        "score_checkpoints": list(lab.score_checkpoints),
        "checkpoint_bests": bests,
        "score": score(bests),
    }
    return history, info


def _forward_fill_column(history: pl.DataFrame, column: str, length: int) -> list:
    """Pad a column to `length` rows by repeating its last value.

    Parameters
    ----------
    history : polars.DataFrame
        Source history.
    column : str
        Column to extract and pad.
    length : int
        Target length; no-op if `history` already has at least this many rows.

    Returns
    -------
    list
        The column's values, padded with its last value (or None if empty)
        up to `length`.
    """
    values = history[column].to_list()
    if len(values) < length:
        pad = values[-1] if values else None
        values = values + [pad] * (length - len(values))
    return values


def _mean_std_all_defined(rows: list) -> tuple:
    """Compute the elementwise (mean, std) across rows, only where every row has a value.

    Averaging only the rows that happen to be defined would make a mean of
    non-decreasing curves drop whenever a new row joins with a low first value
    (e.g. a seed's first Pilot result). Requiring all rows keeps it monotone.

    Parameters
    ----------
    rows : list
        One list per seed; all lists must have the same length.

    Returns
    -------
    means : list
        Elementwise mean across rows, or None where any row is None.
    stds : list
        Elementwise standard deviation, aligned with `means`.
    """
    means: list[float | None] = []
    stds: list[float | None] = []
    for values_at_index in zip(*rows):
        if any(v is None for v in values_at_index):
            means.append(None)
            stds.append(None)
            continue
        mean = sum(values_at_index) / len(values_at_index)
        means.append(mean)
        stds.append((sum((v - mean) ** 2 for v in values_at_index) / len(values_at_index)) ** 0.5)
    return means, stds


def mean_std(values: list) -> tuple:
    """Compute the (mean, std) of a list of per-seed scalars.

    E.g. per-seed best titer.

    Parameters
    ----------
    values : list
        Scalars, one per seed; None entries are dropped before computing.

    Returns
    -------
    mean : float or None
        Mean of the non-None values, or None if there are none.
    std : float or None
        Standard deviation of the non-None values, or None if there are none.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance**0.5


def aggregate_histories(per_seed_histories: list, name: str) -> pl.DataFrame:
    """Aggregate per-seed histories into one mean/std curve.

    One row per experiment index (1, 2, 3, ...), with {col}_mean / {col}_std
    for spent/time/best, for plotting a mean curve with a std band. Runs are
    aligned by experiment index — the one axis every seed shares exactly,
    since each experiment increments the counter by exactly one regardless of
    what the algorithm decided to do. A seed that stops earlier than the
    others has its last values forward-filled so every seed contributes a
    value at every index up to the longest run.

    Parameters
    ----------
    per_seed_histories : list[polars.DataFrame]
        One history per seed, as returned by `run_algorithm`.
    name : str
        Label to attach to the aggregated history's "algorithm" column.

    Returns
    -------
    polars.DataFrame
        One row per experiment index, with n_experiments and {col}_mean /
        {col}_std for spent, time, and best.
    """
    max_len = max(h.height for h in per_seed_histories)
    agg_columns = {}
    for col in ("spent", "time", "best"):
        rows = [_forward_fill_column(h, col, max_len) for h in per_seed_histories]
        means, stds = _mean_std_all_defined(rows)
        agg_columns[f"{col}_mean"] = means
        agg_columns[f"{col}_std"] = stds

    aggregated = pl.DataFrame({"n_experiments": list(range(1, max_len + 1)), **agg_columns})
    return aggregated.with_columns(pl.lit(name).alias("algorithm"))


def run_algorithm_multi_seed(
    algorithm: AlgorithmFn, name: str, seeds: list, lab_kwargs: dict | None = None, verbose: bool = False
) -> tuple:
    """Run one algorithm once per seed.

    Parameters
    ----------
    algorithm : AlgorithmFn
        `(initial_x, initial_y, lab, seed) -> None`.
    name : str
        Label for each run's history.
    seeds : list[int]
        Seeds to run, one independent replicate per seed.
    lab_kwargs : dict or None, optional
        Extra keyword arguments for `DigitalLab`.
    verbose : bool, optional
        Print one progress line per seed, by default False.

    Returns
    -------
    per_seed_histories : list[polars.DataFrame]
        The individual run histories, one per seed.
    aggregated : polars.DataFrame
        See `aggregate_histories`.
    infos : list[dict]
        The per-seed `info` dicts from `run_algorithm`.
    """
    per_seed_histories, infos = [], []
    for seed in seeds:
        history, info = run_algorithm(algorithm, name, seed, lab_kwargs)
        per_seed_histories.append(history)
        infos.append(info)
        if verbose:
            best = history["best"].max() if history.height else None
            best_str = f"{best:.4f}" if best is not None else "none (no Pilot run)"
            status = " ERROR" if info["error"] else (" STOPPED AT TIME LIMIT" if info["time_limit_hit"] else "")
            print(f"  {name} seed {seed}: score {info['score']:.1f}, best Pilot titer {best_str}, "
                  f"{info['wall_time']:.1f} s{status}")

    aggregated = aggregate_histories(per_seed_histories, name)
    return per_seed_histories, aggregated, infos


def per_seed_metrics(per_seed_histories: list, infos: list) -> dict:
    """Compute the score and the best Pilot titers per seed, then (mean, std) across seeds.

    Parameters
    ----------
    per_seed_histories : list[polars.DataFrame]
        One history per seed, as returned by `run_algorithm_multi_seed`.
    infos : list[dict]
        The matching `info` dicts (for the checkpoints).

    Returns
    -------
    dict
        {"score": ..., "best_titer": ..., "best_at_33": ..., "best_at_66": ..., "best_at_100": ...},
        each a (mean, std); a checkpoint without a Pilot run counts 0.
    """
    metrics = {"score": mean_std([i["score"] for i in infos])}
    metrics["best_titer"] = mean_std([h["best"].max() if h.height else None for h in per_seed_histories])
    for k, checkpoint in enumerate(infos[0]["score_checkpoints"]):
        percent = round(100 * checkpoint / infos[0]["score_checkpoints"][-1])
        metrics[f"best_at_{percent}"] = mean_std([i["checkpoint_bests"][k] or 0.0 for i in infos])
    return metrics
