"""Reference baseline: simplified multi-fidelity Bayesian optimization.

Loosely inspired by "Holistic Bioprocess Development Across Scales Using
Multi-Fidelity Batch Bayesian Optimization" (arXiv:2508.10970). This is a
SIMPLIFIED version, not a faithful reproduction: the paper uses a multi-task
(ICM) Gaussian process and Monte Carlo batch acquisition functions with a
learned cost-aware fidelity policy. Here:

- Model: one joint GP over all scales, with the reactor's (log) cost as a
  "fidelity" input feature, so cheap and expensive scales share one model.
  It is fit on log(1 + titer), because titers span orders of magnitude.
- Each iteration it proposes one batch per scale it can still afford, each
  respecting that scale's rules (a 24-well plate with shared conditions and
  no feeds, 4 mini bioreactors, 1 pilot reactor), scored by Expected
  Improvement over the best value seen *at that scale*.
- It runs the proposal with the highest total EI per sqrt(effective cost),
  so cheap scales are favoured without making the mid scale pointless. The
  effective cost is the larger of the batch's share of the remaining budget
  and of the remaining experiment allowance: a plate is cheap in money but
  uses up 24 experiments at once. After
  `PILOT_ONLY_AFTER` of the budget, it only runs Pilot batches, to turn what
  it learned into Pilot results.
- The run is scored at 33 %, 66 % and 100 % of the budget, and a checkpoint
  without a Pilot result counts 0. So if there is no Pilot result yet and a
  cheaper batch would leave no room for one before the first checkpoint, it
  runs Pilot.

Bioreactor batches are greedily diversity-filtered (minimum distance in the
standardized feature space), since a plain top-k tends to pick near-duplicates.
"""

import math
import random
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

N_CANDIDATES = 400  # random bioreactor candidates per scale and iteration
N_PLATE_CONDITIONS = 60  # random shared plate conditions tried per iteration
MIN_BATCH_DIST = 0.75  # min Euclidean distance (standardized feature space) between batch members
PILOT_ONLY_AFTER = 0.5  # fraction of the budget after which only Pilot batches are run


def _encode(x: np.ndarray, lab) -> np.ndarray:
    """Map raw input rows to numeric, unscaled GP features.

    Parameters
    ----------
    x : numpy.ndarray
        Raw input rows: (reactor, celltype, 12 continuous floats).
    lab : DigitalLab
        The lab, for the reactor costs and the clone list.

    Returns
    -------
    numpy.ndarray
        One row per input: log reactor cost, one-hot clone, then the 12 continuous features.
    """
    n_clones = len(lab.celltypes)
    features = np.zeros((len(x), 1 + n_clones + len(lab.bounds)))
    for i, row in enumerate(x):
        features[i, 0] = math.log(lab.reactor_costs[row[0]])
        features[i, 1 + lab.celltypes.index(row[1])] = 1.0
        features[i, 1 + n_clones:] = [float(v) for v in row[2:]]
    return features


def _fit_gp(train_x: list, train_y: list, lab, seed: int) -> tuple:
    """Standardize the features and fit a GP (Matern 5/2, ARD) on log(1 + titer).

    Parameters
    ----------
    train_x : list
        Training input rows.
    train_y : list
        Corresponding titers.
    lab : DigitalLab
        The lab, for encoding.
    seed : int
        Random state for the GP's restart optimizer.

    Returns
    -------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        The fitted GP (predicts log(1 + titer)).
    scaler : sklearn.preprocessing.StandardScaler
        Scaler for encoding future candidates consistently.
    """
    encoded = _encode(np.array(train_x, dtype=object), lab)
    scaler = StandardScaler().fit(encoded)
    kernel = Matern(length_scale=np.ones(encoded.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # hitting a length-scale bound is harmless here
        gp.fit(scaler.transform(encoded), np.log1p(np.maximum(train_y, 0.0)))
    return gp, scaler


def _expected_improvement(mean: np.ndarray, std: np.ndarray, best_f: float) -> np.ndarray:
    """Compute Expected Improvement over `best_f`.

    Parameters
    ----------
    mean : numpy.ndarray
        GP posterior mean at each candidate.
    std : numpy.ndarray
        GP posterior standard deviation at each candidate.
    best_f : float
        Incumbent value.

    Returns
    -------
    numpy.ndarray
        Expected Improvement at each candidate.
    """
    std = np.maximum(std, 1e-9)
    z = (mean - best_f) / std
    return (mean - best_f) * norm.cdf(z) + std * norm.pdf(z)


def _greedy_diverse_pick(encoded: np.ndarray, score: np.ndarray, k: int) -> list:
    """Greedily pick k high-scoring candidates at least `MIN_BATCH_DIST` apart.

    Falls back to the next-best candidates regardless of distance if too few are diverse.

    Parameters
    ----------
    encoded : numpy.ndarray
        Standardized features of the candidates.
    score : numpy.ndarray
        Acquisition score per candidate.
    k : int
        Number of candidates to pick.

    Returns
    -------
    list[int]
        Indices of the picked candidates.
    """
    order = np.argsort(score)[::-1]
    picked: list[int] = []
    for idx in order:
        if all(np.linalg.norm(encoded[idx] - encoded[j]) >= MIN_BATCH_DIST for j in picked):
            picked.append(int(idx))
        if len(picked) == k:
            return picked
    picked += [int(i) for i in order if i not in picked][: k - len(picked)]
    return picked


def _incumbent(all_x: list, all_y: list, reactor: str, mean_fn) -> float:
    """Best log-titer seen at `reactor`, or the best predicted one at observed recipes if none was run there.

    Parameters
    ----------
    all_x : list
        Observed input rows.
    all_y : list
        Observed titers.
    reactor : str
        Scale to get the incumbent for.
    mean_fn : Callable
        Maps raw rows to the GP's predicted log-titer.

    Returns
    -------
    float
        Incumbent in the GP's log space.
    """
    observed = [y for row, y in zip(all_x, all_y) if row[0] == reactor]
    if observed:
        return math.log1p(max([*observed, 0.0]))
    as_reactor = np.array(all_x, dtype=object)
    as_reactor[:, 0] = reactor
    return float(mean_fn(as_reactor).max())


def _propose(  # pylint: disable=too-many-locals
    rng: random.Random, lab, reactor: str, predict, best_f: float
) -> tuple:
    """Build the best batch for one scale and its total Expected Improvement.

    Parameters
    ----------
    rng : random.Random
        Source of randomness for the candidates.
    lab : DigitalLab
        The lab.
    reactor : str
        Scale to propose for.
    predict : Callable
        Maps raw rows to `(encoded, mean, std)`.
    best_f : float
        Incumbent at this scale (log space).

    Returns
    -------
    batch : numpy.ndarray
        `lab.batch_sizes[reactor]` rows.
    total_ei : float
        Sum of the batch's Expected Improvement.
    """
    size = lab.batch_sizes[reactor]
    if reactor == lab.screening_reactor:
        # Plates: try random shared conditions; for each, fill the wells with the clones of highest EI.
        best_batch, best_total = None, -1.0
        for _ in range(N_PLATE_CONDITIONS):
            shared = [rng.uniform(low, high) for low, high in lab.bounds[:4]]
            all_clones = lab.plate(lab.celltypes, *shared)
            _, mean, std = predict(all_clones)
            ei = _expected_improvement(mean, std, best_f)
            order = np.argsort(ei)[::-1]
            picks = [lab.celltypes[i] for i in order[:size]]
            picks += [lab.celltypes[i] for i in order[: size - len(picks)]]  # repeat the best if size > clones
            total = float(np.sort(ei)[::-1][:size].sum())
            if total > best_total:
                best_batch, best_total = lab.plate(picks, *shared), total
        return best_batch, best_total

    pool = np.empty((N_CANDIDATES, 2 + len(lab.bounds)), dtype=object)
    pool[:, 0] = reactor
    pool[:, 1] = [rng.choice(lab.celltypes) for _ in range(N_CANDIDATES)]
    for j, (low, high) in enumerate(lab.bounds):
        pool[:, 2 + j] = [rng.uniform(low, high) for _ in range(N_CANDIDATES)]
    encoded, mean, std = predict(pool)
    ei = _expected_improvement(mean, std, best_f)
    picked = _greedy_diverse_pick(encoded, ei, size)
    return pool[picked], float(ei[picked].sum())


def reference_mfbo(
    initial_x: np.ndarray, initial_y: np.ndarray, lab, seed: int
) -> None:
    """Run the simplified multi-fidelity BO loop described above until no scale can run another batch.

    Parameters
    ----------
    initial_x : numpy.ndarray
        Initial design rows.
    initial_y : numpy.ndarray
        Titers observed for `initial_x`.
    lab : DigitalLab
        Call `lab(x)` with one full batch at one scale; returns `(titers, spent)`.
    seed : int
        Seed for this run's randomness (GP restarts and candidate sampling).
    """
    rng = random.Random(seed)
    all_x = list(initial_x[:len(initial_y)])
    all_y = [float(v) for v in initial_y]

    while lab.affordable_reactors():
        gp, scaler = _fit_gp(all_x, all_y, lab, seed)

        def predict(rows, gp=gp, scaler=scaler):
            encoded = scaler.transform(_encode(rows, lab))
            mean, std = gp.predict(encoded, return_std=True)
            return encoded, mean, std

        reactors = lab.affordable_reactors()
        pilot = lab.objective_reactor
        no_pilot_yet = all(row[0] != pilot for row in all_x)
        pilot_due = no_pilot_yet and lab.spent + lab.batch_cost(reactors[0]) + lab.batch_cost(pilot) > lab.score_checkpoints[0]
        if (lab.spent >= PILOT_ONLY_AFTER * lab.budget or pilot_due) and pilot in reactors:
            reactors = [pilot]

        best_batch, best_score = None, -1.0
        for reactor in reactors:
            best_f = _incumbent(all_x, all_y, reactor, lambda rows: predict(rows)[1])
            batch, total_ei = _propose(rng, lab, reactor, predict, best_f)
            share = max(
                lab.batch_cost(reactor) / lab.remaining_budget,
                lab.batch_sizes[reactor] / lab.remaining_experiments,
            )
            score = total_ei / math.sqrt(share)
            if score > best_score:
                best_batch, best_score = batch, score

        titers, _ = lab(best_batch)
        all_x += list(best_batch[:len(titers)])
        all_y += [float(t) for t in titers]
