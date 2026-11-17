"""Reference: scale-down screening funnel, no model at all.

Mirrors how clone selection and scale-up are often done in practice:

1. Screen clones on a plate. The initial design already has one plate; one more
   plate tests the clones it missed plus the best ones again, under the
   initial plate's conditions moved halfway to the centre of the ranges.
   Clones are ranked by titer relative to their plate's median, since two
   plates with different conditions aren't directly comparable.
2. Narrow down in mini bioreactors: the top clones, one per reactor, with
   Sobol-sampled conditions and feeds, then variations of the best results.
3. Confirm and refine at Pilot: the best mini-bioreactor recipes first, then an
   evolution strategy that perturbs the best Pilot recipes with a shrinking
   step size.

A cheap batch only runs if the money left afterwards could still be spent on
Pilot runs within the remaining experiments (`_keeps_budget_usable`), so the
cheap stages never strand budget when the experiment limit is the bottleneck.
And the best recipe so far goes to Pilot before the first scoring checkpoint
(`_pilot_due`), because a checkpoint without a Pilot result scores 0.
"""

# A single self-contained file (like a submission), so the main loop keeps its state in locals.
# pylint: disable=too-many-locals

import numpy as np
import sobol_seq

MBR_SHARE = 0.25  # of the budget, spent in mini bioreactors
N_PARENTS = 3


def _perturb(rng, lab, parents, n, reactor, scale):
    """Gaussian variations of the parent rows (scale = fraction of each range), clipped to the bounds."""
    picks = rng.integers(len(parents), size=n)
    rows = np.empty((n, 2 + len(lab.bounds)), dtype=object)
    rows[:, 0] = reactor
    rows[:, 1] = [parents[p][1] for p in picks]
    for j, (low, high) in enumerate(lab.bounds):
        base = np.array([float(parents[p][2 + j]) for p in picks])
        rows[:, 2 + j] = np.clip(base + rng.normal(0, scale * (high - low), size=n), low, high).tolist()
    return rows


def _best(data, reactor, k):
    """Return the k best rows measured in the given reactor."""
    ranked = sorted((d for d in data if d[0][0] == reactor), key=lambda d: -d[1])
    return [row for row, _ in ranked[:k]]


def _clone_ranking(data, lab):
    """Rank clones by their best plate titer relative to the median of their plate."""
    plates = {}
    for row, titer in data:
        if row[0] == lab.screening_reactor:
            plates.setdefault(tuple(float(v) for v in row[2:6]), []).append((row[1], titer))
    score = {}
    for wells in plates.values():
        median = max(np.median([t for _, t in wells]), 1e-9)
        for clone, titer in wells:
            score[clone] = max(score.get(clone, 0.0), titer / median)
    return sorted(score, key=score.get, reverse=True)


def _keeps_budget_usable(lab, reactor):
    """Whether one batch in `reactor` can run without leaving money that can no longer be spent.

    Two limits apply: budget and number of experiments. A cheap batch uses up experiments
    without spending much, so it is only worth it if, afterwards, the remaining experiments
    could still absorb the remaining budget as Pilot runs. Otherwise the money left over
    is wasted, and Pilot runs are the better use of the scarce experiments.
    """
    pilot = lab.objective_reactor
    if not lab.can_run(reactor):
        return False
    if reactor == pilot:
        return True
    budget_after = lab.remaining_budget - lab.batch_cost(reactor)
    experiments_after = lab.remaining_experiments - lab.batch_sizes[reactor]
    return budget_after <= experiments_after * lab.reactor_costs[pilot] and experiments_after >= lab.batch_sizes[pilot]


def _pilot_due(lab, reactor, data):
    """Whether a batch in `reactor` now would leave no room for a Pilot run before the first scoring checkpoint.

    A run is scored on its best Pilot titer at 33 %, 66 % and 100 % of the budget, and a checkpoint
    without any Pilot result counts 0. So the first Pilot run has to happen before the first checkpoint.
    """
    pilot = lab.objective_reactor
    if reactor == pilot or any(row[0] == pilot for row, _ in data):
        return False
    return lab.spent + lab.batch_cost(reactor) + lab.batch_cost(pilot) > lab.score_checkpoints[0]


def _run(lab, data, batch):
    """Run one batch and record the results."""
    titers, _ = lab(batch)
    data += [(row, float(t)) for row, t in zip(batch, titers)]


def reference_screen_then_confirm(initial_x, initial_y, lab, seed):
    """Screen clones on a plate, narrow down in mini bioreactors, then evolve at Pilot."""
    rng = np.random.default_rng(seed)
    plate_reactor, pilot = lab.screening_reactor, lab.objective_reactor
    middle = next(r for r in lab.reactor_types if r not in (plate_reactor, pilot))
    data = [(row, float(y)) for row, y in zip(initial_x, initial_y)]

    # 1. One more plate: untested clones first, then the best clones so far.
    plate_size = lab.batch_sizes[plate_reactor]
    if _keeps_budget_usable(lab, plate_reactor) and not _pilot_due(lab, plate_reactor, data):
        tested = _clone_ranking(data, lab)
        wells = [c for c in lab.celltypes if c not in tested] + tested
        wells = (wells * plate_size)[:plate_size]
        previous = [row for row, _ in data if row[0] == plate_reactor]
        centre = np.array([np.mean(b) for b in lab.bounds[:4]])
        shared = (np.array(previous[0][2:6], dtype=float) + centre) / 2 if previous else centre
        _run(lab, data, lab.plate(wells, *shared))

    # 2. Mini bioreactors: the top clones with space-filling conditions, then variations of the best.
    clones = _clone_ranking(data, lab) or list(lab.celltypes)
    size = lab.batch_sizes[middle]
    mbr_budget = lab.spent + MBR_SHARE * lab.budget
    skip = int(rng.integers(1, 1000))  # different Sobol points per seed
    first = True
    while lab.spent + lab.batch_cost(middle) <= mbr_budget and _keeps_budget_usable(lab, middle):
        if _pilot_due(lab, middle, data):
            # Best recipe so far to Pilot, so the first checkpoint has a Pilot result.
            parents = _best(data, middle, 1) or _best(data, plate_reactor, 1)
            _run(lab, data, _perturb(rng, lab, parents, lab.batch_sizes[pilot], pilot, 0.0))
            continue
        if first:
            unit = sobol_seq.i4_sobol_generate(len(lab.bounds), size, skip)
            batch = np.empty((size, 2 + len(lab.bounds)), dtype=object)
            batch[:, 0] = middle
            batch[:, 1] = [clones[i % len(clones)] for i in range(size)]
            for j, (low, high) in enumerate(lab.bounds):
                batch[:, 2 + j] = (low + unit[:, j] * (high - low)).tolist()
            first = False
        else:
            batch = _perturb(rng, lab, _best(data, middle, N_PARENTS), size, middle, 0.1)
        _run(lab, data, batch)

    # 3. Confirm the best mini-bioreactor recipes at Pilot, then evolve from the best Pilot results.
    size = lab.batch_sizes[pilot]
    confirm = _best(data, middle, size)
    if confirm and lab.can_run(pilot):
        batch = np.array(confirm, dtype=object)
        batch = np.vstack([batch, _perturb(rng, lab, confirm, size - len(batch), pilot, 0.03)])
        batch[:, 0] = pilot
        _run(lab, data, batch)
    scale = 0.1
    while lab.can_run(pilot):
        parents = _best(data, pilot, N_PARENTS) or _best(data, middle, N_PARENTS)
        _run(lab, data, _perturb(rng, lab, parents, size, pilot, scale))
        scale = max(0.02, scale * 0.8)
