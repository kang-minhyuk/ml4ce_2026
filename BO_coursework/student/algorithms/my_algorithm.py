"""My algorithm. Written from project.ipynb — edit it there, this file is overwritten."""

import numpy as np

# Full names (like onImperial ID card) and CIDs of everyone in your group. Your results are reported under these names.
STUDENT_NAMES = ["TODO Firstname Lastname CID", "TODO Firstname Lastname CID"]


def my_algorithm(initial_x, initial_y, lab, seed):
    """Spend the lab's budget to find the highest Pilot-scale titer.

    Parameters
    ----------
    initial_x : numpy.ndarray
        Initial design rows (already run and paid for).
    initial_y : numpy.ndarray
        Titers observed for `initial_x`.
    lab : DigitalLab
        Call `titers, spent = lab(x)` with one full batch at one scale
        (`lab.batch_sizes[reactor]` rows).
    seed : int
        Seed all your randomness from this.
    """
    rng = np.random.default_rng(seed)
    while lab.affordable_reactors():  # empty once no scale can run another batch
        reactor = rng.choice(lab.affordable_reactors())
        size = lab.batch_sizes[reactor]
        if reactor == lab.screening_reactor:
            # Plate: one clone per well, one shared temperature, pH and medium, no feeds.
            clones = rng.choice(lab.celltypes, size=size, replace=size > len(lab.celltypes))
            temperature, ph, glucose, glutamine = [rng.uniform(low, high) for low, high in lab.bounds[:4]]
            x = lab.plate(clones, temperature, ph, glucose, glutamine)
        else:
            # Bioreactors: every run has its own clone, conditions, feeds and temperature/pH shifts.
            x = np.empty((size, 2 + len(lab.bounds)), dtype=object)
            x[:, 0] = reactor
            x[:, 1] = rng.choice(lab.celltypes, size=size)
            for i, (low, high) in enumerate(lab.bounds):
                x[:, 2 + i] = rng.uniform(low, high, size=size).tolist()
        titers, spent = lab(x)
