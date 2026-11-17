"""Moderate-medium plate ranking with realistic location error (analysis only).
For every DEV clone: moderate (0.3/0.3) plate titer at the true optimum shifted by (dT, dpH) offsets."""
import json, sys
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
OFF = [(0, 0), (-1, 0), (1, 0), (0, -0.15), (0, 0.15), (-2, 0), (2, 0), (0, -0.3), (0, 0.3), (1, -0.15), (-1, 0.15)]
MEDIA = [(0.3, 0.3), (1.0, 0.05), (0.5, 0.2)]


def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    out = []
    for ci, c in enumerate(sc["clones"]):
        row = []
        for g, a in MEDIA:
            for dt, dp in OFF:
                T = np.clip(c["e_activation"] + dt, *b[0]); P = np.clip(c["ph_opt"] + dp, *b[1])
                row.append(f("MTP", ci, [T, P, lo[2] + g * sp[2], lo[3] + a * sp[3], 0, 0, 0, 0, T, P, T, P]))
        out.append(row)
    return str(sid), out


if __name__ == "__main__":
    ids = json.load(open("splits.json"))["dev"][:100]
    with Pool(20) as p:
        res = dict(p.map(run, ids))
    json.dump(res, open("plate_mod_study.json", "w"))
