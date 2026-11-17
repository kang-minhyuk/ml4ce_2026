"""Plate titer response shape around each clone's T/pH optimum at the rich plate medium (analysis only)."""
import json, sys
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
DT = np.linspace(-6, 6, 13); DP = np.linspace(-0.9, 0.9, 13)


def run(args):
    sid, ci = args
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    c = sc["clones"][ci]
    out = np.zeros((len(DT), len(DP)))
    for i, dt in enumerate(DT):
        for j, dp in enumerate(DP):
            T = c["e_activation"] + dt; P = c["ph_opt"] + dp
            x = [T, P, hi(b, 2), lo[3] + 0.05 * sp[3], 0, 0, 0, 0, T, P, T, P]
            out[i, j] = f("MTP", ci, x)
    return sid, ci, out.tolist()


def hi(b, k):
    return b[k, 1]


if __name__ == "__main__":
    ids = json.load(open("splits.json"))["dev"][:20]
    jobs = [(s, ci) for s in ids for ci in (0, 1, 2)]
    with Pool(20) as p:
        res = p.map(run, jobs)
    json.dump(res, open("plate_shape.json", "w"))
