"""Offline: how well do screening measurements predict a clone's Pilot potential? (analysis only)
For every clone of N DEV scenarios, at the TRUE T/pH optimum: MBR and Pilot titer at the 3 portfolio recipes."""
import json, sys
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
PORT = [(-1.7959, -2.699, 0.6), (-1.5229, -1.699, 0.85), (-2.0969, -2.699, 0.85), (-2.5, -2.2, 0.4), (-1.2, -2.7, 0.3)]
V = json.load(open("clone_value.json"))


def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    out = []
    for ci, c in enumerate(sc["clones"]):
        row = []
        for reactor in ("MBR", "Pilot"):
            for la0, la1, b2 in PORT:
                u = np.array([0, 0, 1, 10 ** la0, 1, 10 ** la1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
                x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
                row.append(f(reactor, ci, x))
        out.append(row)
    return sid, out


if __name__ == "__main__":
    ids = json.load(open("splits.json"))["dev"][:int(sys.argv[1])]
    with Pool(20) as p:
        res = dict(p.map(run, ids))
    json.dump({str(k): v for k, v in res.items()}, open("screen_study.json", "w"))
