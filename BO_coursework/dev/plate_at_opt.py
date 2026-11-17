"""Plate titer (rich medium) at each clone's true T/pH optimum, DEV scenarios (analysis only)."""
import json, sys
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario


def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    out = []
    for ci, c in enumerate(sc["clones"]):
        T = np.clip(c["e_activation"], *b[0]); P = np.clip(c["ph_opt"], *b[1])
        row = []
        for g, a in ((1.0, 0.05), (0.3, 0.3), (1.0, 0.01), (1.0, 0.2)):
            x = [T, P, lo[2] + g * sp[2], lo[3] + a * sp[3], 0, 0, 0, 0, T, P, T, P]
            row.append(f("MTP", ci, x))
        out.append(row)
    return str(sid), out


if __name__ == "__main__":
    ids = json.load(open("splits.json"))["dev"][:100]
    with Pool(3) as p:
        res = dict(p.map(run, ids))
    json.dump(res, open("plate_at_opt.json", "w"))
