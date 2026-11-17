"""Offline: which common MBR recipe best identifies the clone with the highest Pilot potential? (analysis only)
Same training scenarios as plate_signature.py. MBR titer per clone at the TRUE T/pH on a grid of glutamine recipes
(glucose at its upper bound in every feed). -> screen_recipe.npz"""
import json, sys, itertools
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario

LA0 = [-3.0, -2.6, -2.3, -2.1, -1.9, -1.7, -1.5, -1.2]
LA1 = [-3.2, -2.7, -2.3, -1.9, -1.5]
B2 = [0.3, 0.6, 0.85]
GRID = list(itertools.product(LA0, LA1, B2))


def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    rows = []
    for ci, c in enumerate(sc["clones"]):
        T = float(np.clip(c["e_activation"], *b[0])); P = float(np.clip(c["ph_opt"], *b[1]))
        r = []
        for la0, la1, b2 in GRID:
            u = np.array([0, 0, 1, 10 ** la0, 1, 10 ** la1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = T; x[[1, 9, 11]] = P
            r.append(f("MBR", ci, x))
        rows.append(r)
    return sid, rows


if __name__ == "__main__":
    d = np.load("plate_signature.npz"); sids = sorted(set(d["sid"].tolist()))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 5) as p:
        res = p.map(run, sids, chunksize=1)
    M = np.concatenate([np.array(r) for _, r in res]); sid = np.concatenate([[s] * len(r) for s, r in res])
    np.savez_compressed("screen_recipe.npz", M=M, sid=sid, grid=np.array(GRID))
    print(M.shape)
