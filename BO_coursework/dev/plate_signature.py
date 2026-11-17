"""Offline: can cheap measurements predict a clone's Pilot potential better than the rich-plate height / MBR screen?
For every clone of training scenarios (never used for validation), at the TRUE T/pH optimum:
  plates at a grid of media (glucose x glutamine fractions of the bounds), MBR at the 3 portfolio recipes,
  Pilot at the 3 portfolio recipes, Pilot potential (grid-best glutamine).
python plate_signature.py N_SCEN -> plate_signature.npz (analysis / data generation only)"""
import json, sys, itertools
import numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario

GLC = [0.3, 1.0]
GLN = [0.005, 0.02, 0.05, 0.15, 0.4]
PORT = [(-1.7959, -2.699, 0.6), (-1.5229, -1.699, 0.85), (-2.0969, -2.699, 0.85)]
A0 = [.001, .003, .008, .02, .05]; A1 = [.003, .008, .02, .05, .12]; B2 = [.3, .6]


def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    rows = []
    for ci, c in enumerate(sc["clones"]):
        T = float(np.clip(c["e_activation"], *b[0])); P = float(np.clip(c["ph_opt"], *b[1]))
        plate = [f("MTP", ci, [T, P, lo[2] + g * sp[2], lo[3] + a * sp[3], 0, 0, 0, 0, T, P, T, P])
                 for g in GLC for a in GLN]

        def bio(reactor, a0, a1, b2):
            u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = T; x[[1, 9, 11]] = P
            return f(reactor, ci, x)
        mbr = [bio("MBR", 10 ** la0, 10 ** la1, b2) for la0, la1, b2 in PORT]
        pil = [bio("Pilot", 10 ** la0, 10 ** la1, b2) for la0, la1, b2 in PORT]
        pot = max(bio("Pilot", a0, a1, b2) for a0, a1, b2 in itertools.product(A0, A1, B2))
        params = [c[k] for k in sorted(c) if isinstance(c[k], (int, float))]
        rows.append(plate + mbr + pil + [pot] + params)
    return sid, rows, sorted(k for k in sc["clones"][0] if isinstance(sc["clones"][0][k], (int, float)))


if __name__ == "__main__":
    S = json.load(open("splits.json")); used = set(sum(S.values(), []))
    free = [i for i in range(1000) if i not in used]
    rng = np.random.default_rng(5); sids = sorted(rng.choice(free, int(sys.argv[1]), replace=False).tolist())
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as p:
        res = p.map(run, sids, chunksize=1)
    X = np.concatenate([np.array(r) for _, r, _ in res]); sid = np.concatenate([[s] * len(r) for s, r, _ in res])
    np.savez_compressed("plate_signature.npz", X=X, sid=sid, pnames=res[0][2], GLC=GLC, GLN=GLN)
    print(X.shape)
