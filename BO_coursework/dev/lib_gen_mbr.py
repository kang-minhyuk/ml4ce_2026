"""Glutamine-response library: Pilot titer on an absolute (a0, a1, b2) grid for several clones per scenario, at the
true T/pH optimum or a perturbed one (pipeline-like location error). Analysis/data generation only."""
import json, sys, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
A0 = np.geomspace(0.015, 4.0, 13); A1 = np.geomspace(0.06, 10.0, 12); B2 = np.array([6, 12, 18, 25, 32, 40, 48.])
def run(sid):
    rng = np.random.default_rng(sid + 77)
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; out = []
    def x_of(c, a0, a1, b2, dT=0., dp=0.):
        x = np.array([0, 0, b[2, 1], a0, b[4, 1], a1, b[6, 1], b2, 0, 0, 0, 0.])
        x[[0, 8, 10]] = np.clip(c["e_activation"] + dT, *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"] + dp, *b[1])
        return np.clip(x, b[:, 0], b[:, 1])
    pot = [f("Pilot", i, x_of(c, 0.8, 0.1, 30.)) for i, c in enumerate(sc["clones"])]
    order = list(np.argsort(pot)[::-1]); pick = order[:2] + [int(rng.choice(order[2:]))]
    for ci in pick:
        c = sc["clones"][ci]
        if rng.random() < 0.5: dT, dp = 0., 0.
        else: dT, dp = rng.normal(0.4, 0.9), rng.normal(-0.15, 0.2)
        Y = np.array([f("MBR", ci, x_of(c, a0, a1, b2, dT, dp)) for a0 in A0 for a1 in A1 for b2 in B2])
        out.append(dict(sid=sid, ci=int(ci), dT=dT, dp=dp, gmax=float(b[3, 1]), Y=Y.tolist()))
    return out
if __name__ == "__main__":
    S = json.load(open("splits.json")); used = set(sum(S.values(), []))
    free = [i for i in range(1000) if i not in used]
    rng = np.random.default_rng(1); suite = sorted(rng.choice(free, 250, replace=False).tolist())
    sids = suite + S["off_extra"] + list(range(10380, 10680))
    with Pool(21) as p:
        res = p.map(run, sids, chunksize=2)
    lib = [r for rr in res for r in rr]
    np.savez_compressed("gln_lib_mbr.npz", Y=np.array([r["Y"] for r in lib]), sid=[r["sid"] for r in lib],
                        ci=[r["ci"] for r in lib], dT=[r["dT"] for r in lib], dp=[r["dp"] for r in lib],
                        gmax=[r["gmax"] for r in lib], A0=A0, A1=A1, B2=B2)
    print(len(lib), "curves")
