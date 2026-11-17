"""Optimal MBR vs Pilot recipe for the oracle clone (analysis only)."""
import json, numpy as np
from multiprocessing import Pool
from scipy.optimize import differential_evolution
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); DEV = json.load(open("splits.json"))["dev"][:12]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]
    xp = np.array(o["x"])
    r = differential_evolution(lambda u: -f("MBR", ci, u), b, maxiter=60, popsize=8, seed=0, x0=xp, tol=1e-8, polish=False)
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    return (xp - lo) / sp, (r.x - lo) / sp, f("Pilot", ci, r.x) / o["oracle"], -r.fun / f("MBR", ci, xp)
if __name__ == "__main__":
    with Pool(6) as p: res = p.map(run, DEV)
    np.set_printoptions(precision=2, suppress=True)
    print("cols: T0 P0 g0 a0 g1 a1 g2 b2 T1 P1 T2 P2")
    for up, um, tr, gain in res:
        print("pilot", up); print("mbr  ", um, f"MBR-opt on Pilot {tr:.2f}; MBR gain vs pilot recipe {gain:.2f}")
    d = np.array([um - up for up, um, _, _ in res]); print("mean diff mbr-pilot", d.mean(0))
