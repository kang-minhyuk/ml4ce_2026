"""Per-clone landscape study: best titer per scale, transfer between scales, value of shifts/feeds.

python dev/landscape.py SID [SID ...]   (SID -1 = practice lab)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from sim import Sim, load_scenario, practice_scenario

OUT = Path(__file__).resolve().parent / "landscape"
OUT.mkdir(exist_ok=True)


def tied(b, u6):
    t, p, g, q, gf, qf = u6
    return [t, p, g, q, gf, qf, gf, qf, t, p, t, p]


def study_clone(args):
    sid, ci = args
    sc = practice_scenario() if sid == -1 else load_scenario(sid)
    f = Sim(sc)
    b = f.bounds
    res = {"clone": ci}
    # MTP: 4-D, no feeds, no shifts
    r = differential_evolution(lambda u: -f("MTP", ci, [u[0], u[1], u[2], u[3], 0, 0, 0, 0, u[0], u[1], u[0], u[1]]),
                               b[:4], maxiter=25, popsize=10, seed=0, tol=1e-6, polish=False)
    res["mtp_best"], res["mtp_x"] = -r.fun, r.x.tolist()
    u = r.x
    res["pilot_at_mtp"] = f("Pilot", ci, [u[0], u[1], u[2], u[3], 0, 0, 0, 0, u[0], u[1], u[0], u[1]])
    # Pilot, tied 6-D
    b6 = np.vstack([b[:4], b[4:6]])
    r = differential_evolution(lambda u: -f("Pilot", ci, tied(b, u)), b6, maxiter=30, popsize=10, seed=0, tol=1e-6,
                               polish=False)
    res["pilot_tied"], res["pilot_tied_x"] = -r.fun, r.x.tolist()
    # MBR full 12-D and its transfer to Pilot
    r = differential_evolution(lambda u: -f("MBR", ci, u), b, maxiter=30, popsize=8, seed=0, tol=1e-6, polish=False)
    res["mbr_full"], res["mbr_full_x"] = -r.fun, r.x.tolist()
    res["pilot_at_mbr"] = f("Pilot", ci, r.x)
    # Pilot full 12-D, warm-started from the tied optimum
    x0 = np.array(tied(b, res["pilot_tied_x"]))
    init = np.clip(x0 + np.random.default_rng(0).normal(0, 0.1, (96, 12)) * (b[:, 1] - b[:, 0]), b[:, 0], b[:, 1])
    init[0] = x0
    r = differential_evolution(lambda u: -f("Pilot", ci, u), b, maxiter=40, init=init, seed=0, tol=1e-6, polish=False)
    res["pilot_full"], res["pilot_full_x"] = -r.fun, r.x.tolist()
    res["mbr_at_pilot_full"] = f("MBR", ci, r.x)
    return res


if __name__ == "__main__":
    for sid in map(int, sys.argv[1:]):
        sc = practice_scenario() if sid == -1 else load_scenario(sid)
        n = len(sc["clones"])
        with Pool(21) as p:
            rows = p.map(study_clone, [(sid, i) for i in range(n)])
        for r, c in zip(rows, sc["clones"]):
            r.update({k: c[k] for k in ("e_activation", "ph_opt", "mu_max", "y_product_from_total",
                                        "y_product_from_viable")})
        (OUT / f"scen_{sid}.json").write_text(json.dumps({"bounds": sc["bounds"], "rows": rows}))
        print(f"scenario {sid}: done")
