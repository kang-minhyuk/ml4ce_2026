"""Oracle: best noise-free Pilot titer per scenario (for normalising scores). Analysis only.

python dev/oracle.py SID [SID ...]       (-1 = practice lab). Results appended to dev/oracle.json
Stage 1: cheap DE per clone. Stage 2: longer DE on the top clones, warm-started.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from sim import Sim, load_scenario, practice_scenario

OUT = Path(__file__).resolve().parent / "oracle.json"
_SIMS = {}


def sim(sid):
    if sid not in _SIMS:
        _SIMS[sid] = Sim(practice_scenario() if sid == -1 else load_scenario(sid))
    return _SIMS[sid]


def init_pop(b, n, rng, x0=None, scale=0.1):
    lo, hi = b[:, 0], b[:, 1]
    if x0 is None:
        pop = lo + rng.random((n, len(lo))) * (hi - lo)
        pop[:, [2, 4, 6]] = hi[[2, 4, 6]] * rng.uniform(0.8, 1.0, (n, 3))  # glucose high helps convergence
        pop[:, [3, 5, 7]] = hi[[3, 5, 7]] * rng.uniform(0.0, 0.3, (n, 3))
        return pop
    pop = np.clip(x0 + rng.normal(0, scale, (n, len(lo))) * (hi - lo), lo, hi)
    pop[0] = x0
    return pop


def stage1(args):
    sid, ci = args
    f = sim(sid); b = f.bounds
    rng = np.random.default_rng(ci)
    r = differential_evolution(lambda u: -f("Pilot", ci, u), b, maxiter=25, init=init_pop(b, 60, rng), seed=ci,
                               tol=1e-8, polish=False)
    return sid, ci, -r.fun, r.x.tolist()


def stage2(args):
    sid, ci, x0 = args
    f = sim(sid); b = f.bounds
    rng = np.random.default_rng(100 + ci)
    r = differential_evolution(lambda u: -f("Pilot", ci, u), b, maxiter=70, init=init_pop(b, 90, rng, np.array(x0)),
                               seed=ci, tol=1e-8, polish=False, mutation=(0.3, 0.9))
    return sid, ci, -r.fun, r.x.tolist()


if __name__ == "__main__":
    sids = [int(s) for s in sys.argv[1:]]
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    sids = [s for s in sids if str(s) not in done]
    TOP = 4
    with Pool(21) as pool:
        jobs = [(s, c) for s in sids for c in range(len(sim(s).clones))]
        s1 = pool.map(stage1, jobs, chunksize=1)
        by = {}
        for s, c, v, x in s1:
            by.setdefault(s, []).append((v, c, x))
        jobs2 = [(s, c, x) for s in sids for v, c, x in sorted(by[s], reverse=True)[:TOP]]
        s2 = pool.map(stage2, jobs2, chunksize=1)
    for s in sids:
        best = max([(v, c, x) for ss, c, v, x in s2 if ss == s] + by[s])
        done[str(s)] = {"oracle": best[0], "clone": best[1], "x": best[2],
                        "stage1": sorted([v for v, _, _ in by[s]], reverse=True)[:6]}
        OUT.write_text(json.dumps(done))
    print(f"oracle done for {len(sids)} scenarios")
