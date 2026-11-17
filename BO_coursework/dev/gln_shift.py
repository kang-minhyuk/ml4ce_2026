"""Is the MBR -> Pilot shift of the glutamine optimum consistent across clones within a scenario? (analysis only)"""
import json, itertools, numpy as np
from multiprocessing import Pool
from sim import Sim, load_scenario
A0 = [.0005, .001, .002, .004, .008, .016, .03, .06, .12]; A1 = [.002, .005, .01, .02, .04, .08, .15]; B2 = [.25, .4, .6, .85]
GRID = list(itertools.product(A0, A1, B2))
V = json.load(open("clone_value.json")); IDS = [int(k) for k in list(V)[:16]]
def run(args):
    sid, ci = args
    sc = load_scenario(sid); f = Sim(sc); b = np.array(sc["bounds"], float); lo, sp = b[:, 0], b[:, 1] - b[:, 0]; c = sc["clones"][ci]
    out = {}
    for r in ("MBR", "Pilot"):
        vals = []
        for a0, a1, b2 in GRID:
            u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
            vals.append(f(r, ci, x))
        out[r] = np.array(vals)
    return sid, ci, out
if __name__ == "__main__":
    jobs = []
    for sid in IDS:
        v = np.array(V[str(sid)]); jobs += [(sid, int(c)) for c in np.argsort(-v)[:5]]
    with Pool(20) as p: res = p.map(run, jobs)
    by = {}
    for sid, ci, out in res: by.setdefault(sid, []).append((ci, out))
    gains_shift, gains_direct, gains_prior = [], [], []
    pr = GRID.index((.016, .002, .6))
    for sid, lst in by.items():
        # learn the shift on the first (incumbent) clone: map MBR-optimal grid point -> Pilot-optimal grid point in log space
        L = lambda g: np.array([np.log10(g[0]), np.log10(g[1]), g[2]])
        ci0, o0 = lst[0]
        shift = L(GRID[np.argmax(o0["Pilot"])]) - L(GRID[np.argmax(o0["MBR"])])
        for ci, o in lst[1:]:
            pm = L(GRID[np.argmax(o["MBR"])]) + shift
            j = int(np.argmin([np.sum((L(g) - pm) ** 2 * [1, 1, 4]) for g in GRID]))
            best = o["Pilot"].max()
            gains_shift.append(o["Pilot"][j] / best); gains_direct.append(o["Pilot"][np.argmax(o["MBR"])] / best); gains_prior.append(o["Pilot"][pr] / best)
    print(f"Pilot value / clone's Pilot grid-best (other clones, n={len(gains_shift)}): prior {np.mean(gains_prior):.3f} | "
          f"MBR-optimal recipe {np.mean(gains_direct):.3f} | MBR-optimal + shift learned on incumbent {np.mean(gains_shift):.3f}")
