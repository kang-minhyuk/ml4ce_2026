"""Optimal glutamine (a0, a1, b2) with T/pH at the clone optimum; sensitivity (analysis only)."""
import json, numpy as np
from multiprocessing import Pool
from scipy.optimize import differential_evolution
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); DEV = json.load(open("splits.json"))["dev"][:30]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    tc = np.clip(c["e_activation"], *b[0]); pc = np.clip(c["ph_opt"], *b[1])
    def val(a0, a1, b2, g=1.0):
        u = np.array([0, 0, g, a0, g, a1, g, b2, 0, 0, 0, 0.]); x = lo + u * sp
        x[[0, 8, 10]] = tc; x[[1, 9, 11]] = pc; return f("Pilot", ci, x)
    r = differential_evolution(lambda v: -val(*v), [(0, .5), (0, .5), (0, 1)], maxiter=40, popsize=12, seed=0, tol=1e-10, polish=False)
    a0, a1, b2 = r.x; best = -r.fun
    tied = differential_evolution(lambda v: -val(v[0], v[0], v[1]), [(0, .5), (0, 1)], maxiter=30, popsize=12, seed=0, polish=False)
    sens = [val(a0 * m, a1, b2) / best for m in (0.5, 2)] + [val(a0, a1 * m, b2) / best for m in (0.5, 2)] + [val(a0, a1, b2 * m) / best for m in (0.7, 1.3)]
    return [a0, a1, b2, best / o["oracle"], -tied.fun / best, *sens, sp[3], sp[5], sp[7]]
if __name__ == "__main__":
    with Pool(6) as p: res = np.array(p.map(run, DEV))
    np.set_printoptions(precision=3, suppress=True)
    print("a0 a1 b2 | /oracle | tied/free | a0x.5 a0x2 a1x.5 a1x2 b2x.7 b2x1.3 | gln span s,f1,f2")
    print(res)
    print("median", np.median(res, 0))
