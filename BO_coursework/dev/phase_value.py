"""How much do early-phase T/pH and glutamine tuning add over the structured recipe? (analysis only)"""
import json, numpy as np
from multiprocessing import Pool
from scipy.optimize import differential_evolution
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); DEV = json.load(open("splits.json"))["dev"][:16]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    tc = np.clip(c["e_activation"], *b[0]); pc = np.clip(c["ph_opt"], *b[1])
    def x_of(T0, P0, T1, P1, a0, a1, b2, T2=tc, P2=pc):
        u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
        x[[0, 1, 8, 9, 10, 11]] = [T0, P0, T1, P1, T2, P2]; return np.clip(x, b[:, 0], b[:, 1])
    # structured: all phases at optimum, gln grid
    base = max(f("Pilot", ci, x_of(tc, pc, tc, pc, .02, .03, q)) for q in (.25, .4, .6))
    # gln free, T/pH at optimum
    rg = differential_evolution(lambda v: -f("Pilot", ci, x_of(tc, pc, tc, pc, *v)), [(0, .3), (0, .3), (0, 1)], maxiter=30, popsize=10, seed=0, polish=False)
    a0, a1, b2 = rg.x
    # early phases free (gln at its optimum)
    rp = differential_evolution(lambda v: -f("Pilot", ci, x_of(*v, a0, a1, b2)), [b[0], b[1], b[0], b[1]], maxiter=30, popsize=10, seed=0, polish=False)
    # final phase also free
    rf = differential_evolution(lambda v: -f("Pilot", ci, x_of(*v[:4], a0, a1, b2, v[4], v[5])), [b[0], b[1], b[0], b[1], b[0], b[1]], maxiter=40, popsize=10, seed=0, polish=False)
    return base / o["oracle"], -rg.fun / o["oracle"], -rp.fun / o["oracle"], -rf.fun / o["oracle"]
if __name__ == "__main__":
    with Pool(6) as p: res = np.array(p.map(run, DEV))
    print("structured %.3f | +gln %.3f | +early T/pH %.3f | +final T/pH %.3f" % tuple(res.mean(0)))
    print(res.round(2))
