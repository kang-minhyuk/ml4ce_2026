"""Where is the titer-optimal T/pH at each scale vs the growth optimum? (analysis only)"""
import json, math, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from sim import Sim, load_scenario
DEV = json.load(open("splits.json"))["dev"][:20]
PF0 = (0.016, 0.002, 0.6)
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]; out = []
    for ci in range(0, len(sc["clones"]), 6):
        c = sc["clones"][ci]; tt, pt = c["e_activation"], c["ph_opt"]
        if not (b[0, 0] + 1 < tt < b[0, 1] - 1 and b[1, 0] + .2 < pt < b[1, 1] - .2):
            continue
        res = [tt, pt]
        for reactor, med in (("MTP", (1.0, .05)), ("MTP", (.3, .3)), ("MBR", PF0), ("Pilot", PF0)):
            def val(v):
                x = np.zeros(12)
                if reactor == "MTP":
                    x[2], x[3] = lo[2] + med[0] * sp[2], lo[3] + med[1] * sp[3]
                else:
                    u = np.array([0, 0, 1, med[0], 1, med[1], 1, med[2], 0, 0, 0, 0.]); x = lo + u * sp
                x[[0, 8, 10]] = v[0]; x[[1, 9, 11]] = v[1]
                return -np.log(max(f(reactor, ci, np.clip(x, b[:, 0], b[:, 1])), 1e-9))
            grid = [(T, P) for T in np.linspace(tt - 3, tt + 3, 13) for P in np.linspace(pt - .6, pt + .6, 13)]
            g = min(grid, key=val)
            r = minimize(val, g, method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-6, "maxiter": 200})
            res += [r.x[0] - tt, r.x[1] - pt]
        out.append(res)
    return out
if __name__ == "__main__":
    with Pool(20) as p: R = np.array([x for o in p.map(run, DEV) for x in o])
    for k, name in enumerate(["MTP rich", "MTP 0.3/0.3", "MBR prior", "Pilot prior"]):
        d = R[:, 2 + 2 * k: 4 + 2 * k]
        print(f"{name:12s} dT mean {d[:, 0].mean():+.2f} sd {d[:, 0].std():.2f} | dpH mean {d[:, 1].mean():+.3f} sd {d[:, 1].std():.3f}  (n={len(d)})")
