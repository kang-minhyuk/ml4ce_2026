"""Offline: Pilot-scale T/pH response template (relative titer vs offsets from the clone's growth optimum, at its
grid-best glutamine recipe); fit asymmetric widths + soft floor like the plate template. Analysis only."""
import itertools, json, sys, numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
from sim import Sim, load_scenario
DT = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, 3, 4]; DP = [-0.5, -0.3, -0.2, -0.1, 0, 0.05, 0.1, 0.2, 0.3]
def run(sid):
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; lo, sp = b[:, 0], b[:, 1] - b[:, 0]; out = []
    for ci in range(3):
        c = sc["clones"][ci]
        def val(dt, dp, a0, a1, b2, reactor="Pilot"):
            u = np.array([0, 0, 1, a0, 1, a1, 1, b2, 0, 0, 0, 0.]); x = lo + u * sp
            x[[0, 8, 10]] = c["e_activation"] + dt; x[[1, 9, 11]] = c["ph_opt"] + dp
            return f(reactor, ci, np.clip(x, b[:, 0], b[:, 1]))
        g = max(itertools.product([.001, .003, .008, .02, .05], [.003, .008, .02, .05, .12], [.3, .6]), key=lambda z: val(0, 0, *z))
        v0 = val(0, 0, *g)
        if v0 <= 0: continue
        # skip offsets that leave the bounds (clipping would fake a flat response)
        rows = []
        for dt, dp in itertools.product(DT, DP):
            T, P = c["e_activation"] + dt, c["ph_opt"] + dp
            if b[0, 0] <= T <= b[0, 1] and b[1, 0] <= P <= b[1, 1]:
                rows.append((dt, dp, val(dt, dp, *g) / v0, val(dt, dp, *g, reactor="MBR") / max(val(0, 0, *g, reactor="MBR"), 1e-9)))
        out.append(rows)
    return out
def pen(dt, dp, w, fl):
    d = (dt / np.where(dt < 0, w[0], w[1])) ** 2 + (dp / np.where(dp < 0, w[2], w[3])) ** 2
    return fl * (1 - np.exp(-d / fl))
if __name__ == "__main__":
    S = json.load(open("splits.json")); used = set(sum(S.values(), []))
    free = [i for i in range(1000) if i not in used]; ids = np.random.default_rng(9).choice(free, 24, replace=False).tolist()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p: R = [r for rr in p.map(run, ids) for c in rr for r in c]
    A = np.array(R); dt, dp, yp, ym = A.T
    for name, y in [("Pilot", yp), ("MBR", ym)]:
        m = y > 1e-4
        fit = least_squares(lambda th: (np.log(y[m]) + pen(dt[m], dp[m], np.exp(th[:4]), np.exp(th[4]))),
                            np.log([1.5, 3.5, 0.8, 0.2, 5.0]), loss="soft_l1", f_scale=0.3)
        w = np.exp(fit.x[:4]); fl = np.exp(fit.x[4])
        print(name, "widths T-/T+/pH-/pH+", w.round(3), "floor", round(fl, 2), "ratio to plate", (w / np.array([1.25, 3.08, 0.65, 0.15])).round(2),
              "resid sd", np.std(fit.fun).round(3))
    np.save("pilot_tmpl.npy", A)
