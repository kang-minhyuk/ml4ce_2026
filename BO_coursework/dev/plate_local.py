"""Offline: per-clone local search of the plate titer optimum (noise-free plates) vs Pilot value (analysis only)."""
import json, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from sim import Sim, load_scenario
O = json.load(open("oracle.json")); SP = json.load(open("splits.json"))
def run(sid):
    rng = np.random.default_rng(sid)
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    xo = np.array(o["x"])
    def V(T, P):
        x = xo.copy(); x[[0, 8, 10]] = T; x[[1, 9, 11]] = P; return f("Pilot", ci, np.clip(x, b[:, 0], b[:, 1]))
    gmed = (b[2, 1], b[3, 0] + 0.05 * (b[3, 1] - b[3, 0]))
    def Pl(T, P):
        T = float(np.clip(T, *b[0])); P = float(np.clip(P, *b[1]))
        x = np.array([T, P, gmed[0], gmed[1], 0, 0, 0, 0, T, P, T, P]); return f("MTP", ci, x)
    Tt, Pt = c["e_activation"], c["ph_opt"]
    vbest = max(V(Tt, Pt), o["oracle"])
    T0, P0 = Tt + rng.normal(0.5, 1.0), Pt + rng.normal(-0.2, 0.25)
    res = {"start": V(T0, P0) / vbest, "true": V(Tt, Pt) / vbest}
    sT, sP = 1.0, 0.2  # scaling
    for k in (4, 6, 8, 12):
        # Nelder-Mead in scaled coords, maxfev k (includes start)
        fun = lambda z: -np.log(max(Pl(T0 + z[0] * sT, P0 + z[1] * sP), 1e-9))
        hist = []
        def wrapped(z):
            v = fun(z); hist.append((v, z.copy())); return v
        minimize(wrapped, [0, 0], method="Nelder-Mead", options={"maxfev": k, "initial_simplex": [[0, 0], [1, 0], [0, 1]], "xatol": 1e-3, "fatol": 1e-9})
        z = min(hist[:k], key=lambda t: t[0])[1]
        res["nm%d" % k] = V(T0 + z[0] * sT, P0 + z[1] * sP) / vbest
        # coordinate search with step halving
        z = np.zeros(2); fz = fun(z); n = 1; step = np.array([1.0, 1.0])
        while n < k:
            improved = False
            for d in (0, 1):
                for sgn in (1, -1):
                    if n >= k: break
                    zz = z.copy(); zz[d] += sgn * step[d]; v = fun(zz); n += 1
                    if v < fz: z, fz, improved = zz, v, True; break
            if not improved: step /= 2
        res["cs%d" % k] = V(T0 + z[0] * sT, P0 + z[1] * sP) / vbest
        # plate-optimum location error
        res["errT%d" % k] = abs(T0 + z[0] * sT - Tt); res["errP%d" % k] = abs(P0 + z[1] * sP - Pt)
    # exact plate optimum
    r = minimize(lambda z: -np.log(max(Pl(Tt + z[0], Pt + z[1] * 0.2), 1e-9)), [0, 0], method="Nelder-Mead", options={"xatol": 1e-3, "maxfev": 80})
    res["plateopt"] = V(Tt + r.x[0], Pt + r.x[1] * 0.2) / vbest
    return res
if __name__ == "__main__":
    with Pool(20) as p: R = p.map(run, SP["dev"])
    for k in R[0]: print("%-10s %.3f" % (k, np.mean([r[k] for r in R])))
