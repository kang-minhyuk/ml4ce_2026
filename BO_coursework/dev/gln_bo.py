"""Mini-benchmark of the glutamine search alone (oracle clone, true T/pH): python gln_bo.py N_MBR N_PILOT [acq]
Uses i5's BioGP on z = (log10 a0, log10 a1, b2). MBR runs first (EI on MBR scale), then Pilot runs."""
import json, sys
import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
from sim import Sim, load_scenario
sys.path.insert(0, "algos")
O = json.load(open("oracle.json")); DEV = json.load(open("splits.json"))["dev"][:40]
NM, NP = int(sys.argv[1]), int(sys.argv[2]); ACQ = sys.argv[3] if len(sys.argv) > 3 else "ei"
import os
LS = tuple(float(v) for v in os.environ.get("LS", "0.6,0.6,0.3").split(","))
STEP = float(os.environ.get("STEP", "1"))
NGLOB = int(os.environ.get("NGLOB", "200"))
PRIOR = np.array([-2.0, -2.3, 0.6]); BOX = np.array([[-3.3, -0.5], [-3.3, -0.5], [0.1, 1.0]])


def run(sid):
    import i5_funnel as M
    sc = load_scenario(sid); f = Sim(sc); b = f.bounds; o = O[str(sid)]; ci = o["clone"]; c = sc["clones"][ci]
    lo, sp = b[:, 0], b[:, 1] - b[:, 0]
    rng = np.random.default_rng(sid)

    def val(r, z):
        u = np.array([0, 0, 1, 10 ** z[0], 1, 10 ** z[1], 1, z[2], 0, 0, 0, 0.]); x = lo + u * sp
        x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
        return f(r, ci, x)
    Z, S, Y = [], [], []
    gp = M.BioGP(M.CFG, rng, 3)
    best_p = []
    for k in range(NM + NP):
        scale = 0 if k < NM else 1
        if not Z:
            z = PRIOR.copy()
        else:
            y = np.log(np.maximum(Y, 1e-9) + 0.01 * max(Y))
            gp.fit(np.array(Z), np.zeros(len(Z), int), np.array(S), y, LS)
            cand = np.vstack([np.clip(np.array(Z)[np.argsort(-np.array(y))[:3]][rng.integers(0, min(3, len(Z)), 300)]
                                      + rng.normal(0, 1, (300, 3)) * np.array([0.3, 0.3, 0.15]) * STEP * rng.choice([0.3, 1, 2], (300, 1)),
                                      BOX[:, 0], BOX[:, 1]),
                              BOX[:, 0] + rng.random((NGLOB, 3)) * (BOX[:, 1] - BOX[:, 0])])
            mu, sd = gp.predict(cand, np.zeros(len(cand), int), np.full(len(cand), scale))
            yy = gp.y[gp.s == scale] if np.any(gp.s == scale) else gp.y
            inc = yy.max()
            if ACQ == "ei" or scale == 0:
                zz = (mu - inc) / sd; a = (mu - inc) * norm.cdf(zz) + sd * norm.pdf(zz)
            else:
                a = mu + 0.3 * sd
            z = cand[np.argmax(a)]
        Z.append(z); S.append(scale); Y.append(val("Pilot" if scale else "MBR", z))
        if scale:
            best_p.append(max(y for y, s in zip(Y, S) if s) / o["oracle"])
    return best_p


if __name__ == "__main__":
    with Pool(8) as p:
        res = np.array(p.map(run, DEV))
    print(f"NM={NM} NP={NP} {ACQ}: best Pilot/oracle after each Pilot run:", res.mean(0).round(3), "p10 final",
          np.percentile(res[:, -1], 10).round(3))
