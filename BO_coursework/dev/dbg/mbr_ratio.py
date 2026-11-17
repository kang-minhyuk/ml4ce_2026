"""Compare the MBR/Pilot titer relation (prior recipe and grid optimum) of the practice lab with library scenarios."""
import numpy as np, json, re
from sim import Sim, load_scenario, practice_scenario
src = open("final/informed_bayes2.py").read()
cf = {k: float(re.search(r"'%s': ([-0-9.]+)" % k, src).group(1)) for k in ("la0", "la1", "b0")}
A0 = np.geomspace(0.015, 4.0, 13); A1 = np.geomspace(0.06, 10.0, 12); B2 = np.array([6, 12, 18, 25, 32, 40, 48.])
def stats(sc, cis):
    f = Sim(sc); b = f.bounds; lo, hi = b[:, 0], b[:, 1]; out = []
    for ci in cis:
        c = sc["clones"][ci]
        def x_of(a0, a1, b2):
            x = np.array([0, 0, hi[2], a0, hi[4], a1, hi[6], b2, 0, 0, 0, 0.])
            x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
            return np.clip(x, lo, hi)
        q = (lo[3] + 10 ** cf["la0"] * (hi[3] - lo[3]), lo[5] + 10 ** cf["la1"] * (hi[5] - lo[5]), lo[7] + cf["b0"] * (hi[7] - lo[7]))
        pm = f("Pilot", ci, x_of(*q)); mm = f("MBR", ci, x_of(*q))
        G = [(a0, a1, b2) for a0 in A0[::2] for a1 in A1[::2] for b2 in B2[::2] if a0 <= hi[3]]
        P = np.array([f("Pilot", ci, x_of(*g)) for g in G]); M = np.array([f("MBR", ci, x_of(*g)) for g in G])
        out.append([np.log(mm / pm), np.log(M.max() / P.max()), np.log(mm / M.max()), np.log(pm / P.max()), np.corrcoef(np.log(M), np.log(P))[0, 1]])
    return np.array(out)
pr = stats(practice_scenario(), range(len(practice_scenario()["clones"])))
S = json.load(open("splits.json")); lib = np.vstack([stats(load_scenario(s), [0, 1, 2]) for s in S["off_dev"][:40]])
names = ["log MBR/Pilot @prior", "log MBRmax/Pilotmax", "log MBR eff @prior", "log Pilot eff @prior", "corr surfaces"]
for k, n in enumerate(names):
    print("%-22s practice mean %+.3f sd %.3f | off_dev mean %+.3f sd %.3f" % (n, pr[:, k].mean(), pr[:, k].std(), lib[:, k].mean(), lib[:, k].std()))
