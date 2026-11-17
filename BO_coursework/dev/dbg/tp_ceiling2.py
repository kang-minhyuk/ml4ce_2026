"""Offline: Pilot T/pH optimum vs the clone's (e_activation, ph_opt), and the used operating point (analysis only)."""
import json, sys, numpy as np
from sim import Sim, load_scenario
T = json.load(open(sys.argv[1]))
U, B = [], []
for r in T:
    if not r["pilots"]: continue
    sc = load_scenario(r["sid"]); f = Sim(sc); names = [c["name"] for c in sc["clones"]]; bd = f.bounds
    best = None
    for b in r["log"][1:]:
        for x in b["x"]:
            if x[0] == "Pilot":
                ci = names.index(x[1]); v = f("Pilot", ci, x[2:])
                if best is None or v > best[0]: best = (v, ci, np.array(x[2:], float))
    v0, ci, x = best; c = sc["clones"][ci]
    bestv, bt, bp = v0, x[0], x[1]
    for t in np.linspace(c["e_activation"] - 3, c["e_activation"] + 3, 31):
        for p in np.linspace(c["ph_opt"] - 0.5, c["ph_opt"] + 0.5, 26):
            y = x.copy(); y[[0, 8, 10]] = np.clip(t, *bd[0]); y[[1, 9, 11]] = np.clip(p, *bd[1])
            v = f("Pilot", ci, y)
            if v > bestv: bestv, bt, bp = v, t, p
    U.append([x[0] - c["e_activation"], x[1] - c["ph_opt"]]); B.append([bt - c["e_activation"], bp - c["ph_opt"], bestv / v0])
U, B = np.array(U), np.array(B)
print("used - clone opt: T med %.2f mean %.2f sd %.2f | pH med %.3f mean %.3f sd %.3f" % (np.median(U[:, 0]), U[:, 0].mean(), U[:, 0].std(), np.median(U[:, 1]), U[:, 1].mean(), U[:, 1].std()))
print("Pilot best - clone opt: T med %.2f mean %.2f sd %.2f | pH med %.3f mean %.3f sd %.3f" % (np.median(B[:, 0]), B[:, 0].mean(), B[:, 0].std(), np.median(B[:, 1]), B[:, 1].mean(), B[:, 1].std()))
print("gain mean %.4f" % B[:, 2].mean())
