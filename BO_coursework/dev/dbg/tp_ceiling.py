"""Offline: for each run's best Pilot, how much would the best T/pH (glutamine etc. fixed) add? (analysis only)"""
import json, sys, numpy as np
from sim import Sim, load_scenario
T = json.load(open(sys.argv[1]))
g = []; gT = []; dts = []; dps = []
for r in T:
    if not r["pilots"]: continue
    sc = load_scenario(r["sid"]); f = Sim(sc); names = [c["name"] for c in sc["clones"]]
    best = None
    for b in r["log"][1:]:
        for x in b["x"]:
            if x[0] == "Pilot":
                ci = names.index(x[1]); v = f("Pilot", ci, x[2:])
                if best is None or v > best[0]: best = (v, ci, np.array(x[2:], float))
    v0, ci, x = best; bd = f.bounds
    bestv, bt, bp = v0, 0, 0
    for dt in np.linspace(-3, 3, 25):
        for dp in np.linspace(-0.5, 0.5, 21):
            y = x.copy(); y[[0, 8, 10]] = np.clip(x[[0, 8, 10]] + dt, *bd[0]); y[[1, 9, 11]] = np.clip(x[[1, 9, 11]] + dp, *bd[1])
            v = f("Pilot", ci, y)
            if v > bestv: bestv, bt, bp = v, dt, dp
    g.append(bestv / v0); dts.append(bt); dps.append(bp)
g = np.array(g); print("n %d  mean gain x%.4f  median x%.4f  p90 x%.3f  frac>2%% %.2f" % (len(g), g.mean(), np.median(g), np.quantile(g, .9), np.mean(g > 1.02)))
print("best shift dT median %.2f mean %.2f | dpH median %.3f mean %.3f" % (np.median(dts), np.mean(dts), np.median(dps), np.mean(dps)))
