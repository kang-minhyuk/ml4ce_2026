"""Diagnose traced failures: per Pilot clone, its potential (best glutamine grid at true T/pH), the used T/pH vs the
clone's optimum, and the value of the used recipe at the true T/pH. (analysis only)"""
import json, sys, numpy as np
from sim import Sim, load_scenario
from pathlib import Path
T = json.load(open(sys.argv[1]))
A0 = np.geomspace(0.015, 4.0, 13)[::2]; A1 = np.geomspace(0.06, 10.0, 12)[::2]; B2 = np.array([6, 12, 18, 25, 32, 40, 48.])[::2]
for r in T:
    sid = r["sid"]
    sc = json.loads((Path("scen_official") / f"scenario_{sid - 10000:04d}.json").read_text()) if sid >= 10000 else load_scenario(sid)
    f = Sim(sc); b = f.bounds; lo, hi = b[:, 0], b[:, 1]; names = [c["name"] for c in sc["clones"]]
    def pot(ci):
        c = sc["clones"][ci]; best = 0
        for a0 in A0:
            for a1 in A1:
                for b2 in B2:
                    x = np.array([0, 0, hi[2], a0, hi[4], a1, hi[6], b2, 0, 0, 0, 0.])
                    x[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); x[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
                    best = max(best, f("Pilot", ci, np.clip(x, lo, hi)))
        return best
    pots = {}
    print("sid", sid, "oracle %.0f" % r["oracle"], "bounds T", b[0], "pH", b[1])
    for bb in r["log"][1:]:
        for x in bb["x"]:
            if x[0] != "Pilot": continue
            ci = names.index(x[1]); c = sc["clones"][ci]; xx = np.array(x[2:], float)
            if ci not in pots: pots[ci] = pot(ci)
            y = f("Pilot", ci, xx); xt = xx.copy(); xt[[0, 8, 10]] = np.clip(c["e_activation"], *b[0]); xt[[1, 9, 11]] = np.clip(c["ph_opt"], *b[1])
            print("  clone %2d pot %.2f | T %.2f (opt %.2f) pH %.2f (opt %.2f) | y %.2f, at true T/pH %.2f | gln a0 %.3g a1 %.3g b2 %.1f" % (
                ci, pots[ci] / r["oracle"], xx[0], c["e_activation"], xx[1], c["ph_opt"], y / r["oracle"], f("Pilot", ci, xt) / r["oracle"], xx[3], xx[5], xx[7]))
    allp = sorted(((pot(i) / r["oracle"], i) for i in range(len(names))), reverse=True)[:5]
    print("  top clones by potential:", ["%d:%.2f" % (i, p) for p, i in allp])
