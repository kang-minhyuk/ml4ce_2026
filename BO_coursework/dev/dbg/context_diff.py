"""Per-scenario paired diffs on off_big vs scenario features (cached results only; analysis)."""
import json, sys, numpy as np
import evaluate as E
from pathlib import Path
ids = json.loads(Path("splits.json").read_text())["off_big"]
def per(p):
    return np.array([E.normalise(E.job((p, i, None)), i) for i in ids])
base = per("final/informed_bayes.py")
F = []
for i in ids:
    sc = json.loads((Path("scen_official") / f"scenario_{i-10000:04d}.json").read_text())
    lab = sc["lab"]; c = lab.get("reactor_costs", {}); pc = c.get("Pilot", 3000)
    ini = lab.get("initial_design", {"MTP": 1, "MBR": 1})
    spend0 = ini.get("MTP", 0) * 24 * c.get("MTP", 10) + ini.get("MBR", 0) * 4 * c.get("MBR", 500)
    F.append([lab["budget"] / pc, 4 * c.get("MBR", 500) / pc, (lab["budget"] / 3 - spend0) / pc,
              lab["max_experiments"], len(sc["clones"]), sc["noise"]["measurement_noise_c_product"]])
F = np.array(F); names = ["B/pc", "mbrb/pc", "ck1room/pc", "cap", "nclone", "noise"]
for p in sys.argv[1:]:
    d = per(p) - base
    print(Path(p).stem, "mean %+.4f" % d.mean())
    for k, n in enumerate(names):
        q = np.quantile(F[:, k], [0, 1/3, 2/3, 1]); g = np.digitize(F[:, k], q[1:-1])
        print("   %-11s" % n, "  ".join("[%.2g..] %+.4f±%.4f" % (q[j], d[g == j].mean(), d[g == j].std() / np.sqrt((g == j).sum())) for j in range(3)))
