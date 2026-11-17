"""Clone funnel coverage of a dev trace: best plated -> best MBR/Pilot-screened -> best piloted -> final (analysis)."""
import json, sys, numpy as np
from sim import load_scenario
T = json.load(open(sys.argv[1])); V = json.load(open("clone_value.json"))
rows = []
for r in T:
    v = np.array(V[str(r["sid"])]); best = v.max()
    names = [c["name"] for c in load_scenario(r["sid"])["clones"]]
    mbr, plate = set(), set()
    for b in r["log"]:
        for x in b["x"]:
            c = names.index(x[1])
            (mbr if x[0] == "MBR" else plate if x[0] == "MTP" else set()).add(c)
    pil = set(p["clone"] for p in r["pilots"])
    if not pil: continue
    inc = r["pilots"][-1]["clone"]
    rows.append([len(v), len(plate), len(mbr), len(pil), max(v[list(plate)]) / best if plate else 0,
                 max(v[list(mbr | pil)]) / best, max(v[list(pil)]) / best, v[inc] / best])
R = np.array(rows)
print("n=%d clones %.1f plated %.1f mbr %.1f pilot clones %.2f" % ((len(R),) + tuple(R[:, :4].mean(0))))
print("best plated %.3f -> best screened %.3f -> best piloted %.3f -> last-pilot clone %.3f" % tuple(R[:, 4:].mean(0)))
print("frac best clone: plated %.2f screened %.2f piloted %.2f last %.2f" % tuple((R[:, 4:] > 0.999).mean(0)))
