"""Screening coverage per run from a trace file: plates run, clones seen on a plate / in an MBR, potential of best
seen clone vs best clone (analysis only). python dbg/coverage.py traces/X_dev.json"""
import json, sys, numpy as np
CV = json.load(open("clone_value.json"))
R = json.load(open(sys.argv[1]))
rows = []
for r in R:
    v = np.array(CV[str(r["sid"])]); best = v.max()
    names = None
    plates = 0; seenp = set(); seenm = set(); pil = set()
    for b in r["log"]:
        for x in b["x"]:
            c = int(x[1].split("_")[1]) - 1
            if x[0] == "MTP": seenp.add(c)
            elif x[0] == "MBR": seenm.add(c)
            else: pil.add(c)
        plates += sum(1 for x in b["x"] if x[0] == "MTP") // 24 if b["x"] else 0
    n = len(v)
    bp = v[list(seenp)].max() / best if seenp else 0
    bm = v[list(seenm)].max() / best if seenm else 0
    bpl = v[list(pil)].max() / best if pil else 0
    rows.append((r["sid"], n, plates, len(seenp), len(seenm), bp, bm, bpl, r["score"] / (3 * r["oracle"])))
A = np.array(rows, float)
print("mean: nclones %.1f plates %.1f plated %.1f mbr %.1f | best plated %.3f best mbr %.3f best pilot %.3f | score %.3f" % tuple(A[:, 1:].mean(0)))
print("runs where not all clones plated:", int((A[:, 3] < A[:, 1]).sum()), " mean best-plated there %.3f" % A[A[:, 3] < A[:, 1], 5].mean())
for r in sorted(rows, key=lambda t: t[5])[:12]:
    print("sid %4d n %2d plates %2d plated %2d mbr %2d | plated %.2f mbr %.2f pilot %.2f score %.2f" % r)
