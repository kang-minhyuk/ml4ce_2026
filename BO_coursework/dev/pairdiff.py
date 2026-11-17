"""Per-scenario paired differences from cached results: python pairdiff.py A.py B.py --split dev [--k 10]"""
import argparse, json
import numpy as np
import evaluate as E
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--split", default="dev")
ap.add_argument("--k", type=int, default=8); a = ap.parse_args()
ids = json.load(open(E.DEV / "splits.json"))[a.split]
rows = []
for i in ids:
    ra, rb = E.job((a.a, i, None)), E.job((a.b, i, None))
    na, nb = E.normalise(ra, i), E.normalise(rb, i)
    if na is None or nb is None:
        continue
    fa = E.normalise({"score": 1.0}, i) * 3
    rows.append((nb - na, i, [round((x or 0) * fa, 2) for x in ra["checkpoint_bests"]], [round((x or 0) * fa, 2) for x in rb["checkpoint_bests"]]))
rows.sort()
print("worst:"); [print(f"  {d:+.3f} sid {i}  A {ca}  B {cb}") for d, i, ca, cb in rows[:a.k]]
print("best:"); [print(f"  {d:+.3f} sid {i}  A {ca}  B {cb}") for d, i, ca, cb in rows[-a.k:]]
