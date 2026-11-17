"""Pooled paired differences over several splits from CACHED results (run evaluate.py first).
python dbg/pooled.py BASE.py ALG.py ... [--splits dev,off_dev,...]"""
import json, sys, numpy as np
import evaluate as E
from pathlib import Path
args = [a for a in sys.argv[1:] if not a.startswith("--")]
sp = "dev,off_dev,hold,off_hold,off_low,off_wide"
for a in sys.argv[1:]:
    if a.startswith("--splits="): sp = a.split("=", 1)[1]
S = json.loads(Path("splits.json").read_text())
ids = sum([S[s] for s in sp.split(",")], [])
def per(p): return np.array([E.normalise(E.job((p, i, None)), i) for i in ids])
b = per(args[0])
for p in args[1:]:
    d = per(p) - b; d = d[~np.isnan(d)]
    print("%-14s pooled over %s (n=%d): %+.4f +- %.4f" % (Path(p).stem, sp, len(d), d.mean(), d.std() / np.sqrt(len(d))))
