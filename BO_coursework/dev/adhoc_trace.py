"""Ad-hoc batch-by-batch trace on one scenario: python adhoc_trace.py ALGO.py SID"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "../scenario_suite"); sys.path.insert(0, ".")
import numpy as np, run_suite
from trace import Recorder
import digital_lab.lab as L
from digital_lab import experiment
path, sid = sys.argv[1], int(sys.argv[2])
sc = json.loads((run_suite.SUITE / "scenarios" / f"scenario_{sid:04d}.json").read_text())
names = [c["name"] for c in sc["clones"]]
L.CELLTYPES = names
L.LAB_CONFIG_FILENAMES = {**L.LAB_CONFIG_FILENAMES, **{n: f"{n}.json" for n in names}}
L.CONTINUOUS_BOUNDS = [tuple(b) for b in sc["bounds"]]
algo = run_suite.load_algorithm(path); log = []
def wrapped(ix, iy, lab, seed):
    log.append({"x": ix.tolist(), "y": list(iy), "spent": 0})
    return algo(ix, iy, Recorder(lab, log), seed)
_, info = experiment.run_algorithm(wrapped, "t", sc["seed"], {"config_dir": str(run_suite.materialise(sc)), **sc["lab"]})
print("score", round(info["score"]), info["checkpoint_bests"], "bounds T", sc["bounds"][0], "pH", sc["bounds"][1], "glc", sc["bounds"][2], "gln", sc["bounds"][3])
for b in log:
    for r, y in zip(b["x"], b["y"]):
        if r[0] == "MTP": continue
        print(f'{b["spent"]:6d} {r[0]:5s} {r[1]:12s} y={y:8.0f} | ' + " ".join(f"{float(v):6.1f}" for v in r[2:]))
    mt = [(r[1], y) for r, y in zip(b["x"], b["y"]) if r[0] == "MTP"]
    if mt:
        r = b["x"][0]; print(f'{b["spent"]:6d} MTP   cond ' + " ".join(f"{float(v):6.1f}" for v in r[2:6]) + f"  best {max(mt, key=lambda t: t[1])}")
