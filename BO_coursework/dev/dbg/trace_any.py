"""Trace runs: record every batch, then diagnose (clone choice vs recipe quality). Analysis only.

python dev/trace.py ALGO.py --split dev --n 30
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

DEV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEV.parent / "scenario_suite"))
sys.path.insert(0, str(DEV))
import run_suite  # noqa: E402

ORACLE = json.loads((DEV / "oracle.json").read_text())
for _k, _v in json.loads((DEV / "oracle_proxy.json").read_text()).items():
    ORACLE.setdefault(_k, _v)


class Recorder:
    def __init__(self, lab, log):
        object.__setattr__(self, "_lab", lab)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name):
        return getattr(self._lab, name)

    def __call__(self, x):
        y, spent = self._lab(x)
        self._log.append({"x": [[str(v) if i < 2 else float(v) for i, v in enumerate(r)] for r in x],
                          "y": [float(v) for v in y], "spent": int(spent)})
        return y, spent


def run(args):
    path, sid = args
    import digital_lab.lab as L
    from digital_lab import experiment
    from sim import Sim
    from sim import load_scenario
    sc = load_scenario(sid)
    names = [c["name"] for c in sc["clones"]]
    L.CELLTYPES = names
    L.LAB_CONFIG_FILENAMES = {**L.LAB_CONFIG_FILENAMES, **{n: f"{n}.json" for n in names}}
    L.CONTINUOUS_BOUNDS = [tuple(b) for b in sc["bounds"]]
    algo = run_suite.load_algorithm(path)
    log = []

    def wrapped(ix, iy, lab, seed):
        log.append({"x": [[str(v) if i < 2 else float(v) for i, v in enumerate(r)] for r in ix],
                    "y": [float(v) for v in iy], "spent": None})
        return algo(ix, iy, Recorder(lab, log), seed)

    if sid >= 10000:
        d = DEV / ".labs_off" / f"scenario_{sid - 10000:04d}"
        if not (d / "noise_parameters.json").exists():
            d.mkdir(parents=True, exist_ok=True)
            for name, cfg in sc["reactors"].items():
                (d / f"{name}.json").write_text(json.dumps(cfg))
            for c in sc["clones"]:
                (d / f"{c['name']}.json").write_text(json.dumps(c))
            (d / "noise_parameters.json").write_text(json.dumps(sc["noise"]))
        cdir = str(d)
    else:
        cdir = str(run_suite.materialise(sc))
    _, info = experiment.run_algorithm(wrapped, "t", sc["seed"], {"config_dir": cdir, **sc["lab"]})
    f = Sim(sc)
    o = ORACLE[str(sid)]
    pilots = []
    for b in log[1:]:
        for r, y in zip(b["x"], b["y"]):
            if r[0] == "Pilot":
                ci = names.index(r[1])
                pilots.append({"clone": ci, "true": f("Pilot", ci, r[2:]), "spent": b["spent"]})
    # clone quality: the clone's own stage-1-ish optimum is unknown; use rank of clone by oracle stage
    return {"sid": sid, "score": info["score"], "oracle": o["oracle"], "oracle_clone": o.get("clone", -1), "pilots": pilots,
            "log": log, "end": info["end_reason"], "err": info["error"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("algo")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--j", type=int, default=21)
    a = ap.parse_args()
    ids = [int(x) for x in a.split.split(":")[1].split(",")] if a.split.startswith("ids:") else \
        [i for i in json.loads((DEV / "splits.json").read_text())[a.split] if str(i) in ORACLE][:a.n]
    with Pool(a.j) as p:
        res = p.map(run, [(a.algo, i) for i in ids], chunksize=1)
    out = DEV / "traces" / f"{Path(a.algo).stem}_{a.split.replace(':', '_').replace(',', '-')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res))
    for r in res:
        ps = r["pilots"]
        best = max(ps, key=lambda p: p["true"]) if ps else None
        seq = " ".join(f"{p['clone']}:{p['true'] / r['oracle']:.2f}" for p in ps)
        print(f"sid {r['sid']:4d} oracle clone {r['oracle_clone']:2d} | pilots {seq}")
