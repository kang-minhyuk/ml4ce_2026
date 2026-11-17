"""Run algorithms on the scenario suite and compare them.

    python run_suite.py ALGO.py [ALGO2.py ...] [--n 100] [--start 0] [--coursework-only] [--j 8]

ALGO.py must define `my_algorithm(initial_x, initial_y, lab, seed)` (or a function named like the file), exactly like a
coursework submission. Every run uses the official harness (`digital_lab.experiment.run_algorithm`): same scoring,
same 5-minute limit, same read-only lab view. Results are cached in results/ so re-runs only compute what is new.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per run, so parallel runs don't fight over cores

import numpy as np

SUITE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("BO_REPO", SUITE.parent))  # folder containing src/digital_lab
sys.path.insert(0, str(REPO / "src"))
LABS = SUITE / ".labs"
RESULTS = SUITE / "results"


def load_algorithm(path):
    spec = importlib.util.spec_from_file_location(f"_alg_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("my_algorithm", Path(path).stem):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ImportError(f"{path} defines neither my_algorithm nor {Path(path).stem}")


def materialise(sc):
    """Write the scenario's lab config (reactors, noise, clones) as the JSON files the lab reads."""
    d = LABS / f"scenario_{sc['id']:04d}"
    if not (d / "noise_parameters.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        for name, cfg in sc["reactors"].items():
            (d / f"{name}.json").write_text(json.dumps(cfg))
        for c in sc["clones"]:
            (d / f"{c['name']}.json").write_text(json.dumps(c))
        (d / "noise_parameters.json").write_text(json.dumps(sc["noise"]))
    return d


def run_one(args):
    path, sid, time_limit = args
    sc = json.loads((SUITE / "scenarios" / f"scenario_{sid:04d}.json").read_text())
    key = hashlib.sha1((Path(path).read_text() + json.dumps(sc) + str(time_limit)).encode()).hexdigest()[:12]
    out = RESULTS / Path(path).stem / f"scenario_{sid:04d}_{key}.json"
    if out.exists():
        return json.loads(out.read_text())
    import digital_lab.lab as L
    from digital_lab import experiment
    names = [c["name"] for c in sc["clones"]]
    L.CELLTYPES = names  # the lab reads its clone list and bounds from these module constants
    L.LAB_CONFIG_FILENAMES = {**L.LAB_CONFIG_FILENAMES, **{n: f"{n}.json" for n in names}}
    L.CONTINUOUS_BOUNDS = [tuple(b) for b in sc["bounds"]]
    kw = {"config_dir": str(materialise(sc)), **sc["lab"]}
    if time_limit:
        kw["time_limit"] = float(time_limit)
    hist, info = experiment.run_algorithm(load_algorithm(path), Path(path).stem, sc["seed"], kw)
    res = {"algorithm": Path(path).stem, "scenario": sid, "score": info["score"],
           "checkpoint_bests": info["checkpoint_bests"], "best_pilot": float(hist["best"].max() or 0.0) if hist.height else 0.0,
           "wall_time": info["wall_time"], "end_reason": info["end_reason"], "error": info["error"],
           "access_warnings": info["access_warnings"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("algos", nargs="+", help="algorithm files; the first one is the baseline for win rates")
    ap.add_argument("--n", type=int, default=100, help="number of scenarios (default 100)")
    ap.add_argument("--start", type=int, default=0, help="first scenario id")
    ap.add_argument("--coursework-only", action="store_true", help="only scenarios with the coursework batch sizes")
    ap.add_argument("--j", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="parallel processes")
    ap.add_argument("--time-limit", type=float, default=None, help="override the 300 s limit (e.g. to test speed)")
    a = ap.parse_args()

    ids = []
    for f in sorted((SUITE / "scenarios").glob("scenario_*.json")):
        sid = int(f.stem.split("_")[1])
        if sid < a.start:
            continue
        if a.coursework_only and not json.loads(f.read_text())["coursework_compatible_batches"]:
            continue
        ids.append(sid)
        if len(ids) >= a.n:
            break
    t0 = time.time()
    with Pool(a.j) as pool:
        res = pool.map(run_one, [(p, i, a.time_limit) for p in a.algos for i in ids], chunksize=1)

    names = [Path(p).stem for p in a.algos]
    by = {n: {r["scenario"]: r for r in res if r["algorithm"] == n} for n in names}
    lines = ["algorithm,scenario,score,best_at_33,best_at_66,best_at_100,wall_time,end_reason,error"]
    for r in res:
        b = [x if x is not None else "" for x in r["checkpoint_bests"]]
        lines.append(f"{r['algorithm']},{r['scenario']},{r['score']:.1f},{b[0]},{b[1]},{b[2]},{r['wall_time']:.1f},"
                     f"{r['end_reason']},{bool(r['error'])}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "last_run.csv").write_text("\n".join(lines) + "\n")

    print(f"\n{len(ids)} scenarios, {time.time() - t0:.0f} s. Per-run table: results/last_run.csv\n")
    print(f"{'algorithm':28s} {'mean score':>14s} {'ckpt 33/66/100 %':>20s} {'zero ckpt':>9s} {'errors':>6s} "
          f"{'max s':>6s} {'wins vs ' + names[0]:>14s}")
    for n in names:
        r = by[n]
        s = np.array([r[i]["score"] for i in ids])
        ck = np.mean([[b or 0.0 for b in r[i]["checkpoint_bests"]] for i in ids], 0)
        zero = sum(any(b is None for b in r[i]["checkpoint_bests"]) for i in ids)
        err = sum(bool(r[i]["error"]) for i in ids)
        win = np.mean([r[i]["score"] > by[names[0]][i]["score"] for i in ids]) * 100
        print(f"{n[:28]:28s} {s.mean():8.0f} ± {s.std() / np.sqrt(len(s)):4.0f} "
              f"{ck[0]:6.0f} {ck[1]:6.0f} {ck[2]:6.0f} {zero:9d} {err:6d} {max(r[i]['wall_time'] for i in ids):6.0f} "
              f"{win:13.0f}%")
        warn = sorted({w for i in ids for w in r[i]["access_warnings"]})
        if warn:
            print(f"   uses non-public lab attributes: {warn}")
        for i in ids:
            if r[i]["error"]:
                print(f"   first error (scenario {i}):\n" + r[i]["error"][-800:])
                break


if __name__ == "__main__":
    main()
