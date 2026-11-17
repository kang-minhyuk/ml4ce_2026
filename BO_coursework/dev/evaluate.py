"""Evaluate algorithms on a scenario split with oracle-normalised scores.

python dev/evaluate.py ALGO.py [ALGO2.py ...] --split dev|hold|stress|practice [--n N] [--j 21]

Normalised score of a run = score / (3 * oracle * noise_factor), i.e. 1.0 means the true best Pilot recipe was
measured before the first checkpoint. Uses scenario_suite/run_suite.run_one (official harness, same cache).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parent
sys.path.insert(0, str(DEV.parent / "scenario_suite"))
import run_suite  # noqa: E402

ORACLE = json.loads((DEV / "oracle.json").read_text()) if (DEV / "oracle.json").exists() else {}
# hold-out splits: proxy oracle (grid-best glutamine at true T/pH, ≈0.95 of the DE oracle) where no DE oracle exists
PROXY = json.loads((DEV / "oracle_proxy.json").read_text()) if (DEV / "oracle_proxy.json").exists() else {}
for _k, _v in PROXY.items():
    if _k not in ORACLE and int(_k) >= 0:
        ORACLE[_k] = _v
PRACTICE_SEEDS = list(range(42, 62))  # 'practice' = first 5 seeds, 'practice20' = all 20


def noise_factor(seed, sigma):
    return 1 + sigma * np.random.default_rng(seed).normal(0, 1, 28)[-1]


def run_practice(args):
    path, seed, _ = args
    import hashlib
    key = hashlib.sha1((Path(path).read_text() + f"practice{seed}").encode()).hexdigest()[:12]
    out = run_suite.RESULTS / Path(path).stem / f"practice_{seed}_{key}.json"
    if out.exists():
        return json.loads(out.read_text())
    from digital_lab import experiment
    hist, info = experiment.run_algorithm(run_suite.load_algorithm(path), Path(path).stem, seed, {})
    res = {"algorithm": Path(path).stem, "scenario": -1, "seed": seed, "score": info["score"],
           "checkpoint_bests": info["checkpoint_bests"], "wall_time": info["wall_time"],
           "end_reason": info["end_reason"], "error": info["error"], "access_warnings": info["access_warnings"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def run_official(args):
    """Like run_suite.run_one, for dev/scen_official scenarios (sid >= 10000)."""
    import hashlib
    path, sid, time_limit = args
    sc = json.loads((DEV / "scen_official" / f"scenario_{sid - 10000:04d}.json").read_text())
    key = hashlib.sha1((Path(path).read_text() + json.dumps(sc) + str(time_limit)).encode()).hexdigest()[:12]
    out = run_suite.RESULTS / Path(path).stem / f"off_{sid - 10000:04d}_{key}.json"
    if out.exists():
        return json.loads(out.read_text())
    import digital_lab.lab as L
    from digital_lab import experiment
    names = [c["name"] for c in sc["clones"]]
    L.CELLTYPES = names
    L.LAB_CONFIG_FILENAMES = {**L.LAB_CONFIG_FILENAMES, **{n: f"{n}.json" for n in names}}
    L.CONTINUOUS_BOUNDS = [tuple(b) for b in sc["bounds"]]
    d = DEV / ".labs_off" / f"scenario_{sid - 10000:04d}"
    if not (d / "noise_parameters.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        for name, cfg in sc["reactors"].items():
            (d / f"{name}.json").write_text(json.dumps(cfg))
        for c in sc["clones"]:
            (d / f"{c['name']}.json").write_text(json.dumps(c))
        (d / "noise_parameters.json").write_text(json.dumps(sc["noise"]))
    kw = {"config_dir": str(d), **sc["lab"]}
    if time_limit:
        kw["time_limit"] = float(time_limit)
    hist, info = experiment.run_algorithm(run_suite.load_algorithm(path), Path(path).stem, sc["seed"], kw)
    res = {"algorithm": Path(path).stem, "scenario": sid, "score": info["score"],
           "checkpoint_bests": info["checkpoint_bests"], "wall_time": info["wall_time"],
           "end_reason": info["end_reason"], "error": info["error"], "access_warnings": info["access_warnings"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def job(args):
    path, sid, tl = args
    if sid < 0:
        return run_practice((path, PRACTICE_SEEDS[-sid - 1], tl))
    if sid >= 10000:
        return run_official(args)
    return run_suite.run_one((path, sid, tl))


def normalise(r, sid):
    if sid < 0:
        o = ORACLE.get("-1")
        seed, sigma = PRACTICE_SEEDS[-sid - 1], 0.05
    else:
        o = ORACLE.get(str(sid))
        p = (DEV / "scen_official" / f"scenario_{sid - 10000:04d}.json") if sid >= 10000 else \
            (run_suite.SUITE / "scenarios" / f"scenario_{sid:04d}.json")
        sc = json.loads(p.read_text())
        seed, sigma = sc["seed"], sc["noise"]["measurement_noise_c_product"]
    if o is None:
        return None
    return r["score"] / (3 * o["oracle"] * noise_factor(seed, sigma))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algos", nargs="+")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--j", type=int, default=21)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.split in ("practice", "practice20"):
        ids = [-(i + 1) for i in range(5 if a.split == "practice" else len(PRACTICE_SEEDS))]
    else:
        ids = json.loads((DEV / "splits.json").read_text())[a.split]
    if a.n:
        ids = ids[:a.n]
    t0 = time.time()
    jobs = [(p, i, a.time_limit) for p in a.algos for i in ids]
    with Pool(a.j) as pool:
        res = pool.map(job, jobs, chunksize=1)
    names = [Path(p).stem for p in a.algos]
    table = {}
    for (p, i, _), r in zip(jobs, res):
        table.setdefault(Path(p).stem, {})[i] = r
    print(f"{len(ids)} runs per algorithm on '{a.split}', {time.time() - t0:.0f} s")
    print(f"{'algorithm':34s} {'norm':>6s} {'±se':>5s} {'n33':>5s} {'n66':>5s} {'n100':>5s} {'p10':>5s} "
          f"{'raw':>8s} {'zero':>4s} {'err':>3s} {'maxs':>5s} {'meds':>5s}")
    base = None
    summary = {}
    for n in names:
        t = table[n]
        norm = [normalise(t[i], i) for i in ids]
        ok = [i for i, v in zip(ids, norm) if v is not None]
        nv = np.array([v for v in norm if v is not None])
        ck = []
        for i in ok:
            fac = normalise({"score": 1.0}, i)
            ck.append([(b or 0.0) * fac * 3 for b in t[i]["checkpoint_bests"]])
        ck = np.mean(ck, 0) if ck else [np.nan] * 3
        raw = np.mean([t[i]["score"] for i in ids])
        zero = sum(any(b is None for b in t[i]["checkpoint_bests"]) for i in ids)
        err = sum(bool(t[i]["error"]) for i in ids)
        wt = [t[i]["wall_time"] for i in ids]
        p10 = np.percentile(nv, 10) if len(nv) else np.nan
        print(f"{n[:34]:34s} {nv.mean():6.3f} {nv.std() / np.sqrt(max(len(nv), 1)):5.3f} {ck[0]:5.3f} {ck[1]:5.3f} "
              f"{ck[2]:5.3f} {p10:5.3f} {raw:8.0f} {zero:4d} {err:3d} {max(wt):5.0f} {np.median(wt):5.1f}")
        summary[n] = dict(zip(ok, nv.tolist()))
        warn = sorted({w for i in ids for w in t[i]["access_warnings"]})
        if warn:
            print("   non-public lab access:", warn)
        if err and not a.quiet:
            i = next(i for i in ids if t[i]["error"])
            print(f"   first error (scenario {i}):\n" + t[i]["error"][-1500:])
    if len(names) > 1:
        b = summary[names[0]]
        for n in names[1:]:
            common = [i for i in summary[n] if i in b]
            d = np.array([summary[n][i] - b[i] for i in common])
            if len(d):
                print(f"paired diff {n} - {names[0]}: {d.mean():+.3f} ± {d.std() / np.sqrt(len(d)):.3f}, "
                      f"wins {np.mean(d > 0) * 100:.0f}%")
    (DEV / "last_eval.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
