"""Offline policy search over the knobs of an algorithm (successive halving on official-style scenarios).

python policy_search.py BASE.py N_CONFIGS [--seed S] [--j J]
Stage 1: N random configs (+ the base) on off_big[:100]; stage 2: top 12 on off_big[100:400];
stage 3 is done by hand (evaluate.py on the other splits). Variant files go to algos/ps/, results to
logs/policy_search/<base>_<seed>.json. Scores are paired differences vs the base on the same scenarios.
"""
import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import evaluate as E

DEV = Path(__file__).resolve().parent
SPACE = {
    "mbr_per_seg": [[1, 2, 1], [2, 2, 1], [2, 1, 1], [2, 2, 0], [3, 2, 1], [2, 3, 1], [1, 3, 1]],
    "plates_per_seg": [[4, 2, 0], [5, 2, 0], [6, 2, 0], [5, 1, 0], [5, 3, 0], [4, 3, 0], [6, 1, 0]],
    "n_space_plates": [1, 2, 3],
    "pilot_frac": [[0.0, 0.34, 1.0], [0.0, 0.5, 1.0], [0.0, 0.67, 1.0], [0.0, 0.8, 1.0], [0.4, 0.5, 1.0], [0.4, 0.67, 1.0]],
    "bayes_wscale": [1.2, 1.3, 1.4, 1.5, 1.7],
    "bayes_sig": [0.25, 0.35, 0.5],
    "screen_eu": [0.25, 0.5, 0.75, 1.0],
    "voi_k": [6, 8, 10, 12],
    "lib_sig": [0.07, 0.1, 0.15],
    "switch": [0.9, 1.0, 1.1],
}


def make_variant(base, cfg, tag):
    out = DEV / "algos" / "ps" / f"{tag}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = Path(base).read_text().rstrip() + "\n"
    out.write_text(src + "\nCFG.update(" + json.dumps(cfg) + ")\n")
    return str(out)


def scores(paths, ids, j):
    jobs = [(p, i, None) for p in paths for i in ids]
    with Pool(j) as pool:
        res = pool.map(E.job, jobs, chunksize=1)
    out = {}
    for (p, i, _), r in zip(jobs, res):
        v = E.normalise(r, i)
        out.setdefault(p, []).append(np.nan if v is None else v)
    return {p: np.array(v) for p, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--j", type=int, default=18)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    stem = Path(a.base).stem
    ids = json.loads((DEV / "splits.json").read_text())["off_big"]
    s1, s2 = ids[:100], ids[100:]
    cfgs = []
    seen = set()
    while len(cfgs) < a.n:
        k = rng.sample(sorted(SPACE), rng.randint(2, 5))
        cfg = {key: rng.choice(SPACE[key]) for key in k}
        sig = json.dumps(cfg, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            cfgs.append(cfg)
    log = DEV / "logs" / "policy_search" / f"{stem}_{a.seed}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    paths = [make_variant(a.base, c, f"{stem}_s{a.seed}_{n:03d}") for n, c in enumerate(cfgs)]
    base_path = make_variant(a.base, {}, f"{stem}_base")
    sc = scores([base_path] + paths, s1, a.j)
    b = sc[base_path]
    st1 = []
    for p, c in zip(paths, cfgs):
        d = sc[p] - b
        st1.append({"path": p, "cfg": c, "d1": float(np.nanmean(d)), "se1": float(np.nanstd(d) / np.sqrt(len(d)))})
    st1.sort(key=lambda r: -r["d1"])
    json.dump({"stage1": st1}, open(log, "w"), indent=1)
    print("stage 1 top:", flush=True)
    for r in st1[:a.top]:
        print(f"  {r['d1']:+.4f} +- {r['se1']:.4f} {r['cfg']}", flush=True)
    top = st1[:a.top]
    sc2 = scores([base_path] + [r["path"] for r in top], s2, a.j)
    b2 = sc2[base_path]
    for r in top:
        d = sc2[r["path"]] - b2
        r["d2"] = float(np.nanmean(d)); r["se2"] = float(np.nanstd(d) / np.sqrt(len(d)))
    top.sort(key=lambda r: -r["d2"])
    json.dump({"stage1": st1, "stage2": top}, open(log, "w"), indent=1)
    print("stage 2 (held-out 300):", flush=True)
    for r in top:
        print(f"  d2 {r['d2']:+.4f} +- {r['se2']:.4f} (d1 {r['d1']:+.4f}) {Path(r['path']).stem} {r['cfg']}", flush=True)


if __name__ == "__main__":
    main()
