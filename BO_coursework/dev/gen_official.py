"""Generate 'official-compliant' hidden-lab scenarios (only what README / FAQ / notebook state).

python dev/gen_official.py [--n 300] [--seed 7]  ->  dev/scen_official/scenario_XXXX.json (same format as the suite)

- clones: 20-40 (FAQ). Each parameter uniform per clone from a fixed range: optimal temperature 30-40, optimal pH 6-8,
  max growth rate 0.02-0.06 (README). Other ranges: practice clones' observed min-max widened by 10 % of the width on
  each side (the true fixed range contains the observed sample; README says yields vary several-fold).
- batch sizes 24/4/1 (FAQ: won't change). Costs around the official 10/500/3000 (x0.6-1.6, MTP << MBR < Pilot).
- budget 6-12 Pilot runs' worth, experiment cap = initial design + 150-350 (README/FAQ: may change).
- initial design: 0-3 plates, 0-3 MBR batches (FAQ), 40 % keep the default 1 + 1. No Pilot in the initial design.
- noise 2-8 % ('a few percent'); reactor settings +-10 % ('reactor settings may change').
- bounds: practice bounds half the time, otherwise modestly varied (algorithms must read lab.bounds).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "src" / "digital_lab" / "lab_config"
OUT = Path(__file__).resolve().parent / "scen_official"
FIXED_KEYS = ("m_gln", "ph_robustness_factor", "temperature_robustness_factor", "hcp_per_cell", "k_agg_product")
OFFICIAL = {"e_activation": (30.0, 40.0), "ph_opt": (6.0, 8.0), "mu_max": (0.02, 0.06)}
NOISE_KEYS = ["c_product", "x_total", "x_viable", "x_dead", "c_glc", "c_gln", "c_lac", "c_ammonia", "c_aggregates"]


def practice_ranges():
    clones = [json.loads((CFG_DIR / f"celltype_{i}.json").read_text()) for i in range(1, 31)]
    ranges = {}
    for k, v in clones[0].items():
        if k == "name" or k in FIXED_KEYS:
            continue
        vals = np.array([c[k] for c in clones], float)
        lo, hi = vals.min(), vals.max()
        w = hi - lo
        ranges[k] = (max(lo - 0.1 * w, 0.5 * lo), hi + 0.1 * w)  # stay positive (yields are divisors)
    ranges.update(OFFICIAL)
    fixed = {k: clones[0][k] for k in FIXED_KEYS}
    return ranges, fixed


def make(i, rng, ranges, fixed):
    n = int(rng.integers(20, 41))
    clones = []
    for j in range(1, n + 1):
        c = {"name": f"celltype_{j}"}
        for k, (lo, hi) in ranges.items():
            c[k] = float(lo + rng.random() * (hi - lo))
        c.update(fixed)
        clones.append(c)
    reactors = {}
    for r in ("MTP", "MBR", "Pilot"):
        base = json.loads((CFG_DIR / f"{r}.json").read_text())
        base["scale_factor"] = round(base["scale_factor"] * rng.uniform(0.9, 1.1), 4)
        base["growth_inhibition"] = round(min(1.2, base["growth_inhibition"] * rng.uniform(0.9, 1.1)), 4)
        reactors[r] = base
    sigma = float(rng.uniform(0.02, 0.08))
    noise = {f"process_noise_{k}": 0.0 for k in NOISE_KEYS}
    noise.update({f"measurement_noise_{k}": (sigma if k == "c_product" else 0.0) for k in NOISE_KEYS})
    if rng.random() < 0.5:
        tb, pb, gb, qb = [30.0, 40.0], [6.0, 8.0], [0.0, 100.0], [0.0, 50.0]
    else:
        tb = [round(rng.uniform(28, 32), 2), round(rng.uniform(38, 42), 2)]
        pb = [round(rng.uniform(5.8, 6.2), 3), round(rng.uniform(7.8, 8.2), 3)]
        gb, qb = [0.0, round(rng.uniform(70, 150), 1)], [0.0, round(rng.uniform(30, 70), 1)]
    bounds = [tb, pb, gb, qb, gb, qb, gb, qb, tb, pb, tb, pb]
    batch = {"MTP": 24, "MBR": 4, "Pilot": 1}
    while True:
        costs = {"MTP": int(round(10 * np.exp(rng.uniform(np.log(0.6), np.log(1.6))))),
                 "MBR": int(round(500 * np.exp(rng.uniform(np.log(0.6), np.log(1.6))))),
                 "Pilot": int(round(3000 * np.exp(rng.uniform(np.log(0.6), np.log(1.6)))))}
        if costs["MTP"] * 10 <= costs["MBR"] and costs["MBR"] < costs["Pilot"] and \
                costs["MBR"] * 4 < 1.5 * costs["Pilot"]:
            break
    pc = costs["Pilot"]
    budget = int(round(pc * rng.uniform(6.0, 12.0), -2))
    if rng.random() < 0.4:
        init = {"MTP": 1, "MBR": 1, "Pilot": 0}
    else:
        init = {"MTP": int(rng.integers(0, 4)), "MBR": int(rng.integers(0, 4)), "Pilot": 0}
    init_cost = sum(init[r] * batch[r] * costs[r] for r in init)
    while init_cost > 0.5 * budget:  # keep the initial design a minority of the budget
        init["MBR"] = max(0, init["MBR"] - 1)
        init_cost = sum(init[r] * batch[r] * costs[r] for r in init)
    init_exps = sum(init[r] * batch[r] for r in init)
    return {"id": i, "seed": int(rng.integers(1, 2 ** 31 - 1)), "coursework_compatible_batches": True,
            "first_checkpoint_reachable": bool(init_cost + pc <= 0.33 * budget),
            "lab": {"budget": budget, "max_experiments": int(init_exps + rng.integers(150, 351)),
                    "reactor_costs": costs, "batch_sizes": batch, "initial_design": init},
            "bounds": bounds, "reactors": reactors, "noise": noise, "clone_range_stretch": 1.0, "clones": clones}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ranges, fixed = practice_ranges()
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        (OUT / f"scenario_{i:04d}.json").write_text(json.dumps(make(i, rng, ranges, fixed), indent=1))
    print(f"wrote {a.n} scenarios to {OUT}; clone ranges:")
    for k, v in ranges.items():
        print(f"  {k:24s} {v[0]:.4g} .. {v[1]:.4g}")
