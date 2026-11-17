"""Generate the scenario suite: python generate_scenarios.py [--n 1000] [--seed 2026]

Each scenario is a complete, self-contained hidden lab written to scenarios/scenario_XXXX.json:
clones, reactor scale behaviour, measurement noise, bounds, budget, costs, batch sizes, experiment cap,
initial design and the run seed. Nothing here depends on the practice lab's clones; parameter ranges are the
ranges the coursework README describes (and slightly stretched per scenario).
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Per-parameter ranges of the cell-culture model (min, max). Optimal temperature / pH follow the README
# (30-40 C, 6-8); the rest span the practice clones' observed ranges.
CLONE_RANGES = {
    "mu_max": (0.0209, 0.0599), "k_lysis": (0.0311, 0.0484), "k_deg_gln": (0.00100, 0.00149),
    "k_death_max": (0.0102, 0.0149), "k_death_mu": (0.0106, 0.0147), "k_sat_lac": (132.5, 169.3),
    "k_sat_ammonia": (35.09, 39.81), "k_sat_glc": (0.904, 1.598), "k_sat_gln": (0.2116, 0.2973),
    "y_ammonia": (1.05e-08, 1.867e-07), "y_lactate": (0.6016, 0.9878), "y_cells_glc": (1.707e07, 2.916e08),
    "y_cells_gln": (2.570e08, 8.139e09), "m_glc": (0.0, 5e-10), "y_product_from_total": (1.22e-08, 1.936e-07),
    "y_product_from_viable": (3.539e-06, 7.863e-06),
}
OPT_RANGES = {"e_activation": (30.0, 40.0), "ph_opt": (6.0, 8.0)}
CLONE_FIXED = {"m_gln": 0.0, "ph_robustness_factor": 1.1, "temperature_robustness_factor": 1.1,
               "hcp_per_cell": 2e-06, "k_agg_product": 3e-06}
REACTOR_BASE = {"MTP": (0.001, 1.1, 0.5), "MBR": (0.1, 1.0, 1.0), "Pilot": (3.0, 1.3, 0.9)}  # volume, scale, growth
DEFAULT_BATCH = {"MTP": 24, "MBR": 4, "Pilot": 1}
NOISE_KEYS = ["c_product", "x_total", "x_viable", "x_dead", "c_glc", "c_gln", "c_lac", "c_ammonia", "c_aggregates"]


def make_clones(rng, n, stretch):
    clones = []
    for i in range(1, n + 1):
        p = {"name": f"celltype_{i}"}
        for k, (lo, hi) in CLONE_RANGES.items():
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * stretch
            p[k] = float(max(0.0, mid - half) + rng.random() * (mid + half - max(0.0, mid - half)))
        for k, (lo, hi) in OPT_RANGES.items():
            p[k] = float(lo + rng.random() * (hi - lo))
        p.update(CLONE_FIXED)
        clones.append(p)
    return clones


def make_scenario(i, rng):
    n_clones = int(rng.integers(20, 41))
    stretch = float(rng.uniform(0.8, 1.2))
    clones = make_clones(rng, n_clones, stretch)

    # Scale behaviour: each reactor's death scale factor and growth inhibition perturbed by up to +-15 %.
    reactors = {}
    for name, (vol, scale, growth) in REACTOR_BASE.items():
        reactors[name] = {"name": name, "volume": vol, "scale_factor": round(scale * rng.uniform(0.85, 1.15), 4),
                          "growth_inhibition": round(min(1.2, growth * rng.uniform(0.85, 1.15)), 4)}
    noise_level = float(rng.uniform(0.02, 0.08))
    noise = {f"process_noise_{k}": 0.0 for k in NOISE_KEYS}
    noise.update({f"measurement_noise_{k}": (noise_level if k == "c_product" else 0.0) for k in NOISE_KEYS})

    # Bounds (12 settings: T, pH, glc0, gln0, glc/gln feed 1, glc/gln feed 2, T/pH after feed 1, after feed 2).
    t_lo, t_hi = rng.uniform(28.0, 32.0), rng.uniform(38.0, 42.0)
    p_lo, p_hi = rng.uniform(5.8, 6.2), rng.uniform(7.8, 8.2)
    g_hi, q_hi = rng.uniform(70.0, 150.0), rng.uniform(30.0, 70.0)
    tb, pb, gb, qb = [round(t_lo, 2), round(t_hi, 2)], [round(p_lo, 3), round(p_hi, 3)], [0.0, round(g_hi, 1)], [0.0, round(q_hi, 1)]
    bounds = [tb, pb, gb, qb, gb, qb, gb, qb, tb, pb, tb, pb]

    # Batch sizes: half the scenarios keep the coursework sizes (the FAQ says they will not change for grading).
    if rng.random() < 0.5:
        batch = dict(DEFAULT_BATCH)
    else:
        batch = {"MTP": int(rng.choice([12, 24, 48, 96])), "MBR": int(rng.choice([2, 3, 4, 6, 8])),
                 "Pilot": int(rng.choice([1, 1, 2]))}
    # Costs per experiment (MTP << MBR < Pilot always holds).
    costs = {"MTP": int(rng.integers(5, 26)), "MBR": int(rng.integers(250, 901)), "Pilot": int(rng.integers(1800, 4501))}
    if costs["MBR"] * batch["MBR"] >= costs["Pilot"] * batch["Pilot"] * 1.5:
        costs["MBR"] = int(costs["Pilot"] * batch["Pilot"] / batch["MBR"] * 0.8)
    pilot_batch = costs["Pilot"] * batch["Pilot"]
    budget = int(round(pilot_batch * rng.uniform(6.0, 13.0), -3))

    # Initial design: 0-3 plates, 0-3 MBR batches, occasionally one Pilot batch. Most scenarios keep the first
    # checkpoint reachable; ~10 % are deliberately hard (the initial design leaves no room for a Pilot before it).
    hard = rng.random() < 0.10
    for _ in range(200):
        init = {"MTP": int(rng.integers(0, 4)), "MBR": int(rng.integers(0, 4)), "Pilot": int(rng.random() < 0.1)}
        init_cost = sum(init[r] * batch[r] * costs[r] for r in init)
        reachable = init_cost + pilot_batch <= 0.33 * budget
        if reachable != hard and init_cost < budget:
            break
    init_exps = sum(init[r] * batch[r] for r in init)
    max_exp = int(init_exps + rng.integers(100, 351))

    return {
        "id": i,
        "seed": int(rng.integers(1, 2**31 - 1)),
        "coursework_compatible_batches": batch == DEFAULT_BATCH,
        "first_checkpoint_reachable": bool(init_cost + pilot_batch <= 0.33 * budget),
        "lab": {"budget": budget, "max_experiments": max_exp, "reactor_costs": costs, "batch_sizes": batch,
                "initial_design": init},
        "bounds": bounds,
        "reactors": reactors,
        "noise": noise,
        "clone_range_stretch": round(stretch, 3),
        "clones": clones,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(__file__).resolve().parent / "scenarios"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = ["id,seed,n_clones,budget,cost_MTP,cost_MBR,cost_Pilot,batch_MTP,batch_MBR,batch_Pilot,max_experiments,"
            "init_MTP,init_MBR,init_Pilot,noise,coursework_compatible_batches,first_checkpoint_reachable"]
    for i in range(a.n):
        sc = make_scenario(i, rng)
        (out / f"scenario_{i:04d}.json").write_text(json.dumps(sc, indent=1))
        L = sc["lab"]
        rows.append(",".join(str(v) for v in [
            i, sc["seed"], len(sc["clones"]), L["budget"], *L["reactor_costs"].values(), *L["batch_sizes"].values(),
            L["max_experiments"], *L["initial_design"].values(), sc["noise"]["measurement_noise_c_product"],
            sc["coursework_compatible_batches"], sc["first_checkpoint_reachable"]]))
    (Path(__file__).resolve().parent / "scenarios_summary.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {a.n} scenarios to {out}")
