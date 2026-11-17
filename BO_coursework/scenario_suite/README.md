# BO coursework: scenario suite

1,000 hidden-lab scenarios for testing a Bayesian-optimisation algorithm on more than the practice lab. Each scenario
is a different lab: its own clones, scale behaviour, noise, bounds, budget, costs, batch sizes, experiment cap,
initial design and seed. An algorithm that only does well on the practice lab will show it here.

## What varies between scenarios

| Factor | Range |
| --- | --- |
| Number of clones | 20–40 |
| Clone parameters (growth, death, nutrient affinities and yields, lactate/ammonia, product yields) | drawn uniformly per clone; each scenario stretches the ranges by ×0.8–1.2 |
| Clone optimal temperature / pH | 30–40 °C / 6.0–8.0, independent per clone |
| Scale behaviour (death scale factor, growth inhibition of MTP / MBR / Pilot) | ±15 % around the coursework values |
| Measurement noise on titer | 2–8 % |
| Bounds | T low 28–32 / high 38–42 °C; pH low 5.8–6.2 / high 7.8–8.2; glucose max 70–150 mM; glutamine max 30–70 mM |
| Cost per experiment | MTP 5–25, MBR 250–900, Pilot 1,800–4,500 (MTP ≪ MBR < Pilot always) |
| Budget | 6–13 Pilot batches' worth (11k–109k) |
| Batch sizes | 49 % keep the coursework sizes (24 / 4 / 1); the rest use MTP 12–96, MBR 2–8, Pilot 1–2 |
| Experiment cap | initial design + 100–350 |
| Initial design | 0–3 plates, 0–3 MBR batches, sometimes 1 Pilot batch |
| Run seed | different per scenario (noise and initial design) |

Not varied: scoring checkpoints (33 / 66 / 100 % of budget), feed times (day 5 and 10) and the 5-minute limit, as the
course FAQ says these are fixed.

`scenarios_summary.csv` lists every scenario's settings. Two flags help filter:

- `coursework_compatible_batches`: batch sizes are 24 / 4 / 1. The FAQ says they won't change for grading, so use
  `--coursework-only` for a grading-like test and all scenarios for a stress test.
- `first_checkpoint_reachable`: False for 92 scenarios (~9 %), where the initial design leaves no money for a Pilot
  run before the first checkpoint. That checkpoint scores 0 for every algorithm; they test recovery, not a bug.

## Requirements

The coursework repository with its environment installed (`uv sync`). The runner imports the official `digital_lab`
package from `../src` by default; set `BO_REPO=/path/to/BO_coursework` if this folder lives elsewhere.

## Running

From this folder:

```bash
uv run python run_suite.py example_algorithms/reference_screen_then_confirm.py my_algorithm.py --n 100
```

- The first algorithm is the baseline for the win-rate column.
- `--n` / `--start` pick the scenario ids, and `--coursework-only` keeps the 24 / 4 / 1 batch sizes.
- `--j` sets the number of parallel processes (default: CPUs − 1).
- `--time-limit 30` checks that your algorithm still finishes on a much slower machine.

Each run goes through the official harness (`digital_lab.experiment.run_algorithm`), so scoring, the time limit and
the read-only lab view are exactly as in the notebook. Results are cached per algorithm file and scenario in
`results/`; editing a file re-runs only that file. Every run is also written to `results/last_run.csv`. Lab config
files are generated on demand in `.labs/`. Both folders can be deleted at any time.

Rough cost: one run takes 0.1–60 s depending on the algorithm. A GP-based algorithm on 100 scenarios takes about
5 minutes on 10 cores.

Output columns: mean score ± standard error, mean best Pilot titer at each checkpoint, number of runs with a zero
checkpoint, errors, slowest run (s), and how often the algorithm beat the baseline. If an algorithm touches non-public
lab attributes, those accesses are listed.

## Files

| File | What it is |
| --- | --- |
| `scenarios/scenario_XXXX.json` | One complete scenario (clones, reactors, noise, bounds, lab settings, seed) |
| `scenarios_summary.csv` | One row per scenario with its main settings |
| `generate_scenarios.py` | Regenerates the suite (`--n`, `--seed`); seed 2026 gives exactly these files |
| `run_suite.py` | Runs and compares algorithms |
| `example_algorithms/` | The two course reference algorithms and a structured multi-fidelity GP algorithm |

Note: `reference_mfbo.py` (from the course) crashes on scenarios whose plates have more wells than there are clones
(e.g. 96 wells, 35 clones), because it never repeats a clone on a plate. That's a limitation of that algorithm, not of
the suite.
