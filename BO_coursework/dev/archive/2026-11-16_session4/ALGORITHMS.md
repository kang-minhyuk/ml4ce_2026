# Algorithm register

Every algorithm that was ever the best of its kind, with its scores. **Update this file at every milestone**:
1. Add the new file to `logs/register/run_register.sh`.
2. Run the script; results are cached, so only new files are evaluated.
3. Run `python make_register.py` to rewrite the tables between the markers.
4. Add a line to the history below.

Old versions are never deleted. Snapshots of `final/`, `algos/`, NOTES and SUMMARY are in `archive/<date>_<label>/`,
and evaluation logs are in `logs/`.

## How to read the scores
- **Score.** Each number is the normalised score: the summed best Pilot titer at the 33/66/100 % budget checkpoints,
  divided by 3 × the best achievable titer of that scenario. 1.0 would mean finding the optimum before the first
  checkpoint.
- **Splits.**
  - **official-style 400** (`off_big`) is the most precise split (± 0.007 for paired differences); prefer it for
    comparisons.
  - **DEV** = 100 suite scenarios.
  - **off-DEV** = 100 official-style scenarios.
  - **HOLD** and **off-hold** are 100 each, kept for validation.
  - **low budget** = 40 scenarios, 3.5–6 Pilot runs.
  - **wide bounds** = 40 scenarios with wider glutamine/glucose bounds.
- **Mean.** The mean is over these 7 splits.
- **Practice lab.** This is the provided lab (`src/digital_lab/lab_config`), seeds 42–61, normalised by its oracle.
  It is a single lab, so it measures one clone set only. The README says grading runs on a *different, unseen lab*
  (other clones, seeds and initial design), so the 400 official-style scenarios are the better guide. Example:
  `k5_bayes` beats `informed_bayes` on the practice lab (+0.032 ± 0.018, paired, all at checkpoint 1) but loses by
  0.042 on the 400 scenarios.

## Score tables
<!-- REGISTER START -->
### Reference algorithms

| algorithm | official-style 400 | DEV | off-DEV | HOLD | off-hold | low budget | wide bounds | **mean (7 splits)** | **practice lab (20 seeds)** |
|---|---|---|---|---|---|---|---|---|---|
| `reference_mfbo` | 0.149 | – | – | – | – | – | – | **0.149** (1) | 0.127 ± 0.015 |
| `reference_screen_then_confirm` | 0.226 | 0.215 | 0.211 | 0.243 | 0.209 | 0.092 | 0.119 | **0.188** | 0.205 ± 0.014 |
| `mfgp_structured` | 0.304 | 0.333 | 0.287 | 0.339 | 0.291 | 0.210 | 0.260 | **0.289** | 0.269 ± 0.029 |

### Final algorithms (`dev/final/`)

| algorithm | official-style 400 | DEV | off-DEV | HOLD | off-hold | low budget | wide bounds | **mean (7 splits)** | **practice lab (20 seeds)** |
|---|---|---|---|---|---|---|---|---|---|
| `generic_funnel` | 0.429 | 0.426 | 0.401 | 0.440 | 0.450 | 0.251 | 0.455 | **0.407** | 0.488 ± 0.032 |
| `generic_bayes` | 0.475 | 0.456 | 0.447 | 0.467 | 0.486 | 0.281 | 0.506 | **0.445** | 0.602 ± 0.021 |
| `informed_funnel` | 0.636 | 0.609 | 0.598 | 0.646 | 0.615 | 0.398 | 0.542 | **0.578** | 0.667 ± 0.024 |
| `informed_library` | 0.639 | 0.614 | 0.604 | 0.650 | 0.618 | 0.398 | 0.553 | **0.582** | 0.673 ± 0.024 |
| `informed_bayes` | 0.730 | 0.670 | 0.678 | 0.713 | 0.712 | 0.471 | 0.661 | **0.662** | 0.689 ± 0.014 |

### Development milestones (`dev/algos/`)

| algorithm | official-style 400 | DEV | off-DEV | HOLD | off-hold | low budget | wide bounds | **mean (7 splits)** | **practice lab (20 seeds)** |
|---|---|---|---|---|---|---|---|---|---|
| `i3_funnel` | – | 0.452 | 0.396 | – | – | – | – | **0.424** (2) | 0.487 ± 0.031 |
| `i6_funnel` | – | 0.478 | 0.448 | – | – | – | – | **0.463** (2) | 0.560 ± 0.031 |
| `i10_funnel` | – | 0.478 | 0.500 | – | – | – | – | **0.489** (2) | 0.542 ± 0.036 |
| `i14_funnel` | – | 0.563 | 0.555 | – | – | – | – | **0.559** (2) | 0.641 ± 0.022 |
| `i17_funnel` | – | 0.596 | 0.578 | – | – | – | – | **0.587** (2) | 0.667 ± 0.024 |
| `k5_bayes` | 0.688 | 0.644 | 0.635 | – | – | – | – | **0.656** (3) | 0.722 ± 0.018 |
| `k7_voi_s2` | 0.712 | 0.677 | 0.662 | – | – | – | – | **0.684** (3) | 0.699 ± 0.014 |
<!-- REGISTER END -->

## What each algorithm is
| algorithm | file | track | one-line description |
|---|---|---|---|
| `reference_mfbo` | `student/algorithms/reference_mfbo.py` | reference (official) | Official multi-fidelity BO example. |
| `reference_screen_then_confirm` | `student/algorithms/reference_screen_then_confirm.py` | reference (official) | Official screen-then-confirm example. |
| `mfgp_structured` | `algos/mfgp_structured.py` | reference (the user's baseline) | The user's structured multi-fidelity GP. |
| `i3_funnel` | `algos/i3_funnel.py` | informed (milestone) | First plates → MBR → Pilot funnel with a T/pH peak model. |
| `i6_funnel` | `algos/i6_funnel.py` | informed (milestone) | Log-glutamine GP, MBR only for screening, Pilot EI on the top-2 clones. |
| `i10_funnel` | `algos/i10_funnel.py` | informed (milestone) | Clones ranked by measured titer, Pilot T/pH fixed at the estimate. |
| `i14_funnel` | `algos/i14_funnel.py` | informed (milestone) | Segment-advance scheduling, clone racing. |
| `i17_funnel` | `algos/i17_funnel.py` | informed (milestone) | Initial-design clones screened, glutamine portfolio. |
| `informed_funnel` | `final/informed_funnel.py` | informed | Session-1 final (i22): low-budget and robustness fixes. |
| `informed_library` | `final/informed_library.py` | informed | Session-2 final: embedded simulated glutamine-response library (transfer BO, RGPE/HyperBO-style). |
| `k5_bayes` | `algos/k5_bayes.py` | informed (milestone) | `informed_library` + Bayes-optimal T/pH under a fixed response template. |
| `k7_voi_s2` | `algos/k7_voi_s2.py` | informed (milestone) | + value-of-information plate design after 2 space-filling plates. |
| `informed_bayes` | `final/informed_bayes.py` | informed (**current best**) | Session-3 final: + top-10 VOI, posterior-mean ranking, expected-titer screening order, plates 5/2/0. |
| `generic_funnel` | `final/generic_funnel.py` | generic | Same funnel with no recipe knowledge (centre-of-box prior, glucose searched). |
| `generic_bayes` | `final/generic_bayes.py` | generic (**current best generic**) | + Bayes T/pH decision and VOI plates with T/pH widths learned in-run. |

## History
| date | event | effect on the 400-scenario split |
|---|---|---|
| 2026-11-13 | session 1: informed funnel built from `mfgp_structured` | see tables |
| 2026-11-14 | session 2: glutamine library (`informed_library`) | +0.004 to +0.011 over `informed_funnel` |
| 2026-11-16 | session 3: Bayesian T/pH + VOI plates (`informed_bayes`, `generic_bayes`) | +0.091 informed, +0.047 generic |

## References
Methods used in the algorithms:
- Ranking-weighted GP ensemble (RGPE), transfer-learning BO: Feurer, Letham, Bakshy, [arXiv 1802.02219](https://arxiv.org/abs/1802.02219).
- Pre-trained GP priors (HyperBO): Wang et al., [arXiv 2109.08215](https://arxiv.org/abs/2109.08215).
- Knowledge gradient / value of information: Frazier, Powell, Dayanik, "The knowledge-gradient policy for correlated
  normal beliefs", INFORMS J. Computing 21(4), 2009, [doi:10.1287/ijoc.1080.0314](https://doi.org/10.1287/ijoc.1080.0314).
- Expected improvement: Jones, Schonlau, Welch, J. Global Optimization 13, 1998,
  [doi:10.1023/A:1008306431147](https://doi.org/10.1023/A:1008306431147).
- Bayesian decision theory for the operating point (Bayes action under posterior uncertainty): standard; see Garnett,
  *Bayesian Optimization*, Cambridge University Press 2023, chapters 5–7 ([bayesoptbook.com](https://bayesoptbook.com)).

Reviewed and judged not useful here (reasons in SUMMARY.md):
- Multi-fidelity MES [1901.08275](https://arxiv.org/abs/1901.08275); trace-aware KG
  [1903.04703](https://arxiv.org/abs/1903.04703); CAGES [2405.07760](https://arxiv.org/abs/2405.07760).
- Pandora's-box Gittins index [2406.20062](https://arxiv.org/abs/2406.20062); cost-aware stopping
  [2507.12453](https://arxiv.org/abs/2507.12453).
- Top-two Thompson sampling [1910.10945](https://arxiv.org/abs/1910.10945), [2310.19319](https://arxiv.org/abs/2310.19319).
- πBO [2204.11051](https://arxiv.org/abs/2204.11051); BOPrO [2006.14608](https://arxiv.org/abs/2006.14608).
- Batch BO: local penalisation [1505.08052](https://arxiv.org/abs/1505.08052).
- BO for bioprocesses guide [2508.10642](https://arxiv.org/abs/2508.10642); multi-fidelity bioprocess BO
  [2508.10970](https://arxiv.org/abs/2508.10970).

Candidates under test in session 4 (from a literature scan; several were read only as abstracts):
- Simulation-based inference / amortised predictors: Bayesian screening [2009.05418](https://arxiv.org/abs/2009.05418);
  neural posterior estimation [1605.06376](https://arxiv.org/abs/1605.06376); ABC regression adjustment (Beaumont,
  Zhang, Balding, Genetics 162, 2002).
- Non-myopic budgeted BO: [2106.06079](https://arxiv.org/abs/2106.06079), [2412.04882](https://arxiv.org/abs/2412.04882).
- Policy-based adaptive design: DAD [2103.02438](https://arxiv.org/abs/2103.02438); Step-DAD
  [2507.14057](https://arxiv.org/abs/2507.14057).
- Prior-informed best-arm identification [2402.05878](https://arxiv.org/abs/2402.05878); KG with common random
  numbers [1910.09259](https://arxiv.org/abs/1910.09259).
- PFNs4BO [2305.17535](https://arxiv.org/abs/2305.17535): judged unlikely to help (needs a neural network, and the
  library mixture already approximates the posterior).
