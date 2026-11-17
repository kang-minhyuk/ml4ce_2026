# Executive summary: BO coursework algorithm development

## Bottom line
Rule-compliant, single-file algorithms are in `dev/final/`. The best is **`informed_bayes2.py`** (session 4,
2026-11-16 evening; confirmed on 400 untouched scenarios in session 5); the best generic one is **`generic_bayes2.py`**
(session 5). The full register with every split and the 20-seed practice lab is `dev/ALGORITHMS.md`.

| | DEV | official-style DEV | HOLD | official-style hold-out | low budget | wide bounds | practice lab (5 seeds) |
|---|---|---|---|---|---|---|---|
| `mfgp_structured` (your baseline) | 0.333 | 0.287 | 0.339 | 0.291 | 0.210 | 0.260 | 0.200 |
| `generic_funnel.py` | 0.426 | 0.401 | 0.440 | 0.450 | 0.251 | 0.455 | 0.512 |
| `generic_bayes.py` (generic, session 3) | 0.456 | 0.447 | 0.467 | 0.486 | 0.281 | 0.506 | 0.654 |
| **`generic_bayes2.py`** (generic, session 5) | 0.463 | 0.440 | 0.490 | 0.494 | 0.278 | 0.532 | 0.641 (20 seeds) |
| **`informed_funnel.py`** (compact) | 0.609 | 0.598 | 0.646 | 0.615 | 0.398 | 0.542 | 0.672 |
| `informed_library.py` (session 2 best) | 0.614 | 0.604 | 0.650 | 0.618 | 0.398 | 0.553 | 0.682 |
| `informed_bayes.py` (session 3) | 0.670 | 0.678 | 0.713 | 0.712 | 0.471 | 0.661 | 0.692 |
| **`informed_bayes2.py`** (highest) | **0.696** | **0.695** | **0.729** | **0.715** | **0.481** | **0.671** | 0.719 (20 seeds) |

`informed_bayes2.py` = `informed_bayes.py` + (1) the glutamine library is weighted by the clone's own plate titer at the
standard plate medium (a clone's absolute plate titer predicts how much glutamine it needs), which sets the first
Pilot recipe of each clone and the library prior from its 2nd run; (2) plates a budget segment could not afford are
carried into the next segment; (3) two MBR screening batches before the first Pilot. On `off_big` 0.745 vs 0.730
(paired +0.015); on the other 480 scenarios every split improves (pooled about +0.014); practice lab 0.719 vs 0.689.

**Final untouched check (session 5, 2026-11-17).** On 400 brand-new official-style scenarios (`off_fresh`, generated
after every decision), `informed_bayes2.py` scores **0.742** (vs 0.745 on `off_big`, where it was selected), +0.019
± 0.004 over `informed_bayes.py`; `generic_bayes.py` scores 0.470. About 15 further ideas tried overnight (library-
judged clone race, T/pH bias correction, a joint glutamine + T/pH Pilot model, MBR screens as library evidence,
second plate medium, MBR recipe refinement for generic, settings combinations) gave at most +0.003 and none
survived the held-out and fresh checks, so `informed_bayes2.py` remains the final. Details: `NOTES.md` (Session 5).
The generic track improved: a 3-stage policy search gave **`generic_bayes2.py`** (VOI over 12 clones, screening
weight 1.0, narrower glutamine steps), confirmed on every independent set (other 480 +0.0085 ± 0.0038, untouched 400
+0.008 ± 0.004 → 0.478, practice lab 0.641 vs 0.602).

On 400 further official-style scenarios (`off_big`, generated after all session-2 decisions), `informed_bayes.py`
scores 0.730 vs 0.639 for `informed_library.py` (paired +0.091 ± 0.008, wins 68 %; checkpoints 33/66/100 %:
0.55/0.78/0.86 vs 0.44/0.70/0.78). It is better on every split with an oracle (+0.056 to +0.107) and gives
13–15 % more raw titer on the stress and extra splits. Median run 4–5 s, slowest 14 s with 20 runs in parallel.

How to read the table:
- **Score.** Each number is the normalised score: the summed best Pilot titer at the 33/66/100 % checkpoints, divided
  by 3 × the best achievable titer.
- **What 1.0 means.** 1.0 would mean finding the true optimum before the first checkpoint.
- **Splits.**
  - DEV = 100 suite scenarios.
  - Official-style = 100 scenarios generated only from the information in the README/FAQ/notebook.
  - HOLD, the official-style hold-out and the low-budget split (40 scenarios, 3.5–6 Pilot runs of budget) were never
    used for any design decision.
  - Wide bounds = 40 official-style scenarios with glutamine bounds ×4 and glucose ×2; it tests the bound-relative
    priors.
  - Practice lab = the provided lab, seeds 42–46.
- **Paired comparison.** The informed funnel beats `mfgp_structured` in 90–96 % of scenarios on every split. Its
  margin is +0.28 to +0.32 on the 100-scenario splits.
- **Generalisation.** Scores on the hold-out splits are equal to or higher than on DEV, so there is no sign of
  overfitting.

Both algorithms pass `check_my_algorithm`, except that `STUDENT_NAMES` is still a placeholder. They:
- import only numpy, scipy and sobol_seq;
- use only the public `lab` API and read every setting from `lab`;
- never read files.

Runs take about 10 s (at most 65 s with 20 runs in parallel). A time guard stops model refits when less than 20 s
remain. With a 30 s limit the score is unchanged; even with a 12 s limit it still runs and scores 0.355.

To submit, paste `dev/final/informed_bayes2.py` into the notebook's `%%writefile algorithms/my_algorithm.py` cell and
fill in `STUDENT_NAMES`.

## What the informed algorithm does
1. **Plates.**
   - 4 plates at a rich medium (glucose at its upper bound, about 5 % glutamine).
   - The first plates use space-filling temperature/pH conditions; later ones are targeted.
   - A joint asymmetric-peak model gives each clone's temperature/pH optimum and a ranking.
2. **MBR screen.** The best plate clones are screened in mini bioreactors at their estimated optimum, all with one
   common prior recipe. Within a run the noise is one constant factor, so the comparison between clones is exact.
3. **Pilot racing.**
   - The best screened clone gets the first Pilot run.
   - Any clone whose MBR titer beats the incumbent's gets one Pilot run; after that, Pilot titers decide.
   - The incumbent works through a 3-recipe glutamine portfolio, then a GP with expected improvement takes over.
     The GP models log titer from the temperature/pH offsets, log glutamine, clone offsets and correlated MBR/Pilot
     scales.
4. **Budget planner.**
   - Each reachable checkpoint gets a Pilot run, preceded by plates and an MBR screen when affordable.
   - Money that cannot buy a Pilot before a checkpoint is planned with the next segment.
   - Allocations follow the stage reached, not the segment index.

## Key discoveries (noise-free simulator analysis)
- **Glutamine is the dominant recipe lever.** With temperature/pH at the clone's growth optimum, tuning only
  glutamine (start, feed 1, feed 2) reaches the oracle (0.63 → 1.00). Varying temperature/pH between phases adds
  nothing.
- **Glutamine optimum.** The optimum is a sharp cusp in log concentration: glutamine is used up just before the next
  feed, and the simulator allows a negative "debt". It varies about 100× between clones and shifts between MBR and
  Pilot, so it must be searched at Pilot scale.
- **Glutamine portfolio.** A fixed portfolio of 3 complementary recipes beats GP search for a clone's first Pilot
  runs (0.85 vs 0.81 of oracle).
- **Plates locate temperature/pH.** The titer-optimal temperature/pH on rich-medium plates equals the growth optimum
  (±0.3 °C, ±0.03 pH). MBR and Pilot runs at a fixed recipe do not: they are off by 1.3 °C and 0.9 °C (standard
  deviation).
- **Asymmetric sensitivity.** Pilot titer falls steeply when the temperature is below the optimum (−2 °C → 0.33) or
  the pH above it (+0.2 → 0.52). The opposite errors cost little.
- **Clone choice.** The GP was unreliable for choosing between clones. Racing on measured titers gave the largest
  single gain (+0.06).
- **Scheduler bugs.** These mattered as much as the modelling:
  - 9 of 24 traced scenarios never ran an MBR screen.
  - Small budgets led to no Pilot at all, or to blind Pilot runs.
  - Fixing these gave +0.02 to +0.08, with the largest gain at low budgets.

## Progression (DEV / official-style DEV)
| step | DEV | off-DEV |
|---|---|---|
| `mfgp_structured` baseline | 0.333 | 0.287 |
| i3: plates → MBR → Pilot funnel with a T/pH peak model | 0.452 | 0.396 |
| i5/i6: log-glutamine GP, MBR used only for screening, Pilot EI on the top-2 clones | 0.478 | 0.448 |
| i10: clones ranked by measured titer, Pilot temperature/pH fixed at the estimate | 0.502 | 0.500 |
| i13/i14: segment-advance scheduling, **clone racing** | 0.573 | 0.559 |
| i16/i17: initial-design clones screened, **glutamine portfolio** | 0.596 | 0.578 |
| i22: low-budget/robustness fixes (final informed) | **0.609** | **0.598** |

## What did not help (each tested on 100 + 100 scenarios)
- **Generic `mfgp_structured` tweaks:** other dimensionalities, an additive kernel, Thompson sampling, Pilot EI,
  allocation changes.
- **Better temperature/pH location models:** Gaussian-shape per-clone models, own-medium-only fits, focus plates
  around the incumbent's optimum, more plates. They are better offline, but neutral or worse in the pipeline. Knowing
  the true optimum would be worth +0.078, but the plate-fit error (≈0.7 °C / 0.17 pH) seems limited by model shape.
- **Safe-side bias:** shifting the recipe warmer and to lower pH (≤ +0.01).
- **Other Pilot/MBR policies:** a Pilot-only GP; MBR refinement of temperature/pH or glutamine (glutamine does not
  transfer between scales); UCB instead of EI; budget-allocation sweeps (insensitive); a conditional glutamine
  portfolio (+0.01 offline).
- **Perfect clone ranking** (analysis only): +0.033, so screening is not the main bottleneck.

## Session 2: transfer BO and other strategies
- **Glutamine-response library (kept, `informed_library.py`).**
  - I simulated the Pilot (and MBR) titer on an absolute glutamine grid for 1,950 clones from scenarios never used
    for validation.
  - In the algorithm, the incumbent's Pilot titers re-weight the library members, each with a free log offset; the
    next recipe maximises the mixture's expected improvement. This is the idea behind ranking-weighted GP ensembles
    (RGPE) and pre-trained GP priors (HyperBO).
  - Offline, after 4 Pilot runs it reaches 0.961 of the grid optimum, against 0.928 for the fixed portfolio and
    0.947 for a HyperBO-style Gaussian prior.
  - In the full pipeline the gain is smaller (+0.004 to +0.011 on every split, never negative). Only the incumbent's
    later Pilots benefit, and clone choice and temperature/pH location still dominate the loss.
  - Compressed to 300 members × 12 components on a 7×6×4 grid, it costs only 0.003 against the full library.
- **Dead ends this session:**
  - *MBR glutamine probes with a joint MBR/Pilot library.* Offline, one extra MBR run lifts the first Pilot recipe
    from 0.77 to 0.88 of the clone's optimum. In the pipeline it scored −0.007 to −0.049: probes displace clone
    screening, and the temperature/pH estimate moves between the probe and the Pilot.
  - *Ignoring checkpoint 1 to maximise checkpoint 2* (your idea): −0.08 to −0.13. Checkpoint 1 falls from 0.42 to
    0.05–0.15, while checkpoints 2 and 3 gain only 0.01–0.04.
  - *Focus plates by pattern search on the incumbent's plate titers.* The final temperature error improved (median
    0.71 → 0.52 °C), but the score was neutral (−0.006 to +0.006). Tight experiment caps often forbid the extra
    plates.
  - *Other acquisitions and refinements:* log-EI, efficiency-EI, continuous library candidates, likelihood widths,
    and allocation sweeps were all within ±0.005.

## Session 3: Bayesian decision theory for temperature/pH (kept, `informed_bayes.py`)
A loss decomposition of `informed_library.py` (100 DEV traces) showed that at checkpoint 1 the loss is clone choice
0.28, recipe 0.16 and temperature/pH location 0.09 of oracle; wrong T/pH also makes MBR screens collapse, which
corrupts clone choice. Offline studies of the simulator family then showed that the plate T/pH response has the
same shape for every clone and medium (an asymmetric quadratic in T and in pH, additive, saturating at a floor;
widths 1.25 °C below / 3.08 °C above, 0.65 pH below / 0.15 above). That turns localisation into a well-posed
Bayesian problem:
- **Bayes-optimal operating point.** Per clone, a grid posterior over its optimum (uniform prior, peak height
  profiled out, heavy-tailed residuals). The chosen T/pH maximises the posterior-expected relative titer under a
  slightly wider (bioreactor-scale) template, so it hedges toward the safe side when the optimum is uncertain.
  Offline expected relative Pilot titer at the first screen: 0.63 vs 0.45 for the previous peak fit and 0.51 for a
  least-squares template fit. Pipeline: +0.049 ± 0.007 on `off_big`.
- **Value-of-information plates.** After 2 space-filling plates, each plate condition maximises the expected gain in
  that decision utility summed over the top-10 ranked clones (knowledge-gradient-style design). +0.024 more.
- **Ranking and screening.** Clones are ranked by posterior-mean peak height; the MBR screen order adds 0.5 × log of
  the expected relative titer, so clones whose optimum is still uncertain wait for more plate data. Plates per
  segment 5/2/0. +0.019 more.
- **Literature checked** (and why most were not pursued): multi-fidelity MES and trace-aware KG (the funnel already
  fixes fidelity roles, and MBR→Pilot glutamine does not transfer), cost-aware Gittins-index / Pandora's-box stopping
  (budget is always spent to checkpoints), top-two Thompson sampling for best-arm identification (clone racing is
  already exact within one noise factor), πBO / BOPrO priors (subsumed by the explicit template posterior),
  batch q-EI (batches are tiny and fixed), hybrid/parametric-mean GPs (the template is that parametric mean).
- **Dead ends this session:** plate medium (mid vs rich, ±0.017, noise), least-squares template fit (+0.026 DEV,
  +0.002 off-DEV), re-screening collapsed MBR screens (≈0), more VOI clones (15: no gain, slower).

## Generic track
`generic_funnel.py` uses the same machinery with no simulator-derived numbers:
- prior recipe at the centre of the search box, with glutamine on a log scale;
- glucose is searched;
- plates use a mid-range medium;
- no glutamine portfolio.

It beats `mfgp_structured` by +0.09 to +0.16 on the 100-scenario splits (wins 68–83 %), but trails the informed
version by about 0.18. The gap is the value of the recipe knowledge.

**`generic_bayes.py` (session 3)** adds the Bayes temperature/pH decision, value-of-information plates and the
expected-titer screening order, but takes the template widths and floor from the in-run joint peak fit (shared by
all clones, learned from the plate data) instead of the simulator study; clone ranking keeps the peak-fit heights.
It beats `generic_funnel.py` on every split: off_big 0.475 vs 0.429 (+0.047 ± 0.007, wins 62 %), +0.027 to +0.051
on the other oracle splits. Caveat: the hyperparameters (residual scale 0.35, Pilot-utility widening ×1.3, VOI over
10 clones) were tuned on the informed track, so they carry a little simulator experience; none is a clone- or
recipe-specific number.

## Infrastructure in `dev/` (analysis only, not part of the submission)
- `evaluate.py`: oracle-normalised evaluation on any split, with paired differences.
- `oracle.py`: oracles by differential evolution; `proxy_oracle.py`: grid-based proxy oracles, used for the hold-out
  splits.
- `gen_official.py`: official-compliant scenario generator.
- `trace.py`, `diag_funnel.py`, `pairdiff.py`: diagnostics.
- `NOTES.md`: the full experiment log; `algos/`: every variant tried (~160).
