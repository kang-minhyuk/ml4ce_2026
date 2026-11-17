# Algorithm development log

## Setup
- Env: uv + .venv (git missing → dbiolab installed from GitHub tarball of the pinned commit). Run with `.venv/bin/python`.
- Splits (dev/splits.json): DEV = first 100 coursework-compatible (24/4/1) scenarios with id<500; HOLD = first 100
  with id>=500 (untouched until final validation); STRESS = 60 non-coursework batch-size scenarios.
- Oracle (dev/oracle.py): best noise-free Pilot titer per scenario (DE per clone + refine on top 4).
- Noise is a single multiplicative factor per run (lab re-seeds noise with the lab seed each experiment):
  factor = 1 + sigma * default_rng(seed).normal(0,1,28)[-1]. Normalised score = score / (3 * oracle * factor).
- Harness: dev/evaluate.py (official run_algorithm via scenario_suite/run_suite.run_one; cached).

## Simulator facts (analysis only)
- Optimal Pilot recipe (100 DEV oracles): glucose at upper bound for start + both feeds; gln start/feed1 ~1-9 % of
  range; gln feed2 ~25-75 %; T/pH after feed 2 ≈ clone growth optimum (tight), after feed 1 near it, start looser.
- 2nd best clone ≈ 84 % of oracle, 3rd ≈ 76 %.
- MBR-optimal recipe transferred to Pilot ≈ 57-76 % of Pilot optimum (practice lab). Clone ranking MBR vs Pilot
  Spearman ≈ 0.96 (at per-scale optima); plates much weaker (≈ 0.5-0.85).
- Structured heuristic with TRUE clone T/pH optimum: best clone 0.73-0.77 of oracle; picking clone by plate 0.34-0.79
  depending on plate medium (moderate medium 0.3/0.3 best); plate top-k → MBR → best ≈ 0.65-0.70.

## Results (DEV, normalised; 1.0 = oracle found before first checkpoint)
| algorithm | norm | n33 | n66 | n100 | note |
|---|---|---|---|---|---|
| reference_screen_then_confirm | 0.215 | .164 | .217 | .264 | course reference |
| mfgp_structured (user) | 0.333 | .244 | .352 | .402 | strongest baseline |
| a1 (12-D MF-GP, log) | 0.240 | | | | plates encoded with feeds=0 poisoned Pilot recipes |
| a2_tied_m999 | 0.319 | | | | masked feed kernel; tied 6-D |
| a3_* (mfgp + single tweaks) | 0.305-0.332 | | | | none beat mfgp (pilot restriction, mean acq, MBR/plate counts) |
| a4_D8 / snap | 0.296-0.301 | | | | untied feeds hurt in GP framework |

Decomposition of mfgp first Pilot: actual 0.285 of oracle; chosen clone's structured-heuristic value 0.43 (clone rank
~8); recipe efficiency 0.66. → both clone choice and recipe matter.
| a4_D7/D5 (+snap) | 0.299-0.312 | | | | lower-dim spaces in mfgp framework: no gain |
| a4_add_* (additive kernel) | 0.284-0.304 | | | | slower (hits 300 s), worse |
| g_ts_* (Thompson sampling MBR/plates) | 0.295-0.306 | | | | worse; runs hit 300 s under CPU overload |
| g_pilot_ei | 0.329 | | | | ≈ mfgp |

- Plates at RICH medium (glc max, gln 5 %) peak at the clone's true T/pH optimum (±0.4 C, ±0.08 pH); at moderate medium
  (0.3/0.3) T peak can be off by >2 C. Symmetric quadratic RSM fit was poor (T err 2.3 C).
- Official-compliant scenario generator: dev/gen_official.py (300 scenarios, ids 10000+; splits off_dev/off_hold/off_extra).
- LESSON: never run >22 busy processes; time-limit hits under overload corrupt cached results.

## Informed funnel (i2 → i4)
| algorithm | DEV norm | practice (5 seeds) | note |
|---|---|---|---|
| i2_funnel | 0.366 | | MBR before plates; runs hit 300 s only due to CPU overload (load 160) |
| i2_pf | 0.381 (n33 .285 n66 .414 n100 .444) | 0.583 | plates first; +0.048 ± 0.022 vs mfgp, wins 59 % |
| i3_funnel | | 0.532 | i2_pf + analytic gradients (peak fit, GP NLL), peak-fit caching: 43 s → 16 s per run |

- Funnel diagnosis (i3, 40 DEV, structured heuristic with true T/pH optimum as clone value):
  best clone 0.742 → best MBR-tested 0.547 (coverage loss −0.20) → chosen 0.496 (selection −0.05)
  → first Pilot actual 0.399 (recipe −0.10); best Pilot 0.567. Plate-fit error at first Pilot: T 0.9 C, pH 0.20 median.
- Simulator T/pH factor is a Gaussian multiplier on growth rate (shared widths for all clones, clipped at 0.2).
  Plate study (30 DEV, Sobol rich-medium plates): per-clone-amplitude Gaussian peak model locates optima far better
  (8 plates: T 0.25 C / pH 0.05 vs quadratic 0.65 / 0.14); shared-amplitude model ranks clones best
  (top-4 capture 0.85, top-8 0.93). i4 = location from the per-clone model, ranking from the shared one.

## Glutamine is the dominant recipe lever (analysis, noise-free sim)
- With T/pH at the clone optimum in all three phases, optimising only (gln start a0, feed1 a1, feed2 b2) reaches the
  oracle (0.63 → 1.00 on 16 DEV). Early-phase T/pH tuning adds nothing → oracle's phase variation is a flat direction.
- Optimum is a sharp cusp in log concentration (gln just exhausted at the next feed; the simulator lets gln go
  negative, so under-feeding creates a "debt"). Median optimum (fraction of range): a0 0.003, a1 0.027, b2 0.37;
  ×2 on a0 → 0.74, ×0.5 → 0.87. Optimum varies ~100× across clones and differs between MBR and Pilot
  (MBR-optimal grid recipe on Pilot: 0.66). Best fixed Pilot prior (grid, 40 DEV): (a0 .008, a1 .005, b2 .6) → 0.754
  (p10 0.49); per-clone grid-best → 0.95.
- DEV (clean load): i3_funnel 0.452 (n33 .338 n66 .462 n100 .556) vs mfgp 0.333; i4 (Gaussian peak model) 0.423 —
  better optimum location did not help. Earlier i2 numbers (0.37-0.38) were depressed by CPU overload.
- Bug fixed: GP noise lower bound exceeded upper bound when all bio titers equal (ValueError).

## i5 / i6 (log-gln GP, MBR = screening only, Pilot EI on top-2 clones)
| algorithm | DEV | off_dev | note |
|---|---|---|---|
| mfgp_structured | 0.333 | 0.287 | baseline |
| i3_funnel | 0.452 | 0.396 | |
| i5_funnel | 0.462 | | log10 gln start/feed1 separate (5-D GP), prior (0.01, 0.005, 0.6) |
| i5_s2e | 0.478 (n33 .340 n66 .497 n100 .596) | 0.448 (+0.161 vs mfgp, wins 76 %) | + no MBR refinement slots, pilot top-2, EI |
- Glutamine BO alone (oracle clone, true T/pH, gln_bo.py): Pilot-only EI from prior 0.75 → 0.86 (4 runs) → 0.94 (8);
  adding 8 MBR runs first makes it worse (gln optimum shifts between scales).
- Plate ranking power with TRUE T/pH optimum vs per-clone Pilot potential (clone_value.json, grid-best gln):
  rich medium top-1 0.84, top-4 0.94; moderate (0.3/0.3) top-1 0.93; MBR at prior recipe top-1 0.93, top-4 0.99.
  With fitted optima (plate_study2): top-4 ≈ 0.88-0.91 regardless of design; targeted plates ≈ Sobol.
- i5_s2e first-Pilot decomposition with potential: best clone 0.957 → best tested 0.726 → chosen 0.707 →
  actual 0.404 (recipe efficiency 0.57: T/pH errors 1.1 C / 0.24 pH hurt a lot). Best Pilot overall 0.651.

## Bounds and dead ends (Nov 14)
| variant | DEV | off_dev | note |
|---|---|---|---|
| i6_funnel (= i5_s2e, refactored) | 0.478 | 0.448 | reproduces i5_s2e exactly |
| i6_hyb / i7_hyb (Gaussian-peak locations) | 0.471-0.482 | 0.425-0.447 | no gain |
| i7 tp_bias (safe-side T+, pH−) 0.03/0.05/0.08 | 0.467-0.482 | 0.440-0.459 | no gain (noise) |
| i8 (peak fit on own-medium plates only) | 0.456 | 0.441 | worse: initial random-medium plates still help |
| x8_cheatloc (TRUE T/pH optimum, analysis only) | 0.505 | 0.493 | location worth only +0.03/+0.045 |
| x8_cheatrank (TRUE potential ranking, analysis only) | 0.458 | | perfect plate ranking does not help |
- Pilot sensitivity (tp_sens.py): T −2 C → 0.33, −1 → 0.84, +2 → 0.91; pH +0.2 → 0.52, +0.3 → 0.22, −0.3 → 0.95.
- Conclusion: the bottleneck is the Pilot-scale recipe (glutamine) search, not clone screening or T/pH location.

## Pilot policy fixes (i9 → i11)
- Pilot traces (i6): first 1-2 Pilots often went to clones known only from the random initial-design MBR runs, or to
  extrapolated recipes (gln at box edge, dT ±3 C) → titer ≈ 0. GP-mean clone ranking was unreliable.
| variant | DEV | off_dev | note |
|---|---|---|---|
| i9_fp (first Pilot = prior recipe) | 0.464 | 0.483 | |
| i9_tp (Pilot varies gln only) | 0.476 | 0.474 | |
| i9_pg (Pilot-only GP) | 0.427 | 0.448 | worse |
| i10_funnel (clones ranked by MEASURED titer, first Pilot prior, clipped bases) | 0.478 | 0.500 | |
| i10_t1 (Pilot on one clone only) | 0.449 | 0.448 | worse |
| **i10_tp** (+ Pilot T/pH fixed at estimate) | **0.502** | **0.500** | +0.024 / +0.053 vs i6 |
| i11 (rescreen initial clones / MBR T/pH refinement slots) | 0.494-0.498 | 0.501-0.504 | neutral |
| allocation sweep on i10_tp (pilot_frac, mbr_per_seg, plates [4,3,1]/[5,2,0]) | 0.496-0.504 | 0.491-0.502 | insensitive |
| plates [3,2,0] | 0.447 | 0.474 | 4 plates in segment 1 matter |
| i12 (Pilot candidates from Pilot rows only; smaller log steps) | 0.485-0.489 | | worse on DEV |

## Scheduling and clone-racing (i13, i14)
- Trace analysis (i10_tp, 24 DEV): 9/24 scenarios never ran an MBR screen. Cause: per-segment money; once a segment
  had its Pilot and leftover < Pilot cost, the scheduler fired a Pilot that spent the NEXT segment's money first.
  i13 plans the next segment instead (seg_advance). Per-scenario swings are huge (±0.38) → decisions are fragile.
| variant | DEV | off_dev |
|---|---|---|
| i10_tp | 0.502 | 0.500 |
| i13_funnel (seg_advance) | 0.500 | 0.505 |
| i13_mbp (+ MBR before last plates) | 0.516 | 0.510 |
| i13_p3 (+ plates [4,3,1]) | 0.508 | 0.516 |
- Scenario 7 (i13): MBR screen found clone 8 at 4.7× the Pilot incumbent's MBR titer, but GP-EI kept running Pilots on
  the incumbent → the GP does not separate clones reliably. i14: explicit racing on measured titers at the prior.
| i14_funnel (racing) | 0.563 | 0.555 |
| **i14_mbp** (racing + MBR before last plates) | **0.573** (n33 .403 n66 .623 n100 .694, p10 .33) | **0.559** |
| i14 switch 0.9 / 1.1 | 0.567 / 0.563 | 0.552 / 0.553 |
| i16_funnel (+ initial-design clones get a prior-recipe screen) | 0.582 | 0.566 |
- i14_mbp end-of-run decomposition (100 DEV): best potential 0.951 → best screened 0.817 → main Pilot clone 0.788 →
  best on it 0.694 (recipe efficiency 0.88); 9.3 clones screened, 5.9 Pilots (5.0 on main clone).
- Glutamine portfolio (gln_portfolio.py, oracle clone, true T/pH; greedy on 100 DEV): (a0, a1, b2) =
  (.016,.002,.6) → .731 [off .727]; + (.03,.02,.85) → .798 [.766]; + (.008,.002,.85) → .847 [.841];
  + (.002,.005,.25) → .872 [.861]. Beats GP-EI from a prior for the first 3-5 Pilots (EI: .75/.80/.81/.86/.88).
| **i17_funnel** (+ 3-recipe gln portfolio for the incumbent's first Pilots) | **0.596** (n33 .408 n66 .658 n100 .721) | **0.578** |
| i17 portfolio k=1/2/4 | 0.585/0.592/0.595 | 0.567/0.575/0.574 |
- i17 validation: practice 0.672 (mfgp 0.200, 5/5 wins); stress (60 non-24/4/1 scenarios) raw 69.7k vs mfgp 39.0k, 0 errors,
  max 81 s.
- Generic track (g1 = i17 machinery, no simulator-derived numbers; glucose searched; prior at box centre):
  g1_generic (linear centre prior) DEV 0.338 / off 0.334; **g1_genlog** (gln on log scale, prior at log-box centre)
  DEV 0.402 / off 0.382 (+0.069 / +0.095 vs mfgp, wins 65 % / 75 %).
- Bounds on i17 (analysis only): TRUE T/pH location → 0.674 (+0.078); TRUE clone ranking → 0.629 (+0.033).
  With T/pH fixed at Pilot, location quality now matters.
- Location experiments on i17: Gaussian location models in pipeline (own/all plates, 4-6 plates) 0.564-0.588 DEV
  (all ≤ i17) even though offline/real-data location error is slightly lower (gk-all median 0.83 C vs quad 1.0 C).
  Real-pipeline first-Pilot clone location error (quad): median 1.0 C / 0.26 pH, bias +0.5 C / −0.23 pH (safe side).
  Rich-medium plate titer optimum = growth optimum (±0.3 C, ±0.03 pH); MBR/Pilot at the prior gln recipe are NOT
  (sd 1.3 / 0.9 C) → only plates should locate optima.
- i19 (Pilot T/pH free after the portfolio): no change.

## Final files (dev/final/)
- informed_funnel.py = i17 (config folded in + time guard: no model refits when < 20 s remain). Reproduces i17 exactly.
- generic_funnel.py = g1_genlog. Reproduces exactly. Both pass check_my_algorithm except STUDENT_NAMES placeholder.
- Time-limit robustness (informed, 30 DEV): limit 30 s → 0.621 (unlimited 0.610); limit 12 s (never fits) → 0.355; 0 errors.

## Hold-out validation (untouched until now; HOLD normalised with proxy oracle = grid-best gln at true T/pH,
## median ratio to DE oracle 0.998 on DEV)
| algorithm | HOLD | off_hold |
|---|---|---|
| mfgp_structured | 0.339 | 0.291 |
| generic_funnel (g1_genlog) | 0.433 (+0.094, wins 70 %) | 0.442 (+0.151, wins 80 %) |
| informed_funnel (i17) | 0.637 (+0.297, wins 93 %) | 0.598 (+0.307, wins 92 %) |
- DE oracle failed on DEV sid 62 (proxy 4.1× higher) → normalised DEV numbers slightly inflated there for all algos.

## Robustness fixes (i22) — found via edge-case tests (budget 6000-16000 on the practice lab)
- Bugs: (1) a segment that cannot afford a Pilot spent its money on fillers → no Pilot ever (budget 6000 → score 0);
  (2) allocations indexed by segment, so when checkpoint 1 was unreachable the next segment planned 0 plates;
  (3) with no prior-recipe MBR screens, the Pilot clone came from random-recipe initial MBR runs (titer ~100);
  (4) with < 3 own plates the peak fit is unreliable.
- Fixes: advance_unreachable, plan_by_stage, fallback_plates (plate ranking), min_fit_plates (best observed plate
  condition until 3 own plates).
| algorithm | DEV | off_dev | off_low (40 scenarios, budget 3.5-6 Pilots) |
|---|---|---|---|
| mfgp_structured | 0.333 | 0.287 | 0.210 |
| g1_genlog (generic) | 0.402 | 0.382 | 0.198 |
| i17_funnel | 0.596 | 0.578 | 0.319 |
| **i22_funnel** | **0.609** | **0.598** (p10 .356) | **0.398** |
- Ablations (DEV/off): without plan_by_stage 0.608/0.595, without min_fit_plates 0.601/0.583, without advance 0.602/0.596,
  without plate fallback 0.593/0.590.
- Other dead ends today: focus plates around the incumbent's optimum (i20: −0.001 to −0.021); safe-side T/pH shift
  on i17 (i21: ≤ +0.004 off_dev); conditional gln portfolio (offline +0.01 only).

## Final validation (dev/final files; 0 errors on every split, max wall 65 s with 20 parallel runs)
| algorithm | DEV | off_dev | HOLD | off_hold | off_low | off_wide | practice |
|---|---|---|---|---|---|---|---|
| mfgp_structured | 0.333 | 0.287 | 0.339 | 0.291 | 0.210 | 0.260 | 0.200 |
| generic_funnel (g2_genlog) | 0.426 | 0.401 | 0.440 | 0.450 | 0.251 | 0.455 | 0.512 |
| informed_funnel (i22) | 0.609 | 0.598 | 0.646 | 0.615 | 0.398 | 0.542 | 0.672 |
- off_wide = 40 official-style scenarios with glutamine bounds ×4 and glucose ×2 (tests bound-relative priors):
  informed drops ~0.06 but stays well ahead.
- Submission checker caught `self._best_cond` (any `_attr` is rejected) → renamed `best_cond_cache`.
- Late dead ends: fair racing at the prior recipe (i23, ±0.001), runner-up Pilot after portfolio (i24, −0.002/−0.005);
  MBR-optimal gln + incumbent-learned MBR→Pilot shift would give 0.82 vs prior 0.73 for other clones offline, but
  needs a full MBR gln search per clone (unaffordable).

## Session 2 (Nov 14 pm): transfer-BO library, probes, focus plates, skipping checkpoint 1
- Library data: gln_lib.npz / gln_lib_mbr.npz = Pilot / MBR titer on an ABSOLUTE grid (a0 13 x a1 12 x b2 7 = 1092
  points, glucose at upper bound) for 1950 clones (650 unused scenarios: 250 suite ids not in any split, off_extra,
  300 new official-style ids 10380-10679; top-2 + 1 random clone each; half at the true T/pH, half with a
  pipeline-like location error N(0.4, 0.9) C / N(-0.15, 0.2) pH). lib_gen.py / lib_gen_mbr.py.
- Offline policies (lib_bo*.py; train suite -> test official and reverse, best/grid-best after k Pilots):
  | policy | k=2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|
  | fixed greedy portfolio | .846 | .905 | .928 | .942 | .950 |
  | library mixture EI (likelihood weights, free log offset, sig 0.1; RGPE-like) | .840 | .937 | .961 | .973 | .982 |
  | pre-trained low-rank Gaussian prior (HyperBO-like), best rank/nugget | .839 | .930 | .947 | .956 | .963 |
  Library PCA: 1 component 85 %, 3 components 97 %.
- Pipeline (i25 = i22 + library EI for the incumbent): the 2nd Pilot (1 observation) is too exploratory; library from
  the 3rd Pilot (2nd = portfolio[1]) is best. i25_f2 DEV 0.617 (+0.009 +-0.003), off_dev 0.605 (+0.007).
  logEI / efficiency-EI / sig 0.05-0.2 / lib_from 3 / allocation sweeps: all within +-0.003.
- Compressed libraries (pipeline, DEV vs full): 100-400 random curves -0.000..-0.004; PCA 6-16 -0.001..-0.004;
  coarse 7x6x4 grid -0.002; coarse grid + PCA 12 + 300 curves -0.003 (5.8k numbers, embeddable: embed_lib.py).
  Validation of that candidate vs informed_funnel: off_dev +0.006, HOLD +0.004, off_hold +0.004, off_low 0.000,
  off_wide +0.011, practice +0.011 (never negative).
- Joint MBR/Pilot library + MBR glutamine probes (i26): offline one extra MBR probe (low gln start) lifts the first
  Pilot recipe from 0.77 to 0.88 of the clone optimum, but in the pipeline -0.007 to -0.049: probes displace
  screening (7.3 vs 10.8 clones screened) and the clone's T/pH estimate moves between the probe batch and the Pilot
  (probes at stale T/pH are unusable). Offline, probes on the incumbent later in the run add only +0.03 at Pilot 2-3.
- User idea: ignore checkpoint 1, maximise checkpoint 2 (i27_s*): -0.08 to -0.13 (ck1 0.42 -> 0.05-0.15, ck2/ck3
  only +0.01-0.04). Dead end: one Pilot before checkpoint 1 is cheap insurance.
- Focus plates by pattern search on the incumbent's own plate titers (i28; plate optimum = growth optimum; offline
  4-8 plates recover the true-location value): DEV +0.004-0.006, off_dev -0.006..+0.002, p10 worse; final T error
  median 0.71 -> 0.52 C. Many large errors are in scenarios where the experiment cap forbids more plates.
- All zero-checkpoint runs (8 DEV, 14 off_dev) are scenarios whose initial design already spends past checkpoint 1.
- Continuous library candidates (i31, random + local around the best recipe): DEV -0.001, off_dev +0.000 vs grid only.
- FINAL dev/final/informed_library.py (= i22 + portfolio[1] for the 2nd incumbent Pilot + embedded library EI from the
  3rd; embed_lib.py, 7x6x4 grid, PCA 12, 300 members, 87 KB). Validation vs informed_funnel: DEV 0.614 (+0.005),
  off_dev 0.604 (+0.006), HOLD 0.650 (+0.004), off_hold 0.618 (+0.004), off_low 0.398 (0.000), off_wide 0.553
  (+0.011), practice 0.682 (+0.011); stress raw 68.0k (67.9k), 0 errors, max 51 s; 30 s limit unchanged; edge tests
  as before (zero only where one Pilot is unaffordable).

## Session 3 (Nov 16): literature-driven search
- Literature checked: cost-aware MF-BO (MF-MES, cfKG, PBGI/Pandora's box), best-arm identification (top-two TS,
  correlated KG), priors over the optimum (piBO, BOPrO), batch BO (KB, local penalisation, q-EI), BO guide for
  bioprocesses (arXiv 2508.10642: parametric/hybrid mean functions, multi-task GPs).
- Checkpoint decomposition of informed_library (ckdecomp.py, 100 DEV; clone potential = grid-best gln at true T/pH):
  | | actual | no Pilot | clone loss | recipe loss | location loss |
  |---|---|---|---|---|---|
  | ck1 | 0.420 | 0.11 | 0.280 | 0.158 | 0.090 |
  | ck2 | 0.672 | 0 | 0.179 | 0.147 | 0.002 |
  | ck3 | 0.750 | 0 | 0.167 | 0.118 | -0.035 |
  End of run: best clone 0.951 -> best screened 0.907 (coverage) -> incumbent 0.833 (selection, only 54 % pick the
  best screened clone). 35/46 selection failures: the better clone's MBR prior titer was lower, often ~0 because its
  T/pH estimate was grossly wrong (dT up to 7 C, dpH up to 1.8; screened-clone location error median 0.21 of span).
- Plate T/pH response (plate_shape.py, plate_mod_study.py): log titer = h - floor*(1-exp(-d/floor)),
  d = asymmetric quadratic, additive in T and pH; widths CLONE- AND MEDIUM-INDEPENDENT (robustness factors are fixed
  at 1.1 in every scenario): soft template widths (C, pH) below/above: T 1.25/3.08, pH 0.65/0.15, floor ~5.
- Fixed-width template location (offline on traced plates): T err median 1.31 -> 0.72 C at first MBR, all plates
  0.74 -> 0.36 C; pH 0.25 -> 0.13. Pipeline (j1_tq: template location + quad ranking): DEV +0.026 +-0.015,
  off_dev +0.002 +-0.013; j1_tmpl (template ranking too): +0.007 / -0.014. Slow (joint fit) -> vectorised LM fit
  (tmpl_fit.py fit_tmpl_vec, 0.04 s).
- Screening power at TRUE T/pH (screen_study.py, plate_at_opt.py): pick quality (pot of chosen / best) among all
  clones: MBR prior 0.918, Pilot prior 0.955, rich plate 0.824, MODERATE plate (0.3/0.3) 0.927, mid (0.5/0.2) 0.926;
  learned log MBR + 0.35 log rich plate 0.963 -- but with fitted plate heights from real runs it does not help
  (score_offline.py).
- 30/100 DEV runs have no MBR screen before the first Pilot (MBR batch = exactly 4 runs, slack < 4 MBR before ck1):
  first Pilot clone from plate rank only -> potential 0.59 vs 0.78 when screened.
- Plate medium (moderate/mid media rank clones far better at the TRUE optimum: top-1 0.93 vs rich 0.82; T/pH
  template shape identical across media; peak exactly at the optimum 80-83 % vs 92 % for rich). Pipeline
  (k1_*, plate_medium g/gln fraction): (0.5,0.2) DEV -0.017 / off_dev +0.017; (0.3,0.3) -0.007/-0.005;
  (0.5,0.1) +0.005/-0.003; (0.7,0.2) -0.014/-0.003 -> no effect beyond noise.
- Paired-difference noise per 100 scenarios is ~0.015, so +0.01 effects are invisible: generated off_big =
  400 fresh official-style scenarios (ids 10680-11079, seed 2027, gen_big_official.py) with proxy oracles.
- Low-budget failure mode (DEV sid 5): with < min_fit_plates (3) own plates the location is the clone's best
  observed plate condition, i.e. one of 2-4 plate conditions (up to 4 C off); template LS fit without the fallback
  pulls weakly identified clones to the box centre (worse, sid 5 50.8k vs 77.6k).
- Bayes-optimal operating T/pH (bayes_loc.py): grid posterior (31x31) over the clone optimum with the template
  likelihood (height profiled out, Cauchy residuals sig 0.35), uniform prior; action maximises expected relative
  Pilot titer (template widths x1.3). Offline expected relative Pilot titer over all plate-seen clones:
  first MBR: quad 0.445 (p25 0.045), best well 0.398, template LS 0.512, Bayes 0.627 (p25 0.385);
  all plates: quad 0.570, template LS 0.638, Bayes 0.717. 0.03 s per fit. Pipeline variant k5_bayes (j5_base).
- PIPELINE RESULT (k5_bayes = informed_library + Bayes-optimal T/pH location, quad heights for ranking,
  min_fit_plates 0): DEV +0.030 +-0.017, off_dev +0.031 +-0.012, off_big (400) 0.688 vs 0.639 = +0.049 +-0.007
  (wins 61 %, n33 .444->.496, n66 .696->.745, n100 .777->.822, p10 .377->.439). k5_bayes_mf (keeps the best-well
  fallback below 3 own plates): DEV +0.038, off_dev +0.029. j3_tq (template LS location) off_big +0.030;
  k1_m52 (mid plate medium) -0.004.
- off_big tuning (vs informed_library 0.639): k5_bayes_mf +0.050, bayes_wscale 1.6 +0.044, bayes_sig 0.6 +0.044,
  re-screen collapsed MBR screens +0.049, min_fit_plates 2 +0.052 -> location-model knobs flat.
- VALUE-OF-INFORMATION PLATES (bayes_voi.py, j6_base plate_design "voi"): after the space-filling plates, each plate
  condition maximises the summed expected gain in the Bayes decision utility (expected relative Pilot titer) of the
  top-6 ranked clones (weights 1/(1+0.3 i)); 15x15 posterior grid, 11x11 actions and candidates, top-40 support.
  Wells go to clones in rank order. k7_voi +0.064 +-0.007 (0.703), k7_voi_s2 (2 space-filling plates) +0.073
  +-0.007 (0.712; n33 .515, n66 .774, n100 .847, p10 .473), 0.4 s per design.
- VOI design variants (off_big, vs k7_voi_s2 0.712): n_space_plates 1 +0.001, 0 -0.005; voi_k 3 -0.012, 10 +0.008;
  bayes_rank (posterior-mean plate height for ranking) +0.008; plates_per_seg [5,2,0] +0.007, [4,3,1] +0.003.
- j7_base adds screen_eu: MBR screening order by plate height + lambda log(expected relative titer at the action).
- Combined (off_big, vs k7_voi_s2 0.712): k11_all (voi_k 10 + bayes_rank + plates [5,2,0]) 0.729 (+0.017 +-0.006);
  k11_se05 (k11_all + screen_eu 0.5) 0.731 (+0.019 +-0.006); voi_k 15 0.728 (max wall 118 s, no gain);
  plates [6,2,0] 0.722; screen_eu alone on k7_voi_s2: 1.0 +0.007 +-0.003, 0.5 +0.006 +-0.002 (small, consistent).
  k11_se05 vs informed_library on DEV +0.060 +-0.017.
- final/informed_bayes.py = k11_se05 with dead paths removed (quad/gauss/template-LS peak fits, targeted plate
  branch, re-screen); quad fit no longer run (bayes_rank uses posterior heights) -> median wall ~4.5 s.
  check_my_algorithm: only STUDENT_NAMES flagged. edge_test: no errors (tiny_budget / one_pilot 0 as before).
- Generic track (algos/g_base.py): same Bayes location / VOI / screen_eu, but widths and floor come from the
  in-run joint quad peak fit (shared across clones, learned from data, clipped), no simulator constants.
- informed_bayes validation (vs informed_library): off_big +0.091 +-0.008 (0.730), DEV +0.056, off_dev +0.074,
  HOLD +0.063, off_hold +0.094, off_low +0.073, off_wide +0.107, practice +0.010; raw titer stress +16 %, off_extra
  +15 %; 30 s limit DEV 0.670 (unchanged); max wall 14 s.
- GENERIC (off_big vs generic_funnel 0.429): g1_bayes (location only) +0.027, gw 0.75 +0.025; g1_voi (+ VOI, bayes
  rank, screen_eu) +0.042; g1_voi_nobr (quad heights for ranking) +0.047 +-0.007 (0.475) -> final/generic_bayes.py.
  Other splits: DEV +0.030, off_dev +0.046, HOLD +0.027, off_hold +0.036, off_low +0.029, off_wide +0.051, practice
  +0.142 (5 seeds). edge_test clean.

## Session 4 (Nov 16, evening): crazier ideas
- Archive: dev/archive/2026-11-16_session3/ (finals + SHA1SUMS, algos tarball, NOTES/SUMMARY); logs in dev/logs/.
  Register of all algorithms: dev/ALGORITHMS.md (make_register.py, logs/register/run_register.sh).
  evaluate.py: PRACTICE_SEEDS now 42..61; split "practice" = first 5, "practice20" = all 20.
- informed_bayes DEV decomposition (ckdecomp): ck1 actual .513; loss clone .242, recipe .164, location .018,
  missing pilot .110 (all 11 such runs have checkpoint 1 unreachable -> structural); ck3 clone .158, recipe .085,
  location -.032. Location is solved; clone choice and the first Pilot recipe remain.
- Literature agent (ideas): amortised cheap->Pilot predictor (SBI), MBR-conditioned library (already a dead end),
  checkpoint rollout, offline policy search, anchor clone, PI-BAI. See ALGORITHMS.md references.
- plate_signature.py (120 training scenarios, true T/pH): selection of the best clone by MBR at portfolio[0] 0.896
  (top1 .35), at portfolio[2] 0.950 (top1 .60); Pilot p0 .963; plate (1.0,.05) .866, plate (0.3,.15) .903;
  ridge plate+MBR .952. screen_recipe_study.py: portfolio[2] (-2.1,-2.7,.85) is the best common screening recipe
  (test .950). BUT at the pipeline's screening depth (top-4 plate clones) port0 = port2 = 0.904 -> l1_p2 off_big
  +0.001 +-0.003. Lean plate media in pipeline (l2_*): -0.004..+0.001. Plate prefilter is not the pipeline bottleneck.
- Plate residual (potential not explained by plate) correlates with y_cells_glc (+.36), m_glc (-.18).
- lib_signature.py: plate media log-ratios (vs reference plate, same T/pH) predict the Pilot glutamine optimum:
  library kNN first-Pilot relative titer .785 (best fixed) -> .88 (3 media); with the embedded 300-member library,
  noise .05: 1 medium (glc .1, gln .005) .843, 2 media .858 vs port0-like .732. Ratios are robust to T/pH error
  (shift .03 safe side, .15-.2 steep side vs between-clone sd .4-.6).
- l3 signature plates, v1 inserted before the first MBR/Pilot: off_big -0.034 / -0.045, zero-checkpoint runs 47 -> 88
  (a plate inserted before the first Pilot spent the money that Pilot needed before checkpoint 1). v3 = planner
  slots (l6_*).
- Trace: in 18/100 DEV runs a Pilot-tested clone with on average 0.22 higher potential was abandoned; best screened
  clone potential .948 vs final .842. -> l4 inc_rule "lib" (incumbent = library-posterior potential), l5 race_eu
  (screen titer / expected relative titer at the screen's T/pH).
- l4 inc_rule "lib" (incumbent = library-posterior potential): off_big -0.001 (32/400 runs changed: 9 better, 23 worse).
  ORACLE RACING (dbg/oracle_race*.py, cheats with true potentials, analysis only): racing order only +0.004 +-0.006;
  order + incumbent +0.007 +-0.006 (n33 -0.012, n100 +0.029) -> racing among screened clones is NOT the bottleneck.
- l5 race_eu (screen titer / posterior expected rel. titer at the screen T/pH): 0.5 -0.007, 1.0 -0.013, 1.0 + screen
  recipe portfolio[2] -0.009 (off_big). Dropped (racing ceiling is +0.007 anyway).
- glc_check.py: glucose at the upper bound in start and both feeds is optimal for every tested clone (300 clones;
  max gain from less glucose 1.6 %).
- l6 signature plates in planner slots (v3): sig (0.1,.005) -0.011 +-0.002, 2 media -0.014, + lib incumbent -0.012
  (off_big; zero-checkpoint runs back to 47). DEV traces: signature plate runs in only ~40 % of runs (needs all
  segment-0 plates), first-Pilot recipe changes in few of those (floor guard / missing ratios), mixed effect
  (sid 0 15.1k vs 18.2k, sid 3 23.7k vs 23.0k). Upper bound ~+0.007 even if it worked -> shelved.
- KNOB SWEEP m1 (off_big vs informed_bayes 0.730): lib_from 1 -0.004; lib_sig .05/.2 0; mbr [1,2,0] -0.001,
  [1,3,1] -0.001, [2,2,1] +0.005 +-0.003 (n33 .552 -> .566); n_space_plates 3 -0.007; plates [5,2,1] 0, [5,3,0]
  -0.003, [6,2,0] -0.008; pilot_frac [0,.34,1] 0, [0,.67,1] +0.001; screen_eu 1.0 -0.001; bayes_wscale 1.5 +0.002.
  End reasons: ~40 % of runs end with the experiment limit binding (plates use 24 experiments each).
- DEV: 30 % of runs have no MBR screen before the first Pilot (money); their first Pilot clone comes from 3-5 plates
  only (potential 0.26-1.00). Runs with a first MBR batch: best in batch .888, chosen .851.
- SWEEP m2 (off_big vs informed_bayes): mbr [2,2,1] +0.005, [2,1,1] +0.005, [2,3,1] +0.004, [3,2,1] +0.004 (all
  n33 .552 -> .565); [2,2,1]+plates [4,2,0] -0.001; [2,2,1]+pilot_frac [0,.67,1] +0.006; [2,2,1]+wscale 1.5 +0.007
  +-0.003; wscale 1.7 +0.002.
- Register complete (ALGORITHMS.md). Practice lab (20 seeds): informed_bayes 0.689, k5_bayes 0.722 (+0.032 +-0.018
  paired, all ck1), k7_voi 0.713; README: grading uses a different unseen lab -> optimise off_big, report practice.
- n1 (off_big): pilot_eu (first Pilot clone without screens by height + lam log eu) 0.5 0.000, 1.0 -0.000, 1.0+mbr
  [2,2,1] +0.004; voi_decay 0.1 -0.003, 0.8 -0.001; voi_k 8 flat weights -0.002. Plateau.
- pilot_tmpl.py: measured Pilot T/pH response (72 clones, best gln): T -2 .21, -1 .79, +2 .87, +4 .42; pH -0.5 .80,
  +0.1 .87, +0.2 .42, +0.3 .07; near-separable; sharper than Gaussian on the steep sides. Plate template x1.3 mean
  abs err .085 (bias -.007), x1.5 .099 (bias +.054). o1_pu = utility from the measured curves.
- o1_pu (measured Pilot curves as utility) +0.001, + mbr [2,2,1] +0.005. Generic g2: mbr [2,2,1] 0.000, wscale 1.5
  -0.001, both +0.002, bayes_rank -0.005 (flat).
- r1 clone-ranking knowledge gradient added to the VOI plate criterion: lam .3 +0.001, 1 -0.003, 3 -0.004.
- COMBO p1_c221 (mbr [2,2,1] + wscale 1.5 + pilot_frac [0,.67,1]) off_big 0.737, +0.008 +-0.003; p1_c221w (no
  pilot_frac) +0.007; p1_c211 +0.005.
- policy_search.py: successive halving (100 random configs on off_big[:100] -> top 12 on off_big[100:]).
- POLICY SEARCH ps1 (100 random configs of 2-5 knobs, seed 1): stage 1 (off_big[:100]) top d1 +0.013..+0.021, but
  on the held-out 300 (stage 2) the top 12 shrink to d2 -0.004..+0.005 (all within 1.1 se) -> winner's curse;
  knob space is flat. Recurring in the top: bayes_sig 0.5, mbr [2,2,1]/[3,2,1], switch 0.9.
- p1_c221 validation (paired vs informed_bayes): DEV -0.000, off_dev -0.010 +-0.009, HOLD +0.005, off_hold -0.002,
  off_low +0.002, off_wide -0.000, practice20 +0.006 +-0.011 -> pooled over the 480 non-off_big scenarios ~ -0.001.
  The off_big +0.008 was selection on off_big. NOT promoted.
- Context analysis (dbg/context_diff.py): m221's gain sits where MBR is cheap (4 MBR/Pilot cost < .57: +0.013),
  cap low, noise high; all subgroup effects <= 2.5 se.
- End reasons off_big: budget 211, budget+experiment limit 179, experiment limit 10.
- Trace DEV 7 (36 clones, init 2 MBR batches, no plate): money for checkpoint 1 left room for only 1 of the 5
  segment-0 plates; the 4 unrun plates were dropped (not carried) -> 4 plates in the whole run (93 experiments
  unused), the 3 best clones (potential 36.9k/36.2k/32.1k) never screened; 7 Pilots tuned clone 6 (23.3k) -> t1
  carry_plates / carry_mbr (planner carries unaffordable screening into the next segment).
- q1 finer Bayes/VOI grids (vs p1_c221): fine -0.002, voi_nx -0.000; slower (median 21 s). Dropped.
- KEY FINDING (offline, lib_signature.npz + gln_lib.npz): the ABSOLUTE plate titer at the pipeline's own plate
  medium (glc 1.0, gln 0.05 of range) at the clone's optimum predicts the clone's Pilot glutamine optimum
  (high producers need more glutamine). Kernel-weighted library (300 members, s 0.3-0.7) first-Pilot relative
  titer 0.774 (best fixed) -> 0.845; robust: suite->official 0.849, official->suite 0.848, feature noise 0.5 -> 0.842,
  0.7 -> 0.835, shift +-0.2 -> 0.842. No extra experiments needed. GLN_PLATE = log plate titer of the 300 embedded
  members (embed_lib seed 0 selection verified).
- In-run feature quality (dbg/feat_check.py, HOLD, top clones): posterior-mean height (used for ranking!) vs true
  plate peak: corr 0.38, sd 0.66 (compressed by far wells hitting the template floor); max observed 0.80;
  "capped max" max_w(ly + min(pen_w at MAP, 1)) over standard-medium wells: bias 0.02, sd 0.43, corr 0.82.
- v1_* (feature = posterior height): first Pilot recipe from feature-weighted library, prior in lib EI, lib_from 1.
  v2_* (feature = capped max; capR = also rank clones by capped max).
- First-Pilot efficiency split (dbg/loc_split.py, DEV): mean 0.733, with the true T/pH 0.779 -> location is a minor
  part; the first glutamine recipe is the main loss. 21 % of first Pilots per clone < 0.6 of the clone's potential.
- t1 plate carry (off_big vs informed_bayes): cp9 +0.003 +-0.002 (n100 .856->.862), cp2 +0.003, cp9+carry_mbr identical (never active), cp9+mbr[2,2,1] +0.008. Small, consistent, structurally sensible -> keep as component.
- v1 (feature = posterior height, off_big vs informed_bayes): f03 +0.002, f03L +0.000, f03L1 +0.003, f05L +0.003, f07L +0.004 (all +-0.002; n33 only +0.003) -> far below the offline promise (noisy feature in-run).
- Why v1/v2 gain little: offline the embedded coarse library is fine (feature recipe .838 vs portfolio[0] .749, fine
  vs coarse candidates equal). In-pipeline (dbg/feat_ab.py, v2_capF DEV trace, first Pilot per clone, simulator at
  the pipeline's T/pH): chosen recipe / portfolio[0] titer = 1.049 mean (median 1.000, 35 % worse); with the TRUE
  plate feature 1.114. First-Pilot efficiency .733 -> .749 (dbg/loc_split.py). The in-run feature error (sd ~.43)
  costs half the gain.
- dbg/trace_any.py traces official scenarios too. off_dev: feature recipe with the TRUE plate feature / portfolio[0]
  = 1.068 (DEV 1.114); in-run chosen 1.023 -> ceiling of the feature idea on official-style ~+7 % at the first Pilot.
  Feature estimators (off_dev, wells before the clone's first Pilot): cap1 sd .69, cap2 sd .525 (bias .007,
  corr .73), posterior height corr .43 (ranking feature!), CV regression sd .44.
- u1 bayes_sig (off_big): 0.5 +0.001, 0.65 +0.002 (+-0.004). Flat.
- v2 (capped-max feature, off_big): capF +0.001, capF1 (lib_from 1) +0.003; capR (rank clones by capped max)
  -0.011 +-0.006, capFR -0.008 -> posterior-mean height is the better RANKING feature (it shrinks uncertain clones)
  even though capped max tracks the plate peak better.
