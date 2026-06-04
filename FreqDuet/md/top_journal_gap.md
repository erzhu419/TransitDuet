# FreqDuet Top-Journal Gap Backlog

Last updated: 2026-06-03

This file records the remaining gap between the current FreqDuet implementation
and a top-journal-ready paper package. It should be used as the execution
backlog after `dev_manual.md` and `GPT.md`: every item below needs either a
completed implementation/experiment, a documented negative result, or an
explicit paper-scope decision.

## Current Status

FreqDuet is now a working main-path prototype rather than a design sketch.
The current promoted line includes:

- causal harmonic demand decomposition with historical priors;
- LF/HF state separation;
- upper target-headway / timetable policy;
- lower discrete holding policy;
- lower high-frequency credit;
- leakage penalty;
- DriftFB feedback;
- guarded promotion with active and persistent gates;
- trace logger, audit tooling, MI / shock-response style diagnostics;
- scheduler/HPC-ready experiment scripts;
- a 40-episode, 20-seed, four-domain final matrix under the current naming.

Latest current-name 40ep final matrix:

```text
main        1.5446
rawhistory  1.5505
nopromotion 1.5599
allfreq     1.5681
nofreq      1.6281
noleakage   1.9113
```

Per-domain main result:

```text
terminal  1.463
highnoise 1.874
odshift   1.531
rushshift 1.311
```

Interpretation: the main method is best on average and best in terminal/rush,
but it is not yet best in every domain. `rawhistory` is still slightly better in
highnoise, and `rawhistory` / `nopromotion` are slightly better in odshift. This
means the method direction is effective, but the paper evidence is not closed.

2026-06-03 update: the current-name 200ep matrix has been synced and
aggregated. It confirms that `noleakage` is clearly bad, but it also exposed a
long-training weakness in the previous main line: lower action/drift rose after
roughly 80 episodes while `lower_lambda` decayed. The drift-cost repair
candidate closed this gap and has been promoted into `F_freqduet_*_main_hiro`.
Historical pre-repair aliases are kept as `*_main_predriftcost_hiro`.

## Top-Journal Gap Summary

Status legend:

- `[x]` completed enough for the current prototype;
- `[~]` partially done, needs paper-grade completion;
- `[ ]` not yet complete.

### 1. Current-Version Long Training

Status: `[x]` promoted 200ep matrix complete

The 200ep current-name matrix is now available, but it does not close the paper
claim by itself because the current main loses its 40ep advantage in some
domains during longer training.

2026-06-02 start: added the current paper-longtrain runner and paired-delta
summary tooling. The first six 80-run shards were auto-routed by scheduler to
Slurm because each requested 80 CPU workers; they were cancelled after staying
`PENDING` under `QOSMaxCpuPerUserLimit`.

The active launch is the direct-node retry below. It uses 30-core shards to stay
below scheduler's large-CPU Slurm auto-route threshold and to make the tasks
visible as normal scheduler direct-node jobs in `tui-top`.

```text
matrix: 4 domains x 6 methods x 20 paired seeds = 480 runs
episodes: 200
last_k: 100
effective active ranges:
  t5907  node001   0-30
  t5926  node005   30-60
  t5909  node003   60-90
  t5910  node004   90-120
  t5911  node005   120-150
  t5912  node006   150-180
  t5913  node001   180-210
  t5927  node006   210-240
  t5915  node003   240-270
  t5916  node004   270-300
  t5917  node005   300-330
  t5918  node006   330-360
  t5919  node001   360-390
  t5928  node001   390-420
  t5921  node003   420-450
  t5922  node004   450-480
state at launch check: all 16 effective ranges running direct on node001/node003/node004/node005/node006
```

2026-06-03 result: the pre-repair current 200ep summary is under
`FreqDuet/freqduet/results_freqduet/paper_longtrain_current_ep200_wu10`.
Overall paired deltas show that main is decisively better than `noleakage`, but
not decisively better than `nofreq`, `rawhistory`, `allfreq`, or `nopromotion`.
The diagnosis is that long-horizon lower drift was only a reward shaping term,
not a Lagrangian cost. The active repair adds `lower_drift_cost_*` leakage keys
and injects rolling drift excess into lower constrained cost.

The promoted 200ep matrix is under
`FreqDuet/freqduet/results_freqduet/paper_longtrain_promoted_ep200_wu10`.
Overall paired composite deltas for promoted main are:

```text
vs nofreq      -0.0719  CI [-0.1322, -0.0266]
vs rawhistory  -0.0528  CI [-0.0724, -0.0333]
vs allfreq     -0.0629  CI [-0.0913, -0.0346]
vs nopromotion -0.0542  CI [-0.0720, -0.0354]
vs noleakage   -0.3052  CI [-0.3908, -0.2286]
```

Done means:

- run current `main` against `nofreq`, `rawhistory`, `allfreq`,
  `nopromotion`, and `noleakage`;
- use 100 or 200 episodes;
- cover terminal, highnoise, odshift, and rushshift;
- report seed-level paired deltas, mean, std, confidence interval, and best/worst
  seed behavior;
- record whether the 40ep advantage survives long training.

### 2. Systematic Generalization Matrix

Status: `[~]` three-shift 100ep matrix complete; broader scenario families open

Highnoise, odshift, and rushshift are present, but this is not yet a complete
held-out generalization package.

2026-06-03 update: `scripts/run_freqduet_generalization_matrix.sh` now defaults
to highnoise / odshift / rushshift x six methods x 20 paired seeds at 100
episodes. The current promoted 100ep three-shift matrix has been executed under
`results_freqduet/generalization_promoted_ep100_wu10`; broader scenario
families beyond the three current held-out shifts are still open.

Done means:

- evaluate multiple demand-noise levels instead of one highnoise setting;
- evaluate multiple OD profile shifts;
- evaluate multiple rush-pattern shifts;
- include service stochasticity / route perturbation if feasible;
- separate in-distribution seeds from held-out scenario families;
- show whether frequency separation is robust or only tuned to one service day.

### 3. Per-Domain Weakness Repair

Status: `[~]` lower drift repaired; fixed-headway gap still open

The current main wins on average but loses narrowly in highnoise and odshift.
This must either be repaired or explained with statistically defensible
tradeoffs.

2026-06-03 update: 200ep diagnosis pointed to lower-drift/Lagrangian mismatch
rather than only decomposer smoothing. `main_driftcost` beat previous main in
all four domains at 20 seeds / 200 episodes and has been promoted into the main
aliases. Mechanism summary shows lower action dropping from roughly 8-9s to
4.3-4.7s and lower drift penalty dropping from roughly 0.33-0.39 to 0.13-0.15.

2026-06-03 late update: the 100ep promoted generalization matrix plus external
fixed-headway baseline exposed a different weakness: promoted main is robust
against internal ablations, but fixed-headway remains very competitive because
FreqDuet can pay extra composite cost through fleet overshoot/CV. A fleet
no-harm guard was tested without changing the promoted main alias:

```text
candidate           overall vs promoted main      overall vs fixed-headway
noharm full         -0.0450 CI [-0.0792,-0.0106]  -0.0142 CI [-0.0524,+0.0233]
upper-only          -0.0460 CI [-0.0775,-0.0123]  -0.0138 CI [-0.0528,+0.0228]
lower-only          -0.0456 CI [-0.0793,-0.0120]  -0.0156 CI [-0.0481,+0.0147]
adaptive gate       -0.0457 CI [-0.0790,-0.0131]  -0.0133 CI [-0.0404,+0.0156]
adaptive2 gate      -0.0311 CI [-0.0680,+0.0042]  -0.0002 CI [-0.0330,+0.0351]
```

Interpretation: lower-only no-harm is the safest current candidate because it
keeps the rushshift gap to fixed statistically ambiguous and avoids the
upper-only rush degradation. Adaptive gates are documented negative results:
they still activate upper projection in terminal/rush and do not beat
lower-only. The fixed-headway gap is not closed enough for a top-journal claim;
do not promote an adaptive gate without new evidence.

2026-06-03 fixed-gap follow-up: several 100/200ep attempts were run and should
be treated as negative evidence rather than promotion candidates.

100ep fleet-safe scalarization:

```text
fleetstable / fleetfloor30 changed w_fleet as intended but did not close the
fixed-headway gap. Overall vs fixed-headway stayed statistically ambiguous, and
terminal/rush still carried the gap.
```

100ep proactive lower no-harm:

```text
proactive1 overall vs fixed-headway  +0.0295 CI [-0.0043,+0.0633]
proactive2 overall vs fixed-headway  +0.0306 CI [+0.0016,+0.0606]
hfgate     overall vs fixed-headway  +0.0349 CI [+0.0058,+0.0614]
hfgate2    overall vs fixed-headway  +0.0550 CI [+0.0239,+0.0891]
```

Interpretation: proactive shrink can help highnoise overshoot, but hand gates
trigger in OD/terminal/rush trajectories and damage wait/composite.

200ep lower-only / stronger drift follow-up:

```text
loweronly    overall vs fixed-headway +0.0252 CI [-0.0109,+0.0641]
             terminal +0.0691 and rush +0.0782 were significant regressions
driftcost50  overall vs fixed-headway +0.0524 CI [+0.0164,+0.0918]
adaptdrift   overall vs fixed-headway +0.0528 CI [+0.0202,+0.0860]
```

Interpretation: lower-only no-harm does not survive 200ep as a main promotion.
Stronger/adaptive drift cost improves terminal/rush in isolation but destroys
OD/highnoise tradeoff, so it is not a universal repair.

Current decision: do not promote fleetstable, fleetfloor30, proactive1,
proactive2, hfgate, hfgate2, driftcost50, or adaptdrift. The fixed-headway gap
needs a mechanism change rather than another hand threshold: likely a
holding-value-per-fleet-second guard, or Phase-4 terminal/first-stop execution
that moves useful delay off the on-route fleet.

2026-06-03 value-guard candidate: implemented a causal lower holding guard that
clips an on-route holding action only when its estimated passenger/headway value
is below its fleet-pressure cost. The first local 2-seed / 10ep screen is
promising but not sufficient for promotion:

```text
soft valueguard vs main, composite delta:
terminal   -0.1723 CI [-0.4381,+0.0935]
highnoise  -0.2125 CI [-0.3920,-0.0330]
odshift    +0.2342 CI [+0.2329,+0.2355]
rushshift  +0.0519 CI [-0.3089,+0.4127]
overall    -0.0247 CI [-0.0455,-0.0039]
```

Interpretation: adding headway-correction value fixes the earlier OD/rush damage
from a pure passenger-HF guard, but OD overshoot remains unstable. A valueguard
+ fleet-floor combination was also screened locally and rejected because it
removed the highnoise benefit. Next step is a scheduler-visible 100ep/20seed
valueguard screen against main and fixed-headway before any alias promotion.

Done means:

- inspect highnoise and odshift seed traces;
- identify whether the gap comes from decomposer smoothing, promotion timing,
  raw-history fallback, lower credit scale, DriftFB, or upper replan cadence;
- test targeted fixes without degrading terminal/rush;
- if no universal fix exists, document the tradeoff and report domain-specific
  confidence intervals.

### 4. Paper-Grade Decomposer Evidence

Status: `[~]` fixed validation package generated; needs final paper selection

The harmonic causal decomposer and synthetic evaluator exist, but the paper still
needs direct evidence on real/sim demand traces.

2026-06-03 update: `scripts/make_freqduet_decomposer_figures.py` now writes a
fixed decomposer validation package to
`results_freqduet/decomposer_validation/current_trace`, including synthetic
LF/HF/burst truth, harmonic prior sensitivity, and trace LF/HF alignment CSVs
and figures.

Done means:

- plot `lambda_L` against daily peak structure;
- plot `lambda_H` against local burst events;
- show HF energy versus wait/holding spike alignment;
- audit that the decomposer uses only causal history and no future leakage;
- run cutoff/window sensitivity;
- compare harmonic-prior decomposition against raw history, EMA, and non-prior
  variants.

### 5. FreqDuet-Specific Mechanism Figures

Status: `[~]` fixed FreqDuet figure package generated

The existing figure scripts are not yet a complete FreqDuet mechanism package.

2026-06-03 update: `scripts/make_freqduet_mechanism_figures.py` writes
FreqDuet-specific mechanism CSVs and figures to
`results_freqduet/mechanism_figures/current_ep200` and
`results_freqduet/mechanism_figures/promoted_ep200`, including HF-to-holding,
lower drift, promotion active/inactive, method/domain bars, action/state
spectrum, and longtrain drift curves.

Done means:

- generate HF residual to holding lag plots;
- generate lower LF drift distributions;
- show promotion before/after wait, overshoot, and replan behavior;
- show why `allfreq` pushes lower actions too large or too noisy;
- show leakage and DriftFB effects;
- add action-spectrum / state-spectrum diagnostics;
- save all figures through a fixed script with stable output paths.

### 6. Phase 4 Terminal Dispatch Scope

Status: `[~]`

The current promoted path is closer to a target-headway / executable timetable
MVP. `dev_manual.md` also describes a stronger Phase 4 direction involving real
terminal dispatch, actual launch time, and first-stop holding.

Done means one of the following:

- implement and validate the stronger real terminal timetable path as the paper
  main line; or
- explicitly scope the paper method as a target-headway timetable policy and
  move full terminal dispatch to future work / appendix.

If targeting a top transport journal, implementing the stronger Phase 4 path is
preferred unless it destabilizes the method.

### 7. External Baselines

Status: `[~]` FreqDuet classical baseline matrix complete; external RL/TransitDuet baselines open

Current evidence is mostly FreqDuet internal ablations and TransitDuet-family
comparisons. Top-journal claims need stronger outside baselines.

2026-06-03 update: `scripts/run_freqduet_external_baselines.py` now runs
FreqDuet-format classical baselines using the same config/env perturbations and
paper composite output. It currently covers fixed-headway without holding,
rule-based holding, and a simple MPC/forecast triple controller. A 1-episode
smoke passed, but the full 20-seed external baseline matrix has not been run.
Preserved TransitDuet and SUMO-RL-style online RL baselines remain open.

2026-06-03 late update: the full 100ep / 20-seed external classical matrix was
run under `results_freqduet/external_baselines_promoted_ep100`. Fixed-headway is
the strongest classical baseline; rule-holding and the simple MPC/forecast
controller are much weaker. Promoted main and no-harm candidates beat fixed on
OD shift and are competitive overall, but they do not yet decisively dominate
fixed-headway across terminal/highnoise/rushshift. Added
`scripts/compare_freqduet_external_baseline.py` for paired candidate-vs-external
comparisons with bootstrap CIs.

Done means:

- include fixed-headway and rule-based holding baselines;
- include original TransitDuet or closest preserved version;
- include a tuned SUMO-RL-style online RL baseline if feasible;
- include an MPC / forecast-control / classical heuristic baseline if feasible;
- tune baselines enough that they are credible and not strawman comparisons.

### 8. Statistical Rigor

Status: `[~]` paired statistics tooling complete; final paper tables pending

Mean scores are not enough for top-journal claims, especially when domain-level
gaps are close.

2026-06-03 update: current longtrain summarization, candidate comparison, and
promoted-main table generation now write seed-level paired deltas, bootstrap
CIs, and win rates via `scripts/summarize_freqduet_paper_matrix.py`,
`scripts/compare_freqduet_candidate.py`, and
`scripts/build_freqduet_promoted_matrix.py`.

2026-06-03 late update: `scripts/compare_freqduet_external_baseline.py` adds the
same paired-delta/bootstrap-CI workflow for external baselines. Split no-harm,
adaptive, and adaptive2 screens all have seed-level CSVs and paired CI outputs
under `results_freqduet/noharm_*_screen_ep100_wu10`.

Done means:

- preserve seed-level outputs for every table;
- run paired tests against the same seeds;
- report confidence intervals or bootstrap intervals;
- report effect sizes, not only p-values;
- flag negative or ambiguous domains instead of hiding them in averages.

### 9. Reproducibility And Config Hygiene

Status: `[~]` paper manifest started; aliases locked, historical cleanup pending

There are many historical configs, failed ablations, and renamed aliases. This
is acceptable during research but risky for paper reproduction.

2026-06-03 update: `FreqDuet/freqduet/paper_manifest.yaml` now maps current
longtrain, driftcost candidate, held-out generalization, decomposer validation,
and mechanism figures to configs, seeds, scripts, logs, and result paths. The
promoted main aliases now include drift-cost. The remaining work is to move
historical failed configs into a clearly marked layer and keep final table/figure
manifests current as new generalization and baseline results land.

Done means:

- create a canonical paper config layer;
- separate `paper_main`, `paper_ablation`, `paper_generalization`, and
  `historical_failed` configs;
- add a result manifest mapping each table/figure to configs, seeds, episodes,
  and scripts;
- provide one-command scripts for final matrix, long training, generalization,
  mechanism figures, and statistical tables;
- avoid ambiguity between old main aliases and current promotion main.

### 10. Theoretical / Method Framing

Status: `[ ]`

The method framing is strong, but a top-journal paper needs a tighter
assumption-and-mechanism explanation.

Done means:

- formalize frequency-separated HRL as exogenous-state decomposition, not just
  feature engineering;
- state the causal filtering assumption and no-future-leakage property;
- explain why LF demand belongs in upper timetable planning and HF residual
  belongs in lower holding;
- define leakage and promotion as cross-frequency correction mechanisms;
- state limits: non-stationary OD shifts, extreme noise, and promotion false
  positives.

### 11. Realism / Data Evidence

Status: `[ ]`

For a top transport / ITS journal, simulation-only evidence may be insufficient
unless the simulator is strongly calibrated.

Done means:

- document demand calibration source and route assumptions;
- add real or semi-real service-day demand profiles if available;
- hold out route/day/profile families;
- report robustness under fleet-size, dwell-time, and passenger-arrival
  stochasticity changes.

### 12. Paper Tables, Figures, And Negative Results

Status: `[ ]`

The code has many raw results, but the final paper package is not assembled.

Done means:

- produce final main table;
- produce ablation table;
- produce generalization table;
- produce long-training table;
- produce mechanism figures;
- produce decomposer validation figures;
- include a concise negative-results appendix explaining failed variants.

## Recommended Execution Order

1. Lock the current paper-main config and clean aliases.
2. Run current-version 100/200ep long training across the key baselines.
3. Run systematic held-out generalization.
4. Repair or explain highnoise and odshift weaknesses.
5. Build decomposer and mechanism figure scripts.
6. Add paired statistics / bootstrap reporting.
7. Decide and document the Phase 4 terminal-dispatch scope.
8. Add external baselines.
9. Organize paper configs, manifests, and one-command reproduction scripts.
10. Assemble paper tables, figures, and negative-results appendix.

## Working Rule For Future Development

Do not mark a gap closed just because code exists. Mark it closed only when:

- the implementation is in the promoted FreqDuet copy, not only the old
  TransitDuet tree;
- there is a reproducible config or script;
- there are seed-level results;
- the result is either effective or the failure is documented;
- the conclusion is consistent with `dev_manual.md` and `GPT.md`;
- the current branch has been pushed after a good result.
