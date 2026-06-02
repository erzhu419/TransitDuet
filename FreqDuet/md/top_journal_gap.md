# FreqDuet Top-Journal Gap Backlog

Last updated: 2026-06-02

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

## Top-Journal Gap Summary

Status legend:

- `[x]` completed enough for the current prototype;
- `[~]` partially done, needs paper-grade completion;
- `[ ]` not yet complete.

### 1. Current-Version Long Training

Status: `[~]` running

The strongest current evidence is still 40 episodes. Older 100/200ep results do
not fully represent the current soft-promotion main line.

2026-06-02 start: added the current paper-longtrain runner and paired-delta
summary tooling, then submitted the current 200ep matrix as six scheduler/Slurm
shards:

```text
tasks: t5870, t5872, t5874, t5875, t5876, t5877
slurm: 18263, 18264, 18265, 18266, 18267, 18268
matrix: 4 domains x 6 methods x 20 paired seeds = 480 runs
episodes: 200
last_k: 100
state at submission: Slurm PENDING, reason QOSMaxCpuPerUserLimit
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

Status: `[~]`

Highnoise, odshift, and rushshift are present, but this is not yet a complete
held-out generalization package.

Done means:

- evaluate multiple demand-noise levels instead of one highnoise setting;
- evaluate multiple OD profile shifts;
- evaluate multiple rush-pattern shifts;
- include service stochasticity / route perturbation if feasible;
- separate in-distribution seeds from held-out scenario families;
- show whether frequency separation is robust or only tuned to one service day.

### 3. Per-Domain Weakness Repair

Status: `[~]`

The current main wins on average but loses narrowly in highnoise and odshift.
This must either be repaired or explained with statistically defensible
tradeoffs.

Done means:

- inspect highnoise and odshift seed traces;
- identify whether the gap comes from decomposer smoothing, promotion timing,
  raw-history fallback, lower credit scale, DriftFB, or upper replan cadence;
- test targeted fixes without degrading terminal/rush;
- if no universal fix exists, document the tradeoff and report domain-specific
  confidence intervals.

### 4. Paper-Grade Decomposer Evidence

Status: `[~]`

The harmonic causal decomposer and synthetic evaluator exist, but the paper still
needs direct evidence on real/sim demand traces.

Done means:

- plot `lambda_L` against daily peak structure;
- plot `lambda_H` against local burst events;
- show HF energy versus wait/holding spike alignment;
- audit that the decomposer uses only causal history and no future leakage;
- run cutoff/window sensitivity;
- compare harmonic-prior decomposition against raw history, EMA, and non-prior
  variants.

### 5. FreqDuet-Specific Mechanism Figures

Status: `[ ]`

The existing figure scripts are not yet a complete FreqDuet mechanism package.

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

Status: `[ ]`

Current evidence is mostly FreqDuet internal ablations and TransitDuet-family
comparisons. Top-journal claims need stronger outside baselines.

Done means:

- include fixed-headway and rule-based holding baselines;
- include original TransitDuet or closest preserved version;
- include a tuned SUMO-RL-style online RL baseline if feasible;
- include an MPC / forecast-control / classical heuristic baseline if feasible;
- tune baselines enough that they are credible and not strawman comparisons.

### 8. Statistical Rigor

Status: `[ ]`

Mean scores are not enough for top-journal claims, especially when domain-level
gaps are close.

Done means:

- preserve seed-level outputs for every table;
- run paired tests against the same seeds;
- report confidence intervals or bootstrap intervals;
- report effect sizes, not only p-values;
- flag negative or ambiguous domains instead of hiding them in averages.

### 9. Reproducibility And Config Hygiene

Status: `[ ]`

There are many historical configs, failed ablations, and renamed aliases. This
is acceptable during research but risky for paper reproduction.

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
