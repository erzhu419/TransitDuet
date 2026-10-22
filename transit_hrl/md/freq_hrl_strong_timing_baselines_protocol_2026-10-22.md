# Strong Timing Baselines: Prepared, Not Run

Date: 2026-10-22. Entry point: `scripts/run_pointmaze_timing_baselines.py`.
No scheduler submission, native performance run or new scientific result.
Local preparation checks: 32 tests passed, including mocked native replay,
fit/test separation and existing timing/paper-evidence regressions.

## Scope

Freq-HRL is a hierarchy for time-series control with useful slow plans, fast
closed-loop execution and observable evidence that a plan may need revision.
PPO/SAC are possible learning backends, not algorithms it must universally
replace. The relevant question is added utility **inside this applicability
class**, including cases where timescale separation or plan persistence fails.
This suite isolates timing under a shared frozen controller; it does not test
every frequency-routing component or rank complete jointly trained algorithms.

## Comparators

- Fixed start-of-bin timing, the original seeded random-jitter schedule, and the
  best constant offset among 0/5/10/15/20/25.
- Error, age, target-velocity-change and force thresholds: cutpoints are fit-path
  quantiles plus always/never-cross endpoints; selection uses separate full
  episodes. The strongest selected threshold competes with the candidate.
- Original Stage9 current/history fitting recipes, refitted on the new fitting
  bank without changing the archived Stage9 models, results or gates.
- Capacity controls: current-only versus history, each using the existing
  17-dimensional feature view and all 170 quadratic terms plus intercept.
  Ridge penalties are solved separately for the same fixed slope effective
  DF=8, including each fit-path OOF fold. Each threshold is the same q75 of
  OOF predictions. Neither zero-padding nor equal parameter count alone is
  treated as equal fitted complexity.
- Learned termination: current/history heads each use 17 view features plus
  within-bin offset and remaining episode fraction, and the same 19-64-64-1
  Bernoulli architecture (5,505 actor parameters). Both heads receive the exact
  same pooled paired-label bank, optimizer recipe and update budget.

## Learned Termination

Stage10's stochastic policy was active but not consistently useful; MC credit
alone failed. Stage11/12's fixed timing-pair recipes and Stage25's cached linear
cost-sensitive decisions also failed their development gates. This preparation
does not relabel those outcomes or present cost weighting as an untried cure.

Start from a fixed offset-10 policy. For each fit path, uniformly select 12 bins
excluding the first/last, then an actually visited nondeadline check within each
bin. Replay the identical prefix twice: stop now, or skip just this check.
Afterward both arms resume the current policy on their own observations, with
exactly one call per bin. Existing Stage14 adaptive replay supplies this path.
Check identical prefixes/features and reproduction of the factual schedule/ISE.

The label is full-episode `ISE(wait one check) - ISE(stop now)`. Fit deterministic
termination with `abs(label)`-weighted binary cross entropy on `label > 0`.
This uses conditional expected action advantage rather than majority label
sign. No future value or event label is supplied to the actor. Two acquisition/
fit rounds pool the two heads' behavior equally; identical initial behavior is
replayed once. Latest fitted heads, not selection-winning checkpoints, generate
the next round's states. Training and inference use the same deterministic rule.

Select the lowest-ISE checkpoint on timing-selection paths, **including a
collapsed checkpoint if it is best**. Separately report multiple chosen offsets,
stop/wait variation across paths at a common bin and offset, and selection ISE
improvement over the best constant offset. All three are required for development eligibility; no
early-call quota, root deletion or test-time threshold repair is permitted.

## Splits And Next Decision

Use five disjoint roles: controller training, controller selection, timing fit,
timing selection and final test. Reserve new roots/paths before performance work;
revealed Stage9/10 test paths are not fresh confirmation data. Controller
diagnostics receive timing-selection paths only. All rules/checkpoints are
frozen before test access; test outcomes never determine eligibility or refits.

The command defaults to protocol preparation. For example, from `transit_hrl/`:

```bash
PYTHONPATH=. python3 scripts/run_pointmaze_timing_baselines.py \
  --optimizer-seed 1 --train-seeds 11 12 13 \
  --selection-seeds 21 22 --branch-fit-seeds 31 32 33 \
  --timing-selection-seeds 41 42 --trigger-eval-seeds 51 52 \
  --output /tmp/freq_hrl_timing_plan.json
```

These are illustrative IDs, not a reserved experiment roster. The explicit
`--execute` command runs development locally when computation is authorized;
this entry point never submits jobs. First use is a short native preflight,
then independent-root development. If termination remains collapsed or cannot
improve on the clock, retain that failure and stop confirmation of this recipe.
If eligible, freeze a separate new-root confirmation comparing root-level paired
ISE/return, with fixed, strongest threshold and learned baselines. No confirmation
roster or performance decision is created by the current preparation.
Execution writes branch/pair labels to a separate `.fit-cache.json` on the
execution machine for later lightweight refits. Retrieve the compact result/log
first, not this cache, trajectories or a controller checkpoint.

## Limitations

Eligibility is development evidence, not a CI-supported performance claim.
The paired target still contains long-horizon noise; nonlinear fitting and
on-policy coverage are hypotheses, not established repairs. These are custom
learned stopping heads, not faithful TEMPLE/Option-Critic reproductions.
Current/history controls have equal input counts but different information;
the ridge panel matches effective DF, while the neural panel matches parameter
count, not effective DF. Original Stage9 branch acquisition uses simulator event
metadata; paired timing acquisition instead samples visited bins without it.
The termination pair bank is additional supervision, shared between its two
heads, not the same bank as the ridge controls. Replay/selection/test steps are
recorded separately; full controller-selection, optimizer and latency accounting,
own-schedule controller training and a faithful named timing baseline remain
subsequent work. No total-compute or complete-algorithm fairness claim is closed.
