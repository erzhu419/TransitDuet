# FreqDuet Protocol V5 Journey-Feasible Contract

Date locked: 2026-08-08

## Purpose

Protocol V5 is a structural rebuild after Protocol V4 failed its locked
selection. Its paper question is narrower and identifiable:

> Does causal LF/HF demand-state allocation improve fixed-pool passenger
> journey time over matched no-frequency and raw-history controls, without
> degrading service reliability or relying on non-executable dispatch/holding?

The primary endpoint is
`restricted_total_journey_horizon_min`. The fixed-horizon restriction includes
waiting and in-vehicle censoring for every generated passenger, so withholding
service cannot improve the endpoint by removing passengers from the observed
sample.

## Executable method contract

1. The upper reward owns queue passenger-seconds, onboard passenger-seconds,
   and overdue dispatch-backlog trip-seconds. Episode-global wait/CV reward and
   centered hindsight gap credit are disabled.
2. Upper timetable actions use one antisymmetric linear amplitude. Future
   directional headway deltas are projected onto their box bounds with exactly
   zero total delta. The controller can redistribute departures over the
   rolling horizon but cannot create extra departures by globally compressing
   the timetable.
3. Training and evaluation use a physical fixed pool of 12 vehicles. Fleet-size
   robustness is a held-out generalization experiment, not training-time domain
   randomization in the primary comparison.
4. A lower action is permitted only at a deployable arrival event and is capped
   by `max(target headway - observed forward headway, 0)`. The cap uses no
   follower state or future trajectory. The capped action is rounded downward
   to the largest feasible holding bin.
5. Holding cost is weighted by frozen APC-observed onboard load. Promotion,
   DriftFB, frequency reward attribution, and leakage penalties are disabled in
   the V5 main; none may return without a separate single-axis result.
6. Harmonic features use historical priors and causal APC boardings. The upper
   receives LF/forecast features and the lower receives HF residual features.
   The raw-history control is dimension-matched: six upper bins and four lower
   bins.

## Locked development screen

- Configurations: main, no-frequency, raw-history, all-frequency, upper-only,
  lower-only, no-budget, no-guard, no-load-cost, wait-only-credit, and standard
  constrained SAC.
- Training episodes: 80.
- Training seeds: 503, 521, 541, 557, 571, 587, 601, 617.
- Frozen evaluation seeds: 41011, 41017, 41023, 41039.
- Grid: 88 trained policies and 352 frozen rollouts.
- Common random numbers are shared only through evaluation scenario tapes;
  policy initialization, replay, exploration, and environment streams remain
  isolated by contract.

The screen is developmental. These seeds may be reused to diagnose and revise
V5, but they cannot support the final paper claim.

## Locked screen decision

For each control, deltas are `candidate - V5 main`; lower journey is better.

- Frequency support requires both no-frequency and raw-history controls to be
  at least 0.25 min worse in mean restricted journey, with the 95% CI lower
  bound above zero, while V5 main passes all no-harm gates against each control.
- Layer-allocation support additionally requires all-frequency, upper-only, and
  lower-only controls to be at least 0.10 min worse in mean restricted journey,
  with the CI lower bound above zero and the same no-harm gates. Frequency
  support without this second condition cannot support an LF/HF division claim.
- If either frequency control has a journey CI upper bound below zero, the V5
  structure is rejected and must be redesigned; the same rule applies when an
  allocation control is superior.
- Otherwise the result is inconclusive and cannot be promoted.
- A physical/mechanism ablation has performance support only when it is at
  least 0.10 min worse in mean journey and its CI lower bound is above zero.
- Standard constrained SAC may replace the ensemble optimizer only if its
  journey CI upper bound is at most +0.25 min and every no-harm gate passes.

No-harm margins are fixed at +0.50 min restricted wait, +0.005 unserved rate,
+0.02 headway CV, +0.005 denied-trip rate, +15 s mean readiness delay, +0.10
holding passenger-min per generated passenger, and -0.005 for launch and
completion rates.

The main implementation also must show: causal guard enabled in every rollout,
mean absolute projected headway-delta sum at most `1e-5` s, finite journey
outcomes, and nonzero onboard-credit exposure.

## Untouched confirmation reserve

The following seeds are reserved now and must not be inspected during V5
development:

- Training: 701, 719, 733, 751, 769, 787, 809, 827, 853, 877, 907, 929, 953,
  977, 997, 1013, 1031, 1051, 1069, 1091.
- Evaluation: 51001, 51007, 51019, 51031, 51043, 51059, 51071, 51089.

After one V5 candidate is frozen, confirmation uses 200 training episodes and
the complete crossed evaluation. Strong fixed-headway, tuned rule holding,
rolling MPC, and the closest locked TransitDuet-family baseline must use the
same scenario tapes and physical fixed-pool contract.

The paper may claim efficacy only if the untouched confirmation passes. A good
development screen is not itself publishable evidence.
