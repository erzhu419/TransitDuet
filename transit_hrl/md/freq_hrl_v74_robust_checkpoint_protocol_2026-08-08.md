# Freq-HRL v7.4 Robust-Checkpoint Protocol

Date: 2026-08-08

## Motivation

The source-bound v7.3.2 budget ladder completed all 135 registered cells at
64, 96, and 128 iterations. Its terminal decision was
`blocked_budget_limited_at_128`; no final HPO or held-out confirmatory run is
permitted from that decision.

At 128 iterations, six of nine representative method/candidate rows still
selected checkpoints inside the registered final-eighth boundary too often.
The full-method promotion mechanism was selective on only two of five
optimizer replicates for the balanced candidates and on zero of five for the
forecast candidate. These outcomes are treated as protocol-diagnostic
failures, not as positive performance evidence.

Two root causes were identified before defining v7.4:

1. A single validation observation could select a checkpoint, so a late noisy
   spike looked like continued learning.
2. Low-noise promotion predictions were pooled across time and paths. This
   controlled the pooled empirical rate but did not control every independent
   calibration path.

## Frozen v7.4 Changes

All shared PPO, SAC, and TD3 trainers now use the same checkpoint selector in
the v7.4 protocol:

- trailing validation-score mean over eight observations;
- minimum material improvement of `5e-4` over the currently selected score;
- complete fixed-budget training, with no outcome-dependent early stopping;
- full per-iteration history persisted as `training_history.json`;
- selected raw score, smoothed score, last material improvement, and plateau
  tail persisted in every cell summary.

Promotion null calibration now computes a separate empirical threshold on
each independent stationary-low-noise path and uses the maximum pathwise
threshold. Twelve registered support-only calibration seeds replace the three
paths used by v7.3.2. The artifact records every path's sample count, threshold,
and pre/post action rate.

## Source-Bound Budget Decision

The plan is defined by
`full_method_budget_plan_v74.py` and hashed before any v7.4 result is observed.

- budgets: 192 iterations, then 256 only if 192 fails;
- fresh optimizer seeds: `7207, 7211, 7213, 7219, 7229`;
- representative matrix: nine method/candidate rows, 45 cells per budget;
- learning gate: at least four of five replicates select a materially improved
  checkpoint and mean smoothed validation gain is positive;
- stability gate: at least four of five replicates have no material checkpoint
  improvement during the final 32 iterations;
- every representative row must pass before the budget can be selected.

If 192 fails, 256 is mandatory. If 256 also fails, final HPO and confirmatory
evaluation remain blocked. Increasing the budget again requires a new
source-bound protocol and fresh optimizer seeds.

## Compute and Evidence Boundary

All v7.4 jobs are restricted to the dynamic `node001` through `node006` pool,
with one CPU core per independent cell and no hard node binding. These nodes
provide 1,152 schedulable cores in total.

The old `jtl110cpu` records are excluded. They have remained labelled running
after SSH key-exchange failure, have no confirmed remote process-tree state,
and retain stale scheduler telemetry. A scheduler label or cached utilization
field is not accepted as evidence of 110+ hours of valid computation.

Passing this budget protocol establishes only an adequate and auditable
training budget. It does not itself establish superiority, promotion benefit,
or a domain-general claim; those require the separately registered final HPO
and held-out confirmatory analyses.
