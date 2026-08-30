# MuJoCo v17.14 Exhaustive Actor Oracle Outcome

## Decision

Status: `exhaustive_actor_oracle_closes_frozen_linear_fir_grid`.

V17.14 evaluated the 852 frozen v17.13 candidates that had not received an
exact responsibility oracle and merged them with the prior 48. All 900 members
of the pre-registered linear causal FIR grid therefore have exact reused-path
outcomes. No candidate passed the complete advancement gate, so this grid is
closed and no fresh validation path was accessed.

## Frontier

The selected W8, ridge `1e-4`, actor-floor weight 256, gain 1.5, cap 0.01
adapter kept all 120 paths valid, preserved all 113 reference-feasible paths,
and recovered 6/7 actor-floor paths. It recovered 2/2 paths for seed 2802248628
and 4/5 for seed 294864529. All seven floor paths received a nonzero
post-clipping action change. Target normalized MSE was 0.6398, and the maximum
reference-feasible correction RMS was 0.00324.

Every one of the 900 candidates preserved 113/113 reference-feasible paths.
The maximum recovery rose with output gain from 2/7 at gain 0.5 to 3/7 at gain
1.0 and 6/7 at gains 1.5 and 2.0. Across the full grid, 497 candidates recovered
two floor paths, 207 recovered three, 71 recovered four, 48 recovered five, and
77 recovered six. None recovered all seven.

## Remaining Failure

The unresolved path is Hopper-v5, `ood_chirp`, seed 294864529. Its corrected
lower power is 0.002517 versus the frozen 0.00225625 budget, while upper power
remains feasible. This localized failure, after exhausting the linear FIR grid,
supports moving to a causal actor that conditions on observation and regime
state rather than performing further FIR hyperparameter search.

## Efficiency

Scheduleurm task `t85847` ran on node003 with 32 bounded oracle workers. Worker
runtime was 335.0 seconds, scheduler runtime was 360.5 seconds, and peak RAM was
3063 MB. Source paths and actor targets remained server-only; only compact JSON
results were synchronized.

## Claim Boundary

This result closes only the frozen 900-member linear causal FIR development
grid on reused paths. It is not a universal impossibility result and does not
establish reward improvement, online policy learning, fresh-seed
generalization, leakage no-tradeoff, or manuscript support.
