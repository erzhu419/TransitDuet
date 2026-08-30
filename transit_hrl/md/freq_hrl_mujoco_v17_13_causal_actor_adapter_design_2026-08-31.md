# MuJoCo v17.13 Causal Actor Adapter Design

## Motivation

V17.11 closed fixed-total-action router development. V17.12 then showed that
the seven remaining Hopper actor-floor paths have nearby frequency-feasible
targets requiring at most `0.008118` total-action correction RMS. V17.13 tests
whether those acausal targets can be distilled into a deployable causal actor
residual without damaging the 113 paths already feasible.

## Learner

The adapter consumes only current and past upper/lower actor proposals. It is a
multivariate causal FIR residual fitted with normalized weighted ridge. Sample
weights are normalized within each trajectory so that trajectory length does
not determine its influence; actor-floor paths receive a frozen candidate
weight. The output changes the pre-saturation total action and is clipped by a
candidate trust region and the component-sum box.

All hyperparameters are shared across environments. Coefficients are fitted per
environment because action dimensions differ. Selection uses eight
leave-one-seed-out folds, with all five disturbance modes of the held seed kept
outside fitting.

## Frozen Screen

The grid contains 900 combinations of FIR window, ridge penalty, actor-floor
path weight, output gain, and correction limit. A cheap target-fidelity screen
forms the union of three deterministic top-16 rankings. Only this frozen union
receives full-horizon responsibility-oracle evaluation. Per-path sufficient
statistics are cached, and the exact convex oracles run in a bounded process
pool with one numerical thread per worker.

Advancement requires all of the following:

- 120/120 valid corrected paths;
- 113/113 reference-feasible paths remain feasible;
- all seven actor-floor paths and both actor-floor seed groups become feasible;
- actor-floor target normalized MSE no greater than `0.75`;
- reference-feasible correction RMS at most `0.01` on every path;
- every actor-floor target and learned correction remains nonzero after
  environment action clipping.

No fresh path is read by this screen. Passing only authorizes a separately
frozen closed-loop fresh validation.

## Claim Boundary

This is grouped reused-path causal-adapter development. It is not evidence of
reward improvement, online policy learning, fresh-seed generalization,
leakage no-tradeoff, or a final manuscript result.
