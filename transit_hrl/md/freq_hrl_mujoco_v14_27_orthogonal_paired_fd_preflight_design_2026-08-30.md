# MuJoCo v14.27 orthogonal paired finite-difference preflight

## Decision context

v14.26 produced positive aggregate train-versus-holdout direction cosine for
both policy levels in all three environments, and its Walker update reduced
untouched-validation frequency merit by `5.48%` with no reward violation. The
all-environment gate failed because three within-mode direction estimates were
negative. Random SPSA leaves cross-coordinate terms in each path estimate, so
eight paths per mode were insufficient to distinguish estimator variance from
real mode disagreement.

## Frozen intervention design

The random Rademacher registry is replaced by the rows of a Sylvester Hadamard
matrix of order eight. Every raw actor output-bias coordinate remains at
absolute amplitude `0.25`, preserving the v14.24-v14.26 intervention RMS. The
first actor-output-dimension columns form the upper and lower designs. Their
columns are balanced and orthogonal for HalfCheetah, Hopper, and Walker.

Each critic role has 16 fresh roots. Within every disturbance mode, Hadamard
rows zero through seven each occur twice on independent environment paths.
Control, upper plus/minus, and lower plus/minus remain isolated, giving 320
intervention paths in train and 320 in holdout per environment.

For each level and path, the antithetic cost difference estimates one
directional derivative. A full-rank least-squares solve recovers the actor-bias
gradient from the direction matrix. The pooled solve defines the update;
separate mode solves diagnose generalization. Both train and holdout designs
must have full rank. Positive overall and every-mode train-versus-holdout cosine
remain mandatory, so v14.27 removes estimator variance rather than weakening
the v14.26 gate.

The four action-cost critic ensemble remains an independent data diagnostic:
both levels require positive holdout R2 and positive fixed action-permutation
MSE gain. Upper and lower direction blocks are normalized to equal RMS. Exact
design steps remain `1e-7`, `1e-6`, `1e-5`, `3e-5`, and `1e-4`, with the same
reward floor, minimum merit reduction, and worst-violation funnel.

## Independent roles

The 16 critic-train roots, 16 critic-holdout roots, 16 design roots, and 16
validation roots are mutually disjoint and absent from v14.20-v14.26. Design
selects at most one step. Validation is untouched unless a design candidate is
eligible.

## Execution contract

The three cells are HalfCheetah, Hopper, and Walker at optimizer seed
`4196455150`. Each requests 24 CPU cores and 16 GB RAM. Scheduler placement is
dynamic across `node001-node006`, with `require_node=None`; no Slurm or
login-node execution is permitted.

## Evidence boundary

This is adaptive development after v14.26, not confirmatory evidence. All three
independent validation cells must pass before the orthogonal estimator can enter
the shared actor-critic. Roots, matrix order, replicate count, steps, thresholds,
and eligibility rules are frozen before outcome access.
