# MuJoCo v18.2 State-Conditioned Actor Design

## Motivation

V17.14 evaluated all 900 frozen linear FIR adapters and left one Hopper
`ood_chirp` path unresolved. V18.2 tests the specific missing capacity: a
nonlinear residual actor conditioned on the causal lower-policy state, while
retaining the same action trust region and grouped reused-path gate.

## Frozen Core

The actor implementation is frozen at Git revision
`6ebf63c77c5c8ecf2e0784b7361eb90a6d71caf9` with Freq-HRL source manifest
`d5842e0972d59182be55881b52de9b4ac074b5b355aa1b4c929e4b7f0a849099`.
The model receives only the current lower-policy state and current/past upper
and lower proposals. Actor-floor labels and nearest-feasible targets affect the
training loss but are not model inputs.

## Candidate Grid

The complete grid contains 16 two-hidden-layer `tanh` residual MLPs:

- proposal history: 1 or 8 steps;
- hidden width: 32 or 64;
- actor-floor path weight: 64 or 256;
- correction limit: 0.010 or 0.025.

Learning rate is `3e-3`, weight decay is `1e-5`, and every fold runs 120 fixed
epochs from a deterministic seed derived before target access. Loss weights are
normalized by path length so a long zero-target trajectory does not dominate a
short actor-floor trajectory. Environments with no actor-floor target use an
exact zero residual model rather than fitting noise.

## Evaluation

All eight reused seeds define leave-one-seed-out folds. The five disturbance
modes of the held seed are excluded together. Every candidate receives exact
full-horizon responsibility oracles on its out-of-fold corrected totals; there
is no target-fidelity prefilter. Selection prioritizes joint feasibility,
actor-floor recovery, and reference-feasible preservation before fit error.

## Advancement Gate

Fresh closed-loop validation is authorized only if one candidate has 120/120
valid paths, preserves all 113 reference-feasible paths, recovers all seven
actor-floor paths and both floor seed groups, has target normalized MSE at most
0.75, has maximum reference correction RMS at most 0.01, and changes the
post-clipping action on all seven floor paths. Otherwise this panel stops
without fresh access.

## Claim Boundary

This is grouped reused-path actor-target distillation. It cannot establish
reward improvement, closed-loop learning, fresh-seed generalization, leakage
no-tradeoff, or manuscript performance support.
