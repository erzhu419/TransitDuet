# MuJoCo v18.3 Causal Joint Projection Design

## Motivation

V17.14 recovered 6/7 actor-floor paths with a linear causal FIR correction,
while v18.2 recovered only 3/7 with a grouped state MLP. The v18.2 seed reversal
shows that seven acausal target trajectories do not provide stable supervised
support. V18.3 therefore removes target distillation entirely.

## Mechanism

At primitive step `t`, the projector receives the current proposed upper and
lower components and its own projected component history. It computes the exact
current HPF8 upper residual and LPF32 lower residual as affine functions of the
current component pair.

The action rule is lexicographic:

1. project responsibility while preserving the proposed total action;
2. if no causal frequency-feasible split exists for that total, project the
   current upper and lower components onto their nearest feasible sets;
3. only the second branch changes the total action.

No observation, reward, actor-floor label, v17.12 target, future action, or
termination time is available to the projector. The core is domain-agnostic and
depends only on causal linear filters, budgets, and component bounds.

## Frozen Candidates

`joint_projection_instantaneous` bounds every current residual energy. This is
stronger than the endpoint mean-square contract and prevents an early prefix
from exhausting future budget.

`joint_projection_prefix_ledger` permits unused prefix energy to carry forward
and enforces the exact cumulative mean-square budget. It is less conservative
but can enter a future viability dead end, so numerical/path feasibility is a
hard selection criterion.

These are two budget semantics, not fitted hyperparameters. Both use HPF8 RMS
0.075, LPF32 RMS 0.0475, and unit component bounds.

## Reused-Path Gate

The screen reads only the 120 server-only v17.8 baseline component traces. A
candidate must satisfy all of the following before a separate fresh experiment
can be frozen:

- 120/120 numerically valid paths;
- 120/120 directly audited and exact-oracle joint-feasible paths;
- 113/113 reference-feasible paths preserved;
- all seven actor-floor paths and both seed groups recovered;
- a nonzero executed-action correction on every actor-floor path;
- total correction absolute maximum no greater than 0.05;
- reference-feasible correction RMS maximum no greater than 0.01;
- actor-floor correction RMS maximum no greater than 0.015.

Passing this gate authorizes fresh closed-loop validation only. The reused-path
screen cannot support reward, generalization, learned-policy, no-tradeoff, or
manuscript claims.
