# MuJoCo v14.21 distributional actor restoration preflight

## Decision context

v14.20 showed that a closed-loop zeroth-order direction could reduce the frozen
frequency merit on Hopper and Walker, but not on HalfCheetah. Its two design and
two validation roots also left the rank direction exposed to single-trajectory
chaos. Router-only screens v14.18-v14.19 had already failed on Walker, so this
preflight changes the objective estimator rather than tuning either mechanism.

## Frozen mechanism

- The editable parameters remain the joint upper/lower deterministic actor
  output heads. Critics, encoders, routers, and all other parameters are frozen.
- Eight fixed antithetic Rademacher directions estimate a rank gradient.
- Sixteen frozen design roots are crossed with all four disturbance modes,
  producing 64 complete 1000-step paths per candidate.
- Sixteen separate frozen validation roots produce another 64 paths. Design and
  validation roots are unique, disjoint, and fresh relative to v14.20.
- Constraint risk is the mean within each disturbance mode. The four resulting
  mode constraints retain disturbance identity while reducing trajectory-level
  variance.
- A design candidate is eligible only with zero reward violations, at least
  `1e-4` relative frequency-merit reduction, and no more than three times the
  baseline worst frequency violation. The same rule is reapplied unchanged on
  validation paths.

## Execution contract

There are three scheduler cells: HalfCheetah, Hopper, and Walker for optimizer
seed `4196455150`. Each cell requests 16 CPU cores and 8 GB RAM. Candidate
vectors are evaluated by 16 spawned processes; each process reconstructs the
same frozen checkpoint and owns complete environment trajectories. Tasks are
dynamically eligible for `node001-node006` with no required node and no Slurm.

## Gate and evidence boundary

The mechanism advances only if all three environments pass the independent
validation rule. This is an adaptive development preflight after seeing v14.20,
not confirmatory evidence and not a paper claim by itself. Failure rejects this
distributional output-head restoration mechanism rather than triggering another
search over roots, risk aggregation, or step sizes on the same results.
