# MuJoCo v14.22 action-cost critic preflight

## Decision context

v14.21 rejected distributional local search over frozen actor output heads.
Hopper and Walker transferred small improvements, but HalfCheetah reversed from
design to validation despite 64 paths per role. The next mechanism must model
how current actions alter future constraint cost under the current closed-loop
occupancy, rather than infer a direction from complete-path ranking alone.

## Frozen mechanism

- Twelve fresh roots crossed with four disturbance modes provide 48 stochastic
  current-policy paths for critic fitting. Four separate roots provide 16
  untouched critic holdout paths.
- Each policy level uses an ensemble of four bootstrap MLP critics estimating
  duration-discounted native constraint return from cost-state and deployed
  action. Discounting and action coordinates come from the frozen checkpoint,
  not from incomplete result summaries.
- A level passes only when ensemble holdout R2 is positive, true holdout actions
  predict better than one fixed permutation of those actions, and the median
  pairwise cosine between ensemble actor gradients is positive.
- Passing gradients are normalized separately for upper and lower actors before
  joint descent. Five float32-effective parameter RMS steps from `1e-8` through
  `1e-6` are evaluated exactly on 16 fresh design roots crossed with all four
  disturbance modes.
- A design candidate is eligible only with zero reward violations, at least
  `1e-4` relative frequency-merit reduction, and no more than three times the
  baseline worst violation. The unchanged rule is reapplied on 16 disjoint
  validation roots crossed with all four modes.

## Execution contract

The three cells are HalfCheetah, Hopper, and Walker at optimizer seed
`4196455150`. Each requests 24 CPU cores and 16 GB RAM. Complete environment
paths are owned by 24 spawned workers. Scheduler placement is dynamic across
`node001-node006`; no task is node-bound and no Slurm path is used.

## Gate and evidence boundary

The mechanism advances into the shared actor-critic only if all three
environments pass independent validation. This is an adaptive mechanism
preflight after v14.21 and is not confirmatory paper evidence. Failure rejects
the current action-cost critic update; roots, thresholds, step registry, and
risk aggregation will not be retuned against the same outcomes.
