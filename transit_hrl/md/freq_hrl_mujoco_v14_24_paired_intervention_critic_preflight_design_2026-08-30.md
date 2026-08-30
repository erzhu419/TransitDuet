# MuJoCo v14.24 paired-intervention critic preflight

## Decision context

v14.23 improved short-horizon predictive metrics but supported only Hopper.
HalfCheetah moved from a lower-critic failure to an upper-critic failure, and
Walker reversed from design improvement to validation degradation. The shared
weakness is that observational policy trajectories do not identify a stable
local action derivative.

## Frozen mechanism

For each critic root and disturbance mode, collection now produces five paired
deterministic trajectories: an unintervened control, antithetic upper-only
output biases, and antithetic lower-only output biases. Each raw actor-mean bias
has RMS `0.25`; the other policy level remains byte-identical. Environment and
disturbance seeds are shared within each five-way group. Interventions are used
only for critic identification and never for actor design or validation.

Eight train roots yield 160 intervention trajectories and four untouched
holdout roots yield 80. The v14.23 upper-eight/lower-32 cost horizons, critic
architecture, bootstrap ensemble, predictive and action-permutation gates,
gradient agreement gate, actor step registry, reward guard, and exact
closed-loop eligibility rule remain unchanged. Design and validation each use
16 new roots crossed with four disturbance modes.

## Execution contract

The three cells are HalfCheetah, Hopper, and Walker at optimizer seed
`4196455150`. Each requests 24 CPU cores and 16 GB RAM. Scheduler placement is
dynamic across `node001-node006`, with no required node and no Slurm path.

## Gate and evidence boundary

The mechanism advances only if all three environments pass fresh validation.
This is adaptive development after v14.23, not confirmatory evidence. Failure
rejects this paired output-bias identification design; intervention amplitude,
roots, horizons, and selection thresholds will not be tuned after outcomes are
read.
