# MuJoCo v14.25 bias-matched critic preflight

## Decision context

v14.24 made every action-cost critic identifiable and independently supported
Hopper and Walker. HalfCheetah failed only after the inferred action derivative
was propagated through all actor layers: every full-network step violated the
design reward guard or sharply increased frequency merit. The update scope was
broader than the output-bias intervention used to identify the effect.

## Frozen mechanism

Critic collection remains the five-way paired intervention protocol: control,
upper output-bias plus/minus, and lower output-bias plus/minus at RMS `0.25`.
The upper-eight and lower-32 cost horizons, four-member ensemble, holdout R2,
action-permutation and gradient-agreement gates are unchanged.

The actor update now contains only the final upper and lower actor output
biases. Hidden weights, output weights, log standard deviations, critics, and
all other parameters remain frozen. Upper and lower bias-gradient blocks are
normalized separately. Exact design evaluates bias RMS steps `1e-7`, `1e-6`,
`1e-5`, `3e-5`, and `1e-4`; reward and frequency eligibility is unchanged.

Eight critic train roots, four critic holdout roots, 16 design roots, and 16
validation roots are all fresh and disjoint from v14.20-v14.24.

## Execution contract

The three cells are HalfCheetah, Hopper, and Walker at optimizer seed
`4196455150`. Each requests 24 CPU cores and 16 GB RAM. Scheduler placement is
dynamic across `node001-node006`, with no required node and no Slurm path.

## Gate and evidence boundary

All three environments must pass independent validation before bias-matched
updates can enter the shared actor-critic. This is adaptive development after
v14.24, not confirmatory evidence. Failure rejects the output-bias update under
this frozen protocol; roots, bias steps, intervention amplitude, and thresholds
will not be changed after reading outcomes.
