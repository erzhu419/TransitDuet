# MuJoCo v14.23 frequency-horizon critic preflight

## Decision context

v14.22 validated cumulative action-cost directions on Hopper and Walker, but
correctly rejected HalfCheetah before actor search. Its lower cumulative-cost
critic had negative holdout R2, negligible action-permutation gain, and
inconsistent ensemble gradients. Adding samples or weakening that gate would
not identify the missing action effect.

## Frozen mechanism

The only algorithmic change from v14.22 is the supervised cost horizon. Upper
targets contain at most eight macro decisions and lower targets at most 32
micro decisions. Each transition retains its SMDP duration discount and each
target stops at episode boundaries. These horizons align with the deployed
upper and lower frequency windows, reducing unrelated long-horizon occupancy
variance without replacing native constraint cost.

The critic architecture, four-member bootstrap ensemble, stochastic policy
collection, action-permutation gate, actor-gradient agreement gate, joint
upper/lower block normalization, five actor steps, exact reward guard, and
mode-mean closed-loop merit are unchanged. Twelve train roots, four critic
holdout roots, 16 design roots, and 16 validation roots are all fresh and
disjoint from v14.20-v14.22.

## Execution contract

The three cells are HalfCheetah, Hopper, and Walker at optimizer seed
`4196455150`. Each requests 24 CPU cores and 16 GB RAM. Placement is dynamic
across `node001-node006`, with `require_node=None` and no Slurm execution.

## Gate and evidence boundary

All three environments must pass the unchanged independent validation rule
before this mechanism can enter the shared actor-critic. This is adaptive
development after v14.22, not confirmatory evidence. Failure rejects the
frequency-horizon target as implemented; no horizon, root, step, or threshold
will be selected after reading the v14.23 outcomes.
