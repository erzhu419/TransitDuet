# MuJoCo v17.8 Causal FIR Distillation Design

## Why v17.7 Stops

The v17.6 full-horizon oracle recovers 81 of 88 failed v17.4 paths by changing
only the upper/lower responsibility split. V17.7 does not recover the Hopper
smoke path: neither its forecast nor its strict prefix-average budget matches
the endpoint solution. On that path, the oracle exceeds the endpoint upper
budget on 79 of 83 prefixes before finishing inside the budget. Therefore an
all-prefix budget is a different and materially stricter problem.

An inspected-path fit then established a narrower fact: a multivariate causal
FIR using cross-action history can satisfy both endpoint budgets, while a
shared scalar FIR and independent per-action FIR cannot. That is architecture
feasibility on one reused path, not generalization evidence.

## Method Contract

At step `t`, the router receives the total action selected for that same step
and its realized total-action history. For FIR width `W`, it predicts

`upper_t = gain * sum_{lag=0}^{W-1} total_{t-lag} B_lag`,

with missing pre-episode history set to zero. The matrices `B_lag` mix action
dimensions. The prediction is projected only onto the physical intersection
that keeps both upper and complementary lower actions in their registered
boxes. It cannot use observations after `t`, episode return, termination time,
disturbance labels, or oracle outputs at evaluation time.

The coefficient tensor is fitted separately per environment because action
dimensions differ. The causal representation, zero-state rule, ridge fitting,
candidate set, grouped selection, and gates are identical across environments.
This is one domain-general algorithm, not one parameter tensor shared across
incompatible action spaces.

## Reused-Panel Selection

All 120 v17.4 paths are development data. The previously inspected Hopper
`standard/4009024190` path is not treated as held out. Candidate selection uses
eight grouped folds: one evaluation seed and all five of its disturbance modes
are omitted from fitting in each fold. Oracle labels are generated and stored
only on node003.

The frozen candidate grid is:

| Dimension | Values |
|---|---|
| FIR width | 16, 24, 32, 48, 64 |
| normalized ridge penalty | `1e-5`, `1e-3`, `1e-1`, `1` |
| output gain | `0.80`, `0.90`, `1.00`, `1.10` |

One candidate id is selected jointly across all environments. Selection is
lexicographic: numerical validity and upper-budget path count, the worst
environment's oracle-recoverable recall, total recovered failures,
baseline-feasible preservation, worst-environment lower-power ratio, overall
mean lower power, then candidate id.

Fresh-path access requires all inherited v17.7 conditions on out-of-fold
predictions: exact reconstruction and bounds on all 120 paths; endpoint upper
budget on all paths; recovery of at least 65/81 oracle-recoverable failures,
including at least 32 HalfCheetah, 24 Hopper, and 6 Walker2d; preservation of at
least 30/32 baseline-feasible Walker2d paths; and no worse mean lower power than
v17.4 in every environment.

## Fresh Validation

Eight seeds were derived before access with
`derive_seed("mujoco_v17_8_fresh_router_validation_v1", index)` for indices
zero through seven. They are disjoint from all v17.4 optimizer, training,
selection, and evaluation seeds:

`2969266561, 1060853697, 1705453152, 1911126157, 3726666952, 2647745800,
3002649567, 2889178607`.

The selected candidate is refitted on all reused paths, frozen, and evaluated
once on 120 fresh environment/mode/seed paths. The fresh gate requires valid
bounded reconstruction everywhere, endpoint upper-budget compliance
everywhere, at least 75% recall among oracle-feasible paths overall, at least
60% per environment, at least 90% preservation of baseline-feasible paths, and
no worse mean lower power than baseline in each environment.

## Claim Boundary

Both stages freeze the total action, so reward and physical trajectory are
unchanged by construction. Passing v17.8 supports a causal responsibility
router mechanism for subsequent fresh closed-loop policy training. It does not
support a reward improvement, an online learned-policy result, or a manuscript
performance claim. Seven known v17.6 Hopper paths are oracle-infeasible and
require actor-level trajectory change rather than a different split.
