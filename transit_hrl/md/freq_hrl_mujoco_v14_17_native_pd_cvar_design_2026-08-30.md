# Freq-HRL MuJoCo v14.17 Native Primal-Dual/CVaR Screen

## Why v14.16 Was Rejected

The recovered v14.16 run contains 81 valid cells. Its preregistered primary
arm passed neither the engineering gate nor the complete effect gate in any of
the nine environment/optimizer-seed replicates. The strongest diagnostic arm
passed both gates in only two of nine cells and only one of three environments.

The failure is mechanistic rather than a missing-seed problem:

1. Native upper/lower trajectory leakage multipliers were disabled; v14.16
   relied on deployment-time action projection.
2. Upper and lower native costs differed by roughly two orders of magnitude,
   so one raw dual step size could not control both responsibilities.
3. Pathwise selection imposed 96 simultaneous constraints (16 paths times one
   reward and five frequency endpoints), causing frequent fallback to the
   initial checkpoint.
4. Freezing the reward actor during restoration reduced its ability to recover
   return and did not restore cross-environment feasibility.

Running more seeds on the same v14.16 mechanism would estimate a rejected
effect more precisely, not repair it.

## v14.17 Mechanism

v14.17 adds two independent components to the shared Freq-HRL actor-critic:

- **Occupancy-aware native primal-dual:** rollout-derived upper-HF and lower-LF
  costs train their existing cost critics. Dual violations are divided by an
  EMA of their absolute magnitude before multiplier ascent. The first nonzero
  violation initializes the scale, and scale/count state is checkpointed.
- **Signed upper-tail CVaR:** projection and actual closed-loop checkpoint
  selection optimize the upper-tail mean of signed normalized excesses. With
  four independent paths per disturbance mode and `alpha=0.5`, this controls
  the worst two paths without requiring every noisy path to be individually
  below every endpoint threshold.

Legacy defaults remain unchanged. v14.17 must be selected explicitly, never
reported under a v14.15/v14.16 protocol label, and does not freeze the reward
actor.

## Frozen Development Matrix

The matrix uses HalfCheetah-v5, Hopper-v5, and Walker2d-v5; three fresh
optimizer seeds are the independent replication units. Four disturbance modes
and held-out rollout paths are paired observations, not additional replicates.

| Arm | Native PD | Projection | Selector/guard | Purpose |
|---|---|---|---|---|
| `mean_s000_control` | off | off | mean reward | zero-router control |
| `mean_s050_projection_calibration` | off | off | mean reward | router-only calibration |
| `paired_s050_d000_control` | off | off | mode CVaR | matched selector control |
| `l2_path_v1416_comparator` | off | violation L2 | all paths | best v14.16 nonfreeze comparator |
| `native_pd_cvar_select` | on | off | mode CVaR | native-cost contribution |
| `cvar_projection` | off | CVaR | mode CVaR | tail-risk projection contribution |
| `native_pd_cvar_projection` | on | CVaR | mode CVaR | preregistered primary combination |

There are 9 shared anchors plus 63 continuation cells, for 72 cells total. Each
task requests one CPU core, 768 MB RAM, dynamic placement on node001-node006,
and scheduler rerouting when a node is unavailable.

## Decision Rule

Relative to the matched CVaR-selection control, a cell must satisfy all of:

- normalized return change at least `-0.02`;
- at least 5% reduction in effective, raw, and latent lower-LF drift;
- at least 5% reduction in effective and latent upper-HF power;
- a trained checkpoint rather than the initial fallback;
- arm-specific engineering checks for finite/active normalized native duals
  and/or valid actual closed-loop CVaR restoration.

The primary is nominated for a larger fresh-seed development run only if all
three environments pass the complete effect gate and all nine primary cells
pass their engineering gate. This screen cannot authorize a paper claim or a
confirmatory result by itself.
