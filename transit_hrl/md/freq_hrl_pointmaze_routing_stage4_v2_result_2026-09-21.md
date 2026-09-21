# PointMaze Frequency-Routing Stage-4 V2 Result

Date: 2026-09-21

## Outcome

The frozen equal-shape development run
`pointmaze_routing_stage4_v2_development_20260920_r1` completed all 120 cells:
five within-HRL controls, three scenarios, and eight independent optimizer
roots. The analysis contains 1,920 held-out fixed-horizon episodes. The
registered selective-routing gate is **not supported**.

All 120 expected signatures completed without failed retries. Compact
`result.json` files were synchronized from node001, node004, node005, and
node006 with final counts 31, 30, 29, and 30.

The result audit passed:

- all 120 cells and 1,920 held-out episodes were present exactly once;
- every cell performed 18,432 PPO gradient updates;
- all eight optimizer roots and disjoint train, selection, and evaluation seed
  roles matched the frozen registration;
- every episode used 300 steps, 12 upper decisions, and 12 lower option
  boundaries;
- every method used 134-dimensional upper and lower states, hidden size 79,
  and exactly 68,424 trainable parameters;
- matched methods used the registered equal-shape masks and identical initial
  model parameters for each root;
- current physical feedback remained visible to both hierarchy levels;
- all numeric outputs were finite and runtime versions matched;
- all 384 scenario/root/evaluation-seed stress groups were exactly paired,
  with disturbance truth hidden from the actors.

## Primary Results

Mean success and root-level 95% confidence intervals were:

| Scenario | History | Filter | All bands | Routed | Swapped |
|---|---:|---:|---:|---:|---:|
| clean | 0.312 [0.188, 0.437] | 0.320 [0.197, 0.443] | 0.516 [0.335, 0.696] | 0.648 [0.470, 0.827] | 0.812 [0.716, 0.909] |
| fast observation noise | 0.297 [0.145, 0.449] | 0.383 [0.284, 0.481] | 0.430 [0.300, 0.559] | 0.539 [0.356, 0.722] | 0.648 [0.463, 0.834] |
| slow drift + fast action | 0.336 [0.268, 0.404] | 0.359 [0.268, 0.451] | 0.492 [0.366, 0.618] | 0.500 [0.372, 0.628] | 0.703 [0.534, 0.872] |

The registered success contrasts were:

| Contrast | Clean | Observation noise | Action stress |
|---|---:|---:|---:|
| routed - history | +0.336 [0.129, 0.543] | +0.242 [0.074, 0.411] | +0.164 [0.022, 0.306] |
| routed - causal filter | +0.328 [0.178, 0.478] | +0.156 [0.017, 0.296] | +0.141 [0.041, 0.240] |
| routed - all bands | +0.133 [0.062, 0.204] | +0.109 [-0.032, 0.251] | +0.008 [-0.102, 0.118] |
| routed - swapped | -0.164 [-0.365, 0.037] | -0.109 [-0.407, 0.189] | -0.203 [-0.353, -0.053] |

Clean routed-versus-all noninferiority passed. Routed exceeded causal filtering
under observation noise, and routed exceeded raw history in both primary
stresses. The conjunctive gate nevertheless failed because routed-versus-all
was inconclusive in both stresses, routed-versus-swapped was inconclusive under
observation noise, and routed-versus-swapped was contradicted under action
stress.

Independent recomputation from the raw evaluation rows reproduced every
contrast. Routed exceeded history in 8/8 clean roots, 6/8 observation-noise
roots with two ties, and 6/8 action-stress roots with one tie. Against swapped,
routed was positive in only 2/8, 3/8, and 1/8 roots respectively.

## Interpretation

V2 removes the V1 architecture and per-level capacity confound. It therefore
supports a bounded finding: causal multiscale masking can improve a learned
HRL policy over raw history and causal smoothing in this PointMaze protocol.
It does not support the stronger claim that the intended
slow+mid-upper/mid+high-lower assignment is the source of that gain. Giving all
bands to both levels remained statistically indistinguishable in both primary
stresses, and the reversed assignment was better under action stress.

This protocol decomposes endogenous physical-state history. The original
Freq-HRL definition instead decomposes an external time-series driver that is
not controlled by the agent. The clean V2 negative attribution result closes
PointMaze endogenous-history routing as a route to the central paper claim.
Adding roots or tuning cutoffs on these outcomes would be post-hoc gate chasing.

The next mainline experiment must expose endogenous physical state unchanged
to both levels while separately decomposing a causal, actor-visible exogenous
stream. PointMaze remains evidence that ordinary goal-conditioned HRL works;
it is not evidence for general exogenous-state frequency responsibility.

## Claim Boundary

Allowed: under the frozen equal-shape PointMaze protocol, routed multiscale
history improved success over raw history in clean and both stress scenarios,
and over causal filtering in both stresses. The intended selective-routing gate
was not supported.

Forbidden: V2 validates the intended Freq-HRL frequency assignment, proves
selective routing beats generic multiscale features, supports a domain-general
Freq-HRL claim, or justifies relabeling the swapped control as the method.

Machine-readable analysis remains under
`results/pointmaze_routing_stage4_v2_development_20260920_r1/analysis/`.
