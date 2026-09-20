# PointMaze Frequency-Routing Stage-4 Result

Date: 2026-09-20

## Outcome

The frozen development run
`pointmaze_routing_stage4_v1_development_20260920_r1` completed all 120 cells:
five within-HRL routing controls, three scenarios, and eight independent
optimizer roots. The analysis contains 1,920 held-out fixed-horizon episodes.
The registered selective-routing gate is **not supported**.

Every expected signature has a successful result. Five earlier scheduler
attempts were retained as failed records and then completed under the same
unchanged signatures as tasks `t98095`, `t98096`, `t98097`, `t98102`, and
`t98103`. The successful cells were distributed across node001-node006 as
20, 19, 20, 20, 21, and 20. Only compact `result.json` files were synchronized.

The result audit passed:

- 120 unique cells and 1,920 held-out episodes were present;
- each cell performed 18,432 PPO gradient updates;
- all eight optimizer roots and all train/selection/evaluation seed roles
  matched the frozen registration;
- every episode used 300 steps, 12 upper decisions, and 12 lower option
  boundaries;
- parameter-budget ratios ranged from 0.993071 to 1.006635;
- current physical feedback remained visible to both hierarchy levels;
- all numeric outputs were finite and runtime versions matched;
- all 384 scenario/root/evaluation-seed stress groups were exactly paired
  across methods, with disturbance truth hidden from the actors.

## Primary Results

Mean success and root-level 95% confidence intervals were:

| Scenario | History | Filter | All bands | Routed | Swapped |
|---|---:|---:|---:|---:|---:|
| clean | 0.344 [0.229, 0.459] | 0.375 [0.222, 0.528] | 0.484 [0.337, 0.632] | 0.508 [0.422, 0.594] | 0.711 [0.583, 0.839] |
| fast observation noise | 0.258 [0.177, 0.339] | 0.336 [0.196, 0.475] | 0.391 [0.299, 0.482] | 0.516 [0.428, 0.603] | 0.688 [0.609, 0.766] |
| slow drift + fast action | 0.281 [0.173, 0.389] | 0.320 [0.222, 0.419] | 0.344 [0.174, 0.514] | 0.375 [0.287, 0.463] | 0.766 [0.705, 0.826] |

The registered gate components were:

| Contrast | Clean | Observation noise | Action stress |
|---|---:|---:|---:|
| routed - all bands | +0.023 [-0.098, 0.145] | +0.125 [-0.003, 0.253] | +0.031 [-0.117, 0.179] |
| routed - swapped | -0.203 [-0.377, -0.029] | -0.172 [-0.314, -0.030] | -0.391 [-0.532, -0.249] |
| routed - causal filter | +0.133 [-0.045, 0.310] | +0.180 [0.057, 0.303] | +0.055 [-0.107, 0.216] |

Clean routed-versus-all noninferiority passed at the registered 0.10 margin,
and routed beat causal filtering under observation noise. Routed-versus-all
superiority remained inconclusive in both stress scenarios. Most importantly,
routed-versus-swapped was contradicted in all three scenarios. Direct raw-row
recalculation found swapped ahead in 7/8 clean roots, 7/8 observation-noise
roots, and 8/8 action-stress roots.

## Interpretation

The implementation labels were checked at the feature-content level: routed
feeds slow+mid history to the upper policy and mid+high history to the lower
policy; swapped does the reverse. The result is therefore not explained by a
label reversal.

Frequency representation matters in this PointMaze setup, but the assumed
fixed assignment does not. The encoded signal is endogenous physical-state
history rather than an explicit exogenous time series. In this setting, the
upper waypoint policy benefited from mid+high history and the lower actuator
policy from slow+mid history. This rejects a domain-general fixed
`slow -> upper, high -> lower` rule; it does not establish that swapped routing
is universally correct.

The next algorithmic step must separate actor-visible exogenous context from
endogenous physical feedback, or learn routing under a fresh preregistered
protocol. Relabeling the winning swapped control as Freq-HRL would be post-hoc
claim substitution and is prohibited.

## Claim Boundary

Allowed: under the frozen PointMaze endogenous-history protocol, multiscale
features and their level assignment materially affected HRL performance; the
registered routed arm improved over history in clean and observation-noise
conditions and over causal filtering under observation noise, but failed the
selective-routing gate because swapped routing performed better.

Forbidden: Stage 4 validates the intended Freq-HRL routing, proves a universal
frequency-to-level assignment, establishes swapped routing outside this
PointMaze setting, or supplies a confirmatory headline result.

Machine-readable analysis remains under
`results/pointmaze_routing_stage4_v1_development_20260920_r1/analysis/`.
