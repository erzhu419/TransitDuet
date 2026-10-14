# V35: Execution-Actor Cross at a Fixed 200-Episode Context

V34 found rising holding and passenger journey time in all three policies despite
improving absolute headway CV. This diagnostic crosses upper/lower actor states
from episodes 39 and 199 (40 and 200 completed training episodes), separately
within each V34 configuration and training seed. All other loaded state and the
evaluation policy episode remain at 199. Only `policy_net.state_dict()` is
replaced; critics, temperatures, multipliers, sampling streams, belief, objective
weights, cached selectors, configuration, and action semantics stay fixed.
This fixes initial deployment state, not the endogenous within-episode evolution
of queues, demand filtering, holding feedback, and executable plans.
Frozen inference uses the online upper actor; the EMA upper is training-only.

The design is 3 configurations x 12 training seeds x 4 combinations x 12 shared
scenario seeds: 144 independent CPU jobs and 1,728 frozen rollouts, with no
training. The verified optimized V9 snapshot is used on node001-node006 through
scheduleurm. Models and raw results stay on the server; only summaries return.
Smoke tests reproduce unchanged 200/200 evaluations against V34 and exercise an
early-lower mixed actor; non-actor state is checked before/after each transplant.

Write Y(u,l) for an outcome at this common background, with 0=40 and 1=200.
Report lower changes at both upper ages, upper changes at both lower ages,
their averaged effects, and interaction Y11-Y10-Y01+Y00. Averaged upper and lower
effects sum to Y11-Y00; interaction is not added again. Subtracting the original
40-episode V34 outcome from Y00 gives a separate residual for non-actor context
and evaluation stage, not a pure belief-state effect. Both components sum to
the observed 40-to-200 change after unchanged-control reproduction succeeds.

Primary outcomes are restricted journey time, cumulative holding per launched
trip, headway CV, and fleet readiness delay. Use the original crossed-bootstrap
intervals and V34 cohorts. APC holding penalties, regularity reward, and upper
wait/backlog credit are descriptive mechanism outputs. This is reused-seed
attribution, not an independent confirmation or selection of a new main policy.

Regression `t152598` passed 16 tests. Smokes `t152599-t152601` reproduced 390
unchanged-control columns in each configuration and passed mixed-actor frozen
evaluation. Diagnostic source `263ded1317` is frozen server-side. Evaluations
`t152635-t152778` and aggregation `t152779` are complete (all 145 tasks done).
All 1,728 rollouts were aggregated; the unchanged 200/200 controls reproduce
V34. Only four summary files (197,768 bytes) were retrieved, with no checkpoints
or rollout-level data. Results are under
`freqduet/results_freqduet/protocol_v6_v35_actor_cross_b200/summary/`.

## Attribution and next decision

Current-main cell means, all 12 training seeds x 12 evaluation scenarios:

| Upper / lower actor age | Journey (min) | Holding / launched trip (s) | Fleet readiness delay (s) | Headway CV |
| --- | ---: | ---: | ---: | ---: |
| 40 / 40 | 18.6090 | 216.8688 | 225.6251 | 0.227011 |
| 40 / 200 | 19.9638 | 291.6558 | 596.3255 | 0.216278 |
| 200 / 40 | 18.5976 | 216.5894 | 224.2597 | 0.225919 |
| 200 / 200 | 19.9666 | 291.8627 | 598.3806 | 0.216592 |

| Attribution contrast | Journey delta (min), 95% CI | Holding delta (s/trip), 95% CI |
| --- | ---: | ---: |
| Lower, averaged over upper ages | +1.3619 [+0.7707, +1.9901] | +75.0302 [+44.3876, +103.6199] |
| Upper, averaged over lower ages | -0.0043 [-0.0290, +0.0181] | -0.0362 [-0.7459, +0.6485] |
| Interaction (separate contrast, not additive) | +0.0142 [-0.0285, +0.0571] | +0.4862 [-0.7999, +1.7110] |
| Non-actor context / evaluation-stage residual | 0 [0, 0] | 0 [0, 0] |

Lower aging accounts for essentially all observed growth under this intervention.
It increases journey and holding in 11/12 main training seeds; both increases
have positive intervals in every V9/V33 train/evaluation subgroup. A small
holding interaction exists in the V33/V33 subgroup (+1.99 s/trip, CI
[+0.19, +4.41]), so this is not a claim of universally absent interaction.
The noguard and compact controls show the same lower-dominated deterioration:
averaged lower journey / holding effects are +1.5194 min / +84.4227 s and
+1.4116 min / +79.4684 s, respectively. All configurations have zero seed-mean
residual across the 12 recorded outcomes.

For main, the observed journey increase splits into +0.9358 min waiting and
+0.4218 min onboard time. Mean lower action rises from 10.9273 to 14.7051 s,
while the lower-actor CV effect is favorable (-0.01003, CI [-0.01688, -0.00283]).
The causal APC holding penalty is already active and rises from 0.003479 to
0.004753; upper dispatch-backlog cost rises from 0.9631 to 2.5529. These are
descriptive signals, not proof of which training-loss term caused the drift.
Frozen source `673355cae1` disables lower wait attribution in the inherited V5
config; system waiting/backlog costs enter upper interval credit, while lower
reward directly charges local APC holding. This credit boundary is the first
repair hypothesis to test, not evidence that the upper actor caused the growth.

Next, keep upper control and execution semantics unchanged and focus on lower
learning. On matched causal lower states, compare the 40/200 critics' action
preferences, cost values, uncertainty and entropy terms against rollout-level
waiting/onboard/fleet consequences. This distinguishes critic misranking from
an objective that genuinely rewards excessive holding before choosing the fix.
Do not repeat a scalar load-penalty sweep, restore an execution guard, or promote
the early-actor splice as a repaired policy; V33 already rejected training-time
lower freezing. No main alias or paper confirmation gate changes in this step.
