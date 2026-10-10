# Stage159: Reference-Paired Upper Credit

Stage158 found normal state scales, critic deployment rank agreement 25/60,
and raw prefix-reward SD 17.6-88.0 times the forecast-paired SD. Test ONLY the
credit change, keeping Stage157's 34 causal inputs, two residual actions,
20-second coefficient scale, shared SAC, frozen native lower and update budget.

For each training seed, run an action-independent frozen forecast reference
with the same exogenous seed and macro decision clocks. Replace replay reward
by factual minus reference prefix reward; keep factual states/actions/next states.
With gamma=1 and equal initial costs, the sum is exactly
`100*(reference_final_cost-factual_final_cost)` before float32 replay rounding.
The reference final cost is independent of learned upper actions. Thus the
expected physical-cost objective is unchanged; no reference or future features
enter the actor, and no reference simulator is needed for deployment.

Qualification: one full forecast source reproduction; then two-episode RAW
learning with extra reference rollouts discarded, reproducing Stage157's short
training outcomes and learning statistics; then independent paired short learning.
Reset the model/RNG before full training. Same roots397/401 and training seeds
as Stage157; registered Stage157 raw SAC is the comparator, not a new replicate.

120 factual + 120 reference + 80 evaluation episodes/root, 5,500 SAC updates/root.
Last actor only; constant residual fixed from training states before evaluation.
All forty forecast/nominal controls/root must reproduce Stage157 exactly.
Main budget 39,283,200 native ticks plus 209,160 qualification ticks. Two scheduler
tasks, node001-006 unpinned, 1 CPU/3 GB each, server-only checkpoints.
97 focused tests pass: factual-transition preservation, terminal/tail credit,
float32 replay accounting, exact raw-SAC parameter reproduction with extra
reference rollouts in the controlled test, paired SAC updates, analysis budget
rejection and code-only scheduler placement.

Run `native_transit_reference_credit_stage159_development_20261011_r1`:
t141573/root397 DONE on node004; t141574/root401 DONE on node006.
Both workers passed full forecast reproduction, raw short-training reproduction
with discarded reference rollouts, and paired short training with two SAC updates
and actor parameter changes. Each resets the model/RNG for formal paired training.
All 240 factual training, 240 reference and 160 frozen evaluation episodes
qualify; both roots completed 5,500 SAC updates and all 80 forecast/nominal
evaluation controls reproduce Stage157. Main budget: 39,283,200 native ticks.

| Learned minus control | Cost delta (397 / 401) | Wait delta, min (397 / 401) |
| --- | --- | --- |
| forecast | +0.021110 / +0.005702 | -0.01280 / +0.02130 |
| constant residual | +0.000582 / +0.025582 | -0.01185 / +0.01275 |
| registered raw SAC | -0.001553 / +0.015370 | +0.01330 / -0.03825 |

Mean reward SD falls from 93.62/92.74 to 2.59/2.49; critic MSE falls from
4,329/3,990 to 22.63/15.98. This does not establish better value ranking:
the learning targets have changed scale, and physical performance is mixed.
Against forecast the learned actor wins 9/20 and 4/20 scenes. Root397's mean
fleet component increases by 0.020833 from one extra peak bus in one burst
scene. Root401 has no fleet difference: waiting, headway and unserved costs
all worsen. Against constant residual both roots worsen headway; sparse fleet
changes dominate the largest regime differences. Near-zero constant controls
do not explain away the learned actor's state-dependent variation, but that
variation has not earned a reliable advantage.

Next: same-scene reference credit with existing shared PPO, gamma=lambda=1
and complete episodes (Monte Carlo targets), no entropy bonus or Q bootstrap.
Keep native lower, 34 actor inputs, residual authority, scenes and terminal
cost fixed. This is a bounded trainer-replacement test, not proof that Q
bootstrapping alone caused failure; compare forecast AND training-only constant.

## Limitations

This is a two-root development ablation reusing evaluation scenes, not fresh
confirmation, joint two-level training or learned promotion. Reference training
doubles training simulator calls; that cost is counted explicitly. Reduced
reward variance does not guarantee a better critic or deployed policy. The
34-dimensional observation remains a partial simulator summary, and the
critic still optimizes soft stochastic continuation rather than deterministic
deployment; neither approximation is claimed to be fixed by this ablation.
