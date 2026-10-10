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
t141573/root397 RUNNING on node004; t141574/root401 RUNNING on node006.
Worker qualification and final paired-policy performance are pending.

## Limitations

This is a two-root development ablation reusing evaluation scenes, not fresh
confirmation, joint two-level training or learned promotion. Reference training
doubles training simulator calls; that cost is counted explicitly. Reduced
reward variance does not guarantee a better critic or deployed policy. The
34-dimensional observation remains a partial simulator summary, and the
critic still optimizes soft stochastic continuation rather than deterministic
deployment; neither approximation is claimed to be fixed by this ablation.
