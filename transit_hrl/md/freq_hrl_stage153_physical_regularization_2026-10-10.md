# Stage153: Physical-Coordinate Critic Regularization

Stage152 removed input-scale dominance, not useful upper control. For
`u=(a-c)/R`, equivalent physical first-layer weights/bias are `W_a/R` and
`b-(W_a/R)c`. Physical_sum computes their L1; later layers are unchanged.

Compare four service-credit signed-dispatch methods: seconds/sum, unit/sum,
unit/physical_sum and unit/mean. Mean is a separate global weakening control.
Keep coefficient, reward, bounds, execution, lower controller, estimator,
leakage and RE-SAC budget fixed. Record per-state absolute Q differences to
zero as well as episode-mean curves to reveal cancellation.

Fresh roots313/331, 300 train episodes/cell, 30 upper warmup, 2,700 upper and
9,000 lower updates. Separate worker qualification, fresh full training, last
checkpoint only. Pair learned/zero-upper/zero-holding over four scenes in five
regimes. Eight jobs: 176,774,400 native ticks plus 216,000 qualification ticks.
Scheduler only, one CPU/3 GB, node001-006 eligible without pinning. Stage code
only; raw data/checkpoints stay server-only. The 56 focused tests pass, including
physical L1 equivalence and gradient units; archived Stage152 analysis is unchanged.

Run `native_transit_critic_regularization_stage153_development_20261010_r1`:
tasks t140738-t140745 completed; all eight cells passed matched-budget analysis.

Physical_sum learned-minus-own-zero-upper, roots313/331: cost
`-0.019630/-0.020956`, restricted wait `+0.01725/+0.00835` min,
episode reward `+2.841/+4.777`. Commands average `+7.646/+6.602` s;
within-episode proposal SD averages `0.551/1.264` s. All 5,240 commands per
root delay departure; none advance. Physical L1 restores the action first-layer
contribution (`0.0395/0.0469`, comparable to state `0.0387/0.0399`), but
paired absolute Q differences at +60 s remain small (`0.00296/0.00133`).

Equal-root cost gain `0.020293` is driven by fleet cost `-0.020833`; other
components sum to `+0.000540`. Only four of 40 paired episodes change peak
fleet. Persistent-shift cost worsens for root313. Unit/mean is not an alternative
success: own-upper reward differences are `-7.475/+1.054`.

Next: Stage154 frozen learned-versus-constant-phase comparison, including a
source-wide mean command and fixed 5/7/9 s. No new training or seed expansion.
If constants recover the gain, redesign temporal credit/plan representation;
if learned adaptation adds value, confirm that separately on independent scenes.

## Limitations

Two-root mechanism development cannot support a superiority or full HRL claim.
Physical-coordinate regularization preserves a prior, not optimizer trajectories
or a guarantee of control gains. Learned plans and promotion remain unproven.
