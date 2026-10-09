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

Judge the regularizer contrast and same-checkpoint upper intervention separately.
If Q separates actions without useful control, investigate delayed credit/plans
instead of expanding seeds.

## Limitations

Two-root mechanism development cannot support a superiority or full HRL claim.
Physical-coordinate regularization preserves a prior, not optimizer trajectories
or a guarantee of control gains. Learned plans and promotion remain unproven.
