# Stage135: Frozen Native Selection Replication

Stage134 meets its development gate with forecast gains+1.980800/+1.280096
and second-step increments+0.498933/+0.376703 at50/100. Selection changes one
long-period decision, but all four choices match the cheaper fit-only rule.

Freeze Stage134 and use roots410037/410049/410061/410073/410089/410101 from
their Stage131 two-step weights, not the Stage133 three-step weights. Same
26D projection,epsilon0.005,damping1,radius0.000555556,std0.15,authority0.05,
frozen lower/critics and at most one additional update. Four fresh label scenes,
16 calibration scenes and16 independent validation scenes, each with two panels;
64 new evaluation scenes. Validation chooses refresh/radial/no-update before
evaluation. Keep all seven executed controls and exact selected-policy aliases.

Ten primary endpoints: selected minus forecast/two_step/fixed-refresh/fixed-
radial/fit-only-choice at both periods. The fit-only rule chooses the largest
pooled quadratic predicted gain, including no-update gain0; zero ties keep the
source and positive candidate ties choose refresh. Its evaluation uses the
already executed chosen arm, not an evaluation oracle or another simulation.
Root-cluster bootstrap65,536draws,Bonferroni10,familywise alpha0.05 within stage.

Material continuation needs forecast CI lower>0.5 and second-step increment
CI lower>0 at both periods. Fixed-method selection and validation added value
are separate gates; do not claim validation helps if fit-only performs equally.
All roots/periods retained, no evaluation-based selection and no fourth update.

New cost/root:6,496episodes/7,795,200steps, including192validation episodes.
Six roots:38,976episodes/46,771,200steps. Immediate Stage131 cost separate;
earlier source-chain costs remain inherited. Fit-only selection avoids the192
validation episodes/root. Six17CPU/12GiB dynamic tasks on node001-node006;
no hard pins or waiting aggregation task. Four candidate weights/root stay
server-side; only compact JSON pulled.

## Run Receipt

Implementation `ea612cdb44`, registration `eea3ba405f`;12 related tests passed.
The six completed Stage131 source cells and12 upper files are available on
node004; the native runner checks checkpoint protocol and source fits on load.
Run `pointmaze_native_selection_replication_stage135_frozen_20261008_r1`:
`t136631-t136636` accepted, ordered by the six roots above; initial snapshot
queued. Exact task specifications, seed roles, ten endpoints, budget and compact
scheduler receipt are saved with the run. All six tasks are DONE;278,947bytes
of compact JSON were fetched, without checkpoints or native state arrays.

## Result And Next Step

The material continuation gate is supported at both periods. Equal-source-root
bootstrap65,536draws with Bonferroni10 gives:

| Period | Selected - forecast | Corrected CI | Selected - two-step | Corrected CI |
| --- | ---: | --- | ---: | --- |
| 50 | 1.985835 | [1.490466,2.363561] | 0.507803 | [0.384909,0.646290] |
| 100 | 1.079190 | [0.679043,1.333611] | 0.336180 | [0.253050,0.401659] |

These are small gains:0.1724%/0.0947% of mean forecast return. The old absolute
0.5 threshold passes; it does not establish a large relative improvement.
All12 root-period incremental gains are positive. Root410049/100 still has
forecast gain0.461219, below0.5; the registered gate is on the equal-root CI.

Fixed-method selection and validation-added-value gates remain not_closed.
Validation and the cheaper fit-only choice agree in11/12 cases, including all
six at100. Selected-minus-fit gains are0.007728 CI[0,0.030912] at50 and exactly
0 at100; this is not superiority or an equivalence test. Selected-minus-radial
at100 remains negative for roots410073/410089/410101. All controls are retained.

Stage136 now diagnoses the first joint PPO update from this validated upper,
rather than adding a fourth finite-difference update or more selector seeds.
Upper-only, lower-only and joint interventions share the same on-policy batch
and per-level optimizer seeds; source/teacher remain frozen. This isolates
whether joint adaptation preserves the existing gain before a larger redesign.

## Limitations

Previously used source roots: internal frozen-method replication, not untouched
external confirmation. Intervals correct this stage's ten endpoints, not the
adaptive research sequence. Validation adds simulation cost. Per-update radius
matches, not cumulative distance from zero. Stage133's composite HOLD, online
promotion, joint HRL and frequency-specific claims remain unchanged.
