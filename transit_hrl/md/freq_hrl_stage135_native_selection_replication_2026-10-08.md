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

## Limitations

Previously used source roots: internal frozen-method replication, not untouched
external confirmation. Intervals correct this stage's ten endpoints, not the
adaptive research sequence. Validation adds simulation cost. Per-update radius
matches, not cumulative distance from zero. Stage133's composite HOLD, online
promotion, joint HRL and frequency-specific claims remain unchanged.
