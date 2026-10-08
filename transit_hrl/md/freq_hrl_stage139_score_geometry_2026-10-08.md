# Stage139: Score Geometry And Deployment

Stage138 conditional credit removes phase variation but does not reliably improve
native return. Hold credit fixed and separate optimizer geometry from sampled
training versus mean deployment. Roots410037/410049, periods50/100, horizon1200.

Replay the32 original sampled paths/root and restore their cached action credits;
check native returns, decisions, original PPO update and credit diagnostics. No
counterfactual reruns or new labels. Compare the option-credit Adam displacement,
the initial full-batch score gradient, and the existing standardized damped Fisher
mean direction(damping1). Same std0.15, authority0.05 and empirical KL0.000555556;
both signs, all deployed lower/critic/teacher/forecast weights frozen.

Evaluate32 new paired scenes/period under both mean and sampled upper policies.
Lower draws and exogenous measurements match all arms; sampled upper innovations
match within its mode. Forecast has no upper decision and is executed once, then
shared between mode tables; it is not a second independent native observation.
Reward, tracking reduction, gradient alignment and sampled-minus-mean are kept.
No native-return fit, evaluation winner, confirmation CI or automatic extension.

New cost/root32replay+960evaluation=992episodes/1,190,400steps;64forecast table
aliases, zero new label queries. Two roots:1,984episodes/2,380,800steps. Inherited
Stage138 query costs and earlier sources stay separate. Sixteen workers+parent,
12GiB, scheduler dynamic node001-node006. Compact JSON only, no checkpoint/trace
writes. Sampling without collection prevents unused evaluation training batches.

## Run Receipt

Implementation `fdea4f9d9a`, registration `2d5650041c`, pushed before submission.
Seventeen related tests passed, including sampling without collection, score
identity, all three exact KL radii, cached label/update replay, shared-forecast
accounting, noise pairing and the reduced native runner. The four selected warm
upper files remain available on node004. Run
`pointmaze_score_geometry_stage139_probe_20261008_r1`: `t136981` root410037 is
running on node005; `t136982` root410049 is running on node001. Nodes are dynamic,
not pinned. No waiting aggregator, checkpoint pull or local native training.

## Limitations

The score estimates the stochastic-upper objective, not a derivative of mean
deployment. Damping uses the existing standardized empirical geometry; matched
training-state KL does not ensure fresh-state gains. This is an optimizer/objective
diagnosis with old training labels, not new confirmation or full joint Freq-HRL.
