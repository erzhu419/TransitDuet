# V93: release-gap and executed-holding counterfactual

V92 identifies a positive change in release/holding cross-covariance, not a
causal loss of compensation. Test this hypothesis with frozen B200 deployment,
both original early/late actors, native seven-bin actions and feasibility masks.
Do not force a recorded launch time: fixed-pool readiness and the reactive
frozen upper remain in the loop. No actor/critic/dual update or main switch.

Within V89's observable no-local-gain extra-holding group, let R be the current
trip's actual launch time minus the already-departed predecessor's actual
launch time, and H the difference in executed holding since each trip's launch
(current prefix versus predecessor at this stop's departure). For each action
a, Q(a)=((R+H+a-target)/target)^2. Use the full normalized native categorical
laws to calculate E_early Q minus E_late Q. Positive means replacement loses
release-coordinate compensation; nonpositive means it does not. The target
is the current executable target, not a future fitted mean. Missing bilateral
evidence remains late under every arm. Valid eligible contexts must match a
completed predecessor; a missing record stops the experiment.

Four dynamic regimes: native late; preserve compensation (replace nonloss
only); loss-only replacement; replace both (original V89 no-gain-only rule).
Primary control contrast is preserve versus late, with the unchanged joint
passenger-improvement/CV/readiness/completion nonharm criterion. Primary
mechanism contrast adds loss-group replacement to preserve: both minus
preserve. Inspect its CV and release/holding cross-covariance changes alongside
passenger time, journey, holding and readiness. Other arms are controls, not
fallback winners. Report the factorial interaction, intervention doses and
actual terminal dispatch-gap mean, variance and CV-squared.

Use 16 fresh days 1931001-1931016, two original actors and three matched
logical-event samplers: 96 cells, four full-day arms each, 384 rollouts. Pair
inference by demand day, report each actor separately. Capture actual launch,
executed holding and remaining elapsed arrival/departure primitives on-server;
reconstruct CV variance/normalization and service-boundary components exactly.
Only logs and small summaries return locally. All tests, smoke, rollouts and
statistics go through scheduler on node001-node006 in the dedicated CPU env.

Smoke uses consumed V89 days, exactly reproduces both native/full-replacement
metrics, sampling events and executed holding for both actors, checks both
strata occur, and executes each partial intervention before opening fresh days.

Limitations: Q excludes other elapsed deliberately; it is a current-prefix
release-coordinate proxy, not predicted full-day action value. Dynamic regimes
change subsequent states and intervention doses; their contrast is not
natural mediation or a matched-dose subgroup effect. Two fixed actors do not
establish training-seed generalization. Passing this pilot cannot admit a main
controller or change the original gate.

Frozen source `0561d3776e`; scheduler smoke `t172971` on node004 passes eight
focused tests and is running full native/conditional path checks. The initial
queued smoke was cancelled before launch to fix matched-prefix lookup when
another departure overwrites the recorder's latest event; no outcomes were read.
Shards `t172973-t172976` cover `[0,24)`, `[24,48)`, `[48,72)`, `[72,96)` with
24 workers each, planned on node002/004/005/006. They wait for the exact smoke
marker. Aggregate `t172978` waits for those four shard markers. No prospective
result or controller admission is claimed at submission.

All 384 paths and aggregation complete, exit 0/retry 0. Only the 111,134-byte
summary was retrieved. The registered preserve-versus-B200 joint gate passes:
journey -0.29696 min, CI [-0.38760,-0.22441]; CV-squared -0.003102,
CI [-0.005204,-0.001087]; readiness -97.24 s, CI [-111.56,-83.38]; holding
-13.47 s/trip. Both original actors separately improve passenger time, CV and
readiness; unlaunched/incomplete differences are exactly zero. Raw arrival
variance falls 506.55 s2 and terminal dispatch variance falls 527.04 s2;
normalization remains positive, so the CV benefit is not a mean-only artifact.

Both-groups minus preserve saves another 0.21831 min and 70.43 readiness
seconds, but CV-squared +0.000271, CI [-0.001505,+0.001992] is unresolved.
The release/holding cross term increases +0.003600, CI [+0.001783,+0.005366],
but actor 12011 has CV -0.001808, CI [-0.003915,+0.000228], while 933013 has
+0.002351, CI [+0.000325,+0.004397]. A positive covariance change alone does
not establish total CV harm. Both-groups versus B200 also has unresolved CV
nonharm, despite a better passenger mean; it does not replace the primary.

Keep the positive diagnostic and freeze its rule. Next confirm unchanged
preserve and both-groups controls on fresh days across all twelve archived
training seeds, with the ten actors not used in V37-V93 as the primary cohort.
Retain original-two-actor replication separately and crossed actor/day
intervals. Main policy stays frozen; this result is not a fixed-policy win or
a repair of the training procedure's 40-to-200 drift.
