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
fallback winners. Report the factorial interaction and intervention doses.

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
