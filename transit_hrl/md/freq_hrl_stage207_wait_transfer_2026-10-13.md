# Stage207: frozen wait fitting versus transfer

Reuse94 Stage206 GLMs/47 folds. Compare training and held-out mean8,
keeping single-draw training separate;scene training matches root/regime.
Separate own-loss errors,intentional rejection,zero signal and future-half
disagreement. Repeated model evaluations are not new states/seeds.

Run: `native_transit_wait_transfer_stage207_diagnostic_20261013_r1`.
One unbound scheduler task,node001-node006,1CPU/3072MB; no fitting,optimizer
updates or native episodes. Models,bank,targets and draw panels stay remote.

Completed`t148059`,node004,exit0; source`58cdd012ed`;8 tests passed.
One691096-byte report pull; no bank,draw,target,model or checkpoint pulls.
Symmetric own-loss boundary errors on informative mean8 actions: scene
same-root/regime training40.106% -> held41.304%; regime39.319% ->45.652%;
lower-root37.542% ->40.803%. There is already substantial training error,
with additional regime degradation, not just a new-domain failure.
For the same model holding out root401, excess mass loss rises0.181800 ->
0.342727 nats; the reverse root397 fold falls0.278469 ->0.210685.
Lambda2 scene rejects426 expected-safe informative actions:146 intentionally
under its stricter objective,280 contrary to its own boundary;4 zero-signal
rejections are separate. Conservative rejection is not all explained by risk
preference. Symmetric cross-root401 selected harm0.006133333 minus benefit
0.003462500 leaves +0.002670833 min, matching the failed Stage206 gate.

Next: register a causal state-conditioned nonlinear plan/wait head with the
symmetric loss and frozen GLM controls;keep data,splits,costs/menu unchanged.
Require both training-fit and held-out improvements,not a larger lambda or
post-hoc threshold. Stage206's unsuccessful primary remains unpromoted.

Limits: in-sample diagnostics are optimistic and mean8 targets remain noisy.
Loss excess includes regularization and model restrictions; it cannot alone
prove a missing causal feature. No threshold tuning, CI or deployment claim.
Leave-lower-root also changes demand/prefix seeds, not just lower policy;
its degradation is not an isolated causal lower-policy effect.
