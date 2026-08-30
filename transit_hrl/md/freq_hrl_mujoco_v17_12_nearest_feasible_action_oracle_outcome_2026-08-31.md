# MuJoCo v17.12 Nearest-Feasible Action Oracle Outcome

## Decision

Status: `nearest_feasible_targets_authorize_causal_actor_adapter`.

The frozen Euclidean projection processed the unchanged 120-path reused panel.
It preserved all 113 paths already feasible under the v17.6 full-horizon
responsibility oracle and constructed feasible targets for the seven remaining
Hopper actor-floor paths. All nine pre-registered advancement checks passed.

The required frequency-only changes to total action were small: actor-floor
correction RMS averaged `0.002343` and was at most `0.008118`; the largest
absolute correction was `0.036874`. The seven compressed target files remain on
node003 under the registered server-only artifact root. Only the summary and
location manifest were synchronized locally.

## Deployment Diagnostic

The optional nominal total-action box projection also found feasible component
pairs for all seven actor-floor paths, but required much larger correction RMS
(`0.075724` mean, `0.081109` maximum). This profile is a diagnostic of clipping
semantics and does not select the distillation target. The next learner targets
the smaller frequency-only total-action correction.

## Next Stage

V17.12 establishes that the seven router-irreducible paths are not far from the
registered frequency-feasible set once the actor may change total action. This
authorizes grouped causal actor-target distillation on the same reused paths.
Fresh validation remains forbidden until that causal learner passes its frozen
reused-path gate.

## Claim Boundary

This is an acausal target-construction oracle on reused development paths. It
does not establish an online policy, reward improvement, fresh-seed
generalization, leakage no-tradeoff, or manuscript support.
