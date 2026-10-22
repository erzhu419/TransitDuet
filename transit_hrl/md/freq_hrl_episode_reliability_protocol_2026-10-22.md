# Local Value To Episode Reliability: Prepared, Not Run

The [unified preregistration](freq_hrl_unified_preregistration_2026-10-22.md)
governs the two primary comparators, alpha allocation and frozen seed reservation.

Date: 2026-10-22. Fourth evidence item. No scheduler submission, controller
training or performance experiment. The scope remains multiscale control with
useful slow plans and fast feedback, not universal RL optimizer superiority.
Verification: 70 focused tests passed, including unchanged factual execution,
causal-prefix/conditional-future replay, delayed-harm attribution, rare-event
uncertainty, frozen margins and existing timing/budget/paper-evidence regressions.

## Frozen Comparison

Use the PointMaze/Reacher execution kernel and frozen development artifacts from
the total-compute preparation. Choose one primary renewal-value predictor and
the two registered timing comparators per task before confirmation. Keep
their weights, target, fitting, threshold and action generation unchanged.
For budget claims, use only the run IDs admitted by the declared budget search.

The full-episode panel compares the two saved controller/timing pipelines.
The local causal panel uses only the candidate's frozen controller: different
upper-call choices share an identical past and then act on their own resulting
states. Do not confuse that panel with a complete training-algorithm comparison.
The primary outcome is complete-episode ISE; dense return is retained separately.
Episode candidate-minus-comparator differences improve below zero; local/adaptive
probe advantages favor renewing now above zero.

## State Distribution

On a predeclared small subset of held paths, record causal features immediately
before each 50-step bin's action. Run both the candidate and the original jitter
reference on exactly those path/clock locations. This avoids oversampling states
where a policy waits and therefore performs more trigger checks.

Scale features using only the saved fitting bank. Report paired standardized
state displacement, per-feature mean shifts and outside-fitting-range fractions,
separately for the first and second episode halves. On identical sampled clock
locations, compare local prediction error under jitter and under deployment,
including the zero-predictor comparison and target scale. The fixed-clock paired
contrast isolates policy-induced occupancy changes; comparing either population
with event-selected fitting data additionally changes the acquisition distribution.

Evaluate three fixed profiles, separately: unchanged task; regime dwell times
multiplied by0.5; force amplitude multiplied by3 and duration by2. These stress
settings change exogenous dynamics only, not the planner, reward or timing rule.

## Local, Opportunity And Tail Effects

Uniformly sample bin starts independently of scores/outcomes, excluding jitter
call collisions and locations without a complete 200-step continuation. At each
sampled candidate state obtain:

- `L`: keep-minus-renew ISE over50 steps, with no later upper calls, exactly the
  existing training target.
- `W`: wait-one-check-minus-now ISE over the same50 steps, with the frozen timing
  policy continuing on each arm's own states and one call per bin.
- `T`: the same adaptive pair's ISE difference after that window to episode end.
- `G`: the complete adaptive episode difference, plus its first200-step contrast.

All four arms replay the same prefix. The code verifies the exact identity
`G = L + (W - L) + T`. Thus a failure can involve local prediction, the changed
within-window opportunity cost, the tail, or more than one of them. Neither
positive `L` nor its sum over factual decisions is an episode-value certificate.

Verify the unchanged-future factual arm against the original complete episode.
Then redraw independent conditional futures at the same branch point using the
existing driver sampler; preserve current observations, latent state and history,
and share each future between all paired arms. Report repeat means, two future
half-groups and finite-repeat variance corrections for local MSE. Negative
noise-corrected MSE estimates are retained, not clipped. Probes and their future
replicas are diagnostics, never additional independent training roots or episodes.

## Long Horizon And Sparse Risk

Every primary path runs to the original fixed episode end. Retain per-bin ISE,
first/second-half ISE, worst50-step ISE and maximum tracking distance. Operational
failure means distance exceeds a task-specific, predeclared threshold for at
least25 consecutive steps by default. Record event runs, excursion duration,
excess-distance ISE and force-induced control clipping without changing reward.

Report paired failure-episode discordances and the empirical upper-decile episode
ISE mean. The one-sided exact binomial upper bound on harmful discordances is
positive even with zero observations. Conditional on each frozen root, distinct
complete paths provide the event denominator, not primitive steps or overlapping
branches. With zero harmful events its upper bound is `1 - alpha_i**(1/n)`;
choose the path count from a predeclared precision/budget target, not favorable
interim outcomes. A small16-path panel cannot establish a tight rare-risk margin.

The initial design uses eight new optimizer roots per task, with distinct held
path banks across roots and paired identical paths between methods. Freeze the complete
path roster, diagnostic subset, risk threshold, risk-rate margin and tail-ISE
margin before test access. Primary intervals use paired root means, not pooled
query counts. Each task/comparator has family alpha0.0125, allowing two tasks
and two comparators at total alpha0.05 by a union bound. Half the family's alpha covers the three episode
endpoints across profiles; half covers conditional harmful-event bounds across
roots/profiles. Root-level local diagnostics and conditional-path intervals are
descriptive and labelled separately.

The joint episode criterion requires mean ISE improvement, tail-ISE and failure-
rate noninferiority at the frozen margins, sufficient harmful-event precision,
and negative ISE point differences in both held-path halves. All profiles and
expected roots must be present; failed/missing cases block the joint result.
The analyzer refuses changed margins or incomplete declared path rosters.

## Entry Points

From `transit_hrl/`, this writes only a plan, without opening source artifacts:

```bash
PYTHONPATH=. python3 scripts/run_plan_validity_reliability.py \
  --candidate-report /server/results/history.json --candidate-run-id history-root1 \
  --comparator-report /server/results/threshold.json --comparator-run-id threshold-root1 \
  --comparator-method threshold \
  --task pointmaze --optimizer-seed 1 --evaluation-seeds 101 102 103 104 \
  --probe-seeds 101 103 --probes-per-episode 3 --conditional-replicates 4 \
  --risk-distance 2 --risk-rate-margin 0.02 --tail-ise-margin 0 \
  --output /tmp/reliability_root1_plan.json
```

IDs, thresholds and path counts here illustrate syntax, not a registered or
adequately powered experiment. Execution requires `--execute`; it loads frozen
checkpoints and fitting caches on the execution server, never retrains, and
rejects paths exposed to either source's development process. Evaluation costs
use the existing CPU/step ledger and remain separate from deployment costs.
Interrupted/measured cases cannot be silently overwritten.

`scripts/analyze_plan_validity_reliability.py` reads only compact case reports via
`--inputs`, plus `--expected-roots`, `--task`, the two frozen margins and `--output`.
No fitting, replay or checkpoint access occurs during aggregation. Keep raw
artifacts on the server; retrieve compact JSON/logs only.

## Limitations And Next Decision

No new efficacy or reliability evidence exists. Fixed-clock support diagnostics
are not density-ratio bounds. Conditional futures hold simulator latent state
fixed, so local MSE includes ambiguity not resolvable from the actor's observation;
it is not an estimate of the observable-state Bayes error. Operational excursions
and empirical tail means are not physical safety or population-CVaR certificates.
Root t intervals are approximate, especially with few roots. The algebraic
decomposition does not prove that local scores compose into global improvement.

All four evidence items now have preparation paths. When compute is available,
first run a small native frozen-artifact preflight, then development cost pilots,
then freeze the combined confirmation protocol. New failures trigger a separate
development population, not repair on revealed confirmation paths. Stage304's
negative Transit boundary and the historical positive results remain unchanged.
