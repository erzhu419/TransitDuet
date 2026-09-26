# Stage-8C Compact Plan-Validity Result

Date: 2026-09-26

Run: `pointmaze_compact_plan_validity_stage8c_v1_development_20260922_r1`

Protocol: `pointmaze_compact_plan_validity_stage8c_v1_development`

Frozen algorithm revision: `78e8493c6ba4399b3d68df4da4d353aaab03104e`

## Decision

All seven registered root-level 95% confidence intervals are strictly
positive. The frozen conjunction **authorizes Stage-9 budgeted-trigger
development**. This qualifies a causal predictor of local `keep` versus
`renew` branch value. It does not establish deployed-trigger or episode-level
control improvement.

| Registered quantity | Mean [95% CI] |
|---|---:|
| Controller learning ISE gain | 22.049208 [18.643541, 25.454875] |
| Renewal value at regime change +250 ms | 0.099221 [0.086576, 0.111866] |
| Candidate Spearman rank | 0.755987 [0.697901, 0.814073] |
| Candidate selected local value | 0.157427 [0.140909, 0.173944] |
| Candidate utility minus current-only, **primary** | **0.040329 [0.030866, 0.049792]** |
| Candidate rank minus current-only | 0.383245 [0.310722, 0.455768] |
| Candidate rank minus generic dynamic | 0.147923 [0.076170, 0.219676] |

The secondary candidate-utility-minus-generic-dynamic contrast was also
positive: 0.019476 [0.008287, 0.030665]. The primary contrast was positive
in each of the eight roots; no roots were added after inspection.

## Execution Audit

- All eight registered optimizer roots completed; 320 role seeds were unique.
  Each root used 8 training, 8 selection, 8 predictor-fit, and 16 held-out
  branch-evaluation paths.
- Each root had 192 fit and 384 evaluation rows, with four opportunities per
  class on every branch path. All 8 controllers completed 33,792 optimizer
  updates under the registered 267,018-parameter CPU architecture.
- The candidate had 39 causal features; both quadratic comparators had 170.
  Ridge alpha was chosen by eight-group fit-path cross-validation. Event labels,
  future values, and privileged regime context were absent from predictors.
- Every paired branch had identical prefix and feature state, keep/renew upper
  calls of 0/1, zero downstream upper calls, and closed-loop lower control.
  Extra replay was recorded separately: 1,956,944 fit and 3,918,820 evaluation
  primitive steps across the matrix.
- The independent audit recomputed the primary mean and Student-t interval
  directly from held-out rows and matched the frozen analyzer. The local run
  directory contains only eight compact `result.json` files and preregistration;
  no checkpoint or raw trajectory is present.

## Boundary And Next Step

Stage-8B's failed linear predictor is retained as a negative result. Stage-8C
uses fresh seeds and supports the narrower claim that causal plan-validity
interactions rank and select locally useful renewal opportunities better than
the registered current-only and generic-dynamic comparisons.

Stage 9 must train and evaluate a budgeted trigger in the closed control loop.
It must use causal observations, match upper-call budgets against fixed timing,
handle variable-duration options correctly, and test held-out episode ISE and
return. A predictor-only result cannot establish those outcomes.
