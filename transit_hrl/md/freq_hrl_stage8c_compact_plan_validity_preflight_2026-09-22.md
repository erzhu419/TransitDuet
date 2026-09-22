# Stage-8C Compact Plan-Validity Preflight

Date: 2026-09-22

Successful run: `pointmaze_compact_plan_validity_stage8c_v1_preflight_20260922_r2`

Task: `t100474`

## Outcome

The repaired registered cell completed on `node002` in about 69 seconds. The
only server artifact synchronized locally was its 268-KB `result.json`. The
preflight passes and authorizes the unchanged eight-root development matrix.
It is implementation evidence only.

The preceding r1 attempt exposed a deterministic protocol-test defect: branch
evaluation seed `2145301` had no eligible force-pulse opportunity within the
old 240-step preflight horizon. Tasks `t100459` and `t100464` failed at that
guard, and retry `t100471` was cancelled. No result or performance observation
was produced. The repair increased only the preflight horizon to 300 steps and
added explicit opportunity-coverage tests; the formal protocol is unchanged.

## Audit

- The runtime matched the frozen Gymnasium, Gymnasium-Robotics, MuJoCo,
  PettingZoo, SciPy, and PyTorch versions and used CPU execution.
- The controller retained the registered 267,018-parameter architecture and
  completed 32 finite optimizer updates.
- The registered seed roles contained one training, one selection, two
  branch-fit, and two held-out branch-evaluation paths.
- Fit and evaluation each contained exactly 12 rows: one opportunity from each
  of six classes on each of two independent paths.
- The predictor feature counts were 170 for the current-only quadratic, 170
  for the generic causal-dynamic quadratic, and 39 for the registered causal
  validity interactions. Every model selected its ridge alpha using two-group
  leave-one-path-seed-out fit data with finite losses.
- Every keep/renew pair had zero maximum prefix and feature difference. Keep
  made zero upper calls, renew made one, both made zero downstream upper calls,
  and the lower controller remained closed loop.
- No branch row or predictor contained privileged regime context. Event class
  and timing remained diagnostic metadata rather than predictor inputs.
- Extra branch supervision was reported separately as 3,576 fit and 3,682
  evaluation primitive steps. No checkpoint, CSV, or trajectory was copied.
- Scheduler placement was dynamic across `node001`-`node006` with no required
  node; this task happened to run on `node002`.

The one-root analyzer correctly returned `stage9_not_authorized` with unbounded
confidence intervals. Its point estimates are not performance evidence and do
not change the registered formal matrix.

## Next Step

Run exactly the frozen roots `207011`, `207023`, `207037`, `207049`, `207061`,
`207073`, `207089`, and `207101`. Analyze only after all eight compact results
are complete; do not append roots after inspection.
