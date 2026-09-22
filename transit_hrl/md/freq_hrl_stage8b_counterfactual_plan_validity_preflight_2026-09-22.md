# Stage-8B Counterfactual Plan-Validity Preflight

Date: 2026-09-22

Run: `pointmaze_plan_validity_stage8b_v1_preflight_20260922_r1`

Task: `t100429`

## Outcome

The registered cell completed on `node004` in about 36 seconds. Only its
141-KB `result.json` was synchronized. The preflight passes and authorizes the
unchanged eight-cell development matrix. It is software evidence only.

## Audit

- The runtime used Gymnasium 1.2.0, Gymnasium-Robotics 1.4.2, MuJoCo 3.2.7,
  SciPy 1.13.1, and PyTorch 2.5.1+cu121 with `device=cpu` and an MLP policy.
- The history controller had 267,018 trainable parameters, retained three
  compact checkpoint records, and completed 32 finite optimizer updates.
- The predictor-fit and qualification paths each produced exactly six rows:
  one per registered opportunity class. The feature schema had 37 causal
  values and was stored once rather than repeated per row.
- Every keep/renew pair had zero maximum prefix and feature difference. Keep
  made zero opportunity calls, renew made one, both made zero downstream upper
  calls, and both retained closed-loop lower control.
- Candidate features had no future event or regime-label access. Current true
  regime entered only the separately named diagnostic ceiling predictor.
- Extra branch supervision was accounted as 1,628 replayed primitive steps for
  predictor fitting and 1,902 for qualification evaluation.
- The result directory contains only preregistration, result JSON, sync
  manifest, and compact analysis outputs. No checkpoint or raw trajectory was
  copied locally.
- Scheduler placement was dynamic across `node001`-`node006` with
  `require_node=null`; the task happened to run on `node004`.

The one-root analyzer returned `stage9_not_authorized`, with unbounded
single-root intervals as intended. Its numerical signs are not performance
evidence and were not used to alter the frozen development protocol.

## Next Step

Run the fixed roots `206009`, `206021`, `206033`, `206047`, `206063`,
`206071`, `206087`, and `206099`. Analyze only after all eight result files
are complete. Do not add roots after inspection.
