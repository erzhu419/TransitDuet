# Stage247: state-balanced paired GLS

Freeze all Stage246 controls, B targets/shrinkage,198 features,17 candidates,
TSVD ranks1/3/8/16, head contrast, A RMS cap, prefix/future OOF selection and gate.
Primary `state_gls_pooled_safe` normalizes total effective weight to2 per active
state, preserving all within-state pair/head confidence ratios. Secondary
`state_head_gls` sets each active state/head's mass to1, preserving pair ratios
within a head. Inactive states have no feasible pairs and zero contribution.

One real-cache cell precedes the32-cell matrix and report. Seal fits/actions
before fresh-future evaluation. Reproduce all Stage246 selected controls and
metrics. Compare train/held-prefix cost/wait, pair-sign errors and fleet events.
Use paired conditional MC for each new arm minus frozen pair-GLS, sharing states
and future draws. No raw-reference refit, native replay or new labels.

Next: test whether removing cross-state weighting helps without confusing it
with service/risk reallocation. The previous weight audit found no consistent
root401 downweighting; balancing is an intervention, not an established diagnosis.
Limits: reused development data/conditional MC, not seed CI or end-to-end HRL.
Defaults and reserves unchanged; full models remain remote.
