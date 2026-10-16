# V74: frozen-fit generalization diagnosis

Read only the cached V72/V73 fits and V73's unchanged prepared V70 labels.
No fitting, simulation, actor/dual/production-value update, gate change or model
selection. First verify loaded models reproduce all excluded-day errors and
action choices; stop on a mismatch rather than interpret a different predictor.

For each original prefix, average its errors in the eleven fits that included
its demand day; compare with its one whole-day-excluded fit. Report all six
channels, three continuous outputs, MAE/MSE, primary-minus-original contrasts
and fitting/held gaps. Cluster CIs by the twelve original demand days; repeated
in-sample fits are not new observations. Describe each fixed actor and window.

Use saved training-only fleet+plan scalers for feature support. Match actor and
window, exclude the query day, then report per original/fleet/plan block:
nearest variable-feature RMS distance, fraction outside matched training ranges,
and novel values on matched-training-constant columns. In-sample queries have
ten matched days; excluded-day queries have eleven. Geometry is descriptive,
not a new admission score. Keep per-context details server-side.

Run focused loader/pairing/support tests, actual first-fold cached reproduction,
twelve parallel read-only folds and aggregation via scheduler on node001-node006.
Pull only small summary JSON. V70/V73 development gates remain failed.

## Outcome / Next

Source `aff0f5ead7`, scheduler `t165771` on node004, 12 workers, exit 0/no retry.
Four tests and all 144 cached held-prefix error/choice identities pass.
Passenger MAE (in-sample / excluded day): original 1652.58 / 1997.77;
fleet+plan 1439.20 / 2262.90 person-seconds. Primary-minus-original is -213.38
[-455.45,-43.29] in-sample but +265.12 [145.96,383.25] excluded-day.
Excluded MAE worsens on 10/12 days, all four actors and all three windows.
Plan nearest RMS rises 0.261 to 0.305; fleet 0.923 to 0.941. Matched-range
violations/novel-constant counts do not increase overall, with the unequal
reference counts noted above. Geometry does not establish absence of shift.
Only 131193 bytes of summary JSON were collected; no fits or rollouts added.

Next: use fixed low-dimensional, controlled-vehicle-linked turnover/dispatch
commitment features and more independent training days. The current evidence
favors in-sample optimism over failure to fit, not a single anomalous context.
Keep actor/dual/main frozen and preserve failed gates; no posthoc winner.

Limitations: same consumed 32-future mean labels and overlapping fits. Day CIs
are descriptive conditional on saved fits, not retrained-bootstrap inference. This
cannot separately identify label noise, model misspecification and finite-day
estimation error, and is not new-day or controller confirmation.
