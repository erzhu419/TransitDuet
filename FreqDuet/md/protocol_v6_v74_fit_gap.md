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

Next: distinguish insufficient fitting from in-sample optimism/support shift,
then choose structural features or independent training-day expansion.

Limitations: same consumed 32-future mean labels and overlapping fits. This
cannot separately identify label noise, model misspecification and finite-day
estimation error, and is not new-day or controller confirmation.
