# Stage223: inner/final target precision

Cross two/four-draw calibration models with two/four-draw final models. Reuse Stage222 two-draw OOF predictions, its legacy two-draw validation calibration and final four-draw heads. New c2v4 and c4v4 calibration use the same four-draw truths of entirely excluded prefixes; outer evaluation futures/prefixes never select parameters.

Each architecture keeps 600 updates per fit. New budget: 320 inner4 + 64 final2 bundles; 384 analytic cost/GLM fits, 1,152 wait fits (691,200 updates), 384 scalar calibrations, zero new rollouts or actor updates. Final2's complementary half-models are separate sensitivity controls, not an ensemble; final4 is repeated for matched comparisons. Raw/model files stay remote.

## Status

Five new tests and eight Stage222 tests passed (13/13), including common validation magnitudes, real precision fits, outer label isolation and correlated coverage accounting. Scheduler dispatch pending. Default/native policy unchanged.

## Next Decision

Compare legacy2->final2 versus legacy2->final4, then c2v4 versus c4v4 on the same final4 model. Inspect train_same and held_other cost/wait floors plus proper risk. This separates final fitting precision from validation-target precision without enlarging the model class.

## Limitations

c2v4/c4v4 hold out entire prefixes, not additional inner future samples; outer other-half evaluation remains independent of fitting. Inner models still use 54 states versus final60. Subhalf and future comparisons reuse 240 states/eight draws, not new independent seeds or full learned HRL.
