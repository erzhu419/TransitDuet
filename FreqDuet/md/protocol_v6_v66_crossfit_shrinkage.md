# V66: whole-day cross-fitted residual shrinkage

Freeze V59 actors, dual, upper/B200 deployment and V65 production value weights.
Use only V64's 32 training days. Four deterministic folds hold out day indices
`day_index % 4 == fold`, consistently across actors. Per actor/fold, fit both
V65 residual architectures on 24 days, predicting the other eight. Prior,
standardizer and targets use only those 24 days. Preserve full causal inputs,
2,048 steps, batch 256, original LR and paired minibatches. This adds 32 OOF
model fits; do not claim equal total compute to the original V65 fits.

For each actor/model, estimate one scalar alpha in [0,1] minimizing day-balanced
OOF MSE: alpha = clip(mean(r*y)/mean(r*r), 0, 1), where r is the excluded-day
residual prediction and y is terminal CV-squared minus that fold's fitting-day
prior. Zero residual variance chooses alpha=0. This is a convex combination,
not a new deployment gate. Save quadratic day statistics, not OOF checkpoints.
Reuse frozen 32-day V65 values for production: prior + alpha*residual - exact
completed prefix. Alpha=0 recovers the prior; alpha=1 recovers the raw model.

Lock all four calibrations before new-data collection. Then run a full-day
collection/scoring smoke and reuse that day in the fixed confirmation matrix:
32 fresh demand days 760001-760032 per actor, private sampling seeds
`6600000 + case_index * 1000 + day_index`, 128 unique rollouts. Preserve all 11
V65 control channels. Primary: shrunk MLP minus prior. Secondary: each shrunk
model minus its raw version, shrunk linear minus prior, shrunk MLP minus linear.
Average actors within demand day before 10,000-resample, 32-day paired CIs.

Keep the strict gate: each calibrated candidate beats all 11 old controls on
both MAE/RMSE on every fresh day, in all four contexts. No control admission or
actor/dual updates here; calibration passing could admit a later action audit.
All tests, fitting, collection, scoring and statistics use scheduler on
node001-node006. Pull only logs/markers/compact JSON; keep state arrays remote.

Limitations: coefficient-fitting OOF errors are not confirmation evidence;
24-day fold residual scales may transfer imperfectly to 32-day full models.
Intervals condition on the fitting set, calibration and four actors (two
original training seeds). Prediction gains do not establish action effects.
