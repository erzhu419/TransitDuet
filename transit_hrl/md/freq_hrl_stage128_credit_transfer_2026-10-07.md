# Stage128: Native Credit Transfer Across Scenes

Stage127 validates the local-to-episode gradient sum, but complete credit loses
to coarse at root410011 period50 and remains inactive at root410023 period100.
Test cross-scene overfitting, not more time labels or control authority.

Reconstruct Stage127 zero-policy states from its server-only native cache.
Compare the unchanged raw392 mean against a fixed26-dimensional causal
subspace: physical/tracking state6, clocks2, and current/mean/OLS-trend values
of all six stream channels over64 historical samples. This is noninvertible
compression, not an orthogonal coordinate swap. Map compact weights back into
the same392-to8 actor; no inference-time new sensor, search or lower change.
Both methods use the same labels, unit damping and original-state Fisher
radius0.000555556. Raw and compact optimizer metrics differ by construction.

Leave each of four label scenes out, including both of its noise panels. Fit
on the other three; sum held-out local credit and verify whole-policy secants
at fixed0.1 scale. These folds are diagnostic, not hyperparameter selection.
Final directions use all four scenes, followed by identical fits on16 new
calibration scenes/two panels and32 separate evaluation scenes. Keep both
periods50/100, both roots, all negative folds, opposite/blinded controls and
the0.5 practical-gain threshold. Per root:288replay+64fold-audit+320training+
192crossfit+384evaluation=1,248episodes/1,497,600new steps; inherited Stage127
cost reported separately.16workers+parent/12GiB, dynamic node001-node006.
Only compact JSON returns locally; four final upper weights remain server-side.

## Limitations

Four scenes/two development roots cannot establish generalization at paper
scale. The Stage127 scene-transfer hypothesis is not a settled cause; its
label audit and calibration used different perturbation scales. Compression
may remove useful information and must earn its gain on
new scenes. This does not change prior negative gates or establish learned
promotion, frequency-specific value or joint-HRL training.

## Run Receipt

Code `0ef1c8a606`, registration `67db90dff0`; four focused tests and four
Stage127 regression tests passed. Run
`pointmaze_credit_transfer_stage128_pilot_20261007_r1`: `t136457` root410011
and `t136458` root410023 accepted, initially queued. Full task specifications,
fresh rosters and inherited/new budgets are saved in `preregistration.json`.

## Completed Result

Both tasks completed:2,496episodes/2,995,200new steps. Compact-minus-forecast
is+0.935483/+0.524526(root410011 periods50/100),+0.919379/+0.219561(root410023).
Compact-minus-raw is+0.759954/+0.450027 and+0.795817/+0.270056. All compact
steps are active and all eight calibration crossfits are positive. Equal-root
gains+0.927431/+0.372044 clear0.5 only at period50; two development roots do
not provide independent confirmation.

Compact held-scene native secants are positive in15/16folds, raw in12/16.
Keep the negative compact fold(root410023,p100,scene127200004,-0.074825) and
the sign disagreement(root410011,p100,scene127100002: prediction-0.212894,
native secant+0.673550). Local credit is not a uniform held-scene bound.
Freeze this method next; replicate on six other source-policy roots with
fresh label/calibration/evaluation scenes and root-cluster corrected CIs.
