# Stage240: selection stability versus refit consistency

Effective run: `native_transit_refit_consistency_stage240_development_20261015_r2`; fits `t153531`-`t153562`, aggregate `t153565`, all done/0. Five unit tests passed. R1 serialization failure and cancelled retries are archived separately; the fix converts deletion eligibility to Python bool.

## Design

All20 Stage239 candidates/cell frozen; no reselection. All32 outer cells and20 prefix/future matched blocks/cell retained. Five endpoints separate residual state expansion, cached raw-reference replacement, residual future-draw expansion, and final cached reference replacement. All raw references reused; zero raw training. 18,432 objective fits, 1,408 analytic all-MSE paths. All12,800 OOF probe outcome/count records and6,720 full endpoint metric records reproduced exactly; subspace prediction error <=4.99e-16. Only the8,289,145-byte aggregate was pulled; models/panels remain remote.

## Results

Mixed pooled, frozen joint/raw/half-safe candidate, held-state/other-future deltas. Negative is better; common raw is the native60/4 raw controller.

| Endpoint | Cost vs own raw | Wait vs own raw (min) | Cost vs common raw | Wait vs common raw (min) |
|---|---:|---:|---:|---:|
| Cached54/2 | +0.000057301 | -0.000083294 | +0.000824005 | +0.001523073 |
| Residual60/2, fixed54 reference | +0.000027809 | -0.000202708 | +0.000794512 | +0.001403659 |
| Native60/2 | -0.000075849 | -0.000246484 | +0.000407942 | +0.000511719 |
| Residual60/4, fixed60/2 reference | -0.000004354 | -0.000247135 | +0.000479437 | +0.000511068 |
| Native60/4 | +0.000077710 | -0.000333333 | +0.000077710 | -0.000333333 |

Two problems coexist. Six of7 admitted mixed cells lose eligibility after at least one prefix deletion (10/70 deletion checks fail);12/14 source-cell cost benefits have top1-prefix share >=80%. The frozen54/2 candidate already worsens held cost before any refit, despite improving OOF probe cost/wait. Selection/transfer failure therefore precedes refitting.

Reference changes also dominate action changes: for joint/raw/half-safe on held states, residual-only state/draw refits switch0.90%/1.67% of actions; native reference replacements switch22.36%/31.30%. The post-hoc subset of7 admitted mixed cells shows the same pattern (2.06%/3.81% versus22.14%/32.02%). Yet preserving the old reference is not a performance repair: every endpoint still loses held cost to common raw, and native60/4 has the smallest pooled cost harm.

The prior `mixed_b/fold1/even_odd/direction1` rank3@1 regression survives all10 prefix deletions. Its source401 OOF cost benefit is99.66% concentrated in one prefix. Its cached54/2 held cost delta is already +0.000710384 versus own raw. Another deletion filter alone would retain this failure.

## Limitations And Next Step

Reused development states/futures, no fresh CI. Equal matched-block means are not independent replicas. Refits include basis/scales/RMS changes; reference replacement includes head/gate/coordinate changes. Prefix probes become training-exposed at later endpoints and are diagnostic only. Defaults unchanged.

Next: freeze current candidates and models, resample independent futures on the existing training-prefix states first, and test whether the original OOF gains survive without any refit. Keep reserved confirmation cohorts closed and raw references unchanged. Use that result to distinguish noisy selection from reproducible prefix-transfer failure before changing the training/deployment pipeline; do not widen the encoder or add another eligibility threshold yet.
