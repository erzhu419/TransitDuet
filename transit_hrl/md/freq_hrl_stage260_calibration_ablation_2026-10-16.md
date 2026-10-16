# Stage260: Fixed Calibration Prefix Fits

Frozen path: cached full60 -> prefix54 with full60 calibration -> cached
recalibrated prefix54. Same 16 training futures, prior/basis/cap and old keys.
Only the middle endpoint is fitted; no raw/full-model training or selection.

Reliability, state-balanced precision and lambda are frozen/sliced from full60.
All 33 scheduler tasks completed exit 0, no retries; eight tests passed.
640 prefix fits / 5,120 solves and 512 cached-full KKT checks; max KKT 6.34e-12.
Stage259 reproduced within 3.47e-18; path residual at most 1.39e-17.
Pulled an 8,575,025-byte scalar report and small preflight receipt only.

Mixed expanded8 / pooled-safe removed group, held means (12 bank-cells/family):

| Effect | free_old | shrunk |
|---|---:|---:|
| Fit-population cost | +0.000048977 | -0.000003501 |
| Recalibration cost | +0.000028226 | +0.000046502 |
| Fit-population wait/min | +0.000073290 | +0.000106736 |
| Recalibration wait/min | +0.000061988 | +0.000070174 |

Fixed-calibration held cost-prediction RMS gaps remain 0.001829 / 0.001838
versus total 0.001880 / 0.001880. Calibration matters for outcomes, but does
not remove fit-population sensitivity. No cost/wait dual improvement established.

Limitations: full calibration uses the excluded prefix's training labels, so
this is conditional diagnosis, not strict OOF or a deployable policy. Ordered
path effects are not unique causal shares; RMS/action rates do not add. Removed
groups are selection-conditioned; bank0 shrunk recalibration improves waiting,
unlike the pooled mean. Development/repeated exposures are not seed CI.
Stage253/248/169 closed; no deployment/default change.

Next: reuse saved models/calibrations to inspect weighted curvature, prefix
leverage and response directions under fixed calibration. Determine whether
remaining changes concentrate in weakly identified directions before another
representation or penalty change; no new fitting or simulation needed.
