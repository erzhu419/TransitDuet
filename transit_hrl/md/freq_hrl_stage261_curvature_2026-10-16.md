# Stage261: Fixed Calibration Curvature

Reuse Stage260 full/fixed models, calibration and old keys. Inspect the actual
weighted profiled Hessian, joint-head prefix leverage and RMS-cap multipliers.
Low block: numerical null plus lowest quarter of positive modes, extended over
ties. Decompose prediction response into low/rest/cross; no additive RMS shares.

One single-core scheduler task; 32 cached cells / 640 fits / 1,280 primary
prefix-model records. No fitting, selection, evaluation panels or simulation.
Reproduce cached primary actions and Stage260 fit-population cost RMS; verify
the cached quadratic influence equation. Raw ledgers stay remote.

Limitations: development/frozen-calibration diagnosis, not accuracy or utility
proof. Conditional KKT curvature is not constrained sampling covariance. Repeated
exposures are not independent seeds. Stage253/248/169 stay closed.

Next: decide whether a directional stability constraint is warranted from the
response geometry before changing the representation or fitting loss.
