# Stage261: Fixed Calibration Curvature

Reuse Stage260 full/fixed models, calibration and old keys. Inspect the actual
weighted profiled Hessian, joint-head prefix leverage and RMS-cap multipliers.
Low block: numerical null plus lowest quarter of positive modes, extended over
ties. Decompose prediction response into low/rest/cross; no additive RMS shares.

t165633 completed on node006, exit 0, no retries; eight tests passed.
32 cached cells / 640 fits / 1,280 primary records; no new fitting or panels.
Actions/RMS reproduced within 1.17e-15; influence residual at most 6.58e-14.
Only the 841,874-byte scalar report was pulled; raw ledgers remain remote.

Mixed removed group, 12 bank-cells / 120 active prefix-models per family:

| Quantity | free_old | shrunk |
|---|---:|---:|
| Mean low-block parameter-energy fraction | 65.8% | 68.0% |
| Excluded low response RMS | 0.001602 | 0.002307 |
| Excluded rest response RMS | 0.004830 | 0.005031 |
| Held total response RMS | 0.001829 | 0.001838 |
| Held rest response RMS | 0.001174 | 0.001154 |

Mean low-block dimension is 45.1% after ties/null modes. Maximum block leverage
is 0.998 / 0.997; excluded inverse-curvature exposure rises 12.4 / 13.1 times.
Low modes matter, but do not dominate every prediction response. Bank0 is more
low-mode dominated; bank1 has larger remaining-direction effects.

Limitations: development/frozen-calibration diagnosis, not accuracy or utility
proof or a policy ablation. Conditional KKT curvature is not sampling covariance;
RMS parts include cross terms, and repeated exposures are not seeds.
Stage253/248/169 stay closed; no deployment/default change.

Next: cached KKT decomposition of data-removal forcing versus the RMS-dual-change
term. Prefix duals are positive in 107/120 cases in both families. Separate
constraint redistribution from data sensitivity before choosing a directional
penalty; no new fits or simulation needed.
