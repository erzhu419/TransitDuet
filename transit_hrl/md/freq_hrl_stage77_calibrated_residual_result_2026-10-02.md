# Stage77 result: harm attenuated, not eliminated

Historical command-response calibration substantially improved the original
sampled residual curve, but remained worse than the zero-residual controller.

Preflight t118908/t118909 and full t119609-t119617 all exited 0. Six focused
tests passed. Full evaluation covered eight frozen roots, 32 fresh seeds/root,
1,536 native episodes / 1,843,200 steps. All 12 reward/tracking contrasts used
equal-root bootstrap with Bonferroni12 correction.

| Period | Reward contrast | Mean | Corrected CI |
|---|---|---:|---|
| 50 | Original - zero | -154.1862 | [-163.3030, -143.0893] |
| 50 | Calibrated - original | +152.1922 | [141.5470, 161.8918] |
| 50 | Calibrated - zero | -1.9940 | [-3.0068, -0.9174] |
| 100 | Original - zero | -75.6008 | [-82.2789, -64.0247] |
| 100 | Calibrated - original | +72.0834 | [62.3365, 78.9334] |
| 100 | Calibrated - zero | -3.5174 | [-5.7041, -0.7454] |

Tracking error also improved relative to original but worsened relative to zero
at both periods, with corrected CIs excluding zero. Descriptively, calibration
removed 98.7%/95.3% of the mean reward harm. Velocity above the BC speed q99 fell
from 40.94% to 2.36% (P50), and 12.71% to 1.75% (P100).

## Limitations / Next Step

The one-shot ratio did not enforce its nominal command-fidelity scale:
actual historical command-change RMS was 2.22-2.61 times BC RMSE at P50,
1.24-1.54 times at P100. This is nonlinear response, not a passed constraint.
All actors/critics/Adam and forecasters stayed frozen; no traces/checkpoints
were written. This validates a decoder repair, not learned HRL or frequency
superiority. Stage67 HOLD remains.

Next use actual historical nonlinear command response as the constraint,
retaining the same BC-derived target and contracting the coherent curve until
it is met. Freeze that solver before new native seeds; no reward-selected
scale sweep or policy adoption. Compact evidence is in
`results/pointmaze_calibrated_residual_stage77_full_20261002_r1/qualification_compact.json`.
