# Stage9 Timing-Feature Ablation

[Frozen plan](../protocols/plan_validity_stage9_scale_ablation_2026-11-12.json).
Reuse eight shared controllers and their 192 branch labels per root. All
three heads use the same eight instantaneous plan features and four
additional channels. Current uses target position and current force;
single-scale uses 100ms target velocity and 100ms force RMS; mixed-scale
changes only the velocity horizon to 500ms. This isolates that window
substitution, not frequency-separated upper/lower credit or all single
horizons. Stage9 is a frozen descriptive anchor, not a fitted treatment.

Twelve raw inputs, 90 quadratic terms plus intercept, slope DF8, common
leave-one-path-out q75 rule. Fit 216 ridges including folds, no controller
updates or simulation. Freeze all 24 heads before any new path access.
Evaluate on 64 fresh paired paths per root, 2,048 complete episodes at
24 calls each, 2,457,600 native steps. No original HOLD paths.

Mixed-scale support requires improvement against both matched alternatives:
nominal paired root t95 lower bound above zero, at least 2% pooled episode
ISE improvement, and positive effects in both path halves. Apply the same
criteria to the separate single-scale/current comparison. OOF MSE, tail ISE
and latency are descriptive, not replacements for episode ISE. No resizing
the cohort or selecting new windows after evaluation.

## Completed Result

[Fit summary](../results/pointmaze_stage9_scale_ablation_20261112/fit_summary.json),
[evaluation](../results/pointmaze_stage9_scale_ablation_20261112/summary.json)
and [exit records](../results/pointmaze_stage9_scale_ablation_20261112/completion.json).
t214870-t214878 all exited 0. Fit: 216 ridges, 0.53 CPU seconds, zero steps.
Evaluation: 2,457,600 steps, 49,152 calls and 2,231.13 CPU seconds. Only
61,065 fit and evaluation summary bytes were retrieved.

| Contrast | Relative ISE gain | Absolute gain [nominal root 95% CI] | First / last half | Favorable roots |
| --- | ---: | --- | ---: | ---: |
| Single-scale vs current | +6.44% | +0.07906 [0.01870, 0.13942] | +5.58% / +7.33% | 8/8 |
| Mixed-scale vs single-scale | -5.85% | -0.06710 [-0.11534, -0.01886] | -5.37% / -6.34% | 1/8 |
| Mixed-scale vs current | +0.98% | +0.01197 [-0.04469, 0.06863] | +0.51% / +1.46% | 4/8 |

The single-scale/current criterion passes; the mixed-scale joint criterion
fails. Mean ISE is 1.22699, 1.14792, 1.21502 and 0.98527 for current,
single-scale, mixed-scale and original Stage9. The original is descriptively
14.17% better than the best matched head; its additional features and fitting
recipe are not isolated here. Mean early renewals are 15.04, 14.51, 15.13
and 13.58, with exactly 24 calls each. Maximum root p99 is 1.73ms across
methods; one Stage9 step exceeded 10ms.

The tested short-history view improves closed-loop timing. Replacing its
100ms target velocity with 500ms worsens ISE at fixed capacity. This does not
rule out other multiscale representations or identify the source of the
original Stage9 advantage. Keep Stage9 frozen, retain this boundary, and
do not tune more windows on these paths. Original learned-only HOLD stands.
