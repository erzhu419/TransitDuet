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
