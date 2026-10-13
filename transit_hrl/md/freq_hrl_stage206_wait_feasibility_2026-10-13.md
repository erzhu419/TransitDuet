# Stage206: learned draw-level wait feasibility

Freeze Stage204 cost heads, six candidates,47 grouped folds and the600/240
training/evaluation cohort. Replace wait regression with199-parameter GLMs
over the unchanged198 causal features plus an intercept. Let a,b be mean
positive/negative wait magnitudes over available future draws. Minimize
lambda*a*softplus(-z)+b*softplus(z), with count8/1 and training-only scales.
Symmetric lambda1 is the control; fixed lambda2 is primary. Use share<=0.5
then frozen cost minimum, preserving exact identity and ties. No tuning.

Run: `native_transit_wait_feasibility_stage206_development_20261013_r1`.
One preparation,47 fit tasks/two models each,one report;1CPU/3072MB unbound
node001-node006. Draws,targets,models and predictions stay server-only.
No new cost fitting,actor updates or native episodes; report actual L-BFGS
iterations. Compare mean8 outcomes and both future-half directions, retaining
root/regime summaries, false-feasible choices and all negative results.

Results pending. Next decision follows learned gates and high-noise/burst
outcomes, not the label-informed Stage205 oracle or threshold retuning.

Limits: the GLM share is magnitude-weighted, not a calibrated probability of
harm. The symmetric population optimum recovers expected-wait feasibility;
lambda2 requires negative mass to cover twice positive mass. Finite causal
models may still misclassify. This remains grouped supervised development,
not independent-seed CI or end-to-end learned HRL validation.
