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

Completed preparation`t147970`,fits`t147971-t148017`,report`t148028`,all exit0.
94 wait fits,3444 L-BFGS iterations;17 distinct unit tests passed. Only one
666532-byte report pull; no draw,bank,target,model or checkpoint pulls.
All variants pass cost but fail wait. Primary lambda2 is not promoted.
Symmetric scene cost/wait improves from -0.002193371/-0.002673958 min to
-0.002584361/-0.005930729, with both roots improving. However regime/root397
wait is +0.000179167 and cross-lower/root401 +0.002670833 min. Lambda2 reduces
scene false-feasible choices74->7 but overrides133->16; first/last repeated
false-unsafe candidates93->270, and cost savings drop to -0.000966247.
Next: retain symmetric loss as an exploratory control. Diagnose training
versus held-out feasibility errors before adding causal lower-response/state
features; no larger lambda or post-hoc threshold changes.

Limits: the GLM share is magnitude-weighted, not a calibrated probability of
harm. The unrestricted, unregularized pointwise optimum recovers expected-wait
feasibility;lambda2 requires negative mass to cover twice positive mass. Finite causal
models may still misclassify. This remains grouped supervised development,
not independent-seed CI or end-to-end learned HRL validation.
