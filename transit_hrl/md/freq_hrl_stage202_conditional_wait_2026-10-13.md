# Stage202: frozen-cost conditional wait credit

Keep both Stage199 count-weighted cost heads and their rankings unchanged.
Reuse frozen wait as control; fit geometry-only (102 coefficients) and
queue/load-conditioned wait (198 coefficients, primary). Add four prefix
geometry moments and12 causal conditions interacting with eight plan features.
Conditions use queue, onboard/load, concentration, active supply, estimated
return time and LF forecast; no root/scene/regime IDs or future observations.
Fit all15 candidate pairs per training state with ridge0.1, mean8 weight8 and
singleton weight1. Input scales and wait pair-RMS use outer training only.
Zero wait margin, exact identity0, unchanged strict cost/wait gate on both
roots and all three splits; no margin, capacity or hyperparameter search.

Run: `native_transit_conditional_wait_stage202_development_20261013_r1`.
47 fit tasks plus one report,94 wait fits and47 reused cost models;
0 optimizer/actor/native updates,1CPU/3072MB each, unbound node001-node006.
Models, bank, targets and predictions stay server-only; small reports only.
Tests:22 passed, including train-only scales, held-out-label isolation,
physical wait units, exact frozen costs and all47-fold identity gate.
All47 fits`t147610-t147656` and report`t147658` completed with exit0;
94 new wait fits, no native episodes. Only the26477-byte report was pulled.
All three variants pass cost but fail the complete wait gate.
Conditional wait repairs scene/root397: +0.001229167 to -0.003104167 min.
Scene overall cost improves from -0.001641093 to -0.002508395 and wait from
-0.001316667 to -0.005318750 min. Transfer remains weak: regime/root397 wait
is +0.001373958 and lower-root/root401 is +0.002986458 min. Wait pair MSE/zero
improves in scene (0.94998 to0.92502) but worsens in regime (1.09917 to1.13203).
Geometry-only also fails wait, despite passing cost. No variant is promoted.

Next: test stronger shrinkage of the added conditional block, selected using
training-only regime-held-out wait errors. Retain frozen costs, outer splits,
mean8 cohort and strict gate; do not tune margins or select on outer outcomes.
Limits: larger feature sets are not matched-capacity controls; this remains
supervised single-intervention development, not CI or full learned-HRL proof.
