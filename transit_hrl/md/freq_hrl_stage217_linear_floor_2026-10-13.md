# Stage217: fixed-feature training bounds

Weighted zero-ridge SVD projection on the same98 cost features, all600
mean8 targets and original240 weight8/360 weight1. Keep training scales,
three RHS and frozen mean8 GLM. Decompose physical training cost-pair MSE
into projection residual floor and ridge excess; evaluate both frozen-ridge
and projected cost selectors on unchanged47 folds/original240 held states.

Training-original240 only: independently probe both future partitions for
linear representability of stable true-safe cost order (98 coefficients)
and stable wait signs (199 coefficients including GLM bias). Minimum-margin
deficit LPs in the numerical SVD column space include nonnegative slacks;
dual lower bounds distinguish a positive minimum from zero. Witnesses are
references, not policies; safe logits<=0, unsafe>=1, strict cost-order margin1.
Numerically unresolved cases remain separate; they do not discard value bounds.

One unbound node001-node006 task,1CPU/3072MB;47 projection fits, at most188
single-thread LPs, no gradient/actor updates or native episodes. Reference
models/witnesses and data remain remote;small report only. No model promotion.

Initial r1 t151734 failed on HiGHS Unknown, not an expressivity conclusion.
Solver probes t151757/t151761 reproduced failures at ~1e-17 coefficients;
v2/r2 uses the always-feasible slack formulation instead of hard feasibility.

Completed r2 t151780 on node004, exit0, source86657a487b. Seven tests;
47 projections/188 solved LPs, zero unresolved bounds; physical controls
reproduced. Only396982B report plus4386/5568B solver summaries pulled.
Mean per-fold projection/ridge training-MSE ratios: scene88.63%, regime86.97%,
root82.61%. Root401 held MSE0.001072->0.017486 under zero ridge; no promotion.
Exact stable cost-order fit: first/last8/47, even/odd3/47; wait-sign fit4/47,
3/47. Both root-only training folds fit both targets in both partitions;
mixed-root scene folds largely cannot. Sample/constraint counts differ.

Next: matched-prefix-count single/mixed-root bounds, and best-action-only
constraints instead of every pair order. This distinguishes sample-size
effects and unnecessary exact-fit requirements before choosing nonlinear
state-action conditioning versus a decision-focused training objective.

Limits: empirical bounds for the numerical column space of the linear class and
finite mean8/stable-label training set, not impossibility for nonlinear
features or population truth. Exact order/sign fit is stronger than a useful
selector requires. Unregularized references can overfit; held
labels do not choose rank, coefficients, normalization, margins or thresholds.
