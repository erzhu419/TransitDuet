# Stage217: fixed-feature training bounds

Weighted zero-ridge SVD projection on the same98 cost features, all600
mean8 targets and original240 weight8/360 weight1. Keep training scales,
three RHS and frozen mean8 GLM. Decompose physical training cost-pair MSE
into projection residual floor and ridge excess; evaluate both frozen-ridge
and projected cost selectors on unchanged47 folds/original240 held states.

Training-original240 only: independently probe both future partitions for
linear representability of stable true-safe cost order (98 coefficients)
and stable wait signs (199 coefficients including GLM bias). Minimum-L1 LP
witnesses are references, not policies; safe logits<=0, unsafe>=1, strict
cost-order margin1. Feasible/infeasible and numerical failures are distinct.

One unbound node001-node006 task,1CPU/3072MB;47 projection fits, at most188
single-thread LPs, no gradient/actor updates or native episodes. Reference
models/witnesses and data remain remote;small report only. No model promotion.

Limits: empirical numerical bounds for the registered linear class and
finite mean8/stable-label training set, not impossibility for nonlinear
features or population truth. Unregularized references can overfit; held
labels do not choose rank, coefficients, normalization, margins or thresholds.
