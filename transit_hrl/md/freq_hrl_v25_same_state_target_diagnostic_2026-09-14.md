# V25 Same-State Target Diagnostic

The next step after v24 is a controlled objective diagnosis. It uses the
unchanged terminal-reserve projector and fresh random roots, with no saved
policies, retired experimental paths, or environment reward evaluation.

The fixed panel crosses dimensions 3/6, raw Gaussian mean amplitudes 0/1/2.5,
standard deviations 0.2/0.5, and four roots generated from NumPy seed 250091.
Each of 48 cells takes snapshots after 0, 32, and 64 causal prefix steps.
Upper proposals are held for 16 prefix steps. At each snapshot, 8 upper and
16 lower independent samples are crossed while projector history is fixed.

Recorded diagnostics separate upper, lower, and interaction contributions to
raw-space and action-space target variance. The lower-mean plug-in target is
compared with the finite-sample conditional mean, including its Monte Carlo
sampling variance. This is not an exact population bias estimate.

Three gradients use identical samples and projected targets:

1. Current raw-mean fitting: `||mu - atanh(projected_action)||^2`.
2. Fixed-sample raw fitting: `||z_old + mu - mu_old - raw_target||^2`.
3. Fixed-sample action fitting:
   `||tanh(z_old + mu - mu_old) - projected_action||^2`.

An identity-projector control measures gradient noise when no action needs
correction. The action-space formulation must have zero gradient there, and
its derivative must match finite differences. The stochastic residual remains
fixed during an actor update; recomputing its reference mean after each
minibatch would change the objective.

Run one operational preflight and then the registered matrix through
scheduleurm on node001-node006, one CPU and 768 MiB per cell. Only aggregate
JSON is exported. Interpretation is mechanistic: this panel cannot establish
policy performance, choose reward thresholds, or authorize a confirmation run.
Any subsequent training comparison needs a fresh frozen development protocol.
