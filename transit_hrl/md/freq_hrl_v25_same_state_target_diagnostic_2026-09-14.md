# V25 Same-State Target Diagnostic

> Result update, 2026-09-19: the full 48-cell development run completed and
> failed the registered validity, reward, correction, and Hopper correction
> gates. See
> `freq_hrl_mujoco_v25_sample_consistent_upper_result_2026-09-19.md`. The
> protocol text below is retained as the preregistration record.

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

Operational preflight t93594 completed in 6.26 seconds with exit code zero.
Scheduler's short-job classifier did not recognize its custom terminal log
message. The full panel uses the recognized `DONE` marker; the preflight
remains an operational record outside the full-panel results.

## Result and Next Experiment

Full-panel t93601-t93648 completed: 48 JSON exports (874,029 bytes), 144
snapshots, with unchanged projector history throughout. Source revision is
`c11756d306e518b65d386e499798a24a50d8622f`; the run is
`results/mujoco_v25_same_state_target_noise_20260914_r2`.

At prefix lengths 32/64, upper sampling accounts for 97.30%/94.72% of the
upper raw-target variance summed over cells; lower sampling accounts for only
0.52%/1.29%. Relative summed gradient variance for raw-sample versus raw-mean
is 1.380/1.149; action-sample is 0.445/0.364. This does not support a blanket
claim that simply replacing mean fitting with sample fitting reduces variance.
At the initial prefix, lower action-sample gradient variance is 26.96 times
the raw-mean value. Therefore the next training experiment changes upper only.

The identity-projector action-sample gradient is zero. Raw-mean fitting has
nonzero stochastic gradient even when projection makes no action correction.
This motivates the fixed-residual objective, not a policy-performance claim.

The next frozen development compares zero consistency, causal raw-mean,
causal raw-sample (diagnostic only), and causal action-sample (sole candidate).
It uses 4 new optimizer roots, 3 environments, 512 iterations, and 40 evaluation
episodes per cell. A separate 12-cell short preflight precedes the 48-cell
screen. The exact seeds, budgets and adoption gates are in
`scripts/mujoco_v25_sample_consistent_upper_spec.py`. Both baseline rewards and
physical correction magnitudes remain gates; unlike-unit losses are not.

The first training preflight t93665-t93676 rejected its configuration before
training: four training conditions require at least four train and selection
roots. Its registration is retained as r1; r2 supplies one fresh root per
condition in both roles. The full-development seed set already met this
requirement. Algorithm source and performance gates are unchanged.

Corrected preflight r2, t93693-t93704, completed all 12 cells and 60 evaluation
episodes. The frozen analyzer returned `preflight_valid`: source/parameter and
path contracts, matched capacity, finite losses, certificates, fallback, and
prefix budgets passed. Each active arm applied consistency in 4 of 8 iterations
at both levels; the zero arm applied it in none. Small exports total 746,288
bytes, including registration and analysis, with no local checkpoints/history.

Full development `mujoco_v25_sample_consistent_upper_development_20260914_r1`
was submitted as t93706-t93753: 48 cells, 512 iterations each, and 1,920
evaluation episodes. It later completed and failed the registered development
gates documented in the result update above. This diagnostic remains a
preregistration record, not a performance improvement result.

## Limitations

Action-space gradients include the tanh derivative, so lower gradient variance
can reflect rescaling or saturation rather than better learning. These snapshots
use independent Gaussian proposals, not trained policies with upper/lower state
coupling. The training comparison retains the shared per-step checkpoint score
to isolate this loss change, while the development outcome uses episode return.
