# Stage265: Joint MC Noise and KKT Directions

Reuse Stage249 labels and frozen Stage264 model/calibration/operator; no fitting,
simulation, selection or held-outcome evaluation. Center scores per state/bank;
Cq = sum_i(S_i,0/16 + S_i,1/16), using the distinct state/future seed blocks.
Retain action/head covariance and low/rest query cross terms. For
A = H_prefix + mu_prefix I, Ctheta = pinv(A) Cq pinv(A).T.

t165710: node005, exit 0, no retries, 50.38 seconds; eight tests passed.
32 cells / 1,280 records qualified. Stage264 component RMS reproduced within
5.56e-17; covariance and spectral relative residuals <= 1.12e-12.
Pulled the 2,389,167-byte scalar report and small cached scalar lookups;
full direction profiles remain remote.

Mixed removed held queries: 12 bank-cells / 120 active prefix-models per family.
Observed target-swap RMS / plug-in noise RMS: service 1.41 / 1.39, risk .89 / 1.06
(free_old / shrunk). Differences remain comparable to estimated noise.
Focal service ratios are 1.45 / 1.46; its held service-noise share is 91.5% / 84.0%.
Shrinkage reduces scale without materially improving that ratio.

Low-block parameter-noise trace fractions are 86.9% / 90.2% for mixed removed,
but only 37.7% / 42.4% for the focal prefix. Its noise is dominated by the active
16@1 context, whose leading covariance direction carries 78.0% / 75.5% of variance;
the other 1@0.25 context is interior. Raw low-block targeting alone misses much
of the focal amplification.
Focal held low/rest cross terms are -86.2% / -84.8% of full noise variance:
the signed cross terms cannot be dropped.

Limitations: multiplier held fixed; frozen-operator plug-in MC noise, not actual refit
covariance, ground-truth error, p-values or seed CI. Anchor quantities depend on
training; finite labels can miss rare tails. Inverse-KKT gain is coordinate/unit
dependent. 397/913970203 remains the previously designated diagnostic prefix.
The two focal contexts are not a matched rank ablation; signed cross ratios are
not causal contribution shares.
Stage253/248/169 stay closed; no default/deployment change.

Next: compare the frozen-multiplier inverse with the full active-RMS KKT tangent
response on the same caches. Retain joint covariance/cancellation before choosing
noise-direction shrinkage; no fitting or simulation needed for that comparison.
