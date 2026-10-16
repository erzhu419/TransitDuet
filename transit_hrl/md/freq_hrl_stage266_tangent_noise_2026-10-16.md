# Stage266: Active-RMS Tangent Noise

Same Stage265 caches; B = pinv(H_prefix + mu_prefix I),
active T = B - Bu Bu.T/(u.T Bu), interior T = B. All cross terms retained.

t165728: node006, exit 0, no retries, 53.84s; 16 tests passed.
32 cells / 1,280 records qualified; Stage265 baseline exactly reproduced;
KKT residual <= 8.56e-12. Pulled 3.98 MB scalar report, no raw arrays.

Mixed removed held queries, free_old / shrunk: service noise variance falls
30.6% / 31.3%; observed/noise RMS is 1.29 / 1.27. Service variance instead
increases for 7 / 9 of 120 prefix-model exposures: projection is not uniformly safe.

Focal 397/913970203: service variance falls 32.4% / 34.7%, observed/noise RMS
1.33 / 1.36. Active 16@1 parameter-noise trace rises 47.8% / 34.4% while query
noise falls; negative low/rest cross terms matter. Interior 1@0.25 is unchanged.

Limitations: local same-force response, not joint refit error, performance or CI.
Focal noise norm RMS is 4.32 / 2.94 times the radius: not a small perturbation.
Frozen calibration/keys; finite labels can miss tails; contexts are not a matched
rank ablation. Stage253/248/169 closed; no fitting, simulation or default change.

Next: Stage267 should jointly propagate full and prefix score noise:
d_delta = -T_prefix d_g_prefix + T_full d_g_full, retaining shared-label
covariance. Compare with the one-ended diagnostic and check response/radius
scale before choosing shrinkage. Reuse caches; no new fits or simulation.
