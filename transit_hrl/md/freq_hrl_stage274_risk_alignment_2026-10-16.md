# Stage274: Frozen Prefix54 Error Alignment

t165984/node006: done/0, 9.47s; 15 tests passed. Reused 1,920 contexts / 32 cells.
Zero fits, inference or simulations. Pulled 1.03 MB scalars, no raw data.
Stage272/273 replay and decomposition errors <4.69e-17.

delta_MSE = E[delta squared] + 2 E[delta * (raw-target)], delta = prediction-raw.
Primary .01, mixed pooled, opposite16; entries in 1e-6 risk-MSE units:

| Frozen-model query population | Correction squared | Alignment | Delta MSE |
| --- | ---: | ---: | ---: |
| Excluded, original weights | 24.050 | -15.176 | +8.875 |
| Matching held, original weights | 23.861 | -49.275 | -25.414 |
| Excluded, common-action pairs | 26.001 | -25.436 | +0.565 |
| Held, same common-action pairs | 25.425 | -77.477 | -52.052 |

Alignment accounts for 99.45% of the natural gap, 98.91% after common-action
pairing. Energy barely changes. Mixed pooled gap direction agrees across both
fit banks, both evaluation banks and all three penalties. Mixed_a excluded
alignment is harmful (+25.157); mixed_b remains beneficial (-53.385).

Next Stage275: localize fixed-head alignment contributions by prefix/regime/probe;
check concentration and cross-future consistency from caches. No defaults changed.

Limitations: common-action pairs retain 608/1,920 mixed exposures (31.7%); states
remain different. Historical raw-reference conditioning and repeated development
exposures remain; no causal, seed-CI or policy-gain claim.
