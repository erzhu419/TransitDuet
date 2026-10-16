# Stage273: Same Held States, Frozen Prefix54 Heads

t165953/node005: done/0, 75.13s, no retries; 16 focused tests passed.
Reused 192 full60 / 1,920 prefix54 contexts across 32 cells. Zero new fits,
solves, simulations or action selections. Pulled 1.45 MB scalar report only.
Stage272 predictions reproduce exactly; metric/mixture errors <1.67e-16.

Primary .01, mixed pooled, opposite futures; risk MSE change versus raw:

| Frozen head | Full60 | Other-condition removal, 54 | Matching-condition removal, 54 |
| --- | ---: | ---: | ---: |
| Cached RMS fit, no local clip | +4.666% | +4.613% | +12.266% |
| Same fit, local clip | -3.310% | -2.942% | -2.839% |
| Linear, clipped | -3.015% | -2.622% | -2.716% |
| Interaction, clipped | +0.808% | +1.148% | +1.182% |

At fixed 54, matching removal adds only +0.102 percentage points for the primary
clipped head; the two bank directions disagree (-0.211 / +0.400 pp). Same-bank
evaluation gives -0.022 pp. The unclipped head adds +7.653 pp: local clipping
largely contains that sensitivity. Mixed support falls 1 to 0; single-root
support falls 2 to 1, yet the latter contrasts are larger (+1.189 / +1.297 pp).

Other removal versus full60 adds +0.368 pp, combining count and composition.
Earlier prefix-OOF is +1.353%, while matched-removal held is still -2.839%.
The same-query removals do not reproduce that OOF/held sign flip; the pure
sample-count effect remains unidentified. No model/default changes.

Next Stage274: keep each prefix54 model fixed; compare its excluded versus
condition-matched held errors via correction-squared and raw-residual alignment
terms, using caches only. Limitations: descriptive development-cohort contrasts,
repeated state/model exposures, not a pure count effect, seed CI or policy gain.
