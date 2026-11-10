# Calibration Score-Source Result

All four jobs t214466-t214469 exited 0 on node004/node005. Reused saved OOF scores
and eight unchanged heads: zero fits/native steps, 1.304 measured phase CPU seconds.
Retrieved 533,108 public-result bytes, no raw banks or models. OOF cuts reproduced.

Below: full-score-selected cut minus unchanged quantile policy on SAME full scores.
Positive W/G means improvement; R/C denote reference/C0 label heads. Now counts /96.

| Task / root | Head | B W gain | B G gain | Choose-now |
| --- | --- | ---: | ---: | ---: |
| PointMaze / 1972110000 | R | +0.002563 | -0.000293 | 44 |
| PointMaze / 1972110000 | C | +0.001172 | -0.004379 | 49 |
| PointMaze / 1972120000 | R/C | +0.000953 | +0.004386 | 86 |
| Reacher / 1972210000 | R | +0.000302 | -0.001756 | 19 |
| Reacher / 1972210000 | C | +0.000744 | +0.001392 | 3 |
| Reacher / 1972220000 | R | +0.000281 | -0.000185 | 16 |
| Reacher / 1972220000 | C | +0.000169 | -0.003596 | 35 |

Score alignment matters: all eight full-score A W floors now hold, and B W point
gains are positive in both halves for every head. This is not a safety certificate.
G remains unresolved: only PointMaze root2 and Reacher root1 C improve in both
halves; all G prefix intervals cross zero. PointMaze root2 heads choose identical
actions; second-half W is only +0.000040. Reacher root1 C remains near-all-wait
and its W versus actual C0 is -0.000542, not a joint improvement over C0.

Direct full-selected minus OOF-selected comparison holds scores and baseline fixed.
Only three of eight heads have positive pooled G increments; all those intervals
cross zero. PointMaze root1 C goes from A G +0.006502 to B G -0.004379 versus
quantile, negative in both halves. Reacher root2 C goes from 3 to 35 choose-now
actions but still loses B G. Misaligned scores are a contributor, not the sole cause.

Next freeze a small independent-calibration/new-prefix-confirmation protocol for
the unchanged heads. Calibration labels must be separate from head-fitting A and
revealed B. Do not keep searching cuts on this cache. No further jobs are submitted;
PILOT_RECIPE_HOLD and the original confirmation restrictions remain.

## Limitations

Full-score A calibration is in-sample; OOF cuts are not nested policy validation.
B is previously revealed development data on 32 known training prefixes/root,
with two trained roots/task. Prefix intervals are descriptive; these single-action
contrasts do not establish episode, lifecycle-budget or end-to-end HRL performance.

Sources: [summary](../results/plan_validity_score_source_20261110/summary.json),
[frozen protocol](freq_hrl_score_source_2026-11-10.md).
