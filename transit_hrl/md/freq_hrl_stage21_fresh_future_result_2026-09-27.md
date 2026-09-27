# Stage-21 Independent-Future Precision Result

**The frozen two-root gate failed.** Tasks `t101399/101400` completed on
node006/node005 in 60.5/59.6 minutes, submitted at `8f0e4c4acc` as
`pointmaze_fresh_future_stage21_v1_development_20260927_r1`. Both final logs
reached 16/16 states and result completion. The two local result JSONs total
159,955 bytes; no additional download was needed.

| Root | Corrected zero MSE | Stage-19 neural MSE | Frozen ridge MSE | Ridge improvement vs zero |
|---|---:|---:|---:|---:|
| 209011 | 0.002280754 | 0.002671587 | 0.001908796 | +16.31% |
| 209061 | 0.001247787 | 0.055944495 | 0.001396302 | -11.90% |

Positive differences favor ridge. These are the preregistered Bonferroni
98.75% approximate conditional Monte Carlo intervals (four comparisons).

| Root | Control | Control minus ridge MSE | Interval |
|---|---|---:|---:|
| 209011 | Zero | +0.000371957 | [0.000102631, 0.000641283] |
| 209011 | Stage-19 neural | +0.000762791 | [0.000484714, 0.001040868] |
| 209061 | Zero | -0.000148515 | [-0.000276956, -0.000020074] |
| 209061 | Stage-19 neural | +0.054548193 | [0.050732375, 0.058364010] |

The second root now shows harm relative to zero, not an inconclusive interval.
Ridge beats the neural control on both roots but improves only 4/8 and 3/8
path-mean errors versus zero. This closes the precision screen for this frozen
predictor; more draws are not the next step.

Audit passed: all 32 frozen states and predictions match Stage-20; each has
64 finite paired contrasts. All 2,048 new seeds match the registered namespace,
are distinct and disjoint from old draws. Independent squared-error, variance,
Welch interval and gate recomputation agrees. No critic was fitted. Charged
steps are 4,243,200 reconstruction + 2,515,200 replay per root, 13,516,800 total.

Next: retain Stage-9 as the same-task performance reference. Use cached
path-wise residuals to distinguish label-fitting error from cross-path/state
representation failure before another critic design. Do not deploy this
candidate, retune it on these labels or extend its roots/sampling budget.

## Limitations

Inference is conditional on these fixed development states, frozen predictions
and simulator latent state. It is not a policy-return or cross-task result.
This rejects useful accuracy for the current frozen predictor on both roots
jointly; it does not prove continuation credit unlearnable or rule out noisy
training labels as a cause. Stage-20's original failure remains unchanged.
