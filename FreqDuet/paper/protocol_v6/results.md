# Results

## Independent confirmation at 40 episodes

V8 passed the registered effect/no-harm gate for the complete current policy
(Fig. 2; Table 1). Relative to the Protocol V6 reference, headway CV changed by
-0.022 [-0.038, -0.008]. Its
crossed-bootstrap interval excluded zero, whereas the Holm-adjusted
training-seed sign-flip result was
$p=0.125$. Restricted
passenger journey changed by
-0.263 [-0.837, +0.175]
min. The latter interval crossed zero but satisfied the preregistered journey
no-harm margin. Thus V8 is gate-positive under its preregistered criteria; it
is not a familywise-significant effect at 0.05 and does not establish a
passenger-journey benefit.

![Independent confirmation and long-training robustness. Points show paired mean differences between the current policy and the Protocol V6 reference config named `F_freqduet_protocol_v6_noguard_hiro`; bars show 95% crossed-bootstrap confidence intervals over training and evaluation seeds. Both configurations disable the legacy causal holding guard. The current policy additionally uses compact APC/AVL context and the two-sided departure-regularity objective, so this is a combined-policy comparison rather than an isolated guard effect. Lower values favor the current policy. V8 contains 24 paired rollouts (six training seeds by four untouched evaluation seeds) and passed the registered effect/no-harm gate. Its Holm-adjusted training-seed sign-flip result was p=0.125, so the figure labels V8 as gate-positive rather than familywise significant. V9 contains 64 paired rollouts (eight by eight); its passenger-journey interval favored FreqDuet, but the headway-CV effect did not meet the registered magnitude and interval gate, so V9 is reported as not confirmed.](figures/fig2_protocol_v6_confirmation_robustness.png){#fig:protocol-v6-2}

**Table 1. Current policy minus the Protocol V6 reference.** Values are paired
mean differences with crossed-bootstrap 95% confidence intervals. Lower is
better. Holm-adjusted sign-flip p-values use training-seed mean differences.

| Outcome | V8 delta [95% CI] | V8 Holm p | V9 delta [95% CI] | V9 Holm p |
| --- | --- | --- | --- | --- |
| Restricted passenger journey (min) | -0.263 [-0.837, +0.175] | 0.938 | -1.242 [-2.204, -0.536] | 0.047 |
| Restricted passenger wait (min) | -0.185 [-0.470, +0.050] | 0.375 | -1.011 [-1.878, -0.397] | 0.047 |
| Restricted in-vehicle time (min) | -0.078 [-0.376, +0.147] | 1.000 | -0.232 [-0.355, -0.121] | 0.047 |
| Headway coefficient of variation | -0.022 [-0.038, -0.008] | 0.125 | -0.009 [-0.028, +0.006] | 0.031 |
| Unserved passengers (percentage points) | +0.02 [-0.00, +0.11] | 1.000 | +0.00 [+0.00, +0.02] | 0.250 |
| Realized holding (s/launched trip) | -1.7 [-41.8, +29.4] | 1.000 | -37.6 [-59.8, -16.2] | 0.070 |
| Trips denied at least once (percentage points) | +0.48 [-9.21, +8.81] | 1.000 | -1.65 [-3.96, -0.22] | 0.117 |
| Restricted service cost | -0.040 [-0.079, -0.007] | 0.125 | -0.110 [-0.207, -0.042] | 0.047 |

## Long-training robustness was not confirmed

At 200 episodes, the same policy improved restricted journey by
-1.242 [-2.204, -0.536]
min relative to the reference. The headway-CV difference was
-0.009 [-0.028, +0.006]. Seven of
eight training-seed CV differences were negative, but the interval included
zero and the mean improvement did not reach the registered 0.02 threshold.
V9 therefore returned `longtrain_not_confirmed`; the favorable journey result
cannot be relabelled as confirmation of the registered regularity effect.

## External baselines reveal a passenger-regularity trade-off

The source-identical V9 comparison (Fig. 3; Table 2) shows that FreqDuet had
lower headway CV than fixed headway,
-0.219 [-0.238, -0.198], and lower
restricted service cost,
-0.122 [-0.180, -0.050].
However, restricted journey was higher by
+2.470 [+1.874, +3.188]
min. FreqDuet also used
+286.8 [+269.7, +302.0] more
holding seconds per launched trip and had a
+64.10 [+61.97, +66.03]
percentage-point higher denied-trip rate. All trips were eventually launched
and completed in the aggregated learned-policy results, so this denial measure
captures delayed fleet readiness rather than permanent trip cancellation.
Because the restricted service-cost scalar does not directly charge holding or
retried readiness denials, its favorable difference cannot be interpreted as
passenger-time or fleet-readiness superiority.

FreqDuet reduced restricted journey relative to rule holding by
-3.224 [-4.727, -1.885]
min and relative to rule MPC by
-26.511 [-31.003, -21.628]
min. The supported external conclusion is therefore narrower than universal
superiority: FreqDuet reduced restricted journey relative to the two rules and
produced more regular service than fixed headway, while fixed headway remained
better for passenger journey and fleet-readiness burden.

![External baseline trade-off under the V9 source contract. Points show paired mean differences between FreqDuet and each external baseline; bars show 95% crossed-bootstrap confidence intervals over eight training and eight evaluation seeds (64 paired rollouts). Lower values favor FreqDuet. FreqDuet improved regularity and restricted service cost relative to fixed headway but increased passenger journey time. It reduced passenger journey time relative to rule holding and rule MPC. Exact two-sided sign-flip tests and Holm-adjusted values are provided in the source table and are not encoded as significance symbols in the figure.](figures/fig3_protocol_v6_external_tradeoff.png){#fig:protocol-v6-3}

**Table 2. FreqDuet minus external baseline under V9.** Values are paired mean
differences with crossed-bootstrap 95% confidence intervals. Lower is better.
The complete outcome and adjusted-test table is in the Supplementary Material.

| Baseline | Journey min | Headway CV | Denied trips (pp) | Service cost |
| --- | --- | --- | --- | --- |
| Fixed headway | +2.470 [+1.874, +3.188] | -0.219 [-0.238, -0.198] | +64.10 [+61.97, +66.03] | -0.122 [-0.180, -0.050] |
| Rule holding | -3.224 [-4.727, -1.885] | -0.073 [-0.082, -0.062] | -3.17 [-5.42, -1.90] | -0.351 [-0.496, -0.221] |
| Rule MPC | -26.511 [-31.003, -21.628] | +0.006 [-0.006, +0.020] | +61.52 [+59.09, +63.77] | -2.541 [-2.984, -2.060] |

## Physical execution audit

Relative to the Protocol V6 reference, V9 reduced realized holding by
-37.6 [-59.8, -16.2] s
per launched trip and the denied-trip rate by
-1.65 [-3.96, -0.22]
percentage points (Fig. 4). These are full-policy differences, not isolated
effects of the regularity reward. Against fixed headway, however, the current
policy used
+286.8 [+269.7, +302.0] more
holding seconds per launched trip and increased the denied-trip rate by
+64.10 [+61.97, +66.03]
percentage points. All trips were eventually launched and completed in the
aggregated learned-policy results, so denial records delayed fleet readiness
rather than permanent trip cancellation.

![Paired physical outcomes of the current policy. Points show mean paired differences between the full current policy and the Protocol V6 reference configuration; bars show 95% crossed-bootstrap confidence intervals. Negative values favor the current policy. V8 contains 24 paired rollouts and V9 contains 64. The current policy differs from the reference by both compact APC/AVL context and the two-sided departure-regularity objective, so these panels describe the combined policy's physical behavior rather than an isolated legacy-guard effect.](figures/fig4_protocol_v6_physical_outcomes.png){#fig:protocol-v6-4}

## External data support demand-shape realism only

The FreqDuet OD input has a morning peak similar in timing to the bounded MTA
AFC subset, while its normalized hourly profile differs from the Halifax APC
subset (Fig. 5). The balanced audit contains 39 complete MTA station-complex
days (936 rows) and seven complete Halifax routes across 37 route-days (979
rows). Because systems, dates, sampling units, and measurement processes are
unmatched, these comparisons are descriptive checks of demand-shape
plausibility. They are not same-day calibration, route-family policy tests, or
field-effect estimates.

![External passenger-count demand-shape audit. Panel a compares separately normalized hourly demand shapes from the FreqDuet OD input, a complete-day subset of the bounded public MTA AFC cache (936 source rows; 39 station-complex days), and a complete-route subset of the bounded public Halifax APC cache (979 source rows; 7 routes and 37 route-days). Panel b summarizes the corresponding demand-period shares; the FreqDuet input contains 20 origin series. The balanced-cache derivation excludes incomplete pagination fragments. This remains a descriptive audit across unmatched systems and dates, not a population estimate, same-day calibration, field-policy evaluation, or evidence of deployed control benefit.](figures/fig5_protocol_v6_external_realism.png){#fig:protocol-v6-5}
