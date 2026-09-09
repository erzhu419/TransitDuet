# Protocol V6 Figure Captions

## Figure 1 | Causal frequency-to-authority architecture of the current controller

Historical OD intensities initialize a recursive harmonic demand prior, while
online APC arrivals update the filter causally in 60-s bins.
Low-frequency level, slope, and forecast features enter the upper policy, which
replans an executable terminal headway curve every
15 min over a
45-min horizon under a rolling zero-sum
headway budget. Station-local high-frequency innovations and
compact same-time APC/AVL context enter the lower policy, which selects from
seven discrete holding actions between 0 and 45 s. The two-sided regularity
reward uses forward and follower departure gaps frozen before the action. In
the current confirmed configuration, the legacy holding guard, promotion, and
leakage penalty are disabled.

## Figure 2 | Independent confirmation and long-training robustness

Points show paired mean differences between the current policy and the
Protocol V6 reference config named `F_freqduet_protocol_v6_noguard_hiro`; bars
show 95% crossed-bootstrap confidence intervals over training and evaluation
seeds. Both configurations disable the legacy causal holding guard. The
current policy additionally uses compact APC/AVL context and the two-sided
departure-regularity objective, so this is a combined-policy comparison rather
than an isolated guard effect. Lower values favor the current policy. V8 contains 24
paired rollouts (six training seeds by four untouched evaluation seeds) and
passed the registered confirmation gate. V9 contains 64 paired rollouts (eight
by eight); its passenger-journey interval favored FreqDuet, but the headway-CV
effect did not meet the registered magnitude and interval gate, so V9 is
reported as not confirmed.

## Figure 3 | External baseline trade-off under the V9 source contract

Points show paired mean differences between FreqDuet and each external
baseline; bars show 95% crossed-bootstrap confidence intervals over eight
training and eight evaluation seeds (64 paired rollouts). Lower values favor
FreqDuet. FreqDuet improved regularity and restricted service cost relative to
fixed headway but increased passenger journey time. It reduced passenger
journey time relative to rule holding and rule MPC. Exact two-sided sign-flip
tests and Holm-adjusted values are provided in the source table and are not
encoded as significance symbols in the figure.

## Figure 4 | Paired physical outcomes of the current policy

Points show mean paired differences between the full current policy and the
Protocol V6 reference configuration; bars show 95% crossed-bootstrap confidence
intervals. Negative values favor the current policy. V8 contains 24 paired
rollouts and V9 contains 64. The current policy differs from the reference by
both compact APC/AVL context and the two-sided departure-regularity objective,
so these panels describe the combined policy's physical behavior rather than
an isolated legacy-guard effect.

## Figure 5 | External passenger-count demand-shape audit

Panel a compares separately normalized hourly demand shapes from the FreqDuet
OD input, a complete-day subset of the bounded public MTA AFC cache
(936 source rows; 39 station-complex
days), and a complete-route subset of the bounded public Halifax APC cache
(979 source rows; 7 routes and
37 route-days). Panel b summarizes the corresponding
demand-period shares; the FreqDuet input contains 20
origin series. The balanced-cache derivation excludes incomplete pagination
fragments. This remains a descriptive audit across unmatched systems and dates,
not a population estimate, same-day calibration, field-policy evaluation, or
evidence of deployed control benefit.
