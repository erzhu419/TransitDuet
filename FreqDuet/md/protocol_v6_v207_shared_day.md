# V207: shared daily residual co-movement

Original64 saved moments,V205 saved point operator and V206 results only.
Preserve all actors/days,V199 masks/bins/19-day threshold,V191 factors and
original denominators. No query scan,deletions,new data,interval fit or training.

Use the SAME saved point operator as a fixed-reference diagnostic,not a
replacement for V206's refitted statistic. Freeze per-cell coefficients
c_h=average-half mass/point-half mass. Daily centered cross profiles then
reproduce V205/V206 conditional point exactly. Compare their10000 shared-day
draws with SAVED V206 day-only draws;keep the operator/mass response explicit.

Remove each actor/key's half mass-weighted parent mean. Its remaining day mean
splits into a full-parent-mass weighted shared day shift and actor-specific
deviation. All9 cells enter parent means;zero exposure is not an observed zero
shift. Split4 signed cell/parent mean products into shared/shared,shared/other,
other/other. Shared-location cancellation is an identity,not new mechanism evidence.

New diagnostics:within each fixed half,center each actor's nominal daily
within-cell/relative-mean profile by its day mean. Its covariance matrix splits
into common-day and actor-specific matrices. Preserve off-diagonals,including
cross-key covariance. These daily covariance profiles differ from parent mean
levels and are not matched query-time or causal propagation estimates.

Eight server tests cover location cancellation versus covariance change,point
reproduction,shared/specific covariance,full-parent centering,estimator remainder,
nominal scaling and zero exposure. High/manual on node004;small status JSON only.

## Result and next decision

t210969/source8cb299b8b1 completed high/manual on node004,exit0;8 tests passed.
V205/V206 point reproduction max8.60e-13;profile identities max7.13e-13.
Only62264-byte status summary fetched;all64 days/12 actors remain,with no zero
full-parent exposure days. Support remains315/864 cells;all frozen items unchanged.

Shared day shifts explain94.1-99.6% of parent-mean marginal scatter,but shared
location terms cancel(max3.41e-13 in paired draws). This is a centering identity.
Centered daily covariance profiles still have79.4-91.5% shared actor/day variance.
V206 day-response signed allocations are:frozen within-day12.8-34.4%,relative
cell/parent means42.7-76.1%,operator/mass response10.9-23.0%;not independent causes.
Cross-key shared-profile correlations are.138-.302(clearance) and-.046-.154
(moving-launch),descriptive only. Shared across actors does not imply one global
day factor across these two roads.

Decision:no global daily-bias overlay. Remaining uncertainty concerns local
relative-mean/covariance patterns and estimator response,not an unremoved common
location shift. Any further predictor work needs evidence that query-time inputs
can identify those patterns;these posterior daily labels are not deployable inputs.
No interval fit,new confirmation,replay,training or policy change was started.

## Limitations

Limitations:reused development days,offline label diagnostics. Parent-level
scatter,daily-profile variability and V206 bootstrap response are distinct views,
not additive variance shares. No new predictor,significance gate or policy claim.
