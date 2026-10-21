# V171: temporal-tail protocol and support accepted

Original task `t209984` stopped at a multiline-string IndentationError, before
protocol or support outputs. Syntax-only repair `f5d152f0bc`, task `t210103`,
manually dispatched with high priority to node004: eight tests passed, protocol
sealed before support counting, all 24 actor/mode buckets completed.

- One fixed candidate: original V167 key plus query-time stratum hour2/hour13/other, using floor(query_time_s/3600); no boundary, window, horizon or stop-ID search.
- Calibration: existing V166 days2951001-2951064, 12 actors and old/handoff modes. Current work counts states and distinct supporting days only; no residual quantiles or intervals are fitted.
- Fresh confirmation reservation: days3152001-3152032, samplers1711000000-1711000031, shared across actors/modes, 768 future paths. Inventory checked 45 result roots, 18,497 metadata files, 96,415 artifact names and declared generators: no day/sampler conflicts. No automatic replacement days.
- V168 days2852001-2852032 are excluded from fitting, support checks and threshold selection. Prior diagnostics motivate the questions, but this stage reads no confirmation outcomes.
- Keep V151 point predictions, V146, native3600s window, tails0.1/0.1, native support rules, controller/actor/critic/dual and original V167 maps unchanged.
- New-key support remains at least19 distinct calibration days, independently within each actor/mode/key/time stratum. Low-support or unseen keys permanently abstain; future fitting must recompute equal-day/episode/query masses within each new key.
- Preserve all five original scopes and add hour13_moving_launch, defined by query-time stage kind. Include eligible native-unsupported moving queries in its denominator; do not restrict the subgroup to successful interval outputs.
- Calibration support requires every actor/day cell to have a finite estimate and mean finite fraction at least0.8 in both modes/all six scopes. Failure stops this attempt without regrouping, extra days or opening confirmation.
- Future primary comparator is frozen V167 coarse; frozen V167 conditional is a matched secondary comparator. Keep existing absolute coverage, both-tail, finite-fraction and score thresholds, including strict score improvement in all/clearance; the new moving scope must also pass.
- Report hour13 moving calibration/fresh-confirmation tails separately. Descriptive migration uncertainty uses shared actor resampling and independent whole-day resampling across the two cohorts, 5000 draws/seed20260803; calibration remains in-sample.

Support passed in both modes/all six scopes, each with all 768 actor/day cells
present and nonempty finite support. Mean fractions (old/handoff):

| Scope | V171 finite fraction | Frozen V167 |
| --- | --- | --- |
| all | 93.05% / 93.06% | 93.24% / 93.24% |
| hour2 | 90.18% / 90.27% | 92.07% / 92.08% |
| hour13 | 87.06% / 87.06% | 87.64% / 87.49% |
| hour13_moving_launch | 98.66% / 98.79% | 99.37% / 99.30% |

The 3,802 native-unsupported hour13 requests were joined to query metadata in
1,412 files; none were moving-launch. Moving-launch retains 1,071/971 key-support
abstentions, not an output-only denominator. Hour2 has 92-99 admitted keys per
actor/mode out of 156-167; keys with fewer than 19 distinct days still abstain.

Only summary/status.json (102,435 bytes) was retrieved, under
freqduet/results_freqduet/protocol_v6_v171_temporal_tails/protocol_f5d152f0bc/.
Next: separately authorize construction/sealing of one temporal map on the
existing 64 calibration days, recomputing masses within new keys. Do not open
confirmation yet. No collection, interval fitting, replay, training or policy
change occurred; original V167 remains unchanged. This accepts support only,
not coverage, tail performance or controller improvement.
