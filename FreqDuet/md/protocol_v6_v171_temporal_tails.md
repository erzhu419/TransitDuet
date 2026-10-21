# V171: temporal-tail protocol, pending server seal

Implementation `7c38e514fd`; scheduler `t209984`, explicit high priority and two
manual dispatch attempts, restricted to node001-node006. Latest check: queued
for CPU resources, zero retries. Server tests, freshness and support remain pending.

- One fixed candidate: original V167 key plus query-time stratum hour2/hour13/other, using floor(query_time_s/3600); no boundary, window, horizon or stop-ID search.
- Calibration: existing V166 days2951001-2951064, 12 actors and old/handoff modes. Current work counts states and distinct supporting days only; no residual quantiles or intervals are fitted.
- Proposed fresh confirmation reservation: days3152001-3152032, samplers1711000000-1711000031, shared across actors/modes, 768 future paths. Reservation becomes effective only after the queued server metadata inventory finds no conflicts. No automatic replacement days.
- V168 days2852001-2852032 are excluded from fitting, support checks and threshold selection. Prior diagnostics motivate the questions, but this stage reads no confirmation outcomes.
- Keep V151 point predictions, V146, native3600s window, tails0.1/0.1, native support rules, controller/actor/critic/dual and original V167 maps unchanged.
- New-key support remains at least19 distinct calibration days, independently within each actor/mode/key/time stratum. Low-support or unseen keys permanently abstain; future fitting must recompute equal-day/episode/query masses within each new key.
- Preserve all five original scopes and add hour13_moving_launch, defined by query-time stage kind. Include eligible native-unsupported moving queries in its denominator; do not restrict the subgroup to successful interval outputs.
- Calibration support requires every actor/day cell to have a finite estimate and mean finite fraction at least0.8 in both modes/all six scopes. Failure stops this attempt without regrouping, extra days or opening confirmation.
- Future primary comparator is frozen V167 coarse; frozen V167 conditional is a matched secondary comparator. Keep existing absolute coverage, both-tail, finite-fraction and score thresholds, including strict score improvement in all/clearance; the new moving scope must also pass.
- Report hour13 moving calibration/fresh-confirmation tails separately. Descriptive migration uncertainty uses shared actor resampling and independent whole-day resampling across the two cohorts, 5000 draws/seed20260803; calibration remains in-sample.

The server entry seals protocol.json before counting support. No collector, replay,
training or interval construction has been submitted. Next: retrieve only the
small support/freshness summary after t209984 finishes; support alone is not coverage evidence.
