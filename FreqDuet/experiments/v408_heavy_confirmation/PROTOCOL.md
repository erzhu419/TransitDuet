# V408: Frozen V406 Under Heavy Demand

Confirm only the high-demand signal of the corrected V406 single-action
controller as executed in V407, using the frozen server source03 and its
portable V404 dependencies. Hold the public timetable, demand profile
(1.2x), fleet of 12, legal masks, predictor, seven bins and costs fixed.
The 10 s and 15 s rules clip the native bilateral request from the first query.
The candidate still has all original legal bins. No V407 control arm,
training, policy switch or changes to any historical decision.

The primary comparator is a frozen whole-day rule mixture: 60% fixed10 and
40% fixed15. Its expected endpoints are computed from both actual paths on
each demand day, not by mixing commands within a trajectory. On the two
consumed V407 heavy development days, the mixture gives mean CV2 0.08529
versus 0.08508 for frozen V406. This sets a nearby simple-rule operating
point, not a CV2 tolerance; the weight stays fixed on confirmation.
Report both pure rules separately. Do not select a comparator after testing.

Sixteen reserved demand days, three paths each. On day-paired differences
(V406 minus reference), use the same 20,000 bootstrap day draws for all
contrasts. Pass only if the upper endpoints of the two-sided 95% intervals
are strictly negative for journey and nonpositive for actual CV2, readiness
delay and public delay against the same frozen mixture. Report all day
differences, waiting/onboard, holding, unboarded and journey P95. This
mean-effect gate is a new scoped hypothesis, not a revision of the earlier
all-day development screens. No universal or cross-route claim; the pure
rule contrasts do not determine passage. No automatic promotion.

Zero-traffic preflight checks frozen imports, optimized runtime credentials
and reserved day metadata before release. Run 16 day jobs at most, each
executing its three arms sequentially. Failures consume any started path;
no retries or substitutions. An incomplete matrix cannot pass. Require
physical accounting, matching generated populations and exact paired
random streams before using a day. All computation via scheduler high,
manually dispatched on CPU nodes001-006; only compact summaries are local.
