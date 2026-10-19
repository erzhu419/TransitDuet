# Independent validation of frozen V140/V141/V142 uncertainty

Design only. No new rollout, replay or training in V143. The twelve consumed
V137 paths and all V140--V143 readouts are development evidence, not calibration
or confirmation data. No fitting stage or residual correction is proposed.

Before any future collection, register 32 previously unused demand-day seeds
and a separate fixed sampler seed per day, after checking existing experiment
manifests. Use all twelve archived actors and both frozen old/handoff control
paths: 768 paths, no actor training. The same day/sampler is shared within a
paired actor/mode comparison. Repeated paths and queries are not new days.
Keep the models, 3600s/all-past rule, 0.1 tails, V142 rank rule and unsupported
statuses fixed. No outcome-based case selection or extension of the day cohort.

Evaluate the online completed-prefix history within each new day, never merge
that day's future labels or test outcomes into an interval. Capture all eligible
service/ready queries, including missing same-vehicle outcomes and insufficient
support, rather than selecting only resolved or finite intervals. Observe the
post-service label tail until all eligible queries resolve, or report unresolved
denominators explicitly. Retain passive data collection and the existing control
actions; uncertainty never selects an action in this validation.

Primary estimand: actor/day-balanced finite current-stage coverage, separately
for service and ready intervals and each control mode. Average repeated queries
within the same stop episode first. Report query coverage, availability over all
eligible requests, unsupported/unresolved counts, interval width, and both miss
directions alongside it. Missing intervals/outcomes are never filled with zero.
Report hour13 and the rest of the day separately, plus the predefined V143 support
classes; do not choose new temporal bins from test outcomes.

Use the project's 5000 crossed actor/day bootstrap draws for 95% intervals,
preserving modes and all queries within each sampled actor/day cell. A claim of
finite-interval 80% coverage requires the lower one-sided 95% bound to reach 0.8
for both stage types and both modes; insufficient support is inconclusive, not
a pass. Marginal passing does not validate hour13 or any unsupported subgroup.
This is an empirical fixed-actor/demand-distribution validation, not an IID or
field guarantee. Availability and unresolved rates remain separate limitations;
passing finite coverage alone cannot authorize control deployment.

If either coverage fails or availability is inadequate for the intended use,
keep the passive model and retain the negative result. Any later interval-model
change consumes this cohort as development and requires new untouched days for
confirmation; it cannot be rescued by window/tail tuning on these outcomes.
