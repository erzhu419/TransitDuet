# V57: realized trip credit versus full-day action effects

V56 finds 5/9 nonzero joint benefits have the wrong sign at 30 min, including
cases still harmful when both reference vehicles return. Before changing lower
learning, reuse all 864 V36 contexts and seven action bins; add no rollouts.

Keep each context's deployment and continuation actor fixed. Compare the actor,
its recorded energy minimizer, realized remaining-trip reward minus the anchor
model's recorded lambda times realized trip cost, and minimum full-day journey.
Exact ties retain the incumbent, then the smallest tied action. Report journey,
CV and readiness separately, along with local-gain/day-harm action incidence.
Use the existing crossed-bootstrap convention after averaging target times
within each train/scenario; do not treat all action rows as independent samples.

These are hindsight diagnostics. The deterministic discounted trip return does
not contain future stochastic-policy entropy and is not a soft-Q calibration
target. A local/full-day disagreement supports objective/horizon repair, not a
claim of critic bias by itself. The day-minimum comparator can harm other service
outcomes; it is not a deployable controller. No weight is fitted or gate tuned.

Run regression and the audit through scheduler on node001-node006. Export only
summary tables/logs; detailed action records and checkpoints remain remote.
Use this result to specify shared passenger-time and cross-trip credit before
any new 40/200-episode training comparison. The real main policy remains frozen.
