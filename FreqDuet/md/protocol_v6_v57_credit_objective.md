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

No weight is fitted or gate tuned.

Run regression and the audit through scheduler on node001-node006. Export only
summary tables/logs; detailed action records and checkpoints remain remote.
Use this result to specify shared passenger-time and cross-trip credit before
any new 40/200-episode training comparison. The real main policy remains frozen.

## Result And Next Step

Audit `t162238` completed on node004: all 864 contexts and 6048 actions, five
new tests, zero new rollouts. Regression `t162243` passed all seven legacy tests
and inspected the already computed paired intervals. No audit result was refit.

Under main L200/B200, locally best realized trip credit increases daily journey
in 50/144 contexts; the daily journey minimum reduces local credit in 79/144.
Local and global choices agree in 27/144. The local selector adds 9.6875 s at
the intervened action [4.8264,15.0000], improves trip credit by 0.099877
[0.055754,0.146504], but daily journey is +0.000651 min [-0.019722,0.017204]:
average passenger harm is not established. The daily journey selector reduces
trip credit by 0.127843 [0.044937,0.224478] and journey by 0.029350 min
[0.016663,0.048418].

The next control candidate should charge shared system waiting/onboard time
once per elapsed interval, rather than duplicating it across overlapping bus
transitions, and preserve system-value continuation beyond an individual bus's
terminal return. Keep the seven actions, fleet, upper policy and frequency
modules unchanged initially; retain CV and actual readiness as separate service
outcomes. First verify cost conservation, causal state and terminal/bootstrap
semantics in a small server smoke. Only then run matched 40/200 trajectories
and fresh-seed confirmation. Do not merely raise the local holding penalty or
replace the actor with its current critic's preferred action.

## Limitations

These are hindsight development-bank comparisons, not learned-policy gains or
independent confirmation. The deterministic discounted trip return excludes
future stochastic-policy entropy and is not a soft-Q calibration target;
local/global disagreement alone does not identify critic bias. The day-minimum
selector can harm individual CV/readiness outcomes and is not deployable as-is.
