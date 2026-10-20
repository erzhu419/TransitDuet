# V158: prior reliability and error dependence

Read the same V14532-day/12-actor/two-mode archive(768 paths). V151,V154 eight-day
moments and n/(n+1),main,V146(3s,0s),3600s window,.1/.1 tails stay frozen.
No replay,fit,prior-day selection,point/interval candidate or gate adjustment.

First reproduce V156 full-prior query support and MAE/bias endpoints. Analyse
all/active/hour2/clearance/hour13 with follower-target-episode-first means,
then equal actor/day cells and5000-draw crossed bootstrap. First saved pair and
focused tests must pass before remaining383 pairs. Scheduler CPU nodes001--006.

Prior reliability:separate daily-mean spread,within-day variation and the strict
completed current-prefix displacement. Score frozen complete-leg predictions
against retrospective durations;partial legs remain only in full ETA. Preserve
predicted entry hour for prior lookup;actual entry hour is a diagnostic label.

Dependence:compare frozen V151 and eight-day points on matched structured rows.
Report raw error products,centered covariance/correlation,and exact full MSE/
variance budgets. Split covariance into within actor/day cell,actor means,day
means and actor/day interaction. Coarse running/service/residence includes
joint current-ready uncertainty. Pure holding/execution-wait analysis excludes
queries with inseparable current-ready residence;do not equate these cohorts.

Decision after results:determine which shared day/phase and component errors a
full-ETA uncertainty model must preserve. No interval is constructed or tuned.
Limitations:consumed data,not confirmation;fixed8 prior days;daily-mean variance
mixes latent shifts and finite-sample noise. Day-mean ranges are not predictive
intervals. Correlation is not a causal effect. Keep missing support/undefined
correlation explicit. Only small summaries/logs local;all path/cell data server.

Execution e5a6705971,t204550,node006,8 workers,exit0,no retries.8 tests and all384
pairs/768 paths completed;V156 endpoints reproduced. Only808595-byte summary
pulled. Saved-moment readout da5ac49463,t204575,node006:3 tests and original
covariance matrices reproduced,no new forecasts;339663-byte summary only.

Eight-day complete-leg MAE(old):all21.148->14.381s,paired gain6.767[6.670,6.849];
hour2 gain8.351[8.032,8.657];clearance gain0.674[0.453,0.894]. Handoff gains
6.766/8.369/0.669s. Prior daily-mean variance is10.15% of average prior leg
variance overall,24.30% in clearance. These are leg-duration components,not
full-ETA day-mean shares. Clearance prefix entries are77.13% before the query
hour. Daily-mean ranges omit within-day spread;they are not predictive bands.

Eight-day,holding-identified cohort(old;1654172 all/42411 clearance queries):

| Scope | Within actor/day variance %,95% CI | Common day-mean variance %,95% CI |
| --- | --- | --- |
| all | 99.928 [99.905,99.954] | 0.072 [0.045,0.095] |
| hour2 | 97.610 [96.428,98.743] | 2.376 [1.243,3.557] |
| clearance | 95.865 [94.005,97.653] | 4.108 [2.320,5.969] |
| hour13 | 95.411 [93.437,97.379] | 4.562 [2.597,6.541] |

Handoff shares agree(all99.928% within;clearance95.906%). Coarse residence
analysis agrees in direction;its support is not interchangeable with this table.
Running/holding correlations all-0.233[-0.241,-0.225],clearance-0.254[-0.295,-0.214].
Service/holding:-0.231/-0.163. Running/service:all+0.111;clearance+0.014,crossing0.
Holding/execution-wait:-0.820 overall,but execution-wait variance is small.
Clearance running/holding covariance is-248.130s^2:within-240.724 and day-7.338.
Ignoring covariance overstates total variance23.18%[22.00,24.34]overall and
33.77%[28.15,40.34]clearance,not an interval-width or coverage result.

Decision:conditional joint residual blocks around unchanged points,not independent
component intervals or a dominant all-day bias. Preserve local context and day
blocks;keep post-ready residence joint when holding is inseparable. See
protocol_v6_full_eta_uncertainty_design.md. No interval fitted or main promoted.
Within-cell dependence includes stop/state/horizon and local day-by-context
variation;first/second moments do not identify conditional tails. Consumed days
cannot supply fresh confirmation or justify weight/window/quantile tuning.
