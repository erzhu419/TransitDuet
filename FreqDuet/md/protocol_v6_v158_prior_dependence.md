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
hour2 gain8.351[8.032,8.657];clearance gain0.674[0.453,0.894]. Handoff agrees.
Prior daily-mean variance is10.15% of average prior leg variance overall,24.30%
in clearance;this is not the full-ETA day-mean fraction or predictive coverage.

Holding-identified cohort(old;1654172 all/42411 clearance queries):within-cell
variance99.928%[99.905,99.954]all,95.865%[94.005,97.653]clearance;day-mean
fractions0.072%/4.108%. Handoff agrees. Coarse residence has different support.
Running/holding rho=-0.233all,-0.254clearance[-0.295,-0.214];service/holding
rho=-0.231/-0.163. Running/service clearance rho+0.014 crosses0. Holding/ticks
rho=-0.820all,but ticks have small variance. Ignoring covariance overstates full
variance23.18%all,33.77%clearance[28.15,40.34],not interval width or coverage.

Next:state-conditional joint residual blocks around unchanged points,not a
dominant whole-day bias or independent intervals;see
protocol_v6_full_eta_uncertainty_design.md. Within-cell still mixes stop/state/
horizon and local day-by-context variation. Joint tails remain unidentified;
no fit,tuning,confirmation claim or main promotion. Detailed CIs stay in summaries.
