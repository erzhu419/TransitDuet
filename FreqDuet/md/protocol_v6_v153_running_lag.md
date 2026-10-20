# V153: read-only running-time localization

Freeze V151/prior,main,V146(3s,0s),3600s window and .1/.1 tails. Reuse384 saved
confirmation paths;no replay,training or fitting. Scheduler,node001--006 only;
only small summary returned. Reconcile common-support running errors with V152
and exact segment-error budgets before interpreting them.

Report hourly running errors,particularly hour2(7200--10800s),and six most
negative segment contributions in hour2/clearance. Inspect all clearance launches.
For each segment inspect strict-prefix sample age,entry-hour mix and current-hour
completed support. Compare native clocks to same-hour completed samples and
already observed moving-time averages on explicitly conditional support.

Saved selected ETA speed is usually journey-average including residence,not
instantaneous speed. Derive moving-time averages using only past events;do not
infer an unobserved instantaneous speed or use future durations as inputs.
These comparisons diagnose lag and observable motion;they do not validate a
new predictor or uniquely identify future stochastic changes. No tuning here.

Source cf7e297449,scheduler t204485,8 workers,allowed node001--006.
Superseded t204484 cancelled while queued;complete V152 component support is
used rather than assuming physical position alone guarantees a full forecast.
Initial dispatch:all six nodes report0 available CPU;task remains queued,not
failed. Tests and analysis have not run;no V153 result is available yet.
