# V154: archived-day running-prior development

V151/model,main,V146(3s,0s),3600s window and .1/.1 tails unchanged. No replay or
policy training. Use only V151 eight calibration days2751001--2751008,12 actors,
old/handoff;96 pairs/192 existing paths. Do not read its16 confirmation days.

Eight leave-day-out folds:exclude the entire held day in both modes from both
service and running priors. Reference is V151-LODO(seven service-prior days),not
the sealed V151 confirmation model. Running cells:actor/direction/stop/entry-hour;
equal day mass,equal available-mode mass within day. Fixed mean combines each
prior day and one native strictly completed-prefix anchor. Missing prior retains
native. Do not replace an already moving partial leg with a whole-leg mean.
Changing running clocks may change downstream service phase;score full ETA too.

Pre-fixed development requirement,both modes:all/hour2 full-MAE improvement;
clearance/hour13/low-support full-MAE non-harm;running-MAE non-harm in all/hour2/
clearance/hour13. Incomplete crossed support is inconclusive. Empirical per-leg
mixture dispersion is diagnostic,not a calibrated full-ETA interval. All5000
crossed actor/day draws are descriptive LODO development,not fresh confirmation.

Server scheduler,node001--006,8 workers. Tests and first saved pair before the
remaining95 pairs. Running moments sealed before scores;no parameter search.
Pull only small summary;moments,per-cell outputs and original paths stay server.

Source5be4cea534,t204491,node004,exit0,no retries;6 tests passed,96 pairs/192
paths complete. Only93483-byte summary pulled. Relative to V151-LODO,old/handoff
full ETA MAE falls44.729->35.289s /44.723->35.289s;paired delta-9.439/-9.434s,
CI[-9.666,-9.160]/[-9.658,-9.165]. Hour2 delta-11.473/-11.474s;all12 actor and
8 day margins improve in both scopes. All-day running MAE improves10.612/10.601s;
induced service-clock mean change is only0.00014/0.00011s. Hour2 full bias
improves-35.320->-14.154s(old),so substantial residual underprediction remains.

Development gate remains false:clearance full delta-0.883/-0.863s has CI
[-2.507,0.822]/[-2.476,0.827];hour13 also crosses0. Low-support group never
occurs(all replaced legs have7 prior days),so it is untested,not zero harm.
Next freeze weights and revalidate on disjoint existing V145 archives(V155);
no tuning on these eight days or new replay/training. This is useful LODO
development evidence,not an independent positive result or controller gain.
