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
