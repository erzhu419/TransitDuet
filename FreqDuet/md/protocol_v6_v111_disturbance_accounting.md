# V111: consumed-path action and disturbance accounting

V110 is closed: all three service/regularity/readiness gates failed. Do not
rescue its coefficient or enlarge its sample. Main, actor, critic and dual stay
frozen. This stage reads the existing 72 V110 pair/NPZ records on a server through
scheduler, with no simulation, checkpoint loading, training or new demand day.

Reconcile every positive executed holding bin against saved action counts;
reconcile summed executed seconds per launched trip and actual arrival CV2
against V110 endpoints. Initial unobserved zero-hold events are reported
separately. Report the seven commanded-bin counts, zero/45s fractions, executed
holding dose, and the already-saved unrounded request saturation. Average
samplers within actor/day before the existing crossed actor/day intervals.

At arrivals, describe adjacent-stop persistence of nonholding elapsed contrasts
for the same predecessor trip, including prior own holding and the next actual
hold. This is arrival-state accounting, not a reconstruction of ready-time
states. The saved NPZ does not contain follower estimates or feasible masks.

Test the control law's response to balanced current forward/follower gaps with
nonzero cumulative R. At D=0, R=-60s still requests 45s while the original rule
requests zero. Repeated use of a persistent level is therefore not automatically
an incremental disturbance correction. This analytic example establishes a
property of the law, not how often it caused the measured service loss.

The next decision is whether existing records suffice to quantify repeated
compensation. If exact decision evidence is needed, first instrument consumed
V110 paths and reproduce their endpoints; no new gain or fresh-day controller
screen precedes that diagnosis. A structural successor must account for already
executed compensation/current gap response rather than blindly reapply a
cumulative level. Only summary/status.json returns locally; details stay remote.
