# V129: read-only V128 terminal delay and recovery

Read the eight V128 treated terminal records and their eight V127 controls.
No rollout, new day, parameter search, training or main-policy change.
Use the existing CPU environment through scheduler; return only summary JSON.

Reconstruct each launch from its query-time published clock, preceding actual
release +360 s, and selected vehicle's observed dispatch visibility. Retain any
unexplained residual. Balance end-of-direction lateness against inherited delay,
short published gaps, recovery slack and new supply constraints. Follow the same
vehicle's prior trip to expose carried launch lateness. Quantify conditional
immediate release advance, sticky-eligibility clocks and vehicle remapping.

Six synthetic tests cover persistence, recovery, short gaps, turnover inheritance,
unexplained waits and paired return propagation. Match recorded V128 delay
endpoints; split matched return changes into prior launch, holding and other time.
All quantities are consumed-path accounting, not a new controller or an estimate
of downstream policy benefit. Decide on any bounded plan-anchored candidate only
after these records show where delay can actually recover.
