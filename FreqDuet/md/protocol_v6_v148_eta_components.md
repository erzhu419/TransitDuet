# V148: read-only clearance ETA and source-switch component accounting

Read the same768 V146 confirmation paths and V147 cells. Keep V1463s/0s,
V140/V141 formulas,window,tails,actors/critic/dual and deployed controls frozen.
No simulation replay,training,new forecast model or fitting. Only reconstruct
frozen component calculations and require agreement with saved V141 totals.
First reproduce each V147 clearance count,bias/MAE and source-switch count/MAE.

Use exact same-trip/bus target-ready labels. Future running intervals are between
recorded departures/physical launch and arrivals; mandatory service is arrival
to ready; executed holding is the recorded holding-counter difference ending at
departure; remaining ready-to-departure time is execution/policy wait. Truncate
all intervals at query and target ready,excluding target post-ready holding.
Current service requires a strictly earlier observed arrival. A conditional
ready-residence predictor stays joint holding/execution;do not invent a split.

Allocate signed full-gap error to current service,running,downstream service,
holding,execution wait,current-ready joint,and unallocated fallback. Unallocated
forecasts retain their whole error;other entries mean zero allocated contribution,
not measured zero component error. Report structured fraction. For MAE,use
sign(total error)*component error;negative allocations offset other errors.
Both signed errors and MAE allocations must sum to saved full-gap errors.
Also report combined service and post-ready residence totals across phase switches;
these avoid confusing a current/downstream or joint/separate relabeling with gain.

For the frozen matched-identity source/status-switch set,subtract both paths'
component errors. Keep the conditional349-cell group separate from switch burden
over all valid matched decisions. Observed no-switch events contribute zero to
burden,never to a conditional missing outcome. Keep incomplete groups without CI.
Episode-first means and existing5000-draw crossed actor/day statistics;report
axis margins. Tests,saved-pair preflight and all analysis run via scheduler on
node001--006. Details stay remote;only a small final JSON may be fetched.

This is consumed-path additive clock accounting,not intervention mediation or
controller admission. Decide the next repair only after the complete decomposition;
do not fit an offset or change controls on these days.

Read-only follow-up after the main audit:unallocated error dominates switch MAE.
Split its saved raw forecast into ETA minus actual signed arrival gap,service
proxy minus target full service,and saved gap minus ETA/proxy(existing adjustment).
The first term is an arrival clock,not pure running;past target arrivals remain
signed. Reproduce V148's original unallocated bias/MAE budget per cell before
crossed statistics. This is a post-hoc explanation,not refitting or a new gate.

## Results

Main sourcebd0dbe292c,t199046/node006 exit0;12 tests and all384 saved pairs
pass. V147 clearance endpoints and switch cohorts reproduced. Additive clearance
bias contributions(s),old/handoff:current service +0.281/+0.280,running
+2.879/+2.845,downstream service +18.860/+18.885,holding -0.719/-0.715,
execution +0.101/+0.099,ready-joint -0.001/-0.006,fallback +0.140/+0.159.
Sum +21.540/+21.547s. Downstream service95% CIs[17.476,20.128]/[17.501,20.154],
positive for all12 actor/all32 day margins;running/holding bias CIs cross zero.
Structured support99.764/99.765%. Full MAE40.925/40.947s has running allocations
27.087/27.143s,service totals10.486/10.508s,holding3.131/3.079s:systematic bias
and absolute timing error have different leading components.

Switch conditional MAE20.312s(349 cells);whole-grid event burden
0.371[0.296,0.460]s/valid matched decision. Burden allocations:unallocated
0.186[0.151,0.230],holding0.085[0.066,0.108],running0.065[0.046,0.089],
service0.026[0.020,0.033]s. Raw-clock follow-up isolates the unallocated term.
Initial t199047 stopped after tests/saved-pair proof because of a missing spawn
context import;source16b51ea4f0 corrects only that import,without model changes.
Corrected t199048/node006 exit0,4 tests/all384 pairs pass;the original budget
is reproduced. Of conditional20.312s switch MAE,raw arrival clock allocates
14.067s,target service proxy0.020s,existing gap adjustment approximately zero.
Whole-grid raw-clock allocation0.1857[0.1504,0.2299]s/valid matched decision,
positive for all12 actor/all32 day margins. This is not pure running error.
Only57413 bytes of the two final JSON summaries fetched;raw paths stay remote.

Next: prioritize demand-phase-aware downstream service expectations and a common
arrival-clock convention across fallback/component transitions. Do not infer the
cause of service-mean mismatch or achievable control gain from these allocations;
separate candidate design and fresh-day confirmation are required. No changes to
V146,current-stage intervals,point formulas,windows,tails,holding or main here.
