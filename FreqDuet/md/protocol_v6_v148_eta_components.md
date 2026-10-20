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
