# V161: frozen full-ETA interval protocol

Protocol only, using V160's reserved 32 calibration days (2851001-2851032) and
32 confirmation days (2852001-2852032), all 12 actors, both old/handoff modes.
The original matched sampler allocation and whole-day split are unchanged.
No collection, interval fit, replay, training or main-policy change is authorized.

V151 is the point reference; the running shadow is diagnostic only. V146 3s/0s,
the native 3600s window and .1/.1 tails remain fixed. Same-query running, service
and residence residuals stay together. Residence includes inseparable ready
time; opaque cases abstain rather than acquire invented components.

Conditional keys: actor/mode, query and predicted-target demand phase,
direction/target stop, frozen position status/current-stage kind, native support
class, and predicted ETA bands [0,900], (900,1800], (1800,infinity) seconds.
Context is captured before labels. Late load and realized future clocks are not
features. Current service/ready native rank abstention remains unchanged.

At least 19 distinct calibration days per key; otherwise permanent abstention,
with no adaptive pooling. Within a key, weight days equally, episodes equally
within day, queries equally within episode. For D days, k=floor((D+1)/10),
j=D+1-k; invert the weighted signed-total-error CDF at k/D and j/D, without
interpolation. Bounds are [max(0,m-Qhigh), max(0,m-Qlow)]. This day-mass rank
correction is empirical, not an exact conditional coverage theorem.

Seal both conditional and coarse actor/mode/query-phase full-ETA reference
maps on calibration only. Insufficient calibration support stops confirmation
release. Confirmation uses fixed maps once, with no interval updates or tuning.

Confirmation: episode means within each actor/day; complete 12x32 crossed grid;
5,000 crossed actor/day draws, seed 20260803. In both modes, all/active/clearance/
hour2/hour13 must have coverage lower95 >=.8, each miss-tail upper95 <=.1,
finite-fraction lower95 >=.8, and paired interval-score upper95 <=0. Score must
strictly improve in all and clearance. Score is width +10 times tail shortfall.
Opaque/native-low-support queries remain in the finite-fraction denominator;
missing finite labels, reference bounds or crossed cells are inconclusive.
Report actor and day marginals; no selection of successful subsets.

Next: after this protocol is sealed, separately authorize calibration collection
and verify query-time instrumentation. Confirmation stays unopened until the
deterministic calibration maps and support admission are sealed. Interval
success is not evidence for switching the holding policy.
