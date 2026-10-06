# Stage127: Decision-Complete Native Credit

Stage126 fixes local gradient alignment but root410023 still selects no action.
Test missing temporal credit, rather than another geometry or radius change.
Use four fresh native scenes and two full noise trajectories per scene. Gather
eight-coordinate central derivatives at every upper decision, with the same
noise before and after each intervention and full recovery suffix credit.

Compare complete credit with credit at0/300/600/900 only. Both use the same
all-decision states/Fisher matrix, unit damping and fixed0.000555556 radius.
Sparse labels are weighted by inverse time-sampling fraction; neither method
gets a different actor, observation, lower, decoder or0.05 response limit.
All labels are shared: this isolates credit coverage, not compute efficiency.
Sum local derivatives per trajectory and compare with full-policy central
secants at fixed0.1 scale. This checks whether local credit actually explains
the deployed direction, including all later decisions.

Use16 separate training scenes/two noise panels for matched0/+1/-1 fits and
crossfit;32 fresh evaluation scenes, flat/forecast/sparse/opposite/blinded
controls, both roots and periods50/100. The0.5 practical threshold is unchanged.
Per root:288queries/4,896label episodes plus64audit+320training+192crossfit+
384evaluation=5,856episodes/7,027,200steps.16workers+parent/12GiB via dynamic
scheduler node001-node006. Four final upper weights stay server-side; only
compact summaries return locally, no native traces or raw states persisted.

## Limitations

One native derivative update, four training scenes and two development roots
cannot confirm joint HRL, frequency-specific value or learned promotion.
Stage119 used native derivatives on a much weaker advice channel; this tests
decision coverage on the unchanged Stage121 bounded response channel instead.
Negative gradients, failed transfer and inactive steps are retained.

## Run Receipt

Code `7c129e5537`, registration `f8e18cce25`; four focused tests passed,
plus four existing geometry tests. Run
`pointmaze_credit_coverage_stage127_pilot_20261006_r1`: `t136168` root410011
and `t136169` root410023 accepted, initially queued. Fixed tasks and rosters
are in `preregistration.json`; no original FreqDuet code was changed.
