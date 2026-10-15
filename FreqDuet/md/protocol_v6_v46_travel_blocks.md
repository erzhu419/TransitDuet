# V46: completed-leg travel distributions

Keep U200/L200, the seven-bin holding model and APC flow unchanged. Reuse the
12 certified V45 service-day snapshots, at all 51 origins, without rerunning
actors. These demand seeds are now regression data, not untouched confirmation.

Fit only AVL departure/arrival pairs completed strictly before issue, within
the previous three hours. Per directed leg, use the completed-duration mean
(old travel mean if no completed sample). Build 300-second departure-bin means,
carry forward missing bins, and recenter each leg profile to that mean.
Compare completed_mean, independent draws and shared_block circular moving
blocks. The latter uses one historical time offset across vehicles and directed
legs for the full 30-minute forecast; independent draws have identical marginal
distributions but break temporal/cross-vehicle dependence. Initial partial legs
retain remaining-distance scaling. No future road-speed tape is read.

Run 64 paths with separate private travel/holding RNGs. Score 15/30-minute
holding, passenger exposure and dispatch backlog, plus common-clock stop timing
and causal gaps. Missing predicted events retain capped clocks, not deletion;
gap/action scores report available paths. Repeat completed-window calibration
only for observable onboard/backlog/holding channels. Report raw and calibrated
results separately, all-origin and common-clock views, and day-level paired MAE
deltas. Retain historical and V45 raw controls. No policy promotion from this
engineering screen; fresh demand and action-response validation remain required.

Raw all-origin 30-minute screen, per existing cohort: holding MAE no worse than
V45 and historical, absolute bias <=5%, passenger MAE <=105% of historical,
holding 90% band coverage >=80%. Do not select by coverage alone: inspect CRPS,
width and stop-clock errors. Circular boundaries and sparse carried-forward
profiles are empirical approximations, not a claimed calibrated traffic model.
Legs lacking traced departures, including terminal-first legs, retain the old
mean; report support rather than treating those legs as validated distributions.

Smoke: focused causal/physics tests, exact reproduction of the stored V45
64-path forecast with the untouched travel mode, then all three new modes at
one origin. Only after smoke passes dispatch the 612 independent origins via
scheduler on node001-node006. Aggregate on a server; retrieve small summaries.

## Run Status
`4f30a3798b`: final smoke `t155611` passed 40 tests and exact V45 path replay.
Matrix `t155612-t155616`: 612 origins, 307 workers planned across node001/002/
004/005/006; full node003 excluded by live capacity. All completed, exit 0.
Aggregation `t155649` completed, exit 0. Earlier smoke `t155609` passed 39 tests.

All-origin 30-minute holding MAE (development / reused demand / formerly fresh
demand): V45 raw 700/767/739 s, completed_mean 532/590/507 s, independent
409/449/408 s, shared_block 415/453/406 s, historical 491/526/484 s. Shared
blocks improve all 12 day-level MAEs versus V45. Holding CRPS is better than
independent sampling in each cohort, but point and stop-clock gains are mixed;
this supports stochastic travel, not unique identification of shared dependence.
Raw shared-block holding coverage is only 60.3/57.8/65.2%. Reused-demand waiting
MAE is 86,514 person-seconds versus historical 80,630, exceeding the 5% gate.
No variant passes the complete screen. Mature calibrated holding coverage is
93.4/86.0/94.9%, with widths 2,266/2,072/2,405 s; it does not establish reliable
passenger prediction. Canonical small outputs: `protocol_v6_v46_travel_blocks/summary`.

Next: complete terminal-first/last travel observations (four directed legs
currently use fallback), then diagnose APC arrival-rate/left-behind queue
propagation. Keep the actual controller frozen and require new-context and
counterfactual validation before any strategy switch.
