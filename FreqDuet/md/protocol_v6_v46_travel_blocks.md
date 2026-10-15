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
