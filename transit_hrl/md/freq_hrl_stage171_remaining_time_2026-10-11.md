# Stage171: Passive In-Flight Remaining-Time Audit

Stage170's marginal-preserving dependence correction loses all ten conditional
peak-cost checks and is not deployed. Measure the other modeling assumption:
active remaining time = distance fraction * causal mean total duration.

Replay all ten Stage164 fit scenes with passive snapshots at the 44 native upper
callbacks. Record only observable progress, age, direction, load, speed, holding,
dwelling and causal duration mean. Join completion labels after each episode.
Fit one fixed ridge additive correction per root on these five fitting regimes.
No check outcomes enter fitting; no hyperparameter search.

Ten fresh check scenes use seed811000000+root*1000+regime_index*10, disjoint from
the old fit/check and rolling evaluation seeds. Each gets an unchanged native
window control and an audited replay; require identical physics, launches and
lower goals. Together with the ten exact fit source reproductions: 30 frozen
episodes, 1,841,400 ticks, 20 exact control qualifications, zero policy updates.
Compare remaining-time MAE, bias, underprediction and q95 error on every fresh
check regime. Ten unpinned scheduler tasks, node001-006, 1 CPU/3 GB each.
Code/fresh model inputs only; raw records/checkpoints remain remote.
41 focused tests pass, including unchanged preferences/candidate scores, causal
snapshots, same-tick completion labels, exact paired truth, fit/check separation,
physical/source accounting and scheduler staging. No full native simulation ran
locally.

Run `native_transit_remaining_time_stage171_frozen_20261011_r1`, revision
`c3f6d9ab93`: t142748-t142757 are DONE; all 20 exact controls qualify.
Fresh-check MAE is 96.683 -> 85.956 seconds (root397, 2405 observations) and
99.324 -> 91.214 seconds (root401, 2397 observations). All ten root/regime
panels improve. Q95 absolute error is 275.148 -> 241.543 and 285.601 -> 245.987
seconds. Bias is -6.298 -> -3.892 and -9.626 -> +0.111 seconds. No forecast
correction was applied to control during these episodes.

Next: replace only the active-vehicle expected remaining-time term in the
six-candidate occupancy ensemble. Retain future-trip draws, candidate geometry,
scores, weights, physical cost and frozen lower. Require exact source controls
and compare actual cost/wait/fleet against both window and uncorrected selector.

## Limitations

Passive window-policy accuracy does not prove prospective occupancy accuracy,
action-value ranking or allocation improvement. Corrections are not used by the
native selector in this experiment. Per-callback samples share trips/scenes and
are not independent seed replications. A positive result would justify a bounded
control test, not a complete learned Freq-HRL claim.
