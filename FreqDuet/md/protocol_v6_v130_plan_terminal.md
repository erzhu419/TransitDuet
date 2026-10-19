# V130: bounded plan-anchored terminal coordination

Fix one candidate before outcomes: at the existing once-only upper query, with
current same-direction ready inventory, issue P + min(30, floor(max(0,L+360-P)/2))
seconds, where P is that query's published release and L is past actual release.
No ready vehicle, missing past release, expired budget or command already due:
retain P. Never move this trip's deadline later after the query; no hard minimum
spacing. Half-gap response minimizes equally weighted squared phase and gap
errors. The 30 s budget is a conservative legal holding-scale choice, not fitted
to V129 hindsight advances. No sweep. Keep lower targets, weights and physics.

Tests first: independent bound, contraction 30/15/7/3/1/0 after a single supply
shock, no budget restart when late or unavailable, fixed live command and total
delay including control. Actual fleet-induced delay is not claimed bounded.
Then exact raw/split neutral replay on actor12011/day2231004 and two treated
paths. Only after smoke passes, run the remaining six paths on the consumed
V128 two-actor/two-day grid. Reuse eight no-terminal and eight hard-spacing paths.

Primary contrast: split_plan minus split, with original-lower and hard-spacing
contrasts also reported. Existing crossed actor/day CI; passenger improvement,
actual arrival CV2 and inclusive published-clock-to-launch delay no harm, plus
completion. Record capped commands and actual lateness profiles separately.
No automatic promotion, main switch, new day or training. All compute through
scheduler on node001--006; keep full records there and return only small summaries.

## Outcome (2026-10-19)

Tasks t195198--202 exited 0; five tests and both exact neutral replays passed.
All eight treated paths obeyed the 30 s eligibility-shift bound. Mean shift was
1.09--1.90 s/trip; final directional lateness 15--537 s, versus V128's 1093--1964 s.
Paired means against the same lower without terminal intervention:

| Lower | Journey, min | Arrival CV2 | Inclusive launch delay, s |
| --- | ---: | ---: | ---: |
| Original bilateral | +0.00880 | -0.000329 | +2.052 |
| Split-prediction bilateral | -0.02632 | -0.001910 | +0.264 |

For split, crossed CI: journey [-0.09637,+0.00195], CV2 [-0.006173,-0.00000544],
delay [-4.260,+2.324]. CV2 improves in all four cells; most journey gain comes
from actor12011/day2231004. All three registered joint gates fail. No promotion.
Next: read-only diagnosis of the three split cells with increased launch delay
and the cross-actor difference on day2231004; no budget/gain tuning or training.
Final summary (142360 bytes):
`freqduet/results_freqduet/protocol_v6_v130_plan_terminal/summary/status.json`.
