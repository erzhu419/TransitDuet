# Stage184: Causal Full-Tail Summaries

The current 96-feature representation aliases two real candidate actions
with different outcomes (root401/900035901/probe23, candidates3/4).
Their recorded full-tail wait proxy and headway irregularity differ.
Append those two causal summaries, rather than a strength label or outcome.

Reuse Stage183 predictions as control. Freeze its 28 fitting groups and
100-step schedules, proper MSE, targets, train-only scales and optimizer.
Expand 96-16-3 to 98-16-3 (+32 parameters); preserve all original initial
weights/biases and initialize the two added input columns to zero.
Primary is tail_summary:wait_feasible, with the same both-root cost/wait
gate across scene, regime and lower-root splits. Both selectors share fits.

One unbound scheduler task on node001-006: 28 fits/2,800 updates,
one CPU/3GB. No new control, actor, inner fits or native episodes.

Twenty-five focused tests pass (six new, nineteen shared-core regression
tests). The real recorded alias is separated by the new feature map;
no distinct-outcome exact aliases remain in these 120 development states.

## Result and Next Step

t143292 completed exit0 on node005: 28 fits/2,800 updates qualified.
The 1,645,315-byte result was auto-synced; no extra pull was needed.

| Split | root397 cost | root401 cost | Overall wait, min |
|---|---:|---:|---:|
| Scene | +0.0005383 | -0.0047949 | +0.0177667 |
| Regime | +0.0152034 | -0.0052567 | +0.0172083 |
| Lower root | +0.0132508 | -0.0131310 | +0.0059750 |

Primary improves regime/lower-root cost and wait relative to Stage183,
but worsens scene results. All both-root cost/wait gates remain false.
Scene selection predicts -0.0214251 min wait but realizes +0.0177667;
28/51 changed choices are falsely feasible. Posthoc outcome-oracle
diagnosis finds joint improvements in 31/60 root397 and 42/60 root401
states. The action pool has room; eliminating this alias is not sufficient.
Next: ordered full tentative timetables, cross-direction scheduled-launch
state, and independent development-scene coverage for wait-value training.

## Limitations

Two summaries distinguish the observed alias but do not uniquely encode
the full tentative timetable. This is a development representation test,
not learned upper-policy or end-to-end HRL confirmation. The outcome oracle
is diagnostic, not an executable controller. The third/fourth
action-label scenes remain unused unless a separately registered gate passes.
