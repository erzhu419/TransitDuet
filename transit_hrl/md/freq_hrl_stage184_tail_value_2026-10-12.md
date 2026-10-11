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

Twenty-four focused tests pass (five new, nineteen shared-core regression
tests). The real recorded alias is separated by the new feature map;
no distinct-outcome exact aliases remain in these 120 development states.

## Limitations

Two summaries distinguish the observed alias but do not uniquely encode
the full tentative timetable. This is a development representation test,
not learned upper-policy or end-to-end HRL confirmation. The third/fourth
action-label scenes remain unused unless a separately registered gate passes.
