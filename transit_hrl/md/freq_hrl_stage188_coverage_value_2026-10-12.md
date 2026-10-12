# Stage188: Expanded Grouped Value Validation

Stage187 r2 finished480 tasks, all exit0. Qualify the full cohort on a CPU
node and retain its timetables/labels on shared server storage. Combine120
unchanged old states with480 fresh states: 100 groups/600 states/3,600 labels.

Fixed tail_summary versus timetable_sequence, floor_absolute, proper-MSE,
seed178,100 updates per fit and unchanged optimizer. Each mode has100
leave-scene,5 leave-regime,2 leave-lower-root and1 full fit:216 fits/21,600
updates. Both selectors share each fit. No inner tuning, native or actor updates.

Keep the same rootwise cost<0/wait<=0 gate across all three held-out splits.
Report the old120-state subset separately against Stage184/186; comparing
new600-state averages against old120-state averages would mix sample changes
with training-coverage effects. Parallelize folds through scheduler, one CPU/
3GB each, node001-006 unbound. Full bank/models remain server-only.

Six focused tests pass. Next: qualify the bank, run all matched folds and
decide from held-out effects rather than training losses or hindsight oracle.

## Limitations

The action labels still describe one forced action followed by original
greedy continuation, not repeated learned upper control. Six callbacks are
correlated within each scene; two frozen lower roots remain. Additional
development coverage is not independent confirmation or end-to-end HRL.
