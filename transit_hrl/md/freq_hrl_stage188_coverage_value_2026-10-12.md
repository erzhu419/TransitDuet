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

Seven focused tests pass. Next: qualify the bank, run all matched folds and
decide from held-out effects rather than training losses or hindsight oracle.

Run native_transit_coverage_value_stage188_fit_20261012_r1, fit code251473b2db.
t144065 completed exit0 on node006: all480 collections and100 combined groups
qualified, bank34,907,152 bytes server-only, compact receipt4,658 bytes.
Posthoc joint-improvable states:397=201/300,401=191/300. These are future-outcome
oracle diagnostics. The216 fit tasks are t144082-t144297.

## Limitations

The action labels still describe one forced action followed by original
greedy continuation, not repeated learned upper control. Six callbacks are
correlated within each scene; two frozen lower roots remain. Additional
development coverage is not independent confirmation or end-to-end HRL.
