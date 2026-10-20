# FreqDuet Scheduling

- Submit every FreqDuet scheduler task with explicit `priority: "high"`
  (CLI: `--priority high`), including preflight, collection and analysis tasks.
- After submission, manually invoke scheduler `dispatch --task-id ...` for the
  submitted tasks. Do not cancel or duplicate healthy running tasks.
- Run experiments on node001-node006 only. Local execution is orchestration
  and small-summary inspection, not simulation, training or statistical jobs.
- Inspect scheduler retry counters and logs before calling a queued task a
  failed retry. Pull logs or small summaries, never raw caches or checkpoints.
