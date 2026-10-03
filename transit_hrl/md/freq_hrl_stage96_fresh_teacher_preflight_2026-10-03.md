# Stage96 Native Preflight

- t127012 / t127013: done, exit 0, node006. Native build: 12 episodes / 3,600 steps; worker result wall time 8.262 s.
- Final checkpoint, optimizer budgets, warmup actor/upper-value freeze and BC std/upper/value freeze passed. Four teacher-label archives have the registered plan and velocity shapes. No historical artifacts were loaded.
- Initial/final controller diagnostic means: 188.925 -> 132.811. Retained as a negative short-budget diagnostic, not a root-admission or performance gate.
- Retrieved compact JSON only; four checkpoints, forecaster and label archives remain server-only. Scheduler peak RAM was not measured on these short tasks.
- Frozen full cohort is unchanged: eight fresh roots, 384 x 8 native training paths per root, H=1200; total 25,984 episodes / 31,180,800 steps. Native preflight authorizes its launch.

Next: run the full teacher build, then fresh decoder calibration and fresh lower/upper donors. Unseen-teacher reward confirmation has not been tested yet.
