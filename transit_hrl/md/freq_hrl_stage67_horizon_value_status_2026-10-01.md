# Stage67 Status and Next Step

Implemented finite-horizon factored values; three small analytic tests passed. Scheduler preflight t116905/node002 and qualification t116906/node006 completed with exit0: all four ordinary-MC controls reproduce Stage64 training/public parameters, Adam and probe exactly; actor/upper remain frozen. Budget:24 archives,7200 lower reconstructions,32 critic steps per treatment,zero native steps or actor steps.

Preflight is mechanically valid but not a positive efficacy result: factored values slightly worsen all four global and tail errors, and the credit gate fails. Retain this result without changing the frozen full protocol. [Compact preflight](../results/pointmaze_horizon_value_stage67_preflight_20261001_r1/compact_summary.json).

Full eight-root tasks t116907-t116914 are running with dynamic placement on node004/005/006; qualification t116915 waits for all completion markers. Same original calibration/probe paths;3CPU/3GB per root. Only31.5KB preflight JSON pulled; no traces or checkpoints. [Full roster](../results/pointmaze_horizon_value_stage67_full_20261001_r1/task_roster.json).

Next: aggregate every full root, assess global/tail fit and actual GAE/MC gradient credit together. Both gates must pass before any guarded actor/native reward experiment. If either fails, keep HOLD and diagnose the time-dependent residual; no seed/LR/threshold expansion to rescue this candidate.
