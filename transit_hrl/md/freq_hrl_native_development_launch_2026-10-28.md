# Native Development Launch

The user authorized compute on2026-10-28. All execution uses scheduleurm and the
node001-node006 whitelist, without hard pins. No formal confirmation is released.

## Environment

All six compute nodes have Python3.10.14, NumPy2.1.3, PyTorch2.5.1+cu121,
SciPy1.13.1, Gymnasium1.2.0, gymnasium-robotics1.4.2, MuJoCo3.2.7,
PettingZoo1.26.1 and threadpoolctl3.6.0. The reused interpreter is
`/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python`.
No login-node installation or shared-environment modification was needed.
Workers use CPU only, one native thread and2 GiB RAM each.

## Attempts

- t212001-t212006: cancelled before launch. Extra scheduler inputs require
  directories, not the preregistration file. Fixed in `1de8fead50`.
- t212014-t212019: all six native preflights passed on node001/node004;
  measured116.698784 CPU seconds, peak process RSS447-505 MiB.
- t212024-t212035: all twelve pilot workers exited at the runtime-profile gate,
  before any training or simulation. The live BLAS tuples differed from their
  JSON-decoded lists despite identical values. Their public failure records are
  retained:6.647921 measured CPU seconds, all work counters empty. Fixed profile
  serialization and added a JSON-roundtrip regression in `0832c56c18`;
  24 focused tests passed. Worker failures now also appear in the log.
- t212037-t212042: separate `plan_validity_dev_20261028_r2` native preflight,
  with the original seeds, parameters and model-selection rules unchanged.
  All six passed on node001; the corrected gate is complete with matching
  runtime profiles, no failed/missing cells and124.548466 CPU seconds.
- t212043-t212054: the12-cell r2 pilot submitted only after that gate passed.
  PointMaze uses roots1972110000/1972120000 (t212043-t212048); Reacher uses
  roots1972210000/1972220000 (t212049-t212054). Each root's three consecutive
  tasks are history, threshold and termination_history, in that order.
  All12 launched: t212050/t212052/t212054 on node004 and the other nine on
  node001. Actual Python children were alive after about70 seconds, each using
  97.6%-99.2% CPU and459-540 MiB RSS, with no public failure result. Placement
  remains unpinned within the six-node whitelist.

Compact records remain under `results/plan_validity_dev_20261028/` and
`results/plan_validity_dev_20261028_r2/`. Only public JSONs/logs are retrieved;
models, fitting caches and trajectories stay on the server. The initial
123.346705 measured CPU seconds are additional development overhead, not zero
cost or independent successful roots, and must accompany the r2 cost ledger.
These counters exclude scheduler transport and environment-probe overhead.

## Next Gate

Recovery t212077-t212079 completed on2026-10-29, giving12 logical measurements
with the9 retained originals. See the
[complete readout](freq_hrl_native_development_complete_2026-10-29.md).
Reacher stopping competence and interrupted lifecycle costs still block recipe
admission. No confirmation task was submitted.
