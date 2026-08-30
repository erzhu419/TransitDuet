# MuJoCo v17.8 Causal FIR Distillation Outcome

## Decision

Status: `grouped_causal_fir_stopped_before_fresh_path_access`.

The 120 reused v17.4 paths were exported successfully on node003 and evaluated
with eight leave-one-seed-out folds. All five disturbance modes for each held
seed remained outside that fold's fit. The selected shared candidate was
`fir_w64_ridge1e-05_gain0.80`.

The selected candidate was numerically valid and met the endpoint upper-HPF8
budget on all 120 paths, but it recovered only 7 of 81 oracle-recoverable
failures and preserved none of the 32 baseline-feasible Walker2d paths. Its
environment recovery counts were 7/40 HalfCheetah, 0/33 Hopper, and 0/8
Walker2d. The strict advancement gate failed, so none of the eight frozen fresh
validation seeds was accessed.

## Diagnostic Boundary

The out-of-fold `fir_w48_ridge1e-05_gain1.00` candidate recovered 58 failures
and preserved 32/32 baseline-feasible Walker2d paths. It failed the upper budget
on 30 paths: 36/40 HalfCheetah, 14/40 Hopper, and 40/40 Walker2d paths met the
upper budget. This identifies the next mechanism precisely. Global gain
reduction controls upper high-frequency power but also removes the low-frequency
upper component, producing severe lower drift in Walker2d. The next candidate
must regulate only the predicted upper high-frequency innovation.

## Execution Accounting

- Dataset tasks `t85715` through `t85834`: 120 done, 0 failed, node003.
- Prelaunch task `t85835`: cancelled after a missing CPU-work declaration was
  detected; it did not execute the selector.
- Corrected selection task `t85836`: done on node003 with 8 CPU cores.
- Server-only arrays: 120 compressed paths, 5.5 MB total.
- Local synchronization: JSON markers and the selected model/summary only; no
  checkpoint or training `.npz` was pulled.

## Claim Boundary

This is reused-path grouped development evidence with a frozen total action.
It does not establish fresh-seed generalization, reward improvement, closed-loop
learning, leakage no-tradeoff, or a manuscript performance claim.
