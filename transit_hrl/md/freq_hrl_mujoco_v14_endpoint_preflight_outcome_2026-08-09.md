# MuJoCo v14 Endpoint-Aligned Preflight Outcome

## Status

This is a source-bound mechanism preflight, not performance evidence.

- Run: `mujoco_v14_endpoint_aligned_preflight_20260809_r1`
- Algorithm revision: `ae5f0c46078e97d6aa3ea291fe5d011b7b879d7f`
- Freq-HRL manifest: `2beec62e6234c00c1b95c8dd50e5e6744851c8118f11353ca34bdfef5b042659`
- Protocol: `freq_hrl_mujoco_shared_core_v14_endpoint_aligned_training`
- Scheduler tasks: `t75794` through `t75796`
- Dynamic placement: `node001`, `node004`, and `node005`; no node pinning
- Completion: 3/3 result-synced cells and 3/3 checkpoints

## Endpoint identity

For each held-out preflight episode, the online causal filters were compared to
the final batch diagnostics. The largest observed absolute discrepancy was
approximately `1.1e-13`:

| Identity | Observed discrepancy scale |
|---|---:|
| `LowerLFPowerOnlineMean == LowerLFDriftAbs` | `1e-14` to `1e-13` |
| `RawLowerLFPowerOnlineMean == RawLowerLFDriftAbs` | `1e-14` to `1e-13` |
| `UpperHFPowerOnlineMean == UpperHFPowerAbs` | `1e-15` |

The full-method cell reported `joint_behavior`, upper-HF coefficient `2.0`, and
responsibility reconstruction RMS `5.08e-11`. The two-iteration safe-selector
cell fell back to `no_leakage`; this is expected for a path preflight and is not
a comparative result.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `freq_hrl/cell_summary.json` | `7edb725d71924abc8f7817411292918b610e7e6282d141e5fe2ed7acdd3fca9a` |
| `freq_hrl_no_leakage/cell_summary.json` | `28bd4b44bcd74c70caa8a1c1d8ccc2adba2992f8c2ebbcea874c1f6ca1f3b109` |
| `freq_hrl_safe_selector/cell_summary.json` | `dcc65a3dd047af1353ecf811e403a015336ec91fcbce1b3da8b40c2d972ab851` |
| merged `summary.json` | `0fcda8e9410490066a7b92e6237de8e7e5d993f0e96ffc96775107a7843379be` |

The endpoint-aligned revision supersedes the initial `1e7c8d5841` preflight and
is the only admissible source for the v14 behavior screen.
