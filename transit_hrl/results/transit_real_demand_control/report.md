# Real AFC/APC Demand Control Validation

This replays real passenger-demand traces through the shared Transit PPO surrogate control loop.
Boundary: this is real-demand control replay, not native OD/onboard-load simulation.

## Sources

| source | rows | series | window | boundary |
|---|---:|---:|---|---|
| afc | 1000 | 2 | 2024-10-01T00:00:00 to 2024-10-02T00:00:00 | AFC station entries, not onboard load or OD |
| apc | 1000 | 2 | 2026-01-01 to 2026-01-08 | APC route boardings, not onboard occupancy/alighting/OD |

## Paired Checks

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| real_demand_control_control_objective | supported | control_objective | 6 | +1.8114 | +1.2826 | +2.4539 | 1.00 |
| real_demand_control_reward_mean | supported | reward_mean | 6 | +1.6835 | +1.1721 | +2.3034 | 1.00 |
| real_demand_control_wait_proxy | supported | wait_proxy | 6 | -1.6741 | -2.2818 | -1.1728 | 1.00 |
| real_demand_control_LowerLFDrift | supported | LowerLFDrift | 6 | -0.3026 | -0.3978 | -0.1969 | 1.00 |
| real_demand_control_RawLowerLFDriftAbs | supported | RawLowerLFDriftAbs | 6 | -0.0185 | -0.0191 | -0.0179 | 1.00 |
