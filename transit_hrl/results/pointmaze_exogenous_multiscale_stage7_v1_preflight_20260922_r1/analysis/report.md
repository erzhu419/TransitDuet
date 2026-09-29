# PointMaze Exogenous Multiscale Stage-7 Analysis

Runtime protocol: `pointmaze_exogenous_frequency_routing_stage6_v1`
Independent optimizer roots: 1
Fresh multiscale-HRL confirmation: **not_supported**

| Method | Tracking success mean [95% CI] | Return | RMSE |
|---|---:|---:|---:|
| flat_exogenous_history | 0.531 [-inf, inf] | 40.494 | 0.554 |
| flat_exogenous_multiscale_all | 0.500 [-inf, inf] | 39.937 | 0.573 |
| hrl_exogenous_history | 0.750 [-inf, inf] | 45.709 | 0.365 |
| hrl_exogenous_multiscale_all | 0.766 [-inf, inf] | 46.069 | 0.358 |

| Root-paired contrast | Success improvement [95% CI] | Status |
|---|---:|---|
| flat_multiscale_vs_history | -0.031 [-inf, inf] | inconclusive |
| hrl_history_vs_flat_history | 0.219 [-inf, inf] | inconclusive |
| hrl_multiscale_vs_history | 0.016 [-inf, inf] | inconclusive |
| hrl_multiscale_vs_flat_multiscale | 0.266 [-inf, inf] | inconclusive |

Hierarchy x multiscale success interaction: 0.047 [-inf, inf], **inconclusive**.

The optimizer root is the statistical unit. This fixed 16-root confirmation cannot be extended after inspecting its result.
