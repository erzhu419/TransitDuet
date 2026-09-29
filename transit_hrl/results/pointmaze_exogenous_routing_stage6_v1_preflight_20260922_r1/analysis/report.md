# PointMaze Exogenous Frequency-Routing Stage-6 Analysis

Protocol: `pointmaze_exogenous_frequency_routing_stage6_v1`
Independent optimizer roots: 1
Strict Freq-HRL routing claim: **not_supported**

| Method | Tracking success mean [95% CI] | Return | RMSE |
|---|---:|---:|---:|
| flat_exogenous_history | 0.938 [-inf, inf] | 49.428 | 0.286 |
| flat_exogenous_filtered | 0.969 [-inf, inf] | 49.656 | 0.279 |
| flat_exogenous_multiscale_all | 1.000 [-inf, inf] | 50.250 | 0.264 |
| hrl_exogenous_history | 0.641 [-inf, inf] | 45.015 | 0.453 |
| hrl_exogenous_filtered | 0.656 [-inf, inf] | 45.512 | 0.435 |
| hrl_exogenous_multiscale_all | 0.594 [-inf, inf] | 43.351 | 0.494 |
| hrl_exogenous_multiscale_routed | 0.781 [-inf, inf] | 48.020 | 0.337 |
| hrl_exogenous_multiscale_swapped | 0.594 [-inf, inf] | 43.138 | 0.502 |

| Root-paired contrast | Success improvement [95% CI] | Status |
|---|---:|---|
| flat_filtered_vs_history | 0.031 [-inf, inf] | inconclusive |
| flat_multiscale_vs_history | 0.062 [-inf, inf] | inconclusive |
| hrl_filtered_vs_history | 0.016 [-inf, inf] | inconclusive |
| hrl_multiscale_vs_history | -0.047 [-inf, inf] | inconclusive |
| routed_vs_history | 0.141 [-inf, inf] | inconclusive |
| routed_vs_filtered | 0.125 [-inf, inf] | inconclusive |
| routed_vs_all_band | 0.188 [-inf, inf] | inconclusive |
| routed_vs_swapped | 0.188 [-inf, inf] | inconclusive |

Hierarchy x multiscale success interaction: -0.109 [-inf, inf], **inconclusive**.

The optimizer root is the statistical unit; held-out episodes are averaged within root. Failed gate components remain failed.
