# PointMaze Exogenous Frequency-Routing Stage-6 Analysis

Protocol: `pointmaze_exogenous_frequency_routing_stage6_v1`
Independent optimizer roots: 8
Strict Freq-HRL routing claim: **not_supported**

| Method | Tracking success mean [95% CI] | Return | RMSE |
|---|---:|---:|---:|
| flat_exogenous_history | 0.933 [0.877, 0.990] | 229.547 | 0.298 |
| flat_exogenous_filtered | 0.729 [0.635, 0.822] | 213.576 | 0.404 |
| flat_exogenous_multiscale_all | 0.875 [0.791, 0.959] | 230.545 | 0.301 |
| hrl_exogenous_history | 0.880 [0.837, 0.922] | 233.520 | 0.296 |
| hrl_exogenous_filtered | 0.859 [0.826, 0.891] | 228.508 | 0.320 |
| hrl_exogenous_multiscale_all | 0.929 [0.906, 0.953] | 236.712 | 0.275 |
| hrl_exogenous_multiscale_routed | 0.935 [0.899, 0.971] | 239.599 | 0.263 |
| hrl_exogenous_multiscale_swapped | 0.945 [0.927, 0.963] | 239.921 | 0.260 |

| Root-paired contrast | Success improvement [95% CI] | Status |
|---|---:|---|
| flat_filtered_vs_history | -0.205 [-0.326, -0.083] | contradicted |
| flat_multiscale_vs_history | -0.058 [-0.166, 0.049] | inconclusive |
| hrl_filtered_vs_history | -0.021 [-0.060, 0.017] | inconclusive |
| hrl_multiscale_vs_history | 0.050 [0.003, 0.096] | supported |
| routed_vs_history | 0.055 [0.033, 0.077] | supported |
| routed_vs_filtered | 0.076 [0.040, 0.113] | supported |
| routed_vs_all_band | 0.005 [-0.041, 0.052] | inconclusive |
| routed_vs_swapped | -0.010 [-0.050, 0.029] | inconclusive |

Hierarchy x multiscale success interaction: 0.108 [-0.017, 0.233], **inconclusive**.

The optimizer root is the statistical unit; held-out episodes are averaged within root. Failed gate components remain failed.
