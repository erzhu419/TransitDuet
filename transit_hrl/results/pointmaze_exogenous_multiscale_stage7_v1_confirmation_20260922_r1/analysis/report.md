# PointMaze Exogenous Multiscale Stage-7 Analysis

Runtime protocol: `pointmaze_exogenous_frequency_routing_stage6_v1`
Independent optimizer roots: 16
Fresh multiscale-HRL confirmation: **not_supported**

| Method | Tracking success mean [95% CI] | Return | RMSE |
|---|---:|---:|---:|
| flat_exogenous_history | 0.872 [0.836, 0.907] | 225.816 | 0.319 |
| flat_exogenous_multiscale_all | 0.850 [0.778, 0.921] | 223.100 | 0.343 |
| hrl_exogenous_history | 0.906 [0.887, 0.926] | 235.242 | 0.285 |
| hrl_exogenous_multiscale_all | 0.908 [0.874, 0.942] | 235.141 | 0.284 |

| Root-paired contrast | Success improvement [95% CI] | Status |
|---|---:|---|
| flat_multiscale_vs_history | -0.022 [-0.099, 0.055] | inconclusive |
| hrl_history_vs_flat_history | 0.035 [-0.012, 0.081] | inconclusive |
| hrl_multiscale_vs_history | 0.002 [-0.029, 0.033] | inconclusive |
| hrl_multiscale_vs_flat_multiscale | 0.058 [-0.000, 0.117] | inconclusive |

Hierarchy x multiscale success interaction: 0.024 [-0.049, 0.096], **inconclusive**.

The optimizer root is the statistical unit. This fixed 16-root confirmation cannot be extended after inspecting its result.
