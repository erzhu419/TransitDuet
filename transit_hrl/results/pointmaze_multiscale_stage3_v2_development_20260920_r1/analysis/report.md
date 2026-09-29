# PointMaze Multiscale Stage-3 Analysis

Protocol: `pointmaze_multiscale_goal_stage3_v2`
Evaluation rows: 2560
Independent optimizer roots: 8
Freq-HRL mainline gate: **not_supported**

## clean

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| flat_history | 0.461 [0.349, 0.572] | 97.878 | 1.246 |
| flat_multiscale | 0.680 [0.507, 0.853] | 117.752 | 0.914 |
| hrl_history | 0.367 [0.251, 0.484] | 84.766 | 1.505 |
| hrl_multiscale | 0.594 [0.396, 0.791] | 121.064 | 0.902 |
| flat_causal_filter | 0.656 [0.577, 0.735] | 108.114 | 1.157 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Flat representation | supported (+0.219) | supported (+19.874) | supported (+0.332) |
| Hierarchy on raw history | inconclusive (-0.094) | inconclusive (-13.112) | inconclusive (-0.259) |
| Freq routing increment | inconclusive (+0.227) | supported (+36.298) | supported (+0.602) |
| Freq-HRL vs flat multiscale | inconclusive (-0.086) | inconclusive (+3.312) | inconclusive (+0.012) |
| Flat multiscale vs causal filter | inconclusive (+0.023) | inconclusive (+9.638) | supported (+0.243) |
| Freq-HRL vs causal filter | inconclusive (-0.062) | inconclusive (+12.950) | supported (+0.255) |
| Factorial interaction | inconclusive (+0.008) | inconclusive (+16.424) | inconclusive (+0.271) |

## fast_observation_noise

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| flat_history | 0.609 [0.476, 0.743] | 105.358 | 1.170 |
| flat_multiscale | 0.766 [0.648, 0.883] | 120.893 | 0.964 |
| hrl_history | 0.273 [0.108, 0.439] | 77.592 | 1.579 |
| hrl_multiscale | 0.594 [0.463, 0.725] | 119.899 | 0.953 |
| flat_causal_filter | 0.461 [0.393, 0.529] | 103.648 | 1.231 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Flat representation | inconclusive (+0.156) | supported (+15.535) | inconclusive (+0.207) |
| Hierarchy on raw history | contradicted (-0.336) | contradicted (-27.766) | contradicted (-0.409) |
| Freq routing increment | supported (+0.320) | supported (+42.307) | supported (+0.626) |
| Freq-HRL vs flat multiscale | contradicted (-0.172) | inconclusive (-0.994) | inconclusive (+0.011) |
| Flat multiscale vs causal filter | supported (+0.305) | supported (+17.245) | supported (+0.267) |
| Freq-HRL vs causal filter | inconclusive (+0.133) | supported (+16.251) | supported (+0.277) |
| Factorial interaction | inconclusive (+0.164) | supported (+26.772) | supported (+0.419) |

## slow_drift_fast_action

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| flat_history | 0.594 [0.413, 0.775] | 99.830 | 1.288 |
| flat_multiscale | 0.648 [0.544, 0.753] | 109.006 | 1.093 |
| hrl_history | 0.414 [0.314, 0.515] | 84.632 | 1.576 |
| hrl_multiscale | 0.672 [0.545, 0.799] | 112.683 | 1.102 |
| flat_causal_filter | 0.664 [0.533, 0.795] | 94.080 | 1.236 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Flat representation | inconclusive (+0.055) | inconclusive (+9.176) | inconclusive (+0.195) |
| Hierarchy on raw history | inconclusive (-0.180) | contradicted (-15.198) | contradicted (-0.288) |
| Freq routing increment | supported (+0.258) | supported (+28.051) | supported (+0.474) |
| Freq-HRL vs flat multiscale | inconclusive (+0.023) | inconclusive (+3.677) | inconclusive (-0.009) |
| Flat multiscale vs causal filter | inconclusive (-0.016) | inconclusive (+14.925) | inconclusive (+0.143) |
| Freq-HRL vs causal filter | inconclusive (+0.008) | inconclusive (+18.603) | inconclusive (+0.134) |
| Factorial interaction | inconclusive (+0.203) | inconclusive (+18.875) | inconclusive (+0.279) |

## persistent_action_shift

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| flat_history | 0.555 [0.398, 0.711] | 100.747 | 1.217 |
| flat_multiscale | 0.680 [0.577, 0.782] | 122.245 | 0.959 |
| hrl_history | 0.297 [0.183, 0.411] | 84.045 | 1.466 |
| hrl_multiscale | 0.523 [0.363, 0.684] | 112.760 | 1.088 |
| flat_causal_filter | 0.625 [0.467, 0.783] | 97.785 | 1.198 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Flat representation | inconclusive (+0.125) | supported (+21.498) | supported (+0.258) |
| Hierarchy on raw history | contradicted (-0.258) | contradicted (-16.702) | inconclusive (-0.249) |
| Freq routing increment | supported (+0.227) | supported (+28.715) | supported (+0.378) |
| Freq-HRL vs flat multiscale | inconclusive (-0.156) | inconclusive (-9.485) | inconclusive (-0.129) |
| Flat multiscale vs causal filter | inconclusive (+0.055) | supported (+24.461) | supported (+0.239) |
| Freq-HRL vs causal filter | inconclusive (-0.102) | inconclusive (+14.976) | inconclusive (+0.110) |
| Factorial interaction | inconclusive (+0.102) | inconclusive (+7.217) | inconclusive (+0.120) |

## Claim gate

- fast_observation_noise Freq routing success: **supported**
- fast_observation_noise factorial interaction: **inconclusive**
- slow_drift_fast_action Freq routing success: **supported**
- slow_drift_fast_action factorial interaction: **inconclusive**
- Observation-noise Freq-HRL versus causal filter: **inconclusive**
- Clean success noninferiority: **supported**

Cross-stress Freq-HRL requires positive success increments and hierarchy-by-multiscale interactions in both registered primary stress families, superiority to the causal-filter control under observation noise, and clean success noninferiority.
