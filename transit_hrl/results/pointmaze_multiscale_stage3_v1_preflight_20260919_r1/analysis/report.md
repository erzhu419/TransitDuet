# PointMaze Multiscale Stage-3 Analysis

Protocol: `pointmaze_multiscale_goal_stage3_v1`
Evaluation rows: 8
Independent optimizer roots: 1
Freq-HRL mainline gate: **not_supported**

## clean

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| flat_history | 0.000 [-inf, inf] | 24.717 | 0.911 |
| flat_multiscale | 0.000 [-inf, inf] | 23.995 | 0.997 |
| hrl_history | 0.000 [-inf, inf] | 24.313 | 0.957 |
| hrl_multiscale | 0.000 [-inf, inf] | 24.202 | 0.972 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Flat representation | inconclusive (+0.000) | inconclusive (-0.722) | inconclusive (-0.086) |
| Hierarchy on raw history | inconclusive (+0.000) | inconclusive (-0.404) | inconclusive (-0.046) |
| Freq routing increment | inconclusive (+0.000) | inconclusive (-0.111) | inconclusive (-0.014) |
| Freq-HRL vs flat multiscale | inconclusive (+0.000) | inconclusive (+0.207) | inconclusive (+0.025) |
| Factorial interaction | inconclusive (+0.000) | inconclusive (+0.611) | inconclusive (+0.071) |

## mixed_causal_stress

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| flat_history | 0.000 [-inf, inf] | 31.259 | 0.499 |
| flat_multiscale | 0.000 [-inf, inf] | 31.363 | 0.490 |
| hrl_history | 0.000 [-inf, inf] | 30.782 | 0.491 |
| hrl_multiscale | 0.000 [-inf, inf] | 30.770 | 0.492 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Flat representation | inconclusive (+0.000) | inconclusive (+0.103) | inconclusive (+0.008) |
| Hierarchy on raw history | inconclusive (+0.000) | inconclusive (-0.477) | inconclusive (+0.008) |
| Freq routing increment | inconclusive (+0.000) | inconclusive (-0.013) | inconclusive (-0.001) |
| Freq-HRL vs flat multiscale | inconclusive (+0.000) | inconclusive (-0.593) | inconclusive (-0.002) |
| Factorial interaction | inconclusive (+0.000) | inconclusive (-0.116) | inconclusive (-0.010) |

## Claim gate

- Primary-stress Freq routing success: **inconclusive**
- Primary-stress factorial interaction: **inconclusive**
- Clean success noninferiority: **not_supported**

Freq-HRL requires a positive primary-stress success increment, a positive hierarchy-by-multiscale success interaction, and clean success noninferiority; flat representation gains are reported separately.
