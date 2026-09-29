# PointMaze Stage-4 Frequency-Routing Attribution

Protocol: `pointmaze_frequency_routing_stage4_v2`
Evaluation rows: 1920
Independent optimizer roots: 8
Routing attribution gate: **not_supported**

## clean

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| hrl_history | 0.312 [0.188, 0.437] | 89.230 | 1.407 |
| hrl_causal_filter | 0.320 [0.197, 0.443] | 92.370 | 1.399 |
| hrl_multiscale_all | 0.516 [0.335, 0.696] | 104.184 | 1.221 |
| hrl_multiscale_routed | 0.648 [0.470, 0.827] | 115.556 | 1.038 |
| hrl_multiscale_swapped | 0.812 [0.716, 0.909] | 138.145 | 0.750 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Routed vs history | supported (+0.336) | supported (+26.326) | supported (+0.369) |
| Routed vs causal filter | supported (+0.328) | supported (+23.186) | supported (+0.361) |
| Routed vs all bands | supported (+0.133) | supported (+11.373) | inconclusive (+0.183) |
| Routed vs swapped | inconclusive (-0.164) | contradicted (-22.589) | contradicted (-0.288) |
| All bands vs history | inconclusive (+0.203) | inconclusive (+14.953) | inconclusive (+0.186) |

## fast_observation_noise

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| hrl_history | 0.297 [0.145, 0.449] | 76.242 | 1.651 |
| hrl_causal_filter | 0.383 [0.284, 0.481] | 86.518 | 1.450 |
| hrl_multiscale_all | 0.430 [0.300, 0.559] | 96.035 | 1.289 |
| hrl_multiscale_routed | 0.539 [0.356, 0.722] | 114.190 | 1.038 |
| hrl_multiscale_swapped | 0.648 [0.463, 0.834] | 129.894 | 0.933 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Routed vs history | supported (+0.242) | supported (+37.948) | supported (+0.613) |
| Routed vs causal filter | supported (+0.156) | supported (+27.672) | supported (+0.412) |
| Routed vs all bands | inconclusive (+0.109) | supported (+18.155) | supported (+0.251) |
| Routed vs swapped | inconclusive (-0.109) | inconclusive (-15.703) | inconclusive (-0.105) |
| All bands vs history | inconclusive (+0.133) | supported (+19.793) | supported (+0.362) |

## slow_drift_fast_action

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| hrl_history | 0.336 [0.268, 0.404] | 75.692 | 1.658 |
| hrl_causal_filter | 0.359 [0.268, 0.451] | 89.005 | 1.438 |
| hrl_multiscale_all | 0.492 [0.366, 0.618] | 103.314 | 1.117 |
| hrl_multiscale_routed | 0.500 [0.372, 0.628] | 104.122 | 1.240 |
| hrl_multiscale_swapped | 0.703 [0.534, 0.872] | 130.757 | 0.955 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Routed vs history | supported (+0.164) | supported (+28.430) | supported (+0.417) |
| Routed vs causal filter | supported (+0.141) | supported (+15.116) | inconclusive (+0.197) |
| Routed vs all bands | inconclusive (+0.008) | inconclusive (+0.808) | inconclusive (-0.123) |
| Routed vs swapped | contradicted (-0.203) | contradicted (-26.635) | contradicted (-0.286) |
| All bands vs history | supported (+0.156) | supported (+27.622) | supported (+0.541) |

## Claim gate

- fast_observation_noise routed vs all: **inconclusive**
- fast_observation_noise routed vs swapped: **inconclusive**
- slow_drift_fast_action routed vs all: **inconclusive**
- slow_drift_fast_action routed vs swapped: **contradicted**
- Observation routed vs causal filter: **supported**
- Clean routed vs all noninferiority: **supported**

Selective routing is supported only if it beats all-band and swapped routing in both primary stresses, beats causal filtering under observation noise, and is clean-noninferior to all-band HRL.
