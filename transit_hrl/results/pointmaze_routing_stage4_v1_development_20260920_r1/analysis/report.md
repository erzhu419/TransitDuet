# PointMaze Stage-4 Frequency-Routing Attribution

Protocol: `pointmaze_frequency_routing_stage4_v1`
Evaluation rows: 1920
Independent optimizer roots: 8
Routing attribution gate: **not_supported**

## clean

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| hrl_history | 0.344 [0.229, 0.459] | 87.950 | 1.389 |
| hrl_causal_filter | 0.375 [0.222, 0.528] | 91.316 | 1.294 |
| hrl_multiscale_all | 0.484 [0.337, 0.632] | 101.342 | 1.180 |
| hrl_multiscale_routed | 0.508 [0.422, 0.594] | 110.305 | 1.052 |
| hrl_multiscale_swapped | 0.711 [0.583, 0.839] | 130.285 | 0.894 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Routed vs history | supported (+0.164) | supported (+22.356) | supported (+0.337) |
| Routed vs causal filter | inconclusive (+0.133) | supported (+18.990) | supported (+0.243) |
| Routed vs all bands | inconclusive (+0.023) | inconclusive (+8.964) | inconclusive (+0.128) |
| Routed vs swapped | contradicted (-0.203) | contradicted (-19.979) | inconclusive (-0.157) |
| All bands vs history | supported (+0.141) | inconclusive (+13.392) | inconclusive (+0.209) |

## fast_observation_noise

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| hrl_history | 0.258 [0.177, 0.339] | 76.174 | 1.556 |
| hrl_causal_filter | 0.336 [0.196, 0.475] | 76.315 | 1.537 |
| hrl_multiscale_all | 0.391 [0.299, 0.482] | 93.581 | 1.326 |
| hrl_multiscale_routed | 0.516 [0.428, 0.603] | 105.252 | 1.140 |
| hrl_multiscale_swapped | 0.688 [0.609, 0.766] | 132.125 | 0.804 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Routed vs history | supported (+0.258) | supported (+29.078) | supported (+0.417) |
| Routed vs causal filter | supported (+0.180) | supported (+28.937) | supported (+0.398) |
| Routed vs all bands | inconclusive (+0.125) | inconclusive (+11.670) | inconclusive (+0.187) |
| Routed vs swapped | contradicted (-0.172) | contradicted (-26.873) | contradicted (-0.335) |
| All bands vs history | supported (+0.133) | supported (+17.408) | inconclusive (+0.230) |

## slow_drift_fast_action

| Method | Success [95% CI] | Return | Final distance |
|---|---:|---:|---:|
| hrl_history | 0.281 [0.173, 0.389] | 84.895 | 1.438 |
| hrl_causal_filter | 0.320 [0.222, 0.419] | 89.449 | 1.335 |
| hrl_multiscale_all | 0.344 [0.174, 0.514] | 93.990 | 1.350 |
| hrl_multiscale_routed | 0.375 [0.287, 0.463] | 99.603 | 1.269 |
| hrl_multiscale_swapped | 0.766 [0.705, 0.826] | 136.948 | 0.767 |

| Contrast | Success | Return | Final distance |
|---|---|---|---|
| Routed vs history | inconclusive (+0.094) | supported (+14.708) | inconclusive (+0.169) |
| Routed vs causal filter | inconclusive (+0.055) | inconclusive (+10.153) | inconclusive (+0.067) |
| Routed vs all bands | inconclusive (+0.031) | inconclusive (+5.612) | inconclusive (+0.082) |
| Routed vs swapped | contradicted (-0.391) | contradicted (-37.346) | contradicted (-0.502) |
| All bands vs history | inconclusive (+0.062) | inconclusive (+9.096) | inconclusive (+0.087) |

## Claim gate

- fast_observation_noise routed vs all: **inconclusive**
- fast_observation_noise routed vs swapped: **contradicted**
- slow_drift_fast_action routed vs all: **inconclusive**
- slow_drift_fast_action routed vs swapped: **contradicted**
- Observation routed vs causal filter: **supported**
- Clean routed vs all noninferiority: **supported**

Selective routing is supported only if it beats all-band and swapped routing in both primary stresses, beats causal filtering under observation noise, and is clean-noninferior to all-band HRL.
