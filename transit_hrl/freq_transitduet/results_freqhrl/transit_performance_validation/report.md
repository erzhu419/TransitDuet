# Transit Performance Validation

- source: `transit_hrl/freq_transitduet/results_freqhrl/transit_validation_27seed/freqduet_ablation_per_seed.csv`
- target config: `T_freqhrl_terminal`
- seeds: 27
- evaluation unit: per-seed mean over the retained training/evaluation episodes in the copied runner logs
- paired deltas are `target - baseline`; negative is better for composite/wait/cv/overshoot

## Headline

Composite winner: `T_freqhrl_terminal` (1.695). `T_freqhrl_terminal` composite is 1.695.
Raw-wait winner: `T_swapped_terminal` (6.817 min). `T_freqhrl_terminal` wait is 6.917 min.

## Config Summary

| config | seeds | composite | wait | cv | overshoot |
|---|---:|---:|---:|---:|---:|
| T_freqhrl_terminal | 27 | 1.695 +/- 0.413 | 6.917 +/- 2.206 | 0.470 +/- 0.057 | 1.904 +/- 0.702 |
| T_swapped_terminal | 27 | 1.732 +/- 0.385 | 6.817 +/- 2.345 | 0.496 +/- 0.052 | 2.111 +/- 0.617 |
| T_nopromotion_terminal | 27 | 1.772 +/- 0.500 | 7.589 +/- 2.873 | 0.490 +/- 0.048 | 1.941 +/- 0.720 |
| T_nofreq_terminal | 27 | 1.872 +/- 0.475 | 7.929 +/- 3.847 | 0.504 +/- 0.049 | 2.200 +/- 0.606 |
| T_noleakage_terminal | 27 | 1.903 +/- 0.524 | 8.045 +/- 3.370 | 0.500 +/- 0.050 | 2.215 +/- 0.653 |
| T_lf_upper_terminal | 27 | 1.991 +/- 0.567 | 8.578 +/- 3.974 | 0.507 +/- 0.072 | 2.311 +/- 0.602 |
| T_allfreq_terminal | 27 | 2.020 +/- 0.620 | 8.855 +/- 4.390 | 0.500 +/- 0.056 | 2.363 +/- 0.662 |
| T_rawhistory_terminal | 27 | 2.033 +/- 0.712 | 9.321 +/- 5.582 | 0.496 +/- 0.050 | 2.289 +/- 0.678 |
| T_hf_lower_terminal | 27 | 2.041 +/- 0.626 | 9.408 +/- 4.800 | 0.497 +/- 0.060 | 2.207 +/- 0.590 |

## Paired Target Deltas

| baseline | composite delta | composite CI95 | composite win rate | wait delta | wait CI95 | wait win rate |
|---|---:|---:|---:|---:|---:|---:|
| T_hf_lower_terminal | -0.346 | [-0.643, -0.087] | 0.59 | -2.491 | [-4.726, -0.582] | 0.59 |
| T_rawhistory_terminal | -0.338 | [-0.621, -0.070] | 0.63 | -2.404 | [-4.569, -0.423] | 0.63 |
| T_allfreq_terminal | -0.325 | [-0.518, -0.133] | 0.74 | -1.938 | [-3.350, -0.578] | 0.74 |
| T_lf_upper_terminal | -0.296 | [-0.538, -0.065] | 0.63 | -1.661 | [-3.358, -0.125] | 0.56 |
| T_noleakage_terminal | -0.208 | [-0.416, +0.002] | 0.67 | -1.128 | [-2.401, +0.101] | 0.59 |
| T_nofreq_terminal | -0.177 | [-0.416, +0.059] | 0.56 | -1.012 | [-2.991, +0.688] | 0.48 |
| T_nopromotion_terminal | -0.077 | [-0.330, +0.164] | 0.41 | -0.671 | [-2.026, +0.654] | 0.56 |
| T_swapped_terminal | -0.037 | [-0.198, +0.135] | 0.59 | +0.100 | [-1.002, +1.141] | 0.48 |

## Interpretation

- This is now a paired performance validation over the copied Transit runner logs, not just a smoke run.
- The target Freq-HRL config is best on the composite objective among the 9-config matrix.
- Raw wait remains a caveat: `T_swapped_terminal` has slightly lower mean wait, while `T_freqhrl_terminal` wins composite by reducing the combined wait/CV/fleet-overshoot objective.
- Confidence intervals are bootstrap intervals over seeds; they should be treated as validation evidence, not as a final simulator-training proof.
