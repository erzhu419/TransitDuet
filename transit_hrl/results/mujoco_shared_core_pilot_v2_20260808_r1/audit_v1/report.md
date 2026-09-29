# MuJoCo Development Pilot Audit

Analysis: `mujoco_development_pilot_audit_v1`

This report is development-only. Held-out paths inside one optimizer replicate are repeated measurements, not independent training replicates.

## Integrity

- Status: **valid**
- Cells: 36/36
- Independent optimizer replicates per method/environment: 3
- Issues: 0
- Warnings: 36

## Standard Task

| Environment | Method | Independent n | Return mean | Return SD | Validation gain | Selected iteration |
|---|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | flat_ppo | 3 | 1111.546 | 528.464 | 1.053 | 47.000 |
| HalfCheetah-v5 | freq_hrl | 3 | 247.111 | 256.229 | 0.232 | 59.000 |
| HalfCheetah-v5 | freq_hrl_no_leakage | 3 | 307.311 | 361.986 | 0.237 | 60.333 |
| HalfCheetah-v5 | generic_hrl | 3 | 21.891 | 18.309 | 0.009 | 61.667 |
| Hopper-v5 | flat_ppo | 3 | 192.270 | 16.505 | 1.260 | 35.000 |
| Hopper-v5 | freq_hrl | 3 | 142.985 | 8.207 | 0.910 | 63.000 |
| Hopper-v5 | freq_hrl_no_leakage | 3 | 147.742 | 25.237 | 0.884 | 63.000 |
| Hopper-v5 | generic_hrl | 3 | 138.483 | 40.020 | 0.890 | 63.000 |
| Walker2d-v5 | flat_ppo | 3 | 247.436 | 2.155 | 0.995 | 48.333 |
| Walker2d-v5 | freq_hrl | 3 | 266.231 | 6.207 | 0.599 | 29.667 |
| Walker2d-v5 | freq_hrl_no_leakage | 3 | 316.389 | 45.123 | 0.720 | 29.667 |
| Walker2d-v5 | generic_hrl | 3 | 271.610 | 0.874 | 0.596 | 31.000 |

## Primary Paired Diagnostics

Positive improvement favors the treatment. These rows are not paper claims.

| Environment | Treatment | Control | Independent n | Mean improvement | Win rate |
|---|---|---|---:|---:|---:|
| HalfCheetah-v5 | freq_hrl | flat_ppo | 3 | -864.435 | 0.000 |
| HalfCheetah-v5 | freq_hrl | generic_hrl | 3 | 225.220 | 1.000 |
| HalfCheetah-v5 | freq_hrl | freq_hrl_no_leakage | 3 | -60.201 | 0.333 |
| HalfCheetah-v5 | freq_hrl_no_leakage | generic_hrl | 3 | 285.421 | 0.667 |
| HalfCheetah-v5 | generic_hrl | flat_ppo | 3 | -1089.655 | 0.000 |
| Hopper-v5 | freq_hrl | flat_ppo | 3 | -49.285 | 0.000 |
| Hopper-v5 | freq_hrl | generic_hrl | 3 | 4.502 | 0.667 |
| Hopper-v5 | freq_hrl | freq_hrl_no_leakage | 3 | -4.757 | 0.667 |
| Hopper-v5 | freq_hrl_no_leakage | generic_hrl | 3 | 9.259 | 0.667 |
| Hopper-v5 | generic_hrl | flat_ppo | 3 | -53.787 | 0.333 |
| Walker2d-v5 | freq_hrl | flat_ppo | 3 | 18.795 | 1.000 |
| Walker2d-v5 | freq_hrl | generic_hrl | 3 | -5.378 | 0.000 |
| Walker2d-v5 | freq_hrl | freq_hrl_no_leakage | 3 | -50.158 | 0.333 |
| Walker2d-v5 | freq_hrl_no_leakage | generic_hrl | 3 | 44.780 | 0.667 |
| Walker2d-v5 | generic_hrl | flat_ppo | 3 | 24.173 | 1.000 |

## Evidence Warnings

- `checkpoint_file_hash_not_recorded_by_legacy_protocol`

## Gate

A valid pilot may select protocol and compute budget only. Formal evaluation requires fresh optimizer seeds, untouched evaluation seeds, a frozen source manifest, and multiplicity-controlled confirmatory analysis.
