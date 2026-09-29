# MuJoCo v14.15 Restoration Multiseed Development Screen

- Status: `candidate_not_ready_for_confirmation`
- Candidate: `group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3`
- Optimizer seeds: `15`
- Environments: `3`
- Statistical unit: optimizer seed; held-out paths are not replicates.
- Simultaneous primary gate: `False`
- All mode point gates: `False`
- Complete candidate cells: `8/45`
- One-sided Wilson lower bound: `0.103187`

| environment | metric | mean | simultaneous lower | threshold | pass |
|---|---|---:|---:|---:|---:|
| HalfCheetah-v5 | normalized_episode_return | -0.020402 | -0.119111 | -0.020000 | False |
| HalfCheetah-v5 | LowerLFDriftAbs | -0.002217 | -0.100926 | 0.051293 | False |
| HalfCheetah-v5 | RawLowerLFDriftAbs | -0.002217 | -0.100926 | 0.051293 | False |
| HalfCheetah-v5 | LatentLowerLFDriftAbs | -0.003851 | -0.102561 | 0.051293 | False |
| HalfCheetah-v5 | UpperHFPowerAbs | -0.000913 | -0.099622 | 0.051293 | False |
| HalfCheetah-v5 | LatentUpperHFPowerAbs | -0.001587 | -0.100296 | 0.051293 | False |
| Hopper-v5 | normalized_episode_return | 0.021056 | -0.077653 | -0.020000 | False |
| Hopper-v5 | LowerLFDriftAbs | 0.147073 | 0.048363 | 0.051293 | False |
| Hopper-v5 | RawLowerLFDriftAbs | 0.147073 | 0.048363 | 0.051293 | False |
| Hopper-v5 | LatentLowerLFDriftAbs | 0.188510 | 0.089801 | 0.051293 | True |
| Hopper-v5 | UpperHFPowerAbs | 0.198284 | 0.099575 | 0.051293 | True |
| Hopper-v5 | LatentUpperHFPowerAbs | 0.221038 | 0.122328 | 0.051293 | True |
| Walker2d-v5 | normalized_episode_return | -0.000272 | -0.098981 | -0.020000 | False |
| Walker2d-v5 | LowerLFDriftAbs | 0.027336 | -0.071373 | 0.051293 | False |
| Walker2d-v5 | RawLowerLFDriftAbs | 0.027336 | -0.071373 | 0.051293 | False |
| Walker2d-v5 | LatentLowerLFDriftAbs | 0.028724 | -0.069986 | 0.051293 | False |
| Walker2d-v5 | UpperHFPowerAbs | 0.026171 | -0.072538 | 0.051293 | False |
| Walker2d-v5 | LatentUpperHFPowerAbs | 0.029242 | -0.069467 | 0.051293 | False |

This candidate-fixed screen uses 15 optimizer seeds that were not used to select the v14.15 restoration arm and treats optimizer seed, not held-out path, as the statistical unit. It may authorize a new frozen confirmation protocol. It is still development evidence because the algorithm family and candidate were chosen using earlier MuJoCo outcomes.
