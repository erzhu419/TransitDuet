# Freq-HRL Stage114 Upper Local Credit Result

Stage114 kept the Stage113 forecast-anchored Bernstein residual representation, frozen Stage112 lower branch, frozen upper base/std/value/critic, seeds, update count, KL radius, and evaluation roster. The only change was the training signal: each upper decision received its own undiscounted option return, with leave-one-out pairing over the two noise replicas.

Full run: `pointmaze_upper_local_credit_train_stage114_full_20261004_r1`.

- 8 roots, 2 periods, 8 updates, 32 paired evaluation episodes per period and root.
- 9,728 native episodes and 11,673,600 lower steps.
- Mechanical qualification passed; local option-return identity and all freeze/budget checks passed.

| endpoint | mean | corrected CI | status |
|---|---:|---:|---|
| 50 / learned - forecast | `-0.0001531` | `[-0.0009590, +0.0005206]` | inconclusive |
| 50 / learned - learned-blinded | `-0.0001531` | `[-0.0009590, +0.0005206]` | inconclusive |
| 100 / learned - forecast | `+0.0001899` | `[-0.0003279, +0.0006433]` | inconclusive |
| 100 / learned - learned-blinded | `+0.0001899` | `[-0.0003279, +0.0006433]` | inconclusive |

Gate: `upper_gain_gate=not_supported`.

## Interpretation

Decision-aligned credit is now implemented and mechanically verified, but it does not produce a CI-supported native reward gain. Together with Stage113, this rules out a simple explanation based only on whole-episode credit dilution. The remaining upper bottleneck is the narrow four-dimensional residual action/plan coordinate, or the fact that the frozen upper base is not a useful plan-coordinate representation.

No learned upper performance claim is admitted. The next isolated factor is an explicit wider Bernstein plan-coordinate head, initialized at zero residual so the forecast execution remains the registered baseline.
