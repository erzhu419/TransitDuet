# Freq-HRL Stage113 Upper Residual Result

## Protocol

Stage113 retrained only a zero-initialized `390 -> 4` upper residual readout. The Bernstein plan remained anchored to the causal forecast, the Stage112 learned lower branch was frozen, and the original upper base, standard deviation, value functions, and critic were frozen. A learned-blinded arm removed the upper plan at evaluation while retaining the same lower checkpoint.

Full run: `pointmaze_upper_residual_train_stage113_full_20261004_r1`.

- 8 roots, 2 periods, 8 updates, 32 paired evaluation episodes per period and root.
- 9,728 native episodes and 11,673,600 lower steps.
- Mechanical qualification passed; all frozen-path and budget checks passed.

## Result

| endpoint | mean | corrected CI | status |
|---|---:|---:|---|
| 50 / learned - forecast | `+0.0002635` | `[-0.0002008, +0.0007299]` | inconclusive |
| 50 / learned - learned-blinded | `+0.0002635` | `[-0.0002008, +0.0007299]` | inconclusive |
| 100 / learned - forecast | `+0.0001022` | `[-0.0001683, +0.0003915]` | inconclusive |
| 100 / learned - learned-blinded | `+0.0001022` | `[-0.0001683, +0.0003915]` | inconclusive |

Gate: `upper_gain_gate=not_supported`.

## Interpretation

The explicit forecast-anchored residual path is now executable and trainable, but this training rule does not establish a reward improvement. The identical blinded contrasts are expected from the paired evaluation construction: the learned-blinded arm is the forecast execution with the learned lower checkpoint retained. The result therefore does not support a claim that upper residual learning improves native reward or that the learned plan has separated from the forecast strongly enough to matter.

This is a valid negative result, not a smoke-test failure. No upper gain is admitted and no full actor-critic claim is reopened.

## Next Factor

The next isolated change is decision-level upper credit: use local option returns aligned to each upper decision, while keeping the Stage113 forecast-anchored Bernstein representation, Stage112 lower branch, seeds, evaluation roster, and freeze contract fixed. The purpose is to test whether whole-episode MC credit is diluting the upper signal. If that also fails, the remaining issue is upper state/action representation rather than execution wiring.
