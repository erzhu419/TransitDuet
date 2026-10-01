# Stage60 Full Backtracking Result

Full tasks `t110033`-`t110040` and qualification `t110102` completed with exit0. **Decision: valid nonzero first-update control; no native performance launch.** Plain PPO reproduced Stage58, rejection-only reproduced Stage59, and all three final critics/critic Adam matched exactly. Rollback and frozen cost checks passed.

Equal-root means across all eight development roots, first update only:

| Period | Arm | Plain lower mean KL | Backtracking lower mean KL | Action-mean RMS | Clip Fraction | Retained Steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | zero_train | 6.469716 | 0.015906 | 0.026977 | 19.74% | 320/320 |
| 50 | joint_ppo | 5.739111 | 0.017263 | 0.028121 | 21.48% | 320/320 |
| 100 | zero_train | 0.028248 | 0.016120 | 0.027099 | 20.20% | 320/320 |
| 100 | joint_ppo | 0.029707 | 0.015730 | 0.026753 | 19.93% | 320/320 |

All32 lower cases moved and retained1280/1280 steps, versus0/1280 for rejection-only. Lower final conditional mean KL ranged0.009442-0.019823; action-mean RMS0.019480-0.042361. Of1280 lower steps,491 retained full scale and789 retained a scaled proposal; minimum accepted scale1/32. All64 upper steps retained full scale and matched plain PPO.

Backtracking alone evaluated2496 candidates for1344 nominal Adam steps, including1152 extra interpolation trials; guard cost2544 full-batch distribution passes,4266 state snapshots and2304 rollback checks. Retrying added no Adam or critic optimizer steps. Combined with the rejection-only control, guarded cost3936 passes,6954 snapshots and4864 rollback checks; the old control's1280 rejected steps remain in the accounting.

Frozen reconstruction cost:4352 archived episodes,5222400 lower/78336 upper calls,1024 warmup updates and144 observed updates; diagnostic distribution/value passes288/288 and GAE calls144. Executed lower actor/value steps3840/24320 and upper actor/value192/2240; source loads16 clones/eight forecasters. New native/evaluation/fit counts0. Only approximately91KB of compact JSON was pulled.

Next: preregister a one-update paired native evaluation of clone/plain/rejection/backtracking across the same roots, both periods and arms, keeping critics and credit unchanged and using common fresh evaluation paths. Stage60 saved diagnostics, not candidate policy weights, so reconstruction cost must be included when obtaining those policies. Upper critic repair remains separate.

## Limitations

This establishes controlled nonzero movement on the archived first batch, not reward improvement or frequency-specific superiority. The budget applies to batch-mean conditional KL, not pointwise, trajectory or population KL. Existing Stage57 performance gates are unchanged.

[Compact data](../results/pointmaze_backtracking_stage60_full_20261001_r1/compact_summary.json).

