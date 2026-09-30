# Stage57 Protocol

Reuse each Stage55 fixed final clone checkpoint and saved forecaster. Fork zero_train and joint_ppo with exactly identical networks and optimizer states; retain the original clone for evaluation. Zero_train executes zero upper residual from its first warmup/training rollout and learns only lower actor/value after calibration. Joint executes the actual sampled upper residual and learns upper/lower with the unchanged shared native SMDP PPO. Both still infer and charge upper at fixed periods 50/100. No analytic feedback, refitting, new BC or custom policy gradient.

Each arm calibrates both critics on its own execution distribution for 16 iterations x eight episodes, actors and std exactly frozen. Then both get 32 iterations x eight native training episodes and the same lower optimizer budget. Upper actor/value stay fixed after calibration in zero_train; only joint gets upper PPO updates. Common environment paths, stepwise lower noise and per-level shuffle seeds are paired. Initial upper proposals must match; lower trajectories/batches need not match after the execution treatment. Clone remains unchanged.

Retain eight source roots and deterministic/lower_sampled evaluation of all three fixed final policies on 16 fresh paths per root. Warmup, training and evaluation paths are disjoint from each other and Stage54/55/56 paths. Primary deterministic native return has six contrasts: joint-zero_train, joint-clone and zero_train-clone at each period, 65536 equal-root paired bootstrap draws, simultaneous Bonferroni6 intervals. Matched-upper gate requires joint-zero_train positive at both periods; training-gain gate independently requires joint-clone positive at both. No checkpoint, mode, period or seed selection/extension after outcomes.

Full incremental budget: 16588800 native steps (4915200 critic calibration, 9830400 PPO training, 1843200 evaluation), 13824 audits, 248832 upper calls and 16588800 lower calls. Reuse 16 clone checkpoints/eight forecasters; zero new fits, supervised or extra verification steps. Count actual actor/value optimizer steps, parameter changes and execution/audit operations. Upstream Stage55 cost stays separately declared. Preflight: 16800 steps and 56 audits. Tests, native training/evaluation and qualification use scheduler's dynamic node001-node006 pool, nine CPU / 12 GB per full root and two CPU / 4 GB for preflight/qualification. Pull compact JSON only.

## Execution

Source implementation: `6a435ce0e6`; seed/protocol freeze: `e673634741`. Scheduler tests `t108805` passed all 39 tests. Native preflight `t108833` and qualification `t108836` completed successfully: 16800 steps, 56 audits; optimizer steps upper actor/value 16/48 and lower actor/value 32/64. Only the compact qualification JSON was pulled locally.

Full run `pointmaze_matched_upper_stage57_full_20260930_r1`: tasks `t109115` through `t109122` correspond, in order, to roots 310011, 310023, 310037, 310049, 310061, 310073, 310089 and 310101. All eight completed on the dynamic node001-node006 pool. Full qualification `t109157` completed with exit 0 on node005. Matched-upper gate passed; training-gain gate failed. See [Stage57 Result](freq_hrl_stage57_matched_upper_result_2026-09-30.md) for all six preregistered intervals; preflight is not performance evidence.

## Limitations

Lower training/native budgets are matched; upper update cost and total FLOPs differ and are explicitly counted. This addresses deployment-only deletion's co-adaptation issue but remains conditional development on reused roots and teacher initialization. It does not establish frequency-separation, promotion, OOD or independent training-root confirmation, and does not retroactively change Stage55/56 decisions.
