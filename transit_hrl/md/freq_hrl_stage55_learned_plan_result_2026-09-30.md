# Stage55 Result

Source `28a5afa45e`, full freeze `a9f485c6ac`. All 31 tests passed in `t107808`; native preflight `t107809` and qualification `t107842` passed (22800 steps, 76 audits). Full `t107844`-`t107851` completed with exit 0 on dynamically selected node001/004/005/006; full qualification `t107864` passed on node005. No post-freeze source or parameter changes.

Full accounting: 16128000 native steps (153600 labels, 2457600 critic warmup, 9830400 PPO training, 3686400 evaluation), 13440 audits, 241920 upper calls and 16128000 lower calls; zero gate/extra verification steps. Optimizer steps: BC 20480, lower actor/value 40960/51200, upper actor/value 2048/3072. Forecaster: 256 driver paths, 281600 rows, eight ridge solves and zero native fitting steps. Compact local JSON retains all root endpoints, cloning losses, network changes, accounting and pooled arms/modes; raw data and fixed final weights remain remote.

Native deterministic return differences: 65536 paired-root bootstrap draws, simultaneous Bonferroni14 intervals. All eight roots and both periods retained.

| Contrast | Period50 mean [CI] | Period100 mean [CI] |
| --- | --- | --- |
| teacher - frozen | 47.074 [31.340, 62.801] | 44.134 [28.427, 59.136] |
| clone - frozen | 47.385 [31.536, 62.249] | 44.275 [29.328, 57.453] |
| clone - sham | 273.706 [238.195, 307.269] | 192.575 [164.138, 225.867] |
| lower_ppo - clone | -29.095 [-74.229, -4.705] | -3.475 [-24.822, 12.140] |
| joint_ppo - lower_ppo | 51.940 [29.546, 96.574] | 12.583 [-2.671, 29.963] |
| joint_ppo - frozen | 70.230 [53.486, 88.174] | 53.383 [37.481, 68.463] |
| joint_ppo - clone | 22.845 [11.883, 31.150] | 9.108 [-1.237, 17.023] |

Decision: **learnability gate passed; joint gain gate failed**. Learned MLP deployment reproduces the teacher's improvement at both periods without analytic feedback. Joint native PPO improves over frozen at both periods and over clone/lower-only at period50; period100 incremental gain remains inconclusive. Lower-only PPO significantly degrades period50 return. Joint upper changes the executed clipped plan in every root/period/mode; deterministic residual-action RMS averages .370/.372 and executed plan squared-delta sums average 73.214/53.534 per episode. Network updates and executed actions are non-null, but these checks do not turn the failed joint gate into a pass.

Next: freeze these final weights and preregister a same-policy executed-upper ablation: normal versus zero residual, holding learned lower weights fixed, matching inference calls and using fresh paired evaluation paths at both periods. This distinguishes the upper action's effect from lower co-adaptation before changing objectives or training budgets. Retain period100 and the lower-only regression; do not extend Stage55 seeds or select its successful period.

## Limitations

This is conditional development evidence for native residual HRL over a fixed learned forecaster with teacher initialization. Reused roots do not provide independent training-root confirmation; fixed periods do not establish frequency-separation utility, promotion or OOD performance. Training costs differ among arms, and matched evaluation calls do not imply matched FLOPs. The architecture has a learned native control path, not a completed domain-general Freq-HRL claim.
