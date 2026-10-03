# Stage98 Native Fresh-Joint Preflight

- Tasks t128748/t128749 finished on node004 with exit code 0. Exact server-side reaggregation reproduces the saved qualification; status is preflight_passed / mechanical_only.
- Root410011 uses the full Stage96 teacher and Stage97 decoder artifacts. All six learners finished two updates; std, values, source/Adam state, paired scenarios and fixed decoder checks passed.
- Cost: 136 native episodes / 40,800 steps (96 credit, 40 evaluation), 12 policy updates / 20 mean-parameter updates. Exact budget matches; no optimizer steps, critic/forecaster fits, checkpoint or native-trace writes.
- Joint-call realized call-weighted KL spans 0.0009998871-0.0010002856 against nominal 0.001. Runtime 39.95s; RAM sampling was unmeasured.
- Final four-path paired diagnostics, joint-call minus base: period50 -0.08934, period100 -0.02262. Joint-call minus lower-only: +0.00182/-0.00974; minus joint-level: -0.02687/-0.00225. Preserve all negative results; no CI or performance admission at preflight.
- Only 24,044 bytes of server-reduced JSON/log tails were retrieved. No checkpoints or native trajectories were pulled.

Next: launch the already frozen 8-root Stage98 full cohort (27,136 episodes / 32,563,200 steps). Keep all 20 contrasts and four primary endpoints, all final donors and both periods; rebuild matched lower donors afterward without pooling Stage94/95.
