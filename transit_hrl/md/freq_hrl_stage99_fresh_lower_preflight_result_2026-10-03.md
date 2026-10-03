# Stage99 Fresh Lower: Native Preflight

t128790 and t128791 completed on node005 with exit0. Official qualification is preflight_passed / mechanical_gate passed; server-only recomputation matched the official summary.

The full Stage96 teachers, full Stage97 decoders and final update8 Stage98 joint upper were loaded. Four lower learners started from L0 with fixed U0 or UJ. Source/Adam/std/value/upper freezes, independent/common training-noise pairing and independent final evaluation passed.

Exact cost: 224 native episodes / 67,200 steps, 16 lower-mean updates, 144 extra upper-replay forward calls, two joint checkpoint loads and zero checkpoint/trace writes. Native runtime was 62.43 s. Only 18629 bytes of compact JSON were pulled; no CSV, NPZ or checkpoints.

## Diagnostic Results
These one-root H300/two-update/four-evaluation-path contrasts are negative and are retained without a performance gate or CI.

| Registered primary endpoint | Reward difference |
| --- | ---: |
| 50/source_common_minus_source_matched | -0.00504029 |
| 50/joint_common_minus_joint_fixed | -0.00515694 |
| 100/source_common_minus_source_matched | -0.01537389 |
| 100/joint_common_minus_joint_fixed | -0.01475478 |

## Next Step
Launch the already-frozen eight-root full protocol unchanged: H1200, eight updates, four matched lower learners, 26-contrast Bonferroni family and all four positive primary CI lower bounds required for the global conditioning claim. Keep all fixed-final donors even if that claim fails, then test staged upper learning separately. No old-cohort pooling.

Full training was submitted as t128854-t128861 with qualifier t128862, dynamically eligible on node001-006; each training task requests 9 CPU / 8192 MB RAM. No further protocol was launched.

Scope: this preflight verifies native mechanics, not conditioning improvement, full actor-critic learning or frequency superiority.
