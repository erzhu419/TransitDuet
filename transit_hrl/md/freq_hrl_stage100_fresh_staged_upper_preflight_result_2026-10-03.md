# Stage100 Fresh Staged Upper: Native Preflight

t128933 and t128934 completed on node004 with exit0. Official status is preflight_passed, mechanical_gate passed and staged_confirmation mechanical_only; server-only qualification and recomputation matched the official summary.

The full Stage96 teachers / Stage97 decoders and registered final Stage99 U0-trained lowers / Stage98 joint diagnostic were loaded. Both uppers started from new U0. Lower/std/value/source/Adam/decoder freezes, independent upper/lower noise and all11 actor compositions passed.

Exact cost: 152 native episodes / 45,600 steps, eight upper-mean updates, six donor checkpoint loads, zero checkpoint/trace writes and zero upper replay forwards. Native runtime: 40.32 s. 16937 bytes of compact JSON were pulled; no CSV, NPZ or checkpoints.

| Registered primary endpoint | Reward difference |
| --- | ---: |
| 50/staged_common_minus_source_common | 0.00830233 |
| 50/staged_common_minus_staged_independent | -0.52813883 |
| 100/staged_common_minus_source_common | 0.00978235 |
| 100/staged_common_minus_staged_independent | -0.10178510 |

These one-root H300/two-update/four-evaluation-path diagnostics have no CI. Upper updates slightly improve reward over the corresponding fixed lower composition; common-versus-independent branch comparisons remain negative. Neither result changes the frozen training rule or acts as a full-run admission gate.

Next: submit the already-frozen eight-root H1200 full protocol unchanged: 22,016 native episodes / 26,419,200 steps, 256 upper-mean updates, 32 final server-only checkpoints and the 28-contrast Bonferroni family. All four primary CI lower bounds must be positive for the global staged claim; no old-cohort pooling or donor selection.

Scope: mechanically validated teacher-initialized staged MC mean learning on native PointMaze, not a performance claim, full actor-critic or frequency-superiority proof.
