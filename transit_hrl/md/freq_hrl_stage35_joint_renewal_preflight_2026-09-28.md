# Stage-35 Preflight

`t102117`-`t102120` completed all four treatments on separate root310001.
Four native-trajectory audits and four selected-checkpoint replays pass.
Every registered actor/critic updates; all selected checkpoints are iteration1.
Nineteen focused tests pass. Frozen algorithm implementation: `e9575c0170`.

Method cost: 13,200 steps, 243 upper calls, 194 gate calls, zero previews.
Verification: 1,200 steps, 33 upper calls and 22 gate calls. Only compact JSON
is pulled; trajectories and checkpoints remain server-only. This authorizes
execution, not a performance claim.

Full matrix `t102121`-`t102152` was launched on dynamic node001-node006:
8 roots x 4 treatments, 42,124,800 steps. No preflight score selected treatments
or changed settings. Next: complete all cells, audit server-side, then evaluate
the frozen five-endpoint joint gate versus equal-budget trained baselines.
The [full result](freq_hrl_stage35_joint_renewal_result_2026-09-28.md) is now complete.
