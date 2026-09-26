# Stage-10 Learned-Termination Development Result

Date: 2026-09-26

Run: `pointmaze_learned_termination_stage10_v1_development_20260926_r1`

All eight tasks (`t100750`-`t100757`) completed. The registered analyzer
accepted all compact results, exact fixed-controller replays, 24-call episode
budgets, and 230,400 additional trigger-training/selection steps per root.
The paired intervals were independently reproduced from episode rows.

| Registered contrast | Root mean [95% CI] |
|---|---:|
| Fixed minus learned-termination episode ISE | -0.078637 [-0.205896, 0.048623] |
| Learned termination minus Stage-9 candidate episode ISE | 0.584178 [0.534655, 0.633702] |
| Stage-9 candidate minus learned-termination return | 47.431333 [41.822093, 53.040573] |

**Decision: `stage10_development_gate_failed`.** The learned baseline did not
establish ISE improvement over fixed planning, so the candidate's positive
contrast does not establish superiority to a competent learned-termination
baseline. Seven of eight selected deterministic actors made zero pre-deadline
calls over 16 held-out episodes each; the remaining root made nine. The
baseline averaged 0.070312 early calls per episode. Actor weights changed,
but deterministic deployment mostly collapsed to the offset-25 deadline.

This is a development failure on revealed Stage-9 confirmation paths, not a
new confirmation test. The next diagnostic must isolate whether the on-policy
stochastic actor is useful before changing its training objective or claiming
that stronger termination learning fails.
