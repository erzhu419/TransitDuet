# Stage117: Local Plan Gain Through The Current Lower

Freeze all eight final Stage112 learned lower branches, their full physical/history
feedback, the forecaster, task and plan clocks. Use the Stage116 basis-5/eight-coordinate
decoder. At one interior decision, perturb one plan coordinate by +/-0.05; all other
decisions use zero residual. Continue feedback control to the end of the native episode.
Direct forecast and zero residual must execute identically. No policy or critic is trained.

Each root/period uses 12 fresh scenarios, balanced at steps 300/600/900, with two independent
suffix-noise panels sharing the factual prefix and exogenous path. Record option reward,
post-option reward, full suffix reward and actual plan/command response. Select a local
direction on panel A and score its suffix gain on B, then reverse; include the zero action.
Compare suffix-based versus option-only selection. This is a conditional branching probe,
not a causal deployed selector: the selection panel uses simulated future rewards.

The frozen full cohort is eight roots, two periods, 6,912 native episodes and 8,294,400 steps.
All eight endpoints use equal-root bootstrap 65,536 and Bonferroni8 intervals. Require positive
cross-panel suffix gain and gradient-dot lower bounds at both periods before upper training.
Preflight is mechanical only: one root/query per period, 72 300-step episodes, 21,600 steps. No seed
extension or epsilon selection. Schedule dynamically on node001-node006; only logs/compact
JSON return locally, no raw trajectories or checkpoints. A failed gain gate redirects work
to the plan/lower interface, not another residual-capacity sweep.

## Execution

Implementation revision: `2d0f285d52`; 20 focused tests passed.
Native preflight `t135429` and independent aggregation `t135430` passed mechanically.
Frozen full run: `pointmaze_local_plan_gain_stage117_full_20261004_r1`, workers
`t135432`-`t135439`, aggregation `t135440`. All tasks completed; corrected local-plan
gain and gradient-repeatability gates passed at both periods. This is conditional
headroom, not a learned-policy gain. No preflight metric revised the full protocol.
The matched credit training follow-up is Stage118.
