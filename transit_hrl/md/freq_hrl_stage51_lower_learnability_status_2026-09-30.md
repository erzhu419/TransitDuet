# Stage51 Status

Implementation `40f1a9089d`; preflight/full preregistrations frozen and pushed in `9e0591c3ec` before native execution. Protocol: `freq_hrl_stage51_lower_learnability_protocol_2026-09-30.md`.

- Unit `t106839` found a configuration-string comparison error and a missing gate in the fixture. Both were fixed; `t106842` passed all six tests, including signed CI aggregation and full-budget reconstruction.
- Native preflight `t106846` and qualification `t106847` completed. Compact summary saved under `results/pointmaze_lower_learnability_stage51_preflight_20260930_r1/qualification_summary.json`: 9600 steps, 32 audits, 16 supervised updates, five gradient computations; zero original RL updates and zero extra verification simulation.
- Full run `pointmaze_lower_learnability_stage51_full_20260930_r1`: all eight tasks started through scheduler. `t106849/t106853` on node005; `t106850/t106854` on node001; `t106851/t106855` on node006; `t106852/t106856` on node004. Allowed pool is node001-node006, without node pinning. Each task reserves 9 CPU/12 GiB and uses eight persistent rollout workers.
- Full total: 2304000 native steps, 1920 audits and 10240 supervised updates. Only compact JSON is pulled; raw trajectories, gradients and checkpoints remain on servers.

Next: aggregate all eight complete cells with `scripts/analyze_pointmaze_lower_learnability_stage51.py --run-name pointmaze_lower_learnability_stage51_full_20260930_r1`. Interpret the six deterministic return contrasts and two independent-credit cosines using the frozen Bonferroni intervals. No root extension, gain tuning, checkpoint selection or joint-policy adoption based on preflight. Full scientific results are pending; Stage50 remains unchanged.
