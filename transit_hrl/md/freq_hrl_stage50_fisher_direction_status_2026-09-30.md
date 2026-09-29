# Stage50 status

Project source/manuscripts/compact evidence were pushed on their existing branches;
runtime trajectories, weights and downloaded reference PDFs remain outside Git.

- Unit/regression: t106209, node004, exit0;19 tests passed (69.877s).
- Native preflight: t106212, qualification t106216, both exit0.
- Preflight:20,400 native steps,68 audits; all8 updates accepted within KL0.1.
- Direct arms:4 full-batch score backwards,26 Fisher-vector products,20 CG
  iterations,8 parameter proposals. Critic pairing and cost accounting passed.
- Full run: `pointmaze_fisher_direction_stage50_full_20260930_r1`, t106218-t106225;
  all8 roots running on dynamically selected node001/004/005/006 at this snapshot.
- Frozen full budget:7,680,000 native steps,6,400 audits;8 Bonferroni-adjusted
  paired endpoints. No native performance conclusion yet.

Next: qualify all8 cells through the scheduler with
`scripts/analyze_pointmaze_fisher_direction_stage50.py --run-name pointmaze_fisher_direction_stage50_full_20260930_r1`,
pull only `qualification_summary.json`, and decide from the registered Fisher-vs-
Adam, Fisher-vs-frozen and Fisher-vs-Euclidean final reward intervals.
