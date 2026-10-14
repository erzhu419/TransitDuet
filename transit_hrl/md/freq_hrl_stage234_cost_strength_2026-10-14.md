# Stage234: cost rank crossed with residual strength

## Protocol

- Run: `native_transit_cost_strength_stage234_development_20261014_r1`.
- Freeze the raw gate, Stage230 outer paths, Stage226 outer references and all 21 Stage233 controls. Same 240-state/eight-draw development cohort, two anchors, prefix/future splits and prior focus; no new capacity, native episode or confirmation cohort.
- Cross ranks 0, 1, 3, 8, 16 with strengths 0, 0.25, 0.5, 0.75, 1. Interpolate anchored cost baseline and full-rank prediction; leave logits unchanged. Rank0 and strength0 are exact raw aliases, merged into one of 17 distinct candidates. Exact ties prefer zero, then lower rank, then weaker strength.
- Keep cost-MSE, per-source cost/wait improvement and improvement plus nonincreased wait-worsening selectors. Select only from clean inner54/6-prefix and opposite2-draw validation inside outer training4. Reproduce all 640 inner fit ledgers and 64 full-strength Stage233 selections exactly before scoring new controls.
- 32 fits plus one report through scheduler, unbound node001-node006, one CPU/thread each. Same 640 small raw refits / 384,000 updates and 1280 SVDs; 21,760 inner candidate evaluations, no extra weight fits. Pull only aggregate JSON, not models/panels.

## Results

- All 32 fits and the report completed. The 21 frozen controllers reproduced 161,280 choices and 322,560 physical scores with zero differences; all 640 inner fit ledgers and 64 full-strength selections reproduced exactly. Fourteen unit tests passed. Only the 1,338,217-byte aggregate report was pulled.
- Raw-anchor mean selection chose weaker strength in 10/16 mixed-view cells, full strength in 2 and raw hold in 4. On pooled `held_other`, versus raw, cost delta was **+0.0000914682**, wait delta **-0.0013033854 min**, and wait-worsening fraction delta **-0.00729167**. Cost/wait jointly improved in 3/8 future-direction summaries, down from Stage233's 4/8. Weakening does not resolve the tradeoff.
- `mixed_b/root401` wait delta changed from +0.0003364583 to **-0.003521875 min**, but cost delta worsened from +0.0000550865 to **+0.0001170760**. `mixed_a/root401` cost delta became **+0.0003999385**. The source-level regression was not eliminated. Single-source root397 mean selection still increased cost; its safe selector improved the three aggregate measures, but not every future direction.
- For raw-anchor mean selection, pooled cost deltas were `train_same=-0.0000428203`, `train_other=-0.0005455195`, `held_same=-0.0001973076`, `held_other=+0.0000914682`. This points to a prefix/future interaction, not a general lack of training-set fit. Cost-MSE selection improved pooled held-other cost/wait, but still increased mixed_b/root401 cost; it is not a uniform replacement. The pre-existing focus remains raw hold for both improvement selectors. Defaults remain unchanged.

## Next

Fix the candidate grid and decompose selected-candidate gains by validation prefix, future direction and outer scope. Test whether deleting one validation prefix changes eligibility/rank/strength, and whether gains are concentrated in a few prefixes. Separate selection instability from transfer of the refitted controller before changing capacity or opening fresh confirmation cohorts.

## Limitations

Reused development data and a larger selection grid are not fresh-seed CI or end-to-end Freq-HRL evidence. Inner constraints need not generalize to held-out outcomes.
