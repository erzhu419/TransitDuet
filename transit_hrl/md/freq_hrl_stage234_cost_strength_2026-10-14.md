# Stage234: cost rank crossed with residual strength

## Protocol

- Run: `native_transit_cost_strength_stage234_development_20261014_r1`.
- Freeze the raw gate, Stage230 outer paths, Stage226 outer references and all 21 Stage233 controls. Same 240-state/eight-draw development cohort, two anchors, prefix/future splits and prior focus; no new capacity, native episode or confirmation cohort.
- Cross ranks 0, 1, 3, 8, 16 with strengths 0, 0.25, 0.5, 0.75, 1. Interpolate anchored cost baseline and full-rank prediction; leave logits unchanged. Rank0 and strength0 are exact raw aliases, merged into one of 17 distinct candidates. Exact ties prefer zero, then lower rank, then weaker strength.
- Keep cost-MSE, per-source cost/wait improvement and improvement plus nonincreased wait-worsening selectors. Select only from clean inner54/6-prefix and opposite2-draw validation inside outer training4. Reproduce all 640 inner fit ledgers and 64 full-strength Stage233 selections exactly before scoring new controls.
- 32 fits plus one report through scheduler, unbound node001-node006, one CPU/thread each. Same 640 small raw refits / 384,000 updates and 1280 SVDs; 21,760 inner candidate evaluations, no extra weight fits. Pull only aggregate JSON, not models/panels.

## Results

Pending. Assess whether weaker interventions improve cost/wait without moving the source-level regression elsewhere, retaining all four scopes and root/future breakdowns.

## Limitations

Reused development data and a larger selection grid are not fresh-seed CI or end-to-end Freq-HRL evidence. Inner constraints need not generalize to held-out outcomes.
