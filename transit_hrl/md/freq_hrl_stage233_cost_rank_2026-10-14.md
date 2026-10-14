# Stage233: raw-gate-specific cost rank selection

## Protocol

- Run: `native_transit_cost_rank_stage233_development_20261014_r1`.
- Freeze the Stage230 outer cost paths and Stage226 outer raw references. Retain all 15 Stage232 raw/raw-gate rank, MSE and inherited-selection controls and the same 240-state/eight-draw development cohort.
- Rebuild 20 clean inner references per cell: 54 training states from nine prefixes and two training draws, six held-prefix states on the other two draws, entirely inside the outer training half. Reproduce Stage223 OOF cost/logit predictions exactly. Fit the same two-anchor TSVD design at ranks 0, 1, 3, 8, 16; use only its cost column and the raw logits for decisions.
- Compare cost-MSE, strict cost/wait improvement at each source root, and the same improvement plus nonincreased wait-worsening fraction at each root. Pool physical means before dividing by training-only source RMS; exact ties prefer lower rank. Keep raw when no rank strictly improves both cost/wait means; zero residual is exactly raw.
- Select before scoring outer held-prefix or opposite-future outcomes. Preserve all four scopes, source/future splits and the prior rank8 focus; no new capacity, outer training, episode or confirmation cohort.
- 32 fits plus one report through scheduler on unbound node001-node006, one CPU/thread each. 640 small raw reference refits / 384,000 updates and 1280 inner SVDs; outer 64 paths/32 references unchanged. Only aggregate JSON is pulled; weights/panels stay remote.

## Results

Eight tests passed. Fixed repeated parent-contract construction along the active Stage218-232 dependency chain: all 11 modified contracts retain exactly the same contents. For one Stage232 contract with a fixed Stage217 leaf, leaf calls fall from 508,032 to 2 (3.496 s to 0.000379 s). No cached mutable protocol or algorithm change; this measures metadata construction, not RL training speed.

- t152602-t152633 fits and t152634 report all exited 0. All 640 clean reference refits reproduced old OOF predictions exactly; all 115,200 frozen-choice and 230,400 physical-score comparisons are exact. Pulled 611,894 aggregate bytes; no weights/panels.
- Raw-anchor mean-selected improves held-other cost/wait and wait-worsening fraction in both mixed views. Mixed pooled cost/wait is **-0.00227058/-0.00480417 min**, versus raw **-0.00224017/-0.00445885 min**; wait-worsening fraction is **14.79% versus 15.00%**. Seven of 16 mixed cells select nonzero ranks, versus one under inherited raw-anchor selection; 4/8 future directions strictly improve both means. Single-source root401 also improves all three pooled outcomes, with dual-metric wins in 3/4 directions.
- The added inner risk constraint does not establish no-tradeoff: raw-anchor safe-selected mixed pooled cost/wait is **-0.00203621/-0.00466953 min**, with wait-worsening fraction 14.58%. Cost is worse than raw, and mixed_a worsens both mean cost/wait. A training-only constraint does not ensure held-out performance.
- Important boundaries: raw-anchor mean-selected single-source root397 cost worsens by +0.00035528, and its wait-worsening fraction rises by 0.625 percentage points. Within mixed_b, root401 worsens cost by +0.00005509, wait by +0.00033646 min and that fraction by 0.417 percentage points; aggregate gains come from root397. The prior rank8 focus is correctly held raw by both new selectors rather than retained using its positive outer outcome. Default remains unchanged.

## Next

Keep the raw gate and outer references fixed; cross rank with cost-residual strength at 0, 0.25, 0.5, 0.75, 1 using the same clean prefix/future inner selection. Retain mean-only and risk-constrained selectors, exact zero/raw controls and all source/future scopes. Test whether a smaller intervention can preserve gains while reducing source-level regression before adding capacity or opening confirmation cohorts.

## Limitations

This is supervised one-action suffix development, not fresh-seed CI or end-to-end Freq-HRL validation. A training-only constraint is not an outer held-out no-tradeoff guarantee.
