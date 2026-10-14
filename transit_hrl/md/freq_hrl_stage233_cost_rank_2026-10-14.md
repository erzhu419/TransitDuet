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

Pending. Compare the two new control selectors against inherited selection and raw to separate controller-specific selection mismatch from remaining prefix/future generalization failure.

## Limitations

This is supervised one-action suffix development, not fresh-seed CI or end-to-end Freq-HRL validation. A training-only constraint is not an outer held-out no-tradeoff guarantee.
