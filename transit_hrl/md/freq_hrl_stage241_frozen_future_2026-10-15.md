# Stage241: frozen OOF actions under independent futures

## Question

Stage240 found held-prefix cost harm already at the cached 54/2 endpoint.
Freeze Stage239 candidates, references, residuals and actions to distinguish
future-selection noise from reproducible effects confined to known prefixes.

## Protocol

- Run: `native_transit_frozen_future_stage241_development_20261015_r1`.
- Seal all 32 cells, 640 inner blocks and 21 controller action vectors first.
- Reproduce 12,800 selected OOF outcome/count records before native collection.
- Reuse the existing 240 development states in 40 complete prefix groups.
- Eight independent post-probe futures; six actions share each future tape.
- Compare old OOF two draws, old unused four draws, fresh halves 4+4 and fresh eight.
- Report paired physical cost, wait, mean-label wait harm, draw-level wait harm,
  gate regret, admitted-only effects and per-source prefix concentration.
- No new reference/residual/RL fits, new prefix scenes or confirmation access.
- Ten native preflight cells cover both roots and all five regimes first.
- Full budget: 1 seal + 1,920 draw + 1 report tasks; 15,360 native episodes.
- Scheduler: node001-node006, unpinned, one CPU/task; no local native runs.
- Checkpoints, sealed actions, labels and full rollouts stay on the server.

## Status And Next Step

Six focused fixture tests pass. Seal t153586 completed on node004 with exit zero:
32 cells, 240 states, 40 prefixes and 12,800 reproduced records, exact difference 0.
Mixed joint/raw/half-safe cost delta: old OOF -0.00007526, unused futures -0.00022560.
Wait delta: -0.00057839 -> -0.00006302 min; root397 unused wait is +0.00008073 min.
Thus cached cost improvement survives, but source-wise wait nonharm already fails.
Native preflights t153587-t153596 completed with exit zero: 80 episodes, 10 exact
resampled identities, 70 prefix/RNG matches, 80 frozen-weight and window-ledger checks.
Duration 185-214 seconds/task; measured peak RSS <=702 MB. No training updates.
The remaining 1,910 draws are registered as t153597-t155506 for node001-node006,
one CPU/task without node binding; scheduler dispatch is expanding the active pool.
The report is submitted only after all 1,920 draws complete with exit zero.
The full result will distinguish future-selection noise from prefix-transfer failure.
Only aggregate evidence and compact receipts are kept locally; checkpoints/actions/rollouts remain remote.

## Limitations

This is conditional replication on reused development prefixes, not fresh-seed CI
or evidence that complete end-to-end learned Freq-HRL outperforms its controls.
