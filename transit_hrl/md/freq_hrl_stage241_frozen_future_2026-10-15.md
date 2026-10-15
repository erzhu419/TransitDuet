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

All 1,922 tasks completed with exit zero, including aggregate t155529. No retries.
All 1,920 draws qualified: 15,360 frozen episodes, 1,920 exact identities and
13,440 prefix/RNG matches. All 12,800 cached OOF records reproduce exactly.
Only the 9,101,988-byte aggregate was pulled; models/panels/rollouts remain remote.

Mixed pooled joint/raw/half-safe; negative is better, wait is in minutes:

| Futures | Cost delta | Wait delta |
|---|---:|---:|
| Old OOF2 | -0.000075258 | -0.000578385 |
| Old unused4 | -0.000225604 | -0.000063021 |
| Fresh half0 | -0.000025337 | -0.000214974 |
| Fresh half1 | +0.000046963 | -0.000116406 |
| Fresh8 | +0.000010813 | -0.000165690 |

Joint cost improvement is future-unstable even without refits. Root397 fresh cost
improves (-0.000053348), root401 worsens (+0.000074973); admitted-only signs agree.
All-MSE/original/pooled is a positive secondary result: both roots in both fresh
halves have cost/wait/risk nonharm. Its pooled fresh cost/wait are -0.000078987/-0.000710547.
The regression cell improves pooled fresh cost/wait, but root397 worsens both;
root401 still gets 96.3% of its cost benefit from one prefix. This is not a clean repair.

Next: reuse all fresh panels and cached models to compare seen and held prefixes
under fixed 54/2 and full 60/4 endpoints, with no new training or native episodes.

## Limitations

This is conditional replication on reused development prefixes, not fresh-seed CI
or evidence that complete end-to-end learned Freq-HRL outperforms its controls.
