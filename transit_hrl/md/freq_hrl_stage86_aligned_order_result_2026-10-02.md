# Stage86: aligned ordering did not close the performance gap

Eight roots, 11,776 episodes / 14,131,200 steps; mechanical gate passed; wall350-372s/root.
Only staged-aligned retrained; four Stage85 final baselines reused exactly. Lower uses even source chunks, upper odd.
All18 reward endpoints use equal-root bootstrap65536 / Bonferroni18.

| Contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| staged-aligned minus paired joint | -0.000014 [-0.01104, 0.01290] | -0.00714 [-0.03712, 0.02627] |
| staged-aligned minus alternating | +0.00218 [-0.00871, 0.01522] | -0.00211 [-0.02502, 0.02680] |
| staged-aligned minus lower-only | -1.5839 [-2.0243, -1.2660] | -1.7222 [-3.1678, -0.5446] |
| staged-aligned minus zero | +2.7080 [1.9443, 3.5775] | +3.9428 [2.4867, 5.5800] |

No CI-supported order benefit after correcting per-level dataset assignment; lower-only remains stronger.
Close ordering as the proposed remedy. Keep Stage84's positive direct upper effect and Stage85/86's negative joint-budget results.
Next: preregister decision-call-weighted KL with shared full MC gradient batches. The current cumulative nominal proxy per native step is K_lower+K_upper/period: dual0.00408/0.00404 versus lower-only0.008.

## Limitations
This repairs Stage85 on the same fixed training rosters with fresh evaluation, not independent training replication.
The old equal budget was a sum of level-mean KL, not call-weighted trajectory trust-region exposure. The proxy difference is not proof that it caused the reward deficit; gradient episode counts also differ from lower-only.
Teacher initialization, fixed decoder/std, Stage67 critic-route HOLD and the closed frequency-superiority claim remain unchanged.
