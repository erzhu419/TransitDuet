# Stage85: ordering did not close the lower-only gap

t120865-t120873 all exit0; eight roots, 35,840 episodes / 43,008,000 steps; mechanical gate passed.
896 mean updates, 64 server-only final checkpoints; wall1,104-1,139s/root, peak4,734-5,070MiB.
All22 reward contrasts use equal-root bootstrap65536 / Bonferroni22.

| Contrast | period50: mean [CI] | period100: mean [CI] |
| --- | --- | --- |
| paired joint minus lower-only | -1.4915 [-1.7972, -1.1709] | -1.6772 [-2.6913, -0.7574] |
| alternating minus paired joint | -0.00203 [-0.00323, -0.000697] | -0.00447 [-0.01195, 0.00177] |
| staged minus lower-only | -1.4329 [-1.7597, -1.1507] | -1.6239 [-2.6661, -0.9263] |
| staged minus paired joint | +0.05868 [-0.2485, 0.2879] | +0.05324 [-1.0754, 0.9891] |

Every trained policy beats zero-residual, but all dual methods lose to lower-only at the registered total budget.
The aligned paired/alternating comparison gives no ordering benefit; the period50 negative effect is tiny.

## Limitations
Staged assigned chunks0-7 to lower and8-15 to upper, while paired/alternating used even/odd chunks. Its contrasts combine order and per-level dataset assignment, not order alone.
Paired joint is not Stage83's shared-full-batch joint. Teacher initialization, fixed decoder/std and Stage67 critic-route HOLD remain.

Next: train only staged-aligned on the same even/odd level rosters; reuse all Stage85 final baselines and evaluate all variants on fresh Stage86 seeds. Keep these negative results unchanged.
