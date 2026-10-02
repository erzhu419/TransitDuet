# Stage86: repair staged per-level dataset alignment

Stage85's staged arm changed both order and level-specific training datasets; its pure-order interpretation was invalid.
Retain Stage85 results. Train only staged-aligned from the original source and reuse the four final Stage85 baselines.

- Lower takes Stage85 chunks0,2,...14; upper takes1,3,...15, exactly the paired/alternating per-level scenario/noise rosters.
- Within each level, chunk order is unchanged; execute all eight lower updates before all eight upper updates.
- Same native horizon1200, 32 episodes/gradient, KL0.0005/update, 512 training episodes and cumulative nominal KL0.008/period/policy.
- Stage78 decoder, both std, values, optimizers and forecaster frozen; no baseline retraining or intermediate selection.
- Evaluate source, zero, paired joint, alternating, staged-original, staged-aligned and lower-only on 32 fresh Stage86 seeds/root/period.
- Eight fixed roots, periods50/100; all18 reward contrasts use equal-root bootstrap65536 / Bonferroni18.
- Preflight: one root, horizon300, aligned source chunks0,2,1,3 with two scenarios/batch, four fresh evaluation seeds; mechanical only.
- Full budget:8,192 training +3,584 evaluation episodes, 14,131,200 steps. Read64 baseline checkpoints; save16 aligned final weights server-only.
- Scheduler dynamic node001-006 placement, nine CPU/8192MiB per root, eight rollout workers; local compact JSON/markers only.

Primary decision: staged-aligned versus paired/alternating isolates order with matched level rosters; compare separately against lower-only and staged-original.
If this repair still loses to lower-only, close ordering as a remedy and address the allocation/credit mechanism instead.

## Limitations
This is a post-Stage85 design repair on the same fixed training rosters with fresh evaluation, not independent training replication.
Teacher initialization, fixed decoder/std, nominal conditional rather than trajectory KL, Stage67 critic-route HOLD and the closed frequency-superiority claim remain unchanged.
