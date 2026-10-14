# Stage224: bind calibration to the final controller

Stage223 c2v4/final4 restored mixed cost/wait benefit to 8/8, but its final training decisions still violated OOF-only floors. This stage tests final-controller boundary drift directly.

Reuse the exact final4 models and c2v4 OOF predictions. Compare unchanged GLM/raw/risk/OOF controls with final-only and joint OOF+final constraints. Each source root must meet its own raw-controller cost and wait means; alpha=1 is always feasible. All modes minimize the same OOF physical-mass loss.

## Protocol

32 scheduler cells on unbound node001-node006; 1 CPU/3 GB each. Six scalar calibrations/cell; zero network updates, native episodes or actor updates. Final constraints use only 60 training states and their four training futures. Outer prefixes/other futures are evaluation only. Download only the compact report.

## Results

All calibration tasks t152070-t152101 and report t152102 completed with exit0. Six new/five Stage223 tests passed. Fixed tuple/list protocol roundtrip; two focused tests reran successfully, without repeating calibration. The 800 controls reproduce within 1.39e-17. Only the 585,050-byte report was pulled; no raw/models or default replacement.

Final and joint constraints satisfy actual training floors in all 96 architecture/cell combinations each, by construction. OOF-only conditioned calibration violates final floors in 18/32 cells. Refit floor transfer is a real problem, but fixing it does not improve held performance.

Conditioned mixed held cost/wait are both negative in raw8/8, OOF-only8/8, final-only5/8 and joint7/8 future-direction summaries. Joint mixed_a even_odd/1 wait becomes +0.00250833 min, versus raw -0.00100208. Joint has zero training-choice changes versus raw in both mixed views, yet changes held choices.

Equal-view held conditioned proper loss: raw0.06078162, OOF-only0.05133240, final0.05427036, joint0.05711428. Joint mixed cost improves versus raw (-0.00241664 vs -0.00224017), but wait worsens (-0.00398646 vs -0.00445885 min). These are tradeoffs, not an improved replacement.

## Next Decision

Keep raw and Stage223 c2v4 controls. Stop adding in-sample floors to the scalar shrinkage. Next measure wait-gate error on selectable negative-predicted-cost contenders versus all candidates, using existing prefix/future splits; if decision-relevant error differs, test action-weighted wait training at the same 600-step/final4 budget. No new confirmation scenes.

## Limitations

Development data reuse the same 240 states/eight future draws, not fresh independent seeds. Source roots also differ in lower checkpoints and demand/prefixes. This is one-action supervised selection with the original greedy suffix, not end-to-end learned HRL.
