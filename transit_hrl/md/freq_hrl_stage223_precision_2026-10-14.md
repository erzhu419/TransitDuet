# Stage223: inner/final target precision

Cross two/four-draw calibration and final models, keeping architecture/600 updates fixed. c2v4/c4v4 use common four-draw truths of excluded prefixes. Protocol/budget: `results/native_transit_precision_stage223_development_20261014_r1/preregistration.json`.

## Results

Fits t152035-t152066 and report t152067 completed with exit0. Five new/eight Stage222 tests passed. All 32 cells qualified; 800 control comparisons reproduced within 1.39e-17. Only the 1,278,977-byte report was pulled, no raw/models. Default/native policy unchanged.

Matching two-draw final fitting does not close transfer: raw held conditioned risk rises to 0.07114293 versus four-draw 0.06078162; mixed joint cost/wait benefit is 4/8 versus 8/8. Its two training subhalves show opposite cost/wait outcomes in mixed_b. Matching inner4/final4 also yields only 6/8.

Keeping the same inner2 predictions but validating on four draws (c2v4/final4) restores 8/8 versus legacy 7/8. The previously failing mixed_b even_odd/0 wait changes +0.00328125 -> -0.00143750. Calibration-label precision matters; model precision mismatch alone is not a sufficient explanation.

c2v4/final4 held risk is 0.05133240, 15.5% below raw. Pooled mixed cost/wait improve versus raw by -0.00015026/-0.00018646 min, but there are view-specific tradeoffs and root401 wait worsens by +0.00000521 min. This is a promising development result, not uniform no-tradeoff or new CI-supported evidence.

## Next Decision

Keep c2v4/final4 as the calibration candidate and raw as control. Next bind selected-action constraints to the actual final fitted controller as well as OOF predictions, to address refit/decision-boundary drift rather than merely matching draw counts or retuning alpha.

## Limitations

c2v4/c4v4 hold out entire prefixes, not additional inner future samples; outer other-half evaluation remains independent of fitting. Inner models still use 54 states versus final60. Subhalf and future comparisons reuse 240 states/eight draws, not new independent seeds or full learned HRL.
