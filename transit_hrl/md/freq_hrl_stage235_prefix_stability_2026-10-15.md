# Stage235: validation-prefix influence and outer transfer

## Protocol

- Run: `native_transit_prefix_stability_stage235_development_20261015_r1`. Freeze Stage234's 17 candidates, two anchors, three selectors and all outer actions on the same 240-state/eight-future development cohort. Confirmation cohorts remain closed.
- Replay the same 640 clean raw references / 384,000 updates and 1280 analytic SVDs because per-prefix curves were not retained. Require all 64 full inner selections and raw OOF references to reproduce. No new model capacity, outer fit, prediction or native episode.
- Report signed cost/wait contributions for every validation prefix and top1/top3 shares of positive benefit within each source. Delete both validation directions of one prefix, recompute existing OOF selection, and separately evaluate the original fixed candidate. Also rescore each future half alone. Preserve raw hold and cost-MSE/mean/safe contrasts.
- 640 prefix deletions, 128 future-half rescores, 1920 selector-deletion comparisons. Compare fixed candidates with unchanged Stage234 actions in `train_same/train_other/held_same/held_other`, separately by root397/root401. Mixed views contain two Transit sources, not two application domains.
- Scheduler only, unbound node001-node006, one CPU/thread per task; pull aggregate JSON, not raw curves, panels or weights.

## Results

Pending. Decide whether validation gains depend on a few prefixes, future-half pooling, or disappear after outer refit/transfer before expanding the grid.

## Limitations

Deleting validation records does not remove that prefix from other inner training fits. This measures selection influence, not a new independent refit. Inner54/2-draw and outer60/4-draw references differ, so an inner-to-outer change alone cannot identify the cause of transfer failure. Reused development data do not provide fresh-seed CI or end-to-end HRL proof.
