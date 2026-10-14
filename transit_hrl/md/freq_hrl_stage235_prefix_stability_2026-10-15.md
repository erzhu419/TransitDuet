# Stage235: validation-prefix influence and outer transfer

## Protocol

- Run: `native_transit_prefix_stability_stage235_development_20261015_r1`. Freeze Stage234's 17 candidates, two anchors, three selectors and all outer actions on the same 240-state/eight-future development cohort. Confirmation cohorts remain closed.
- Replay the same 640 clean raw references / 384,000 updates and 1280 analytic SVDs because per-prefix curves were not retained. Require all 64 full inner selections and raw OOF references to reproduce. No new model capacity, outer fit, prediction or native episode.
- Report signed cost/wait contributions for every validation prefix and top1/top3 shares of positive benefit within each source. Delete both validation directions of one prefix, recompute existing OOF selection, and separately evaluate the original fixed candidate. Also rescore each future half alone. Preserve raw hold and cost-MSE/mean/safe contrasts.
- 640 prefix deletions, 128 future-half rescores, 1920 selector-deletion comparisons. Compare fixed candidates with unchanged Stage234 actions in `train_same/train_other/held_same/held_other`, separately by root397/root401. Mixed views contain two Transit sources, not two application domains.
- Scheduler only, unbound node001-node006, one CPU/thread per task; pull aggregate JSON, not raw curves, panels or weights.

## Results

- All 32 fits and the report completed through scheduler (t153154-t153186). Seven tests passed; the updated summary test also passed. All 64 full inner selections reproduced exactly; raw-reference OOF differences were zero. Only the 1,099,891-byte aggregate JSON and small log/scalar outputs were pulled.
- Mixed-view raw-anchor mean selection intervened in 12/16 cells. Across their 24 source-cell comparisons, one prefix supplied at least 80% of positive cost benefit in **24/24**, with median share **1.0**; **13/24** had only one improving prefix. Wait benefit was also concentrated (median top1 share 0.973727). These are positive-benefit shares, not net gains or independent samples.
- Deleting one prefix changed selection in **32/160** checks; the original intervening candidate lost eligibility in **26/120**. Of those losses, **20** had material positive cost/wait deltas and **6** were zero/roundoff boundaries. Split-future eligibility was lost in **16/24** intervening-half checks: **10** material regressions and **6** boundaries. No intervening mixed-cell candidate remained strictly improving on both halves. The original strict rules are unchanged; `1e-12` is used only to distinguish numerical boundaries in this interpretation, including one wait delta near `7e-17`.
- Single-source root401 was more stable: raw-mean deletion eligibility losses **1/80**, half losses **4/16**, with held-other cost/wait deltas **-0.0005441792 / -0.0012354167 min**. Root397 remained cost-regressive. This is source-dependent instability, not evidence that every residual model fails.
- The unchanged mixed raw-mean outer results still move from `train_other` cost delta **-0.0005455195** to `held_other` **+0.0000914682**. Prefix concentration and future-half fragility are directly observed; changes of reference model, fit size and prefix distribution remain confounded in the inner-to-outer comparison. Defaults remain unchanged.

## Next

Use the existing fixed candidates and stored diagnostic records for future-only, prefix-only and combined admission contrasts, allowing numerical/zero nonharm boundaries while retaining pooled strict improvement. Reuse all outer actions and score all four scopes/source roots; no new grid or training. If stability merely reduces everything to raw hold, record that as abstention rather than an algorithm gain, then isolate reference-refit versus prefix-coverage effects.

## Limitations

Deleting validation records does not remove that prefix from other inner training fits. This measures selection influence, not a new independent refit. Inner54/2-draw and outer60/4-draw references differ, so an inner-to-outer change alone cannot identify the cause of transfer failure. Reused development data do not provide fresh-seed CI or end-to-end HRL proof.
