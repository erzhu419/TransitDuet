# Stage227: physical checkpoint aggregation repair

Stage226 produced five selected source/anchor/cell cases where fold-normalized scores hid a positive physical increment. Keep its negative results and all trained weights.

Pool physical validation cost/wait means first. Divide each source mean by one fixed positive, training-only RMS (held-case-weighted mean of fold training RMS). Keep the same worst-component criterion, earlier ties and raw-hold option. MSE checkpoints and all network weights remain unchanged.

32 scheduler reselections plus one report; no fitting, actor update, native episode or new raw/model pull. Reuse64 final paths and all six snapshots. Verify720 unchanged physical/prediction controls. Save only new selection metadata remotely, not duplicate models.

## Results

Reselections t152180-t152211 and report t152212 completed with exit0. Four new tests passed. All720 unchanged controls reproduce exactly. Only the484,306-byte aggregate report was pulled, no raw/models. Zero new network updates; default unchanged.

Nine selected checkpoints changed (two original-anchor/seven raw-anchor); selected internal source physical violations fall from five to zero. The aggregation fix is complete; all64 source paths and MSE checkpoints are unchanged.

Outer mixed cost/wait both negative versus original: raw/c2v4 8/8, original-selected7/8, raw-selected6/8. Strict simultaneous improvement versus raw is only1/8 original-selected and0/8 raw-selected. Raw-selected pooled mixed cost/wait (-0.00194555/-0.00384297 min) are worse than raw (-0.00224017/-0.00445885). Do not replace raw.

Raw-selected root401 pooled cost/wait improve versus raw by -0.00030328/-0.00080365 min, but only2/4 future-direction summaries strictly improve both; one has a cost tradeoff and one holds raw. Root397 wait worsens. This is a bounded local result, not cross-source support.

## Next Decision

Keep the corrected physical aggregation and raw/c2v4 controls. Next test a low-capacity regularized linear joint residual against the same nonlinear paths, with unchanged physical targets, anchor/reference protocol and internal prefix/future selection. Stage226's zero-step MSE winners and severe last600 overfit do not justify longer or larger nonlinear training.

## Limitations

This is a posthoc protocol correction on the same development data, not fresh confirmation. Inner/final refit transfer and nonlinear overfit remain separate issues; no end-to-end native HRL or independent CI claim.
