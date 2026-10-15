# Stage243: variance-aware, group-shrunk service/risk critic

Run `native_transit_variance_critic_stage243_development_20261015_r1`:
all33 tasks done/0, no retries (t155610, t155617-t155647, t155651);13 tests passed.
65,280 combined-MSE OOF records reproduce exactly; split OLS differs by <=3.71e-16.
Zero raw-reference fits/native episodes/actor updates; only the1,298,200-byte aggregate
was pulled. Models, panels and full fit records remain remote.

## Method And Results

Fit separate nonfleet-service/fleet-risk residuals with training-only paired MC variance,
prefix variance pooling and leave-group-out residual-signal shrinkage toward raw.
Same198-feature TSVD space,17 rank/strength candidates, frozen wait gate, original
anchor and total-residual RMS budget; k total-cost and2k component function dimensions.
Unchanged pooled-safe/half-safe OOF selection; primary `variance_group_pooled_safe`.
All fits/actions sealed before fresh-future evaluation. Mixed held60/fresh8 versus raw:

| Pooled-Safe Arm | Cost Delta | Wait Delta (min) | Cost MSE |
|---|---:|---:|---:|
| Raw | 0 | 0 | 0.001598575 |
| Combined / split OLS | +0.000062695 | +0.000188151 | 0.001616719 |
| Variance GLS | +0.000052930 | -0.000223698 | 0.001551933 |
| Group shrinkage | 0 | 0 | 0.001597592 |
| Variance + group | +0.000000293 | +0.000002344 | 0.001598506 |

Variance GLS reduces held-fresh MSE by2.92% and improves mean wait in both roots,
but fleet +0.000075955 outweighs service -0.000023025 in total cost. Root397 harms
cost while root401 improves. Joint shrinkage admits2/16 mixed cells and changes only
0.104% of full-held action comparisons;103/160 full-fit group risk reliabilities are zero.
Its OOF fresh cost -0.000082500 does not survive full refit as meaningful control gain.

## Limits And Next Step

Reused development states/futures, not fresh seed CI or end-to-end HRL; defaults and
Stage169 reserves unchanged. No total-cost superiority established. Next: matched-count
disjoint-future critic targets under frozen raw references, to separate signal-estimation
bias from genuine absence of transferable risk signal; retain current GLS/OLS controls.
