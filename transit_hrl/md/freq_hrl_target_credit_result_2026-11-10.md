# Matched Target Result

All four main tasks t214189-t214192 exited 0; source replays and prefix checks
passed. Each root used 27 ridge fits, no controller/termination updates. Main
cost: 5,301.383 CPU seconds, 6,305,600 simulator steps; retrieved 111,772 public
bytes plus 27,018 derived-cache bytes. Parent decision: HOLD.

## Complete Episodes

Mean ISE on eight new paths/root; lower is better:

| Task | Frozen history | Threshold | Matched L | Matched W | Matched G |
| --- | ---: | ---: | ---: | ---: | ---: |
| PointMaze | 0.837544 | 0.896140 | 0.934507 | 0.881294 | 1.155200 |
| Reacher | 4.206199 | 4.190341 | 4.262345 | 4.241529 | 4.261325 |

W improves L by 5.69%/0.49%, but loses to frozen history by 5.22%/0.84%.
Both tasks' first root loses to L and second root wins; pooled W-L improves in
both episode halves. PointMaze root2 W-L CI is [-0.26601,-0.00187], yet
W-history CI is [+0.01089,+0.19443]. PointMaze root1 G-L CI is
[+0.18121,+0.71432]. Reacher G-L cancels across roots and changes direction
between episode halves. Frozen history still beats PointMaze threshold by 6.54%.

## Prediction And Attribution

W own-target MSE divided by zero-predictor MSE:

| Task/root suffix | Training-prefix independent B | Path-OOF B | New prefix A+B |
| --- | ---: | ---: | ---: |
| PointMaze/110000 | 0.346 | 0.868 | 1.119 |
| PointMaze/120000 | 0.310 | 0.833 | 0.738 |
| Reacher/210000 | 0.123 | 1.456 | 4.912 |
| Reacher/220000 | 0.480 | 1.318 | 1.573 |

W transfers to independent futures on training prefixes, not reliably to new
prefixes. Reacher W held-future signs agree 100%; future noise is not the sole
problem. Only 24 fitting states/head exist. G held-future variance is about
1,610/13,791 times W on Reacher; sign agreement is 83.3%/54.2%. G loses to zero
prediction on three of four new-prefix roots. Its negative corrected training-B
MSE on Reacher root2 is an unstable estimate, not a negative true error.

Cached attribution replaces W predictions with conditional means at the SAME
cut: Reacher gains are +0.04970/+0.02637 and +0.02973/+0.02430 across future
directions. Root1 can keep the same 8/24 stops while improving their placement.
Zeroing the fitted threshold is mixed on root1 and changes no action on root2.

## Next Step And Limits

Next: retain history/L controls; compare W prefix coverage and effective
complexity under matched future budgets with another reserved holdout. Keep G
as the final outcome. No new fitting or simulation followed this readout.

Limits: two roots/task, eight paths/root; CIs are descriptive unadjusted 95%
within-root intervals. Cached actions are single interventions, not deployable
episode gains; one Reacher state holds 89.8% of G future squares. History/new-L
also differ in fit bank/regularization. Termination competence and interrupted
lifecycle costs remain unresolved. Private cost snapshots lagged; authoritative
public costs above are complete. The worker now refreshes private costs after
ledger closure; old files remain unchanged.

Sources: [frozen summary](../results/plan_validity_target_credit_20261110/summary.json),
[descriptive cache readout](../results/plan_validity_target_credit_20261110/cache_readout.json),
[protocol](freq_hrl_target_credit_repair_2026-11-10.md).
