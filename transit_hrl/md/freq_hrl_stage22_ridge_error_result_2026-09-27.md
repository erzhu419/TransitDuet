# Stage-22 Frozen-Ridge Error Result

Preflight `t101429` completed on node004; full tasks `t101432/101433` completed
on node006/node005 at revision `8721fa2776`. Full results total 31,419 bytes.
Each root used eight influence solves, with zero parameter updates, controller
training or new environment steps. Fifteen focused tests passed.

| Root | Fresh training-state MSE | Fresh held-out MSE | Signal-mapping MSE estimate | Zero MSE |
|---|---:|---:|---:|---:|
| 209011 | 0.001258930 | 0.001908796 | 0.002007467 | 0.002280754 |
| 209061 | 0.001158135 | 0.001396302 | 0.001619595 | 0.001247787 |

Training-state error improves over zero by 44.80%/7.18%, but held-out error
improves 16.31% on 209011 and worsens 11.90% on 209061. On the latter root,
the training-label realization term is 0.000021758, the signal-mapping term
0.001619595 and their interaction -0.000245050. The sum reproduces Stage-21.
The signal-mapping point estimate remains 29.80% worse than zero: improving
label precision alone is not the evidence-justified next intervention.

Code/cache inspection also finds that the linear endpoint difference cancels
27 of 40 columns on both roots, including target velocity and force-history
features. Only 13 columns vary between arms. The remaining-time multiplier
still acts on the differences; physical errors also retain indirect context.

Next: test a fixed causal context-by-state interaction in the shared value
difference, retaining antisymmetry, training-only normalization and whole-path
holdout. Keep the frozen linear baseline and Stage-9 performance reference;
do not rescue the current candidate by tuning alpha or adding seeds.

## Limitations

These are retrospective conditional point estimates using already-seen labels,
not new qualification intervals or policy improvements. Signal mismatch
combines representation, regularization and finite-state coverage. The context
cancellation is a structural limitation, not proof of the sole failure cause.
