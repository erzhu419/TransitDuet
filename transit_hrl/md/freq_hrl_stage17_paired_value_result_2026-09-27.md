# Stage-17 Paired-Value Qualification Result

Tasks `t101251/101252` completed on node004/node006. Each root used 96 cached
pairs, eight path-held-out folds per objective and 1,024 critic updates.
No evaluation paths or new environment samples were used. Retrieved: 96,385
bytes of JSON. Matched fits share states, initialization and normalization.

| Root | Zero MSE | Matched absolute MSE | Paired MSE | Stage-16 MSE |
|---|---:|---:|---:|---:|
| 209011 | 0.002280245 | 0.003101872 | 0.002655346 | 0.002666972 |
| 209061 | 0.002958750 | 0.003812204 | 0.002962862 | 0.002917300 |

**Qualification failed on both roots.** Paired loss reduces MSE by
14.40%/22.28% versus the matched absolute-value control, but remains
16.45%/0.14% worse than zero. Root 209061 also loses to Stage-16.
No deployment test, retuning or additional roots follow from this screen.

The loss change helps relative to the matched control but is insufficient
for useful continuation prediction. State insufficiency and future-target
variability remain unresolved alternatives, not established causes. The next
diagnostic should distinguish them before another critic redesign; Stage-9
remains the validated same-task performance reference.
