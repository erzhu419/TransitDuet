# Stage244: independent-future critic targets and calibration

Run `native_transit_disjoint_critic_stage244_development_20261015_r1`: all33 tasks
done/0, no retries (t155663, t155671-t155701, t155702);20 tests passed.
All118,800 Stage243 control metric fields reproduce exactly. Zero raw-reference
fits/native episodes. Only the3,038,313-byte aggregate was manually pulled.

## Design And Results

Cross target A/B with variance/signal calibration A/B under frozen raw references.
A trained raw; B is its old unused complement. Each estimate uses2 inner/4 full
draws, same54/60 states,198 features,17 candidates and frozen A-MSE RMS cap.
The three new modes share A-union-B; same uses only A. OOF `old_other` is opposite
A2; full-fit `old_other` is B4. New Stage241 futures are evaluation-only.
Primary: `disjoint_variance_group_pooled_safe`; unchanged OOF selection/gate.
Mixed full-held/fresh8 joint variance/group arms versus raw; negative is better:

| Target / Calibration | Cost Delta | Wait Delta (min) | Changed Actions |
|---|---:|---:|---:|
| A / A | +0.000000293 | +0.000002344 | 0.104% |
| A / B | +0.000002250 | +0.000027995 | 0.729% |
| B / A | 0 | 0 | 0% |
| B / B | -0.000129902 | -0.000213411 | 0.625% |

Risk reliability zeros fall103/160 ->38/160 group occurrences; median0 ->0.1354.
B/B reduces fresh held all-action MSE0.945% and feasible MSE0.316%. Pooled cost/wait
improve in both fresh halves, but root401 harms both (+0.000010853/+0.000074740).
Only6 states change;2 sampled fleet-event draws dominate cost benefit. Conditional
MC SE is0.000085228 cost/0.000167817 wait (1.52/1.27 SE from zero).
B-OLS hits A's cap60/64 times, but only1/5 selected joint full fits is cap-active.

## Next And Limits

Fix the independent-future protocol; compare feasible-action ranking/regret losses
to MSE with the same capacity, focusing on root401 misselection. Widening the RMS
cap alone is not the main explanation. These are reused development futures and
conditional sampling SE, not new confirmation or training-seed CI. Superiority
remains unestablished; defaults and Stage169 reserves unchanged. Full fits remain remote.
