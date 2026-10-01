# Stage66 Result and Next Intervention

All eight roots completed; t116896/node006 passed complete-roster qualification. All64 actual Stage65 pre-update critic/GAE probes reproduced exactly, and all model/Adam states stayed unchanged. Scalar/batched old-logp differences are at most2.43e-5, not a large on-policy mismatch. Only192.9KB compact statistics pulled; no traces/weights.

Equal-root means for mc_normalized (credit-only gradients):

| Period / Arm | GAE/MC Sign Disagreement | Mean-Gradient Cosine | Last-Decile Value Bias |
| --- | ---: | ---: | ---: |
| 50 / zero_train | 31.9% | 0.642 | +44.37 |
| 50 / joint_ppo | 32.7% | 0.630 | +37.98 |
| 100 / zero_train | 31.3% | 0.702 | +41.81 |
| 100 / joint_ppo | 31.1% | 0.737 | +38.35 |

Mean-gradient alignment improves over raw GAE in all four equal-root groups and stays positive in32/32 normalized cases. Variance-gradient alignment does not improve uniformly;5/32 normalized log_std cosines are negative. Last-decile value bias is positive in all32 normalized cases (+27.15 to+54.05). The critic already receives remaining episode time through value_context/time_context; missing clock is not the problem.

Next isolate a finite-horizon critic parameterization: V(s,t)=m_gamma(H-t)*f(s,t), with m_gamma(n)=sum(gamma**k,k=0..n-1). Fit normalized MC/m_gamma with the same supervised budget; verify global and last-decile errors and actual credit before another native reward trial. This proposed intervention is not implemented or adopted yet; no seed/LR/KL sweep follows this result.

## Limitations

Empirical MC gradients are not true reward gradients; terminal bias has not been proven to cause Stage65's reward result. No new performance or frequency claim follows. Incremental cost:256 archives,307200 lower/4608 upper reconstructions plus307200 extra critic calls,640 score forward/1920 backward batches. New sampling, optimization, critic fitting and checkpoint writing remain zero. [Compact data](../results/pointmaze_actor_credit_stage66_full_20261001_r1/compact_summary.json).
