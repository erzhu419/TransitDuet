# Freq-HRL Unified Top-Journal Evidence Matrix

Unified matrix records current evidence quality; it is not itself a performance validation run.

- supported: `2`
- partial: `4`
- not supported or missing: `0`

| id | claim | status | evidence | remaining gap |
|---|---|---|---|---|
| C1 | Native learned promotion improves reward and wait | partial | best=native_promotion_v21 best_reward=native_promotion_v21 best_wait=native_promotion_v24_fixed reward=supported reward_noharm=supported wait=inconclusive wait_noharm=supported | Wait CI must be supported together with reward in the same native run. |
| C2 | Native real AFC/APC demand improves score/reward without wait/alighting loss | supported | score=supported reward=supported wait=inconclusive wait_noharm=supported alighted=inconclusive alighted_noharm=supported | Alighting/wait no-harm is supported; strict improvement CIs still need stronger throughput-seeking validation. |
| C3 | Large L2/L3 order-book replay path exists | partial | l2_supported_checks=8 l3_positive_checks=8 manifest_coverage={} | Current path has L2 matching and synthetic/CSV-capable L3 FIFO replay; top-journal claim still needs larger real venue L2/L3 feeds. |
| C4 | Advanced encoder evidence spans Quant and Transit | supported | supported_domains=['order_book_l2', 'trading_synthetic', 'transit_real_demand', 'transit_synthetic_demand'] | Public market needs paired multi-window CIs; L3 remains mixed. |
| C5 | Leakage no-tradeoff holds beyond surrogate | partial | no_tradeoff_domains=['transit_real_surrogate'] partial_domains=['native_real_demand', 'native_real_demand_alighting_safe_v2', 'trading_constraint', 'trading_ppo_primal_dual', 'transit_ppo_primal_dual'] | Native real-demand needs LowerLFDrift metrics and alighting-safe improvement. |
| C6 | Formal theory appendix covers main protocol claims | partial | examples=['credit_residual_bound_example', 'leakage_bound_example', 'paired_ci_radius_example', 'promotion_detection_delay_bound_s', 'promotion_false_positive_bound_example'] | Turn proof sketches into polished manuscript appendix text with assumptions near theorem statements. |
