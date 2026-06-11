# Leakage No-Tradeoff Matrix

No-tradeoff is supported only when leakage/drift reduction and performance noninferiority are both supported in the same domain. Native real-demand artifacts expose wait/alighting/reward and LowerLFDrift checks; they are no-tradeoff evidence only when both drift reduction and performance noninferiority are supported.

## Adaptive Native Selector

- status: `strict_supported`
- selected domain: `native_real_demand_service_response_v7`
- selected verdict: `no_tradeoff_strict_supported`
- boundary: Adaptive native leakage selection is evidence-gated: it can select a native real-demand profile only when that same profile has drift reduction plus reward/wait/alighting/throughput no-harm. A strict paper claim additionally requires all required performance metrics to be CI-supported, not only noninferior.

| domain | verdict | checks | drift checks | performance checks | supported | noninferiority | positive mixed | summary positive | summary no-harm | not supported |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_real_demand | partial | 8 | 0 | 7 | 4 | 0 | 2 | 0 | 0 | 1 |
| native_real_demand_alighting_safe_v2 | partial | 36 | 2 | 8 | 9 | 0 | 0 | 0 | 0 | 16 |
| native_real_demand_alighting_throughput_v5 | partial | 37 | 2 | 8 | 13 | 0 | 0 | 0 | 0 | 14 |
| native_real_demand_alighting_wait_v4 | partial | 37 | 2 | 8 | 9 | 0 | 0 | 0 | 0 | 25 |
| native_real_demand_service_response_v7 | no_tradeoff_strict_supported | 50 | 2 | 10 | 28 | 0 | 0 | 0 | 0 | 20 |
| native_real_demand_throughput_safe_wait_v6 | partial | 39 | 2 | 10 | 13 | 0 | 0 | 0 | 0 | 23 |
| trading_constraint | partial | 4 | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 1 |
| trading_ppo_primal_dual | summary_only_noharm | 4 | 1 | 3 | 0 | 0 | 0 | 1 | 3 | 0 |
| transit_ppo_primal_dual | summary_only_noharm | 4 | 1 | 3 | 0 | 0 | 0 | 2 | 2 | 0 |
| transit_real_surrogate | no_tradeoff_strict_supported | 5 | 2 | 3 | 5 | 0 | 0 | 0 | 0 | 0 |
