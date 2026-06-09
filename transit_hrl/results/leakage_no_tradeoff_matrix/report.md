# Leakage No-Tradeoff Matrix

No-tradeoff is supported only when leakage/drift reduction and performance noninferiority are both supported in the same domain. Native real-demand artifacts currently expose wait/alighting/reward checks but not LowerLFDrift, so they are performance/no-harm evidence unless native drift metrics are added.

| domain | verdict | checks | drift checks | performance checks | supported | noninferiority | positive mixed | summary positive | summary no-harm | not supported |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_real_demand | performance_noharm_only | 8 | 0 | 7 | 4 | 0 | 2 | 0 | 0 | 1 |
| native_real_demand_alighting_safe_v2 | partial | 36 | 2 | 9 | 9 | 0 | 0 | 0 | 0 | 16 |
| trading_constraint | partial | 4 | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 1 |
| trading_ppo_primal_dual | summary_only_noharm | 4 | 1 | 3 | 0 | 0 | 0 | 1 | 3 | 0 |
| transit_ppo_primal_dual | summary_only_noharm | 4 | 1 | 3 | 0 | 0 | 0 | 2 | 2 | 0 |
| transit_real_surrogate | no_tradeoff_supported | 5 | 2 | 3 | 5 | 0 | 0 | 0 | 0 | 0 |
