# Trading Phase-0 Logging Audit

- log: `transit_hrl/results/trading_phase0_audit/phase0_trace.jsonl`
- records: 180
- checked arrays: 900
- max abs reconstruction error: 0.000e+00
- passed: True

The audit records `x_raw`, causal `x_bin`, frequency state `z_t`, upper action `a_U`, lower action `a_L`, plan target, action effects, rewards, and entity IDs. The reconstruction check replays only logged `x_bin` values through the tracker and compares the reconstructed frequency state with logged `z_t`.
