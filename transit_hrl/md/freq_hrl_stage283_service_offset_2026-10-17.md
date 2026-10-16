# Stage283: Frozen Service Offset Versus Slopes

- Run: `native_transit_service_offset_stage283_20261017_r1`.
- Reuse Stage282 coefficients/predictions. Split each uncut service correction into its fitted intercept on nonoriginal actions and the remaining feature slopes; both original-action corrections stay zero. No geometry, normalization, penalty, radius or coefficient changes.
- Evaluate offset-only and slopes-only using the same training-only service clip, all three frozen risks and unchanged feasibility. Replay all 24 old controllers; report the old clipped/MC controls plus 18 new ablations. Freeze actions before evaluation.
- Attribute uncut MSE change to offset/slopes energies and all three signed cross terms; preserve bias contributions and assigned/opposite16 plus fixed8+8. Clipped components are not assumed additive. This is a fixed-parameter contrast, not a uniquely identified causal explanation.
- Partition each controller's change from its own unchanged-risk control into unchanged, original-to-other, other-to-original and other-to-other. Report cost/fleet/service/wait contributions with the full-state denominator, including zero-action-change states.
- Same 32 cells / 2048 queries / 84480 repeated exposures; train, retained, excluded, full-held and prefix-head-held, every view/bank. Primary mixed excluded/opposite and held/opposite; compact switch detail for those two populations.
- One scheduler CPU / 3072 MiB, any node001-node006 without pin. Zero new fits, raw/risk training, simulations or actor updates; full query ledgers stay remote, scalar report only.
- Verification: five tests passed for frozen split/anchors/clip nonadditivity, signed MSE terms and denominators, all four switch types, complete cell/group replay with no fits or early outcome access, and completed-parent scheduler gate. Both cold CLI startups passed; submission pending.
- Limitations: fixed-parameter, reused-development single-intervention diagnosis, not unique causal attribution, new-prefix confirmation, seed CI or full HRL validation. MC remains an information benchmark. Stage253/248/169 stay closed; defaults unchanged.
