# Stage163: Causal Supply-Coupled Rolling Allocation

Stage162's positive strengths all worsen mean cost on both roots. Stop constant
strength tuning and do not train alpha from that frontier. Isolate a supply term
in a shared causal candidate selector, with the physical objective unchanged.

Both selectors use five strengths between original nominal interval preferences
and the current causal forecast, projected onto the same current feasible daily
endpoints. This differs from Stage162's accumulated tentative-interval baseline:
zero can restore uncommitted nominal spacing, not just freeze prior compression.
The next six trips remain irrevocable; all adjacent intervals stay 240-480 s.

Predict joint-direction occupancy from active buses' route progress, future known
departures (other-direction committed/tentative plans included), and direction
trip-duration EWMA (.2) updated only from completed real trips. Initial duration
uses current route speeds/distances plus 4 s per segment. The copied simulator
records a passive launch/completion/direction ledger, including trips whose bus
has since been reused; it changes no dispatch rule, action, reward or RNG.

Demand-only minimizes forecast-weighted interval waiting divided by 600;
supply-aware adds `max(predicted_peak-12,0)^2/12`. Exact ties choose lower alpha.
The predicted wait uses uniform arrivals within each forecast interval, not a
capacity/OD simulator. Interval CV is diagnostic only: actual headway, unserved,
completion and fleet components remain in the unchanged measured physical cost.

Five conditions: existing window forecast, rolling0, rolling1, demand-only and
supply-aware. All 120 source controls must reproduce Stage162 physical outcomes
AND planned/actual schedules after adding the passive ledger. Two roots, five
regimes, all four registered scenes: 200 frozen episodes, 12,276,000 native ticks,
zero training updates. Analyze supply minus demand (isolated fleet term), both
minus window forecast, actual fleet/nonfleet costs, wait and selection/prediction
diagnostics. Do not substitute predicted improvement for physical improvement.

Ten unpinned scheduler cells on node001-006, 1 CPU/3 GB each. Only compact result
JSON sync; raw CSV and checkpoints remain remote. Performance pending. If supply
coupling earns net value, next learn the allocation action with its own constant
control; otherwise diagnose duration/capacity prediction before any seed expansion.
123 focused tests pass, including joint occupancy ties/active buses, causal
completion EWMA, supply-term selection, reversible tentative spacing, native
commitment/tail execution, matched analysis and scheduler/worker budgets.
Local tests use stub native environments, not full native training or episodes.

Run `native_transit_supply_allocation_stage163_frozen_20261011_r1`:
t142104-t142113 launched on node001/004/005/006. Source reproduction and final
physical performance remain pending; no local native episodes or checkpoint
downloads were used.

## Limitations

Frozen development mechanism, not learned authority or independent confirmation.
Point-estimate duration/occupancy and waiting proxies omit future speed shocks,
capacity/OD effects and dispatch-cap delays; they are not fleet guarantees.
