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
JSON sync; raw CSV and checkpoints remain remote. All ten tasks are complete;
200 episodes and all 120 physical AND schedule controls qualify. Supply coupling
has not earned a net benefit against window forecast on either root.
123 focused tests pass, including joint occupancy ties/active buses, causal
completion EWMA, supply-term selection, reversible tentative spacing, native
commitment/tail execution, matched analysis and scheduler/worker budgets.
Local tests use stub native environments, not full native training or episodes.

Run `native_transit_supply_allocation_stage163_frozen_20261011_r1`:
t142104-t142113 completed on node001/004/005/006. Relative to window forecast:

| Root | Supply cost delta | Wait delta (min) | Fleet cost delta | Cost wins/20 |
| --- | ---: | ---: | ---: | ---: |
| 397 | +0.1794995 | -0.81465 | +0.2500000 | 4 |
| 401 | +0.0835695 | -0.70835 | +0.1458333 | 10 |

Supply minus demand cost is -0.00003255 / -0.02427585. Root401's benefit is
concentrated in persistent shift; other regime means are unchanged or worse.
The fleet-cost term distinguishes candidates at only 10/880 and 21/880 decisions,
and changes the wait-only argmin at 8/880 and 17/880. Alpha1 still accounts for
865/880 and 859/880 choices. At the first decision all candidate peaks are 5-6,
so every fleet penalty is zero despite eventual actual episode peaks of 14-15.
Later maximum-prediction agreement is not a predictive calibration result: the
predictor includes the already realized historical peak. Initial duration is a
free-flow prior; actual completed-trip EWMA at the last decision averages about
2,100-2,255 s. This motivates an independent training-only duration calibration,
not another strength sweep or learned-alpha training on this negative frontier.

Next: Stage164 collects real single-trip times on separate fitting/check scenes,
pools a direction prior across all five training regimes per root, and checks
it before a new closed-loop comparison. No local native episodes or checkpoint
downloads were used.

## Limitations

Frozen development mechanism, not learned authority or independent confirmation.
Point-estimate duration/occupancy and waiting proxies omit future speed shocks,
capacity/OD effects and dispatch-cap delays; they are not fleet guarantees.
