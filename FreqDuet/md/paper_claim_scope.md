# FreqDuet Paper Claim Scope

Last updated: 2026-08-08 CST

## Active Scope

The manuscript is on submission hold while Protocol V5 is evaluated. The old
paper-main V1 and its composite endpoint are historical controls, not the
current method or headline evidence.

| Canonical term | Meaning |
| --- | --- |
| FreqDuet V5 | Journey-feasible frequency-separated HRL at commit `f2a5ae8e18`. |
| primary endpoint | `restricted_total_journey_horizon_min`; lower is better. |
| strong fixed-headway | Exact V5 fixed-headway comparator on shared scenario tapes. |
| physical safety | Holding exposure, denied-trip rate, readiness delay, launch/completion rates, unserved rate, and fleet/headway constraints. |
| historical V1/V4 evidence | Legacy service-cost experiments retained for diagnosis and negative-results reporting. |
| external realism audit | Public AFC/APC/OD/load/AVL evidence; not a field-effect estimate. |

## Provisional Method Statement

FreqDuet V5 assigns causal low-frequency demand state to executable upper-level
headway planning and high-frequency residual state to arrival-event holding,
while enforcing a fixed fleet, a zero-total headway budget, causal holding
feasibility, and passenger-journey credit.

This is a method description only. It is not yet a performance claim.

## Claims Allowed Before V5 Confirmation

1. The V5 implementation has a frozen source/config/seed contract and a
   crossed common-random-number evaluation protocol.
2. The harmonic decomposer, causal state interfaces, exact timetable planner,
   holding guard, and passenger-journey diagnostics are implemented and tested.
3. The corrected V4 experiment is a documented negative result showing why the
   old service-cost claim was insufficient.
4. Public external data support simulator-realism and route/day protocol audits
   within the existing data-availability boundaries.

## Claims Requiring Completed V5 Evidence

- Frequency features outperform no-frequency and dimension-matched raw history.
- LF-to-upper and HF-to-lower allocation outperforms all-frequency or one-layer
  controls.
- V5 matches or outperforms fixed-headway on passenger journey time.
- The causal guard, load-weighted holding cost, headway budget, and complete
  upper credit each have a positive effect.
- The selected policy generalizes across untouched seeds, perturbation
  families, route families, and service days.

## Claims Not Supported By This Project

- Real-world deployed wait-time or journey-time improvement.
- Complete same-day AFC/APC/AVL/OD field calibration.
- MTA Bus Time as APC or onboard-load data.
- Results from FreqHRL or TransitHRL as FreqDuet evidence.
- The closest preserved TransitDuet-family control as an unmodified checkout of
  the original TransitDuet repository.

## Evidence Map

| Evidence | Current interpretation |
| --- | --- |
| `protocol_v4_selection_outcome_2026-08-08.md` | Corrected negative result; invalidates the old headline effect claim. |
| `protocol_v5_journey_feasible_contract_2026-08-08.md` | Active preregistered method and decision contract. |
| V5 development-screen aggregate | Pending; determines whether frequency and allocation claims survive. |
| V5 untouched 200ep confirmation | Sealed and pending. |
| V5 external-baseline comparison | Pending learned rows; external comparator rows are collected. |
| External AFC/APC/AVL audits | Realism support only, within documented source limits. |

## Writing Rule

Every result sentence must name or unambiguously inherit the protocol, endpoint,
comparator, seed level, and uncertainty statement. Legacy composite numbers
must be labelled historical and cannot be mixed into V5 tables.
