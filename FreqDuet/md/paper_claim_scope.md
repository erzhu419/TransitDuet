# FreqDuet Paper Claim Scope

Last updated: 2026-09-10 CST

## Active Scope

The active method is Protocol V6. The canonical controller is
`F_freqduet_protocol_v6_confirmed_main_hiro`; V8 is its successful independent
40-episode confirmation and V9 is its failed 200-episode robustness gate. Both
must appear together in any current paper account. The old paper-main V1 and
its composite endpoint are historical controls, not headline evidence.

| Canonical term | Meaning |
| --- | --- |
| FreqDuet V6 | Journey-feasible frequency-separated HRL with executable timetable, causal APC/AVL state, fixed fleet, and coherent sampled holding actions. |
| primary endpoint | `restricted_total_journey_horizon_min`; lower is better. |
| current best controller | `F_freqduet_protocol_v6_confirmed_main_hiro`, confirmed at 40 episodes in V8. |
| strong fixed-headway | Exact V6 fixed-headway comparator on shared scenario tapes. |
| historical V1/V4 evidence | Legacy service-cost experiments retained for diagnosis and negative-results reporting. |
| external realism audit | Public AFC/APC/OD/load/AVL evidence; not a field-effect estimate. |

## Supported Method Statement

FreqDuet V6 assigns causal low-frequency demand state to executable upper-level
headway planning and high-frequency residual plus compact APC/AVL state to
arrival-event holding. The current controller uses a fixed fleet, a rolling
zero-total headway budget, a discrete holding alphabet, and a two-sided
departure-regularity reward computed from pre-action evidence. Its legacy
holding guard, promotion mechanism, and leakage penalty are disabled.

## Supported Result Claims

- V8 confirms a headway-CV reduction against the Protocol V6 reference config,
  with the registered passenger-journey no-harm condition satisfied. The
  contrast bundles compact APC/AVL context and the two-sided regularity
  objective; it is not an isolated legacy-guard effect.
- In V9, the same controller improves passenger journey relative to that reference,
  but its preregistered long-run headway effect is not confirmed.
- Under the V9 external comparison, FreqDuet is more regular and has lower
  restricted service cost than fixed headway, but fixed headway has materially
  lower passenger journey time.
- FreqDuet improves passenger journey relative to rule holding and rule MPC in
  that V9 comparison.
- Public external data support simulator-realism audits, not field efficacy.

## Claims Not Established

- Long-run confirmation of the V8 regularity effect.
- Passenger-journey superiority or statistical equivalence to fixed headway.
- Universal superiority of frequency separation over every internal control.
- An isolated causal effect of frequency separation versus same-stage NoFreq,
  RawHistory, or AllFreq controls; those arms are absent from the current
  confirmatory package.
- Generalization across completed route-family or service-day policy matrices.
- Real-world deployed wait-time or journey-time improvement.
- Complete same-day AFC/APC/AVL/OD field calibration.
- MTA Bus Time as APC or onboard-load data.
- Results from FreqHRL or TransitHRL as FreqDuet evidence.

## Evidence Map

| Evidence | Current interpretation |
| --- | --- |
| `protocol_v6_locked_contract_2026-08-08.md` | Active physical, causal, endpoint, and submission contract. |
| V8 confirmation gate | Successful independent 40-episode effect evidence for compact weight two. |
| V9 long-training gate | `longtrain_not_confirmed`; mandatory robustness result. |
| V9 learned-versus-external table | Valid fixed/rule/MPC trade-off comparison under V6. |
| V28-V32 records | Negative development appendix; no controller promotion. |
| External AFC/APC/AVL audits | Realism support only, within documented source limits. |

## Writing Rule

Every result sentence must name or unambiguously inherit the protocol, endpoint,
comparator, seed level, and uncertainty statement. V8 and V9 may not be pooled,
and legacy composite numbers must be labelled historical.
