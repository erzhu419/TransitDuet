# Terminology Ledger

| Canonical term | First-use definition | Avoided variants | Decision |
| --- | --- | --- | --- |
| FreqDuet | Frequency-separated hierarchical bus timetable and holding controller | FreqHRL, TransitHRL, TransitDuet | Use only for the Protocol V6 method in this manuscript. |
| Protocol V6 | Frozen physical, causal, endpoint, and evaluation contract | current protocol, final protocol | Capitalize and name the version. |
| current controller | `F_freqduet_protocol_v6_confirmed_main_hiro` | main, V1, best model | Use the exact identifier when provenance matters. |
| Protocol V6 reference | `F_freqduet_protocol_v6_noguard_hiro` | no-guard baseline, control | State that it retains the harmonic path and disables the legacy guard. |
| fixed headway | Deterministic strong external comparator under shared V9 scenario tapes | fixed policy, fix policy | Use `fixed headway` in prose and `fixed_headway` only for code identifiers. |
| automatic passenger counting (APC) | Station-level observed passenger counts | ridership data | Define once; do not conflate with AFC. |
| automated fare collection (AFC) | Fare-system transactions used in the external profile audit | APC, OD | Define once; not onboard load. |
| automatic vehicle location (AVL) | Event-time vehicle position and movement context | GPS state | Define once. |
| headway coefficient of variation (headway CV) | Standard deviation divided by mean over valid headway events | regularity score | Lower values mean more regular service. |
| restricted passenger journey | Horizon-censored total journey minutes per generated passenger | journey time | Use the full term at first mention. |
| delayed fleet readiness | A scheduled trip denied at least once and later retried | cancellation, unlaunched trip | Never describe as permanent cancellation when all trips launch. |
| V8 | Independent 40-episode confirmation, 6 training by 4 evaluation seeds | short run | Keep separate from V9. |
| V9 | Independent 200-episode robustness test, 8 training by 8 evaluation seeds | long run | Keep separate from V8; status is `longtrain_not_confirmed`. |
