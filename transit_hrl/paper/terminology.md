# Plan-Validity Manuscript Terminology

Active vocabulary, 2026-10-22. The previous frequency-responsibility ledger is
retained in `archive/2026-08-31/terminology.md` for that different method.

| Term | Definition in this manuscript |
| --- | --- |
| plan | A waypoint emitted by the upper policy and held until the next actual upper call |
| upper policy / lower policy | Learned goal policy / goal-conditioned physical actuator policy |
| plan-validity predictor | An operational predictor of the keep-versus-renew tracking-loss difference; not a validity probability |
| renewal advantage | `keep ISE - renew ISE` over the paired 50-step branch horizon; positive favors renewal |
| causal-history features | Features computed from current and past observations; causal means temporally available, not causal discovery |
| budgeted replanning | First threshold crossing or deadline, exactly one upper call per 50-step bin |
| planning budget | Upper-call count; distinct from total FLOPs, wall time and training/label-acquisition cost |
| ISE | Integrated squared tracking error, `0.01 * sum(tracking_distance**2)` |
| optimizer-seed root | Independent PointMaze controller-training replicate; evaluation paths are nested within it |
| prefix | Native Transit history before a queried action; changing only its future does not create a new prefix |
| conditional MC CI | Paired future-sampling interval on fixed Transit prefixes in the single-intervention studies |
| episode CI | Full-scene uncertainty in Stage304; its bootstrap and normal intervals estimate the same sampling source |
| base / zero_abstain | The full P8/S8 zero-event abstention rule before additional service or joint filtering |
| service-only / joint | Frozen Transit service-score / service-plus-learned-fleet-score filtering of the same proposal |
| raw | Each Transit profile's frozen reference selector; not flat RL or a no-control policy |
| single intervention | One queried timetable action followed by the frozen suffix, not repeated online selection |
| supported | The study's registered criterion passed; not a statement of universal safety or dominance |
| Freq-HRL | Historical project and broader frequency-responsibility hypothesis; not the name of every positive result |

ISE advantages in Table1 are comparator minus candidate; return advantages are
candidate minus comparator. Transit differences are first controller minus
comparator, where lower cost/wait is better. Cost is a composite normalized
endpoint, not money. PointMaze and Transit estimands are not pooled.
