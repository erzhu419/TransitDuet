# MuJoCo v18.4 Receding-Horizon Joint Projection Outcome

## Decision

Status: `receding_joint_projection_stops_before_fresh_path_access`.

V18.4 completed the frozen four-candidate screen on all 120 unchanged reused
paths. The selected H16 damped-velocity projector remained numerically valid,
but failed direct online feasibility, actor-floor recovery, reference
preservation, and every action trust-region gate. No fresh path was accessed.

## Direct Versus Exact Feasibility

The selected candidate made only 69/120 realized upper/lower traces directly
joint feasible. It preserved 67/113 reference-feasible paths and recovered 2/7
actor-floor paths: 1/2 for seed 2802248628 and 1/5 for seed 294864529.

The independent full-horizon oracle nevertheless found all 120 corrected total
traces feasible. This is not an online success. It means each corrected total
could be decomposed by an acausal full-trajectory solver, while the components
actually produced by the causal projector failed on 51 paths. The two-stage
audit prevented that offline existence result from being misreported as
realized hierarchical responsibility separation.

The other candidates were weaker on the direct audit. H16-hold and both H32
candidates reached only 40/120 direct-feasible paths and recovered no
actor-floor path. They were not exact-audited and are recorded as unaudited,
not exact-infeasible.

## Terminal-Debt Failure

The selected projector accumulated 40,962 prefix budget violations. Its fixed
16-step optimization could promise compensation near the end of each forecast,
execute only the first action, and move that compensation forward again at the
next replan. This horizon shifting left the realized endpoint upper-frequency
power over budget on 51 paths. The failure is recursive feasibility, not a
choice between horizon 16 and 32.

The action changes also remained unacceptable. Global absolute correction was
0.9888. Reference-feasible correction RMS averaged 0.1279 and reached 0.2507;
actor-floor correction RMS averaged 0.2370 and reached 0.2525. These exceed the
frozen 0.05 absolute, 0.01 reference-RMS, and 0.015 actor-floor-RMS gates by
large margins.

## Efficiency

Scheduleurm task `t85972` completed on node003 with 16 single-threaded workers.
Worker runtime was 1,583.4 seconds, including 1,145.3 seconds for 480 direct
path evaluations and 438.1 seconds for the selected candidate's 120 exact
audits. Scheduler runtime was 1,631.5 seconds and peak RAM was 1,925 MB. Only
the 168 KB preregistration and summary directory was synchronized.

## Next Mechanism

The next design must enforce a nonshiftable terminal-debt certificate or a
recursive invariant set. It must distinguish the upper HPF debt that can be
paid by future smooth actions from debt that is merely pushed beyond a moving
horizon. Horizon or forecast-mode tuning alone is rejected by this result.

## Claim Boundary

V18.4 is development-only. It does not establish behavior-preserving online
projection, reward preservation, fresh-seed generalization, learned control,
leakage no-tradeoff, or manuscript support.
