# MuJoCo v18.3 Causal Joint Projection Outcome

## Decision

Status: `causal_joint_projection_stops_before_fresh_path_access`.

V18.3 evaluated both frozen label-free causal budget semantics on the unchanged
120-path development panel. The instantaneous projector passed every direct and
exact frequency-feasibility check, but failed all action trust-region gates.
No fresh path was accessed.

## Feasibility

The selected instantaneous projector kept all 120 paths numerically valid,
made all 120 projected upper/lower traces directly joint feasible, and left all
120 corrected total-action traces feasible under the independent full-horizon
oracle. It preserved all 113 reference-feasible paths and recovered all seven
actor-floor paths, including 2/2 for seed 2802248628 and 5/5 for seed 294864529.
This is the first causal reused-path mechanism in this sequence to close the
7/7 feasibility count without target labels.

The prefix-ledger candidate also reached 120/120 endpoint feasibility, but only
42 paths retained stepwise component-feasibility validity and it recorded 451
nonconverged steps. This confirms that spending cumulative budget without a
future viability reserve can enter causal dead ends.

## Trust-Region Failure

The instantaneous mechanism changed the total action on 13,234 primitive steps.
Its global absolute correction maximum was 1.8076. Reference-feasible correction
RMS averaged 0.1375 and reached 0.2935; actor-floor correction RMS averaged
0.2819 and reached 0.3019. These values are orders of magnitude above the frozen
0.05 absolute, 0.01 reference-RMS, and 0.015 actor-floor-RMS gates.

The result therefore establishes a useful feasibility construction but rejects
strict per-step residual enforcement as a behavior-preserving deployment rule.

## Efficiency

Scheduleurm task `t85971` completed on node003 with 16 single-threaded workers.
Worker runtime was 750.0 seconds, scheduler runtime was 771.7 seconds, and peak
RAM was 2,183 MB. Only the compact summary and preregistration were synchronized.

## Next Mechanism

The next projector must amortize current budget debt over a causal 16/32-step
forecast, execute only the first projected action, and replan. This can retain
the label-free feasibility signal while avoiding the globally aggressive
instantaneous correction.

## Claim Boundary

V18.3 is development-only. Its 7/7 reused-path feasibility does not establish
reward preservation, fresh-seed generalization, learned control, leakage
no-tradeoff, or manuscript support.
