# MuJoCo v17.1 Headroom-Homotopy Outcome

## Decision

`headroom_homotopy_preflight_not_supported`

This is valid development evidence, not confirmatory evidence. All 15 frozen
scheduleurm cells completed on node004 with dynamic placement and no failed
attempts. The local analysis used only 1.9 MB of synchronized summaries and
held-out rows; checkpoints and training histories remain server-only.

## Frozen Result

All four candidate arms passed the core lower-frequency mechanism in all three
environments:

- trained checkpoint: 3/3 per arm;
- complete-macro zero sum: 3/3 per arm;
- active projection: 3/3 per arm;
- exact responsibility reconstruction: 3/3 per arm;
- at least 10% raw lower-LF reduction versus smooth direct: 3/3 per arm;
- at least 10% raw lower-LF reduction versus the candidate latent proposal:
  3/3 per arm.

The reward claim failed:

- headroom exact: reward noninferiority 0/3;
- headroom homotopy: 1/3;
- homotopy plus 0.5 promotion: 1/3;
- homotopy plus 1.0 promotion: 1/3.

The half-gain promotion arm was the strongest candidate. It improved the joint
frequency merit in Hopper and Walker2d and kept upper HF nonworsening in those
two environments. It still missed the reward floor in HalfCheetah and Hopper;
HalfCheetah upper-HF power increased by 489% and its joint merit worsened.
Consequently no global arm was eligible for fresh multiseed expansion.

## Diagnosis

Headroom feasibility and a training homotopy do not solve the structural cost
of forcing the lower action to have zero mean inside every 16-step upper macro.
Locomotion needs slow actuator responsibility that the regular-rate upper plan
cannot infer before the lower proposal exists. A one-macro-delayed promotion
recovers part of this demand in Walker2d and Hopper, but it is too late for
HalfCheetah and can force high-frequency compensation into the upper policy.

`AdditiveActionClipRate` also exposed a diagnostic defect. It counts every
float32 component sum above exactly one, but this run did not record the excess
magnitude. Eleven candidate cells therefore failed the zero-tolerance rate gate
without evidence that the excess was material. The next protocol must report
maximum and RMS clip excess. This diagnostic issue cannot rescue v17.1 because
the reward gate fails independently.

## Next Architecture

Do not tune promotion gains or repeat strict zero-DC with more seeds. The next
mechanism should:

1. preserve the total executed control path exactly;
2. move a frozen smooth DC responsibility from lower to upper as an additive
   gauge transfer;
3. keep latent policy actions separate from canonical responsibilities;
4. make the transfer state independent of homotopy strength at the endpoints;
5. report action-bound excess magnitude, not only a zero-tolerance hit rate.

This directly tests whether smooth responsibility migration can obtain the
v16 no-harm property without the macro-hold upper-HF failure.

## Claim Boundary

Allowed: v17.1 enforced exact lower macro zero sum and strongly reduced raw
lower-LF diagnostics in a frozen one-optimizer development panel; half-gain
promotion partially recovered reward in Hopper and Walker2d.

Forbidden: v17.1 validates reward no-tradeoff, cross-environment upper/lower
frequency separation, exact plant-level headroom after float32 execution,
fresh-seed confirmation, or a final Freq-HRL algorithm.
