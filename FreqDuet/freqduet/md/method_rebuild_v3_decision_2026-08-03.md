# FreqDuet Method Rebuild Decision (2026-08-03)

## Decision

Do not freeze the paper yet. Continue with a bottom-up method repair before
the confirmatory 200-episode paper matrix.

## Evidence

- The compact continuous upper planner is promising: service cost `1.21970`
  versus `1.23690` for the previous four-dimensional main, but its paired CI
  still crosses zero.
- The original exact categorical upper planner is significantly worse and can
  collapse to the zero curve, so discrete action libraries are not promoted.
- A physical lower-state encoding improves the weak discrete-history reference
  from `1.32828` to `1.25413`, primarily by removing unstable holding. The
  near-zero lower action shows that state scaling is necessary but not yet a
  complete controller.
- Correct trip reset worsens the weak reference by `+0.01253`, CI
  `[+0.00039,+0.02678]`. The old score therefore partly depends on invalid
  cross-trip action/state persistence and cannot be retained for publication.
- Trip-end holding finalization alone has exactly zero effect because the old
  upper state reads stale bus fields instead of lifecycle-owned feedback.

## Structural Defects Being Repaired

1. The continuous actor emits a command, EMA changes it, and replay stores the
   changed action while policy optimization evaluates the raw command.
2. EMA dynamics depend on the previous plan, but that plan is absent from the
   compact upper state.
3. Promotion creates variable upper decision intervals while SAC uses one
   fixed discount per transition.
4. Upper holding state slots use reused-bus residue rather than completed-trip
   lifecycle data.
5. The lower controller carries pending actions across trips and bootstraps
   through its final observable state despite having no subsequent controlled
   transition.
6. Critic L1 uses an unnormalized parameter sum. At initialization its default
   contribution is about `10.5`, far larger than the observed upper Q MSE.
7. Upper `beta_ood` and `weight_reg` exist in the trainer but were not wired
   from configuration.

## Promotion Gate

The rebuilt method is promoted only if it passes paired frozen-policy tests on
service cost, wait, headway CV, fleet overshoot, completion, action stability,
and seed failures. Mean improvement alone is insufficient. The winning
structure then receives multi-domain 200-episode confirmation against the
external fixed-headway baseline.
