# Same-Algorithm Second Task: Prepared, Not Run

The [unified preregistration](freq_hrl_unified_preregistration_2026-10-22.md)
supersedes earlier sample-count suggestions and defines the two freeze stages.

Date: 2026-10-22. No scheduler submission, controller training or performance run.
Only unit tests and a six-step native Reacher interface check are authorized here.
Preparation verification: 43 targeted tests passed, including legacy execution/
label equivalence, shared PPO initialization/credit, and paper-evidence regressions.

## Task And Scope

Use **ReacherTracking-v1**, a derived task on Gymnasium **Reacher-v5** physics.
The plant is an articulated two-joint arm controlled by joint torques, rather
than PointMaze's translating point agent. The upper policy proposes Cartesian
endpoint waypoints; the lower policy acts every 10ms. An elliptical external
target changes its hidden direction/speed at persistent regime transitions;
measured short torque disturbances and an irrelevant distractor also occur.
This supplies slow useful plans, fast feedback and occasions for plan revision.

The stock [Reacher implementation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/reacher_v5.py)
supports `frame_skip`; the [plant XML](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/assets/reacher.xml)
uses a 0.01s integration step. Set `frame_skip=1`, not the stock default 2, and
extend the native time limit to the shared 1,200-step episode. Physics, motors,
joint constraints and the integrator remain the library implementation.
This is not an evaluation of the original static-target Reacher score.

## What Stays Identical

- Primary target: `ISE(keep) - ISE(renew)` over 50 primitive steps, with no
  additional upper calls in either branch and closed-loop lower control.
- Observation semantics: current physical state, 2D target, 2D measured action
  disturbance and 2D distractor. Same causal history builder and 37 monitoring
  features; same existing 39-feature history and 170-feature current predictor.
- Acquisition: the existing six event classes, event lags, fit-path counts,
  blocked jitter calls and extra-supervision accounting. Simulator event clocks
  select offline queries but never enter deployed features.
- Fit: call the original `fit_budgeted_trigger_predictors` through the same
  comparison function; same alpha grid, fit-path grouped ridge selection,
  within-fit OOF predictions and q75 threshold. No second-task threshold search
  for the primary candidate.
- Timing: first threshold crossing at offsets 0/5/10/15/20, otherwise deadline25;
  first call at0; exactly one call per50-step bin, 24 calls per full episode,
  and realized option lengths25-75. Scoring at the deadline retains Stage9's
  implementation convention.
- Controller: the same goal-conditioned PPO class, network dimensions, default
  hidden width128, optimizer settings, intrinsic progress reward, jitter training,
  actual-duration upper credit and lower boundaries at actual waypoint changes.
  Each plant learns its own weights; this is algorithm transfer, not zero-shot
  transfer of PointMaze weights.
- Controls: fixed, random jitter, best constant offset, strongest simple threshold,
  original current/history recipes, effective-DF matched ridge, and matched neural
  stopping heads. They use the first preparation's shared fitting/selection loop.

`experiments/plan_validity_transfer.py` contains one execution/branch loop for
both plants. The new entry point does not choose a different objective or
selector for Reacher. Existing frozen PointMaze implementations/results remain
unchanged; deterministic fixtures test equivalence to their execution and labels.

## Physical Adaptations

Reacher target and endpoint coordinates are divided by the fixed 0.21m arm
reach. Physical channels are wrapped joint angles divided by pi and angular
velocities divided by10 rad/s: four genuine state coordinates, no dummy padding.
The target ellipse is `(0.65*cos(phase), 0.4875*sin(phase))` in normalized space;
the inherited regime speed modes become angular speeds. Disturbance amplitudes
remain in normalized actuator coordinates and are added before control clipping.

Relative upper goals are projected to normalized radii0.10-0.95. This annulus
respects the stock arm's 0.1/0.11m links and +/-3rad elbow limit, instead of
issuing inaccessible square-corner waypoints. The adapter affects only physical
goal feasibility, not the renewal target, fit or stopping rule. As in Stage9,
the post-action endpoint is scored against the pre-action target; the returned
next observation contains the next target. Reward remains `exp(-distance)`.

## Preparation Entry Point

From `transit_hrl/`, this writes a plan only:

```bash
PYTHONPATH=. python3 scripts/run_plan_validity_second_task.py \
  --optimizer-seed 1 --train-seeds 11 12 13 \
  --selection-seeds 21 22 --branch-fit-seeds 31 32 33 \
  --timing-selection-seeds 41 42 --trigger-eval-seeds 51 52 \
  --output /tmp/freq_hrl_second_task_plan.json
```

These are illustrative IDs, not reserved experiments. Default tasks are PointMaze
and Reacher tracking under one configuration; `--tasks reacher_tracking` selects
only the second plant. Execution requires explicit `--execute`; neither command
submits jobs. Use the existing Linux runtime requirements, not a login-node
installation. Fit-label caches remain on the execution machine, separately from
the compact result. Retrieve result/log first, not trajectories or checkpoints.
The compact result retains controller-selection learning curves and a clearly
labelled fitting-bank waypoint/value summary for diagnosing plan authority.

## Next Decision After Computation Resumes

First run native training/replay preflight with a reduced iteration count, keeping
the renewal target and clocks unchanged. Check learning, actual waypoint effects,
shared-prefix replay and the noncollapse diagnostics. A controller that ignores
its plan, invalid physical timing or missing query classes blocks confirmation;
it is not repaired by selecting favorable test paths.

Then freeze eight new optimizer roots per task, with eight controller-training,
eight controller-selection, eight timing-fit, eight timing-selection and512
test paths per root, including8 predeclared diagnostic paths. Five roles are
disjoint. The development entry point never executes the reserved test paths.
Freeze every candidate before test
access; revealed historical confirmation paths are not new confirmation data.
Aggregate paired full-episode ISE/return by optimizer root, not by query/path.
Report each task separately, using paired root-level t intervals as in Stage9.
Freeze the required comparisons before test access; the joint claim requires
all of them to pass, including the comparison with a genuinely competent learned
stopping policy. Report both evaluation half-panels without choosing the better
one. Failed roots remain in the report, without outcome-based filtering.

## Limitations

No second-task performance or noncollapsed stopping result exists yet. Reusing
event clocks intentionally holds exogenous timing structure constant while changing
the plant and target geometry; it is not independent real-world data or a broad
cross-domain benchmark. Raw ISE units differ: multiply Reacher normalized ISE by
`0.21**2` for square-metre-seconds, and never pool it with PointMaze raw ISE.
The neural stopping comparator has separate full-episode paired credit; it does
not replace the primary candidate's unchanged short-window keep/renew target.
Own-schedule algorithm comparisons, named-method reproductions and
local-value-to-episode reliability remain separate gaps. Total-compute accounting
and budget selection now have a [prepared path](freq_hrl_total_compute_budget_protocol_2026-10-22.md),
without measured efficiency evidence.
