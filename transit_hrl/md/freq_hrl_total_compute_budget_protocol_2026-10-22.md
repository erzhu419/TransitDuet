# Total Compute Budget: Prepared, Not Run

Date: 2026-10-22. This prepares the third evidence item for the bounded
plan-validity paper. No scheduler submission, training or performance run.
Preparation verification: 57 focused tests passed, covering ledger arithmetic,
failure costs, budget selection, simulator-excluded latency and existing
execution/fit/paper-evidence regressions. Unit fixtures are not cost measurements.

## Comparison And Budget

Use the same PointMaze/Reacher tasks, jitter-controller recipe, renewal target,
fit helpers and timing rules prepared for the first two evidence items. Compare
fixed, jitter, best constant offset, simple threshold, original history/current
ridge, effective-DF matched ridge and the two learned termination heads.
Each method trial builds its controller from scratch: cached pretraining and
labels do not become free resources. The existing shared-controller suite remains
a mechanism-isolation comparison, not a total-compute efficiency result.

For each task, optimizer root and method, admit a frozen configuration only if:

```text
search_CPU = sum(offline process CPU seconds of ALL declared attempted trials)
lifecycle_CPU(N) = search_CPU + N * selected_validation_control_CPU_per_episode
search_environment_steps = sum(actual primitive steps of ALL attempted trials)

lifecycle_CPU(N) <= CPU_cap
search_environment_steps <= environment_step_cap
validation_control_wall_p99 <= latency_cap
```

Choose the feasible configuration with lowest development ISE, then lower CPU,
then run ID. Test outcomes never enter admission or selection. CPU is the primary
compute unit; environment steps and latency are additional constraints, not
substitutes for CPU. Report outcome-versus-budget curves, realized costs and
unused budget, rather than padding cheap methods or equating equal iterations
with equal compute. Count deployment simulator cost separately from controller
cost; final research evaluation cost is also reported separately.

## What Is Charged

The exclusive nested ledger includes environment/model setup, controller rollout
and PPO training, checkpoint selection and diagnostics, branch prefix replay and
continuations, grouped ridge/OOF fits or neural supervision, timing selection,
latency calibration and artifact export. It records actual primitive steps,
planner/lower/score calls, label pairs and completed optimizer updates alongside
CPU and wall time. Nested phases sum without charging the same interval twice.

Fixed/jitter methods acquire no branch labels. Constant-offset search pays for
six development schedules. Simple thresholds collect causal features at the
same query states without unused keep/renew continuations, then pay for threshold
search. Ridge methods fit only their own view. The learned termination protocol
uses a two-head behavior mixture, so both heads' fitting and acquisition costs
are charged to its cold-build pipeline. Within a trial, repeated requests for
the same selection result reuse that result; no external training cache is free.

Latency spans observation-history processing, features, scoring, upper inference,
goal decoding, lower inference and action conversion. Simulator advancement is
excluded from this latency but included in offline CPU. Pool all validation
control steps for p50/p95/p99/max; also report episode-first latency, missed 10ms
deadlines and CPU per episode. Fixed schedules perform no trigger scoring;
simple thresholds compute only their required scalar at deployment.

Use one CPU thread per worker, enforce native threadpool limits, and compare
the same CPU/runtime profile. Do not bind workers to a named node. This is
warm-worker accounting: interpreter/library startup and process/model-loading
cold-start latency are outside the measurement. No GPU efficiency claim follows.

## Development And Freeze

The initial development resource ladders are controller iterations192/384/768,
renewal events per class1/2/4, termination pairs per path4/8/12 and deployment
horizons N=1/100/1000. These are declared design points, not an automatic sweep
or a requirement to run their Cartesian product. Fix the actual trial roster
and seed roles before measurement. After development-only cost calibration,
freeze numeric CPU/step/latency caps, N, configurations, required comparisons
and a fresh confirmation population before test access. Use the same N and
caps across methods within each task/root comparison. Each frontier point has
its own declared search roster; do not omit already tried configurations from
an ongoing search to make its measured cost fit a smaller cap.

The selector charges unsuccessful and losing trials. Missing records and unknown
costs block admission rather than becoming zero-cost runs. A normal failed attempt
keeps its partial CPU/step ledger. An interrupted process leaves a running record
and cannot silently overwrite it on retry; use a new output ID and retain both
plans. If the interrupted cost cannot be recovered, that search is incomplete.
These are measured admission caps, not a preemptive training-time kill mechanism:
over-budget search is reported as infeasible, not retroactively called matched.

## Prepared Entry Points

From `transit_hrl/`, this writes only a plan:

```bash
PYTHONPATH=. python3 scripts/run_plan_validity_compute_budget.py \
  --method history --tasks pointmaze reacher_tracking \
  --optimizer-seed 1 --train-seeds 11 12 13 \
  --selection-seeds 21 22 --branch-fit-seeds 31 32 33 \
  --timing-selection-seeds 41 42 --trigger-eval-seeds 51 52 \
  --output /tmp/compute_history_i384.json
```

The IDs are illustrative, not a reserved experiment. Explicit `--execute` would
measure only development paths and save the frozen rule/controller; even then
the command does not evaluate test paths or submit jobs. Reserved test seeds
remain disjoint. Install `requirements-compute-budget.txt` only on an execution
node when experiments resume. No installation is performed by either entry point.

`scripts/select_plan_validity_compute_budget.py` reads compact measurement JSONs
via `--inputs`, the complete roster via `--plans`, and explicit
`--cpu-cap-seconds`, `--environment-step-cap`, `--deployment-episodes`,
`--latency-p99-cap-seconds` and `--output`. Its output freezes admitted run IDs
and retains failures; it does not access checkpoints, raw trajectories or test
outcomes. Keep fit caches/checkpoints on the server; retrieve logs or compact
JSONs first.

## Limitations And Next Decision

Code preparation is not evidence that the candidate is faster or better at a
matched total budget. Historical call counts cannot reconstruct missing training
or scoring timings. Own-schedule controller training remains a separate algorithm
comparison; the present comparison holds the jitter-training recipe fixed.
Named-method reproductions and local-to-full-episode reliability remain open.

When computation resumes: run a small development-only cost pilot, freeze the
budget frontier, then evaluate admitted configurations on fresh paired roots
and full episodes. Report performance and total costs by task/root, not by query.
The fourth item now has a [prepared episode-reliability path](freq_hrl_episode_reliability_protocol_2026-10-22.md).
Its branch diagnostics are evaluation expenditure, not free training supervision
or deployment control cost.
