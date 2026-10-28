# Native Preflight And Development Pilot

Prepared on2026-10-28; no scheduler task submitted, remote installation or
performance run. Parameters and seed roles come from the
[unified registration](freq_hrl_unified_preregistration_2026-10-22.md).
Local verification:87 focused tests passed; scheduler submission was mocked.

## Prepared Work

Preflight has6 cells:2 plants x1 reserved root x3 primary methods. Each uses
300 steps,2 controller iterations,1 renewal query/class and2 timing pairs/path;
the50-step target,10ms clock, optimizer/network and timing contract stay fixed.
Pilot has12 cells:2 plants x2 separate roots x3 methods, using the unchanged
1,200-step/384-iteration/four-query/12-pair recipe. Confirmation roots and paths
are excluded from both stages. The generated `cells` sections list every
resolved argument and all five seed roles, rather than relying on CLI defaults.

Native preflight reloads the saved model, reproduces development validation,
checks actual option durations/lower boundaries, same-prefix keep/renew and
adaptive factual replay, conditional futures and risk telemetry. Distance1 is
only a telemetry fixture, not a calibrated operational risk cutoff. Learning
and learned-termination eligibility are pilot diagnostics, not preflight pass
criteria. A completed pilot can still have an ineligible stopping baseline.

Every worker requests1 CPU,2048 MiB RAM and0 VRAM, with native threadpools limited
to1. The whitelist is node001-node006; there is no hard/preferred node. Each
stage is deliberately bounded to6/12 simultaneous cells, not a1152-worker run.
Automatic requeue/resume is disabled: interrupted attempts and their costs must
be reconciled before another registration. The scripts use the existing
scheduleurm bulk API, not Slurm.

## Resume Commands

From `transit_hrl/`, these commands prepare intents only:

```bash
RUN=plan_validity_dev_20261028
PYTHONPATH=. python3 scripts/submit_plan_validity_development_scheduleurm.py \
  --run-name "$RUN" --stage preflight \
  --output "results/$RUN/inputs/preflight_submission.json"
PYTHONPATH=. python3 scripts/submit_plan_validity_development_scheduleurm.py \
  --run-name "$RUN" --stage cost_pilot \
  --preflight-gate "results/$RUN/preflight_gate.json" \
  --output "results/$RUN/inputs/pilot_submission.json"
```

The reviewed blueprints are
[`preflight`](../protocols/plan_validity_preflight_scheduler_2026-10-28.json) and
[`pilot`](../protocols/plan_validity_pilot_scheduler_2026-10-28.json).
When compute is authorized, repeat only the preflight command with `--submit`.
After its6 public result files arrive, run:

```bash
PYTHONPATH=. python3 scripts/analyze_plan_validity_development.py \
  --run-name "$RUN" --stage preflight --output "results/$RUN/preflight_gate.json"
```

Only then repeat the pilot command with `--submit`. The submitter and worker
both check the gate's complete cohort, parameters, seeds and runtime profile;
file existence alone cannot release pilot. No pilot task is queued early.
Analyze the completed pilot with the same analyzer's `--stage cost_pilot` and
a separate output. Read learning curves, fitting-bank plan/query diagnostics,
termination eligibility and measured costs before any recipe lock.

## Storage And Accounting

Each cell has a server-only `private` directory for model/cache/measurement
artifacts. Scheduler result syncing points only at its `public` directory,
containing a compact `result.json`; logs/public JSONs are the first retrieval.
Fitted rule weights stay in the private report; the public result carries only
the rule summary and a server-side source pointer.
The existing result/data staging exclusions remain active. Pilot inputs stage
only the registration and compact preflight gate, not private checkpoints.

The outer worker CPU ledger includes its pipeline, fitting-bank readout and
native replay audits exactly once. The inner pipeline breakdown is retained
separately, not added to total CPU a second time; actual step/call counters are
relayed to the outer ledger. Both generated manifests expose `expected_runs`,
so all preflight/pilot public records can later be charged as shared development
costs over the declared8-root confirmation cohort. Missing/failed cells remain
visible and cannot become zero-cost successful runs.

## Pending Execution Checks

The existing configured Linux interpreter is a candidate, not remotely verified.
Override `--python-executable` if required; any dependency installation must run
on node001-node006 through the scheduler, not on a login node. No installer is
embedded. The2 GiB request is provisional; preflight publishes peak process RSS
for sizing, not summed node memory or a PSS measurement. Runtime failures block
pilot, while positive native preflight establishes software qualification only.
Actual cost caps, risk distances and scientific evidence remain pending.
