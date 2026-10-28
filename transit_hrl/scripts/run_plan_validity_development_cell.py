"""Execute one registered native preflight or pilot cell on a scheduler worker."""

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_development import (
    METHODS, PHASES, attempt_directory, cell_arguments, development_cells, expected_runs, load_plan,
    native_preflight, require_preflight_gate,
)
from freq_hrl.experiments.plan_validity_preregistration import TASKS
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_compute_budget import prepare, run_trial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=PHASES, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--preflight-gate", type=Path)
    parser.add_argument("--attempt", default="")
    options = parser.parse_args(argv)
    plan = load_plan(options.preregistration)
    cell, = [cell for cell in development_cells(plan, options.stage) if cell["task"] == options.task
             and cell["optimizer_seed"] == options.optimizer_seed and cell["method"] == options.method]
    args = cell_arguments(plan, options.run_name, cell, root=ROOT, attempt=options.attempt)
    gate = None
    if options.stage == "cost_pilot":
        if options.preflight_gate is None:
            parser.error("cost_pilot requires a qualified native preflight gate")
        gate = json.loads(options.preflight_gate.read_text())
        require_preflight_gate(plan, options.run_name, gate)
    public = ROOT / attempt_directory(options.run_name, cell, options.attempt) / "public" / "result.json"
    if args.output.exists() or public.exists():
        raise FileExistsError("preserve this measured/interrupted cell; register a separate attempt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    public.parent.mkdir(parents=True, exist_ok=True)
    protocol = prepare(args)
    args.output.write_text(json.dumps({"status": "running", "protocol": _json_ready(protocol), "records": []}) + "\n")
    ledger, audit = ComputeLedger(), None
    record = {**protocol["expected_runs"][0], "status": "failed", "accounting_complete": False,
        "configuration": {key: value for key, value in vars(args).items() if key not in ("output", "dry_run")}}
    try:
        with ledger.recording("development_worker"), single_thread_runtime() as profile:
            record["hardware_profile"] = profile
            record["accounting_complete"] = True
            if gate is not None and profile != gate["hardware_profile"]:
                raise ValueError("pilot runtime differs from the qualified preflight runtime")
            with measured_phase("cold_pipeline"):
                record = run_trial(args, options.task)
            for name, value in record["offline_cost"]["counts"].items():
                ledger.count(name, value)
            record["pipeline_offline_cost"] = record["offline_cost"]
            if record["status"] == "validation_complete" and options.method == "history":
                with measured_phase("fitting_bank_readout"):
                    labels = json.loads(Path(record["fit_cache_file"]).read_text())["branch_labels"]
                    record["fitting_bank_diagnostics"] = {"queries": len(labels),
                        "category_counts": dict(Counter(row["category"] for row in labels)),
                        "mean_waypoint_change": float(np.mean([row["renew_waypoint_delta"] for row in labels])),
                        "mean_absolute_local_advantage": float(np.mean([abs(row["renew_ise_advantage"]) for row in labels])),
                        "scope": "fitting_bank_not_held_control_evidence"}
            if record["status"] == "validation_complete" and options.stage == "preflight":
                with measured_phase("native_replay_audit"):
                    audit = native_preflight(record)
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        print(f"{options.stage} {options.task} {options.method}: {type(exc).__name__}: {exc}", flush=True)
    total = ledger.summary()
    # Outer CPU already includes the nested pipeline; only its work counters were relayed.
    record["offline_cost"] = total
    record["logical_run_id"] = next(row["run_id"] for row in expected_runs(plan, options.run_name, options.stage)
        if row["task"] == options.task and row["method"] == options.method and row["optimizer_seed"] == options.optimizer_seed)
    result = {"protocol": protocol, "stage": options.stage, "run_name": options.run_name,
        "attempt": options.attempt,
        "status": "complete" if record["status"] == "validation_complete" else "failed",
        "records": [record], "native_preflight": audit,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "performance_claim_supported": False}
    content = json.dumps(_json_ready(result), indent=2) + "\n"
    args.output.write_text(content, encoding="utf-8")
    public_record = dict(record)
    if "selected_rule" in public_record:
        rule = public_record.pop("selected_rule")
        public_record["selected_rule_summary"] = {key: value for key, value in rule.items() if key != "model"}
    public_result = {**result, "records": [public_record], "private_source_report": str(args.output)}
    public.write_text(json.dumps(_json_ready(public_result), indent=2) + "\n", encoding="utf-8")
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
