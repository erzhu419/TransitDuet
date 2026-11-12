"""Preflight and audit a shared-controller Stage9 strong-timing comparison."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from freq_hrl.experiments.compute_budget import ComputeLedger, ControlLatency, count_work, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_budgeted_trigger import rollout_budgeted_trigger
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule, select_rule, termination_readiness
from scripts.run_plan_validity_reliability import load_controller, load_record


METHODS = ("history", "threshold", "termination_history", "best_clock")
ISE = "tracking_squared_error_integral"


def same_controller(left, right):
    a, b = left.state_dict(), right.state_dict()
    return a["config"] == b["config"] and all(
        a[m].keys() == b[m].keys() and all(torch.equal(a[m][k], b[m][k]) for k in a[m])
        for m in ("upper_actor", "lower_actor", "upper_value", "lower_value"))


def load_development(cell, design):
    source = {}
    controller = args = None
    for method in METHODS[:3]:
        ref = cell["sources"][method]
        report = load_record(Path(ref["report"]), ref["run_id"])
        if (report["task"], report["method"], report["optimizer_seed"], report["configuration"],
            report["hardware_profile"], report["accounting_complete"]) != (
            "pointmaze", method, cell["optimizer_seed"], {**cell["expected_configuration"], "method": method},
            design["hardware_profile"], True):
            raise ValueError("pilot source identity, configuration or accounting differs")
        model, options = load_controller(report)
        if controller is None:
            controller, args = model, options
        elif not same_controller(controller, model):
            raise ValueError("strong timing rules must share identical controller weights")
        rule = TimingRule(**report["selected_rule"])
        if {k: v for k, v in rule.to_dict().items() if k != "model"} != ref["selected_rule_summary"]:
            raise ValueError("pilot rule differs from the registered frozen artifact")
        source[method] = report
    return controller, args, source, {m: TimingRule(**r["selected_rule"]) for m, r in source.items()}


def preflight(cell, controller, args, source, rules, evaluate):
    cached = {}

    def selection(rule):
        if rule.name not in cached:
            cached[rule.name] = [evaluate(rule, seed) for seed in args.timing_selection_seeds]
        return cached[rule.name]

    with measured_phase("frozen_selection_reproduction"):
        best_clock, clock_table = select_rule(
            [TimingRule(f"clock_{i}", "clock", threshold=i / 25) for i in range(0, 26, 5)], selection)
        clock_rows = selection(best_clock)
        select_rows = {method: selection(rule) for method, rule in rules.items()}
        for method, rows in select_rows.items():
            if not np.isclose(np.mean([r[ISE] for r in rows]), source[method]["validation_ise"], rtol=1e-10, atol=1e-10):
                raise ValueError(f"selection ISE did not reproduce: {method}")
        readiness = termination_readiness(select_rows["termination_history"], clock_rows)
        old = source["termination_history"]["termination_readiness"]
        if any(readiness[k] != old[k] for k in (
            "behaviorally_noncollapsed", "state_conditional_at_same_bin_and_offset", "eligible_for_fresh_confirmation"
        )) or not np.isclose(readiness["selection_ise_gain_over_best_clock"], old["selection_ise_gain_over_best_clock"],
                              rtol=1e-10, atol=1e-10):
            raise ValueError("original learned termination competence did not reproduce")
    with measured_phase("original_stage9_native_parity"):
        rule = rules["history"]
        if rule.kind != "legacy" or rule.view != "causal_validity_interactions":
            raise ValueError("candidate is not the original Stage9 history trigger")
        seed = cell["preflight_seed"]
        row = evaluate(rule, seed)
        legacy = rollout_budgeted_trigger(controller, seed=seed, mode=rule.view,
            fitted_predictors={rule.view: rule.model}, env_id=args.env_id, horizon=args.horizon,
            time_scale=time_contract(args), maximum_subgoal_delta=args.maximum_subgoal_delta,
            max_offset_steps=args.max_offset_steps, check_stride_steps=args.check_stride_steps,
            task_options=_task_options(args))
        count_work("environment_steps", args.horizon)
        count_work("planner_calls", legacy["upper_decision_count"])
        if (row["decision_steps"] != legacy["decision_steps"]
            or row["trigger_score_checks"] != legacy["trigger_score_checks"]
            or any(not np.isclose(row[k], legacy[k], rtol=1e-10, atol=1e-10) for k in (ISE, "episode_return"))):
            raise ValueError("shared timing runner changed the original Stage9 episode")
    return {"best_clock": best_clock.to_dict(), "clock_selection": clock_table,
            "termination_selection_readiness": readiness, "legacy_parity": True,
            "selection_roster": list(args.timing_selection_seeds), "audit_accessed": False}


def panels(rows, *, methods=METHODS):
    size = len(rows[methods[0]])
    seed_sets = [[r["seed"] for r in rows[m]] for m in methods]
    if size < 2 or size % 2 or len(set(seed_sets[0])) != size or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError("evaluation panels need the same unique ordered paired paths")
    output = {}
    for label, indices in (("all", slice(None)), ("first", slice(size // 2)), ("last", slice(size // 2, None))):
        output[label] = {}
        for method in methods:
            sample = rows[method][indices]
            ise = np.array([row[ISE] for row in sample])
            tail = np.sort(ise)[-max(1, int(np.ceil(.10 * len(ise)))):]
            output[label][method] = {"mean_ise": float(ise.mean()), "tail_ise": float(tail.mean()),
                "mean_return": float(np.mean([r["episode_return"] for r in sample])),
                "mean_score_checks": float(np.mean([r["trigger_score_checks"] for r in sample])),
                "mean_early_calls": float(np.mean([sum(t % 50 < 25 for t in r["decision_steps"][1:]) for r in sample])),
                "calls": len(sample[0]["decision_steps"]), "paths": len(sample)}
    return output


def audit(cell, design, args, source, rules, evaluate, ledger, preflight_directory):
    all_preflights = [json.loads((ROOT / c["directory"] / preflight_directory / "public/result.json").read_text())
                      for c in design["development_cells"]]
    if any((r["status"], r["phase"], r["optimizer_seed"], r["legacy_parity"], r["audit_accessed"]) !=
           ("complete", "preflight", c["optimizer_seed"], True, False)
           for r, c in zip(all_preflights, design["development_cells"])):
        raise ValueError("all original Stage9 preflights must pass before new audit paths")
    report, = [r for r in all_preflights if r["optimizer_seed"] == cell["optimizer_seed"]]
    rules["best_clock"] = TimingRule(**report["best_clock"])
    used = set().union(args.train_seeds, args.selection_seeds, args.branch_fit_seeds,
                       args.timing_selection_seeds, args.trigger_eval_seeds, [cell["preflight_seed"]])
    seeds = cell["audit_seeds"]
    if used.intersection(seeds):
        raise ValueError("audit paths were accessed by development fitting or selection")
    rows, budgets = {}, {}
    for method in METHODS:
        latency = ControlLatency()
        with measured_phase(f"audit_{method}"):
            rows[method] = [evaluate(rules[method], seed, latency=latency) for seed in seeds]
        summary = latency.summary(episodes=len(seeds), deadline_seconds=.01)
        if method == "best_clock":
            phases = source["history"]["pipeline_offline_cost"]["phases"]
            shared = [v for k, v in phases.items() if k == "offline/controller" or k.startswith("offline/controller/")]
            selection = report["measured_cost"]["phases"]["preflight/frozen_selection_reproduction"]
            # Conservative: charge the clock all preflight selection replays, including reproduction.
            cpu = sum(v["cpu_seconds"] for v in shared) + sum(
                v["cpu_seconds"] for k, v in report["measured_cost"]["phases"].items()
                if k.startswith("preflight/frozen_selection_reproduction"))
            steps = sum(v["counts"].get("environment_steps", 0) for v in shared) + selection["counts"]["environment_steps"]
        else:
            cost = source[method]["offline_cost"]
            cpu, steps = cost["cpu_seconds"], cost["counts"]["environment_steps"]
        cap = design["budget"]
        lifecycle = cpu + cap["deployments"] * summary["cpu_seconds_per_episode"]
        budgets[method] = {"offline_cpu_seconds": cpu, "offline_steps": steps, "lifecycle_cpu_seconds": lifecycle,
            "latency": summary, "admitted": lifecycle <= cap["cpu_seconds"] and steps <= cap["offline_steps"]
                and summary["wall_p99_seconds"] <= cap["p99_seconds"]}
    expected = len(METHODS) * len(seeds) * args.horizon
    if (ledger.summary()["counts"].get("environment_steps") != expected
        or any(r["upper_decision_count"] != 24 for rs in rows.values() for r in rs)):
        raise RuntimeError("development audit did not close its paired episode budget")
    return {"audit_accessed": True, "audit_seeds": seeds, "panels": panels(rows), "budgets": budgets,
            "termination_selection_readiness": report["termination_selection_readiness"],
            "termination_audit_readiness": termination_readiness(rows["termination_history"], rows["best_clock"])}


def readout(design, reports):
    if len(reports) != len(design["development_cells"]) or any(
        (r["status"], r["phase"], r["optimizer_seed"], r["audit_seeds"]) !=
        ("complete", "audit", c["optimizer_seed"], c["audit_seeds"])
        for c, r in zip(design["development_cells"], reports)):
        raise ValueError("all registered audit episodes are required")
    checks = {}
    for comparator in METHODS[1:]:
        def gain(panel):
            a = np.mean([r["panels"][panel]["history"]["mean_ise"] for r in reports])
            b = np.mean([r["panels"][panel][comparator]["mean_ise"] for r in reports])
            return float((b - a) / b)

        roots = [float(1 - r["panels"]["all"]["history"]["mean_ise"] /
                       r["panels"]["all"][comparator]["mean_ise"]) for r in reports]
        pooled, halves = gain("all"), [gain(p) for p in ("first", "last")]
        checks[comparator] = {"relative_gain": pooled, "halves": halves, "root_gains": roots,
            "passed": pooled >= design["minimum_ise_gain"] and min(halves + roots) > 0}
    budget_ok = all(all(v["admitted"] for v in r["budgets"].values()) for r in reports)
    termination_ok = all(r[k]["eligible_for_fresh_confirmation"] for r in reports
        for k in ("termination_selection_readiness", "termination_audit_readiness"))
    return {"comparisons": checks, "budget_passed": budget_ok, "termination_eligible": termination_ok,
            "confirmation_builds_allowed": budget_ok and termination_ok and all(v["passed"] for v in checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "audit", "readout"), required=True)
    parser.add_argument("--cell", type=int)
    parser.add_argument("--preflight-directory", default="preflight")
    opts = parser.parse_args()
    design = json.loads(opts.manifest.read_text())
    if opts.phase == "readout":
        reports = [json.loads((ROOT / c["directory"] / "audit/public/result.json").read_text())
                   for c in design["development_cells"]]
        print(json.dumps({**readout(design, reports), "roots": reports}, separators=(",", ":")))
        return 0
    cell = design["development_cells"][opts.cell]
    phase_directory = opts.preflight_directory if opts.phase == "preflight" else opts.phase
    directory = ROOT / cell["directory"] / phase_directory / "public"
    directory.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    result = {"phase": opts.phase, "optimizer_seed": cell["optimizer_seed"], "status": "running"}
    print(f"{opts.phase}: {cell['optimizer_seed']} loading frozen timing pipelines", flush=True)
    try:
        with ledger.recording(opts.phase), single_thread_runtime() as profile:
            if profile != design["hardware_profile"]:
                raise ValueError("runtime differs from the charged source profile")
            with measured_phase("load_frozen_artifacts"):
                controller, args, source, rules = load_development(cell, design)
            spec, scale = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id), time_contract(args)

            def evaluate(rule, seed, latency=None):
                return run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale,
                                   rule=rule, latency_recorder=latency)[1]

            if opts.phase == "preflight":
                result.update(preflight(cell, controller, args, source, rules, evaluate))
            else:
                result.update(audit(cell, design, args, source, rules, evaluate, ledger, opts.preflight_directory))
        if ledger.summary()["counts"]["environment_steps"] != design[f"{opts.phase}_steps_per_root"]:
            raise RuntimeError("worker did not close its registered native-step budget")
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(result["error"], flush=True)
    result["measured_cost"] = ledger.summary()
    (directory / "result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(f"{opts.phase}: {cell['optimizer_seed']} {result['status']}", flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
