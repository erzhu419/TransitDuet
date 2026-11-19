"""Experiment B3: Real wall-clock compute and tracking Pareto with a CEM upper subgoal planner."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from freq_hrl.experiments.compute_budget import (
    ComputeLedger,
    ControlLatency,
    measured_phase,
    single_thread_runtime,
)
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import (
    PlanValidityTaskSpec,
    time_contract,
)
from freq_hrl.experiments.plan_validity_unprivileged_and_pareto import (
    BudgetScheduleSpec,
    QUANTILE_CALIBRATION_GRID,
    default_pareto_specs,
    quantile_rule_candidates,
    run_multi_budget_episode,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import (
    TimingRule,
    select_rule,
    threshold_candidates,
)
from freq_hrl.experiments.route_b_baselines import (
    CEMSubgoalControllerWrapper,
    RouteBTimingRule,
)
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.run_stage9_unprivileged_queries import RUN as UNPRIVILEGED_RUN
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_route_b_expensive_cem_planner_20261118"
PLAN = Path("protocols/plan_validity_route_b_expensive_cem_planner_2026-11-18.json")
RUNNER = "scripts/run_route_b_expensive_planner.py"
METHODS = (
    "periodic",
    "distance_threshold",
    "unprivileged_uniform",
)


def prepare(old: dict) -> dict:
    used = historical_path_seeds(old)
    specs = default_pareto_specs(1200)
    cells, jobs = [], []
    for i, c in enumerate(old["confirmation_reservation"]):
        root = int(c["optimizer_seed"])
        base = 1_982_400_000 + i * 10_000
        seeds = list(range(base, base + 32))
        if used.intersection(seeds):
            raise ValueError("CEM planner evaluation seeds overlap an existing role")
        used.update(seeds)
        public = f"results/{RUN}/{root}/public"
        cells.append({
            "optimizer_seed": root,
            "timing_selection_seeds": list(c["timing_selection_seeds"]),
            "evaluation_seeds": seeds,
            "public": public,
        })
        job = deepcopy(old["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        job.update(
            project="Freq-HRL-RouteB",
            description=f"RouteB CEM expensive upper planner root {root}",
            signature=f"Freq-HRL/routeB-cem-planner/{RUN}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public),
            local_result_dir=str(ROOT / public),
            wait_for_files=[str(ROOT / PLAN), str(ROOT / MANIFEST)],
        )
        jobs.append(job)
    return {
        "run_name": RUN,
        "source_manifest": str(MANIFEST),
        "unprivileged_run": UNPRIVILEGED_RUN,
        "cem_config": {
            "population_size": 64,
            "elites": 12,
            "cem_iters": 3,
            "horizon_steps": 25,
        },
        "budgets": [
            {
                "calls_per_episode": s.calls_per_episode,
                "bin_steps": s.bin_steps,
                "max_offset_steps": s.max_offset_steps,
                "check_stride_steps": s.check_stride_steps,
            }
            for s in specs
        ],
        "quantiles": list(QUANTILE_CALIBRATION_GRID),
        "methods": list(METHODS),
        "cells": cells,
        "jobs": jobs,
        "scope": "real_wall_clock_and_tracking_pareto_under_CEM_subgoal_planner",
        "original_confirmation_released": False,
    }


def run_cell(plan: dict, cell: dict, old: dict) -> dict:
    root = int(cell["optimizer_seed"])
    home = directory(old, root)
    hist_rec = json.loads((home / "build/private/history.json").read_text())
    dist_rec = json.loads((home / "build/private/threshold.json").read_text())
    dist_cache = json.loads(Path(dist_rec["fit_cache_file"]).read_text())["feature_rows"]
    dist_candidates = [
        r for r in threshold_candidates(dist_cache)
        if r.kind == "threshold" and r.feature == "tracking_distance"
    ]

    unpriv_path = ROOT / "results" / plan["unprivileged_run"] / str(root) / "private/unprivileged_rules.json"
    unpriv_bundle = json.loads(unpriv_path.read_text())
    unpriv_entry = unpriv_bundle["rules"]["unprivileged_uniform"]
    unpriv_base = TimingRule(**unpriv_entry["rule"])
    unpriv_oof = np.asarray(unpriv_entry["oof_scores"], dtype=np.float64)

    base_controller, args = load_controller(hist_rec)
    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    scale = time_contract(args)
    cem_cfg = plan["cem_config"]
    sel_seeds = list(cell["timing_selection_seeds"])
    eval_seeds = list(cell["evaluation_seeds"])
    quantiles = tuple(plan["quantiles"])

    by_budget: dict[str, dict] = {}
    for b_dict in plan["budgets"]:
        bspec = BudgetScheduleSpec(**b_dict)
        b_key = str(bspec.calls_per_episode)

        # Calibrate scalar thresholds on fast base controller selection paths (or CEM directly)
        sel_cache: dict[str, list[dict]] = {}

        def eval_sel(rule: TimingRule) -> list[dict]:
            if rule.name not in sel_cache:
                cem_sel = CEMSubgoalControllerWrapper(
                    base_controller,
                    maximum_subgoal_delta=args.maximum_subgoal_delta,
                    dt_seconds=scale.dt_seconds,
                    seed=root + int(b_key),
                    **cem_cfg,
                )
                sel_cache[rule.name] = [
                    run_multi_budget_episode(
                        cem_sel,
                        task_spec=spec,
                        seed=s,
                        args=args,
                        feature_time_scale=scale,
                        budget_spec=bspec,
                        rule=rule,
                    )
                    for s in sel_seeds[:4]
                ]
            return sel_cache[rule.name]

        with measured_phase(f"select_cem_B{b_key}"):
            best_dist, _ = select_rule(dist_candidates, eval_sel)
            best_unpriv, _ = select_rule(
                quantile_rule_candidates(
                    unpriv_base, unpriv_oof, prefix=f"cem_unpriv_B{b_key}", quantiles=quantiles
                ),
                eval_sel,
            )

        selected_rules = {
            "periodic": TimingRule(f"periodic_B{b_key}", "clock", threshold=0.0),
            "distance_threshold": best_dist,
            "unprivileged_uniform": best_unpriv,
        }

        eval_summary: dict[str, dict] = {}
        with measured_phase(f"eval_cem_B{b_key}"):
            for method in METHODS:
                cem_eval = CEMSubgoalControllerWrapper(
                    base_controller,
                    maximum_subgoal_delta=args.maximum_subgoal_delta,
                    dt_seconds=scale.dt_seconds,
                    seed=root + 100 + int(b_key),
                    **cem_cfg,
                )
                latency = ControlLatency()
                ep_rows = [
                    run_multi_budget_episode(
                        cem_eval,
                        task_spec=spec,
                        seed=s,
                        args=args,
                        feature_time_scale=scale,
                        budget_spec=bspec,
                        rule=selected_rules[method],
                        latency_recorder=latency,
                    )
                    for s in eval_seeds
                ]
                ises = np.asarray([r[ISE] for r in ep_rows], dtype=np.float64)
                rets = np.asarray([r["episode_return"] for r in ep_rows], dtype=np.float64)
                planner_ms = np.asarray([r["planner_wall_seconds"] * 1e3 for r in ep_rows])
                lower_ms = np.asarray([r["lower_wall_seconds"] * 1e3 for r in ep_rows])
                score_ms = np.asarray([r["trigger_score_wall_seconds"] * 1e3 for r in ep_rows])
                total_ctrl_ms = planner_ms + lower_ms + score_ms
                eval_summary[method] = {
                    "mean_ise": float(ises.mean()),
                    "mean_return": float(rets.mean()),
                    "planner_ms_per_episode": float(planner_ms.mean()),
                    "lower_ms_per_episode": float(lower_ms.mean()),
                    "trigger_score_ms_per_episode": float(score_ms.mean()),
                    "total_control_ms_per_episode": float(total_ctrl_ms.mean()),
                    "mean_score_checks": float(
                        np.mean([r["trigger_score_checks"] for r in ep_rows])
                    ),
                    "latency": latency.summary(episodes=len(ep_rows), deadline_seconds=0.01),
                }
        by_budget[b_key] = {"budget": b_dict, "methods": eval_summary}

    return {"evaluation_seeds": eval_seeds, "budgets": by_budget}


def readout(plan: dict, records: list[dict]) -> dict:
    if len(records) != len(plan["cells"]) or any(r["status"] != "complete" for r in records):
        raise ValueError("CEM readout requires all eight completed roots")

    budget_keys = [str(b["calls_per_episode"]) for b in plan["budgets"]]
    by_budget: dict[str, dict] = {}
    for b_key in budget_keys:
        ise_means = {
            m: np.asarray([r["budgets"][b_key]["methods"][m]["mean_ise"] for r in records])
            for m in METHODS
        }
        ms_means = {
            m: np.asarray(
                [r["budgets"][b_key]["methods"][m]["total_control_ms_per_episode"] for r in records]
            )
            for m in METHODS
        }
        planner_ms = {
            m: float(
                np.mean([r["budgets"][b_key]["methods"][m]["planner_ms_per_episode"] for r in records])
            )
            for m in METHODS
        }
        score_ms = {
            m: float(
                np.mean([
                    r["budgets"][b_key]["methods"][m]["trigger_score_ms_per_episode"]
                    for r in records
                ])
            )
            for m in METHODS
        }
        contrasts = {}
        for ref in ("periodic", "distance_threshold"):
            diff = ise_means[ref] - ise_means["unprivileged_uniform"]
            contrasts[f"unprivileged_vs_{ref}"] = {
                "paired_ise_gain": paired_mean_interval(diff),
                "relative_ise_gain": float(diff.mean() / ise_means[ref].mean()),
                "favorable_roots": int(np.sum(diff > 0.0)),
            }
        by_budget[b_key] = {
            "calls_per_episode": int(b_key),
            "mean_ise": {m: float(ise_means[m].mean()) for m in METHODS},
            "total_control_ms_per_episode": {m: float(ms_means[m].mean()) for m in METHODS},
            "planner_ms_per_episode": planner_ms,
            "trigger_score_ms_per_episode": score_ms,
            "contrasts": contrasts,
        }

    # Joint accuracy + real wall-clock savings vs Periodic(B=24)
    p24_ise = np.asarray([r["budgets"]["24"]["methods"]["periodic"]["mean_ise"] for r in records])
    p24_ms = np.asarray(
        [r["budgets"]["24"]["methods"]["periodic"]["total_control_ms_per_episode"] for r in records]
    )
    joint_savings = {}
    for b_key in ("12", "15", "20", "24"):
        u_ise = np.asarray(
            [r["budgets"][b_key]["methods"]["unprivileged_uniform"]["mean_ise"] for r in records]
        )
        u_ms = np.asarray([
            r["budgets"][b_key]["methods"]["unprivileged_uniform"]["total_control_ms_per_episode"]
            for r in records
        ])
        joint_savings[f"unprivileged_B{b_key}_vs_periodic_B24"] = {
            "paired_ise_gain": paired_mean_interval(p24_ise - u_ise),
            "relative_ise_gain": float((p24_ise - u_ise).mean() / p24_ise.mean()),
            "paired_wall_ms_saved": paired_mean_interval(p24_ms - u_ms),
            "relative_wall_time_saved": float((p24_ms - u_ms).mean() / p24_ms.mean()),
        }

    return {
        "run_name": plan["run_name"],
        "scope": plan["scope"],
        "root_count": len(records),
        "frontier": by_budget,
        "joint_accuracy_and_wall_clock_vs_periodic24": joint_savings,
        "roots": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "readout"))
    parser.add_argument("--cell", type=int)
    opts = parser.parse_args()
    old = json.loads((ROOT / MANIFEST).read_text())
    if opts.phase == "prepare":
        emit(ROOT / PLAN, prepare(old))
        return 0
    plan = json.loads((ROOT / PLAN).read_text())
    if opts.phase == "readout":
        records = [
            json.loads((ROOT / c["public"] / "result.json").read_text())
            for c in plan["cells"]
        ]
        summary = readout(plan, records)
        summary_path = ROOT / "results" / RUN / "summary.json"
        if not summary_path.exists():
            emit(summary_path, summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "roots"}, indent=2))
        return 0

    cell = plan["cells"][opts.cell]
    public_dir = ROOT / cell["public"]
    public_dir.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile, ledger.recording("cem_expensive_planner"):
            if profile != old["hardware_profile"]:
                raise ValueError("runtime differs from original frozen builds")
            result = {**run_cell(plan, cell, old), "status": "complete"}
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(
        public_dir / "result.json",
        {
            **result,
            "optimizer_seed": cell["optimizer_seed"],
            "measured_cost": ledger.summary(),
        },
    )
    print(cell["optimizer_seed"], result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
