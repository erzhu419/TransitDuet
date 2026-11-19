"""Evaluate multi-budget planning Pareto curves on frozen PointMaze controllers."""

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
    out_of_fold_legacy_scores,
    quantile_rule_candidates,
    run_multi_budget_episode,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import (
    TimingRule,
    select_rule,
    threshold_candidates,
)
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.run_stage9_unprivileged_queries import RUN as UNPRIVILEGED_RUN
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_stage9_budget_pareto_20261118"
PLAN = Path("protocols/plan_validity_stage9_budget_pareto_2026-11-18.json")
RUNNER = "scripts/run_stage9_budget_pareto.py"
METHODS = (
    "periodic",
    "best_clock",
    "distance_threshold",
    "unprivileged_uniform",
    "privileged_stage9",
)


def prepare(old: dict) -> dict:
    used = historical_path_seeds(old)
    specs = default_pareto_specs(1200)
    cells, jobs = [], []
    for i, c in enumerate(old["confirmation_reservation"]):
        root = int(c["optimizer_seed"])
        base = 1_982_200_000 + i * 10_000
        seeds = list(range(base, base + 64))
        if used.intersection(seeds):
            raise ValueError("pareto evaluation seeds overlap an existing role")
        used.update(seeds)
        public = f"results/{RUN}/{root}/public"
        unpriv_public = ROOT / "results" / UNPRIVILEGED_RUN / str(root) / "public/result.json"
        cells.append({
            "optimizer_seed": root,
            "timing_selection_seeds": list(c["timing_selection_seeds"]),
            "evaluation_seeds": seeds,
            "public": public,
        })
        job = deepcopy(old["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        job.update(
            project="Freq-HRL-RouteA",
            description=f"Stage9 multi-budget Pareto root {root}",
            signature=f"Freq-HRL/routeA-pareto/{RUN}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public),
            local_result_dir=str(ROOT / public),
            wait_for_files=[
                str(ROOT / PLAN),
                str(ROOT / MANIFEST),
            ],
        )
        jobs.append(job)
    return {
        "run_name": RUN,
        "source_manifest": str(MANIFEST),
        "unprivileged_run": UNPRIVILEGED_RUN,
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
        "scope": "multi_budget_planner_call_pareto_frontier_on_shared_frozen_PointMaze_controllers",
        "original_confirmation_released": False,
    }


def run_cell(plan: dict, cell: dict, old: dict) -> dict:
    root = int(cell["optimizer_seed"])
    home = directory(old, root)
    if (home / "evaluate").exists():
        raise ValueError("original confirmation test directory must remain untouched")
    hist_rec = json.loads((home / "build/private/history.json").read_text())
    dist_rec = json.loads((home / "build/private/threshold.json").read_text())
    hist_cache = json.loads(Path(hist_rec["fit_cache_file"]).read_text())["branch_labels"]
    dist_cache = json.loads(Path(dist_rec["fit_cache_file"]).read_text())["feature_rows"]

    unpriv_path = ROOT / "results" / plan["unprivileged_run"] / str(root) / "private/unprivileged_rules.json"
    unpriv_bundle = json.loads(unpriv_path.read_text())
    if unpriv_bundle["optimizer_seed"] != root or unpriv_bundle["evaluation_accessed_at_freeze"]:
        raise ValueError("unprivileged rule bundle failed freeze verification")
    unpriv_entry = unpriv_bundle["rules"]["unprivileged_uniform"]
    unpriv_base = TimingRule(**unpriv_entry["rule"])
    unpriv_oof = np.asarray(unpriv_entry["oof_scores"], dtype=np.float64)

    stage9_base = TimingRule(**hist_rec["selected_rule"])
    stage9_oof = out_of_fold_legacy_scores(hist_cache, stage9_base.model)

    dist_candidates = [
        r for r in threshold_candidates(dist_cache)
        if r.kind == "threshold" and r.feature == "tracking_distance"
    ]

    controller, args = load_controller(hist_rec)
    eval_seeds = list(cell["evaluation_seeds"])
    sel_seeds = list(cell["timing_selection_seeds"])
    forbidden = set(
        args.train_seeds
        + args.selection_seeds
        + args.branch_fit_seeds
        + args.timing_selection_seeds
        + args.trigger_eval_seeds
    )
    if forbidden.intersection(eval_seeds):
        raise ValueError("pareto evaluation overlaps source roles")

    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    scale = time_contract(args)
    quantiles = tuple(plan["quantiles"])

    by_budget: dict[str, dict] = {}
    for b_dict in plan["budgets"]:
        bspec = BudgetScheduleSpec(**b_dict)
        b_key = str(bspec.calls_per_episode)
        sel_cache: dict[str, list[dict]] = {}

        def evaluate_selection(rule: TimingRule) -> list[dict]:
            if rule.name not in sel_cache:
                sel_cache[rule.name] = [
                    run_multi_budget_episode(
                        controller,
                        task_spec=spec,
                        seed=s,
                        args=args,
                        feature_time_scale=scale,
                        budget_spec=bspec,
                        rule=rule,
                    )
                    for s in sel_seeds
                ]
            return sel_cache[rule.name]

        with measured_phase(f"select_B{b_key}"):
            clock_grid = [
                TimingRule(
                    f"clock_B{b_key}_{off}",
                    "clock",
                    threshold=off / bspec.max_offset_steps,
                )
                for off in range(0, bspec.max_offset_steps + 1, bspec.check_stride_steps)
            ]
            best_clock, _ = select_rule(clock_grid, evaluate_selection)
            best_dist, _ = select_rule(dist_candidates, evaluate_selection)
            best_unpriv, _ = select_rule(
                quantile_rule_candidates(
                    unpriv_base, unpriv_oof, prefix=f"unpriv_B{b_key}", quantiles=quantiles
                ),
                evaluate_selection,
            )
            best_stage9, _ = select_rule(
                quantile_rule_candidates(
                    stage9_base, stage9_oof, prefix=f"stage9_B{b_key}", quantiles=quantiles
                ),
                evaluate_selection,
            )

        selected_rules = {
            "periodic": TimingRule(f"periodic_B{b_key}", "clock", threshold=0.0),
            "best_clock": best_clock,
            "distance_threshold": best_dist,
            "unprivileged_uniform": best_unpriv,
            "privileged_stage9": best_stage9,
        }

        eval_summary: dict[str, dict] = {}
        with measured_phase(f"eval_B{b_key}"):
            for method in METHODS:
                latency = ControlLatency()
                ep_rows = [
                    run_multi_budget_episode(
                        controller,
                        task_spec=spec,
                        seed=s,
                        args=args,
                        feature_time_scale=scale,
                        budget_spec=bspec,
                        rule=selected_rules[method],
                        latency_recorder=latency,
                        risk_spec={"distance": 1.0, "duration_steps": 25},
                    )
                    for s in eval_seeds
                ]
                ises = np.asarray([r[ISE] for r in ep_rows], dtype=np.float64)
                rets = np.asarray([r["episode_return"] for r in ep_rows], dtype=np.float64)
                first_half, last_half = np.split(ises, 2)
                eval_summary[method] = {
                    "selected_rule_name": selected_rules[method].name,
                    "selected_threshold": float(selected_rules[method].threshold),
                    "mean_ise": float(ises.mean()),
                    "first_half_ise": float(first_half.mean()),
                    "last_half_ise": float(last_half.mean()),
                    "mean_return": float(rets.mean()),
                    "mean_score_checks": float(
                        np.mean([r["trigger_score_checks"] for r in ep_rows])
                    ),
                    "mean_early_calls": float(
                        np.mean([r["trigger_early_calls"] for r in ep_rows])
                    ),
                    "sustained_failures": int(
                        sum(r["reliability"]["sustained_failure"] for r in ep_rows)
                    ),
                    "latency": latency.summary(episodes=len(ep_rows), deadline_seconds=0.01),
                }
        by_budget[b_key] = {
            "budget": b_dict,
            "methods": eval_summary,
        }

    return {
        "evaluation_seeds": eval_seeds,
        "budgets": by_budget,
    }


def readout(plan: dict, records: list[dict]) -> dict:
    if len(records) != len(plan["cells"]) or any(
        r["status"] != "complete"
        or r["optimizer_seed"] != c["optimizer_seed"]
        or r["evaluation_seeds"] != c["evaluation_seeds"]
        for c, r in zip(plan["cells"], records)
    ):
        raise ValueError("pareto readout requires all eight completed roots")

    budget_keys = [str(b["calls_per_episode"]) for b in plan["budgets"]]
    frontier: dict[str, dict] = {}
    for b_key in budget_keys:
        means = {
            m: np.asarray(
                [r["budgets"][b_key]["methods"][m]["mean_ise"] for r in records],
                dtype=np.float64,
            )
            for m in METHODS
        }
        returns = {
            m: np.asarray(
                [r["budgets"][b_key]["methods"][m]["mean_return"] for r in records],
                dtype=np.float64,
            )
            for m in METHODS
        }
        contrasts: dict[str, dict] = {}
        for cand in ("unprivileged_uniform", "privileged_stage9"):
            for ref in ("periodic", "best_clock", "distance_threshold"):
                diff = means[ref] - means[cand]
                ci = paired_mean_interval(diff)
                contrasts[f"{cand}_vs_{ref}"] = {
                    "paired_ise_gain": ci,
                    "relative_ise_gain": float(diff.mean() / means[ref].mean()),
                    "favorable_roots": int(np.sum(diff > 0.0)),
                }
        frontier[b_key] = {
            "calls_per_episode": int(b_key),
            "mean_ise": {m: float(means[m].mean()) for m in METHODS},
            "mean_return": {m: float(returns[m].mean()) for m in METHODS},
            "mean_score_checks": {
                m: float(
                    np.mean([
                        r["budgets"][b_key]["methods"][m]["mean_score_checks"]
                        for r in records
                    ])
                )
                for m in METHODS
            },
            "contrasts": contrasts,
        }

    # Cross-budget efficiency comparisons against Periodic at B=24 calls
    periodic_24 = np.asarray(
        [r["budgets"]["24"]["methods"]["periodic"]["mean_ise"] for r in records],
        dtype=np.float64,
    )
    cross_budget_vs_periodic24: dict[str, dict] = {}
    for b_key in ("12", "15", "20", "24"):
        for cand in ("unprivileged_uniform", "privileged_stage9", "distance_threshold"):
            cand_ise = np.asarray(
                [r["budgets"][b_key]["methods"][cand]["mean_ise"] for r in records],
                dtype=np.float64,
            )
            diff = periodic_24 - cand_ise
            cross_budget_vs_periodic24[f"{cand}_B{b_key}_vs_periodic_B24"] = {
                "paired_ise_gain": paired_mean_interval(diff),
                "relative_ise_gain": float(diff.mean() / periodic_24.mean()),
                "call_reduction_pct": float(1.0 - int(b_key) / 24.0),
                "favorable_roots": int(np.sum(diff > 0.0)),
            }

    return {
        "run_name": plan["run_name"],
        "scope": plan["scope"],
        "root_count": len(records),
        "evaluation_paths_per_root": 64,
        "total_paired_paths": 512,
        "frontier": frontier,
        "cross_budget_vs_periodic24": cross_budget_vs_periodic24,
        "roots": records,
        "original_confirmation_released": False,
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
        with single_thread_runtime() as profile, ledger.recording("budget_pareto"):
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
