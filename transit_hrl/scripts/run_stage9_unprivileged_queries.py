"""Evaluate label-free (unprivileged) renewal-query acquisition on PointMaze."""

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
    run_episode,
    time_contract,
)
from freq_hrl.experiments.plan_validity_unprivileged_and_pareto import (
    collect_unprivileged_renewal_labels,
    out_of_fold_legacy_scores,
)
from freq_hrl.experiments.pointmaze_budgeted_trigger import (
    fit_budgeted_trigger_predictors,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE, panels
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_stage9_unprivileged_queries_20261118"
PLAN = Path("protocols/plan_validity_stage9_unprivileged_queries_2026-11-18.json")
RUNNER = "scripts/run_stage9_unprivileged_queries.py"
METHODS = (
    "periodic",
    "jitter",
    "distance_threshold",
    "unprivileged_uniform",
    "unprivileged_observable",
    "privileged_stage9",
)
CANDIDATES = ("unprivileged_uniform", "unprivileged_observable")
REFERENCES = ("periodic", "jitter", "distance_threshold", "privileged_stage9")


def prepare(old: dict) -> dict:
    used = historical_path_seeds(old)
    cells, jobs = [], []
    for i, c in enumerate(old["confirmation_reservation"]):
        root = int(c["optimizer_seed"])
        base = 1_982_100_000 + i * 10_000
        seeds = list(range(base, base + 64))
        if used.intersection(seeds):
            raise ValueError("unprivileged evaluation seeds overlap an existing role")
        used.update(seeds)
        public = f"results/{RUN}/{root}/public"
        cells.append({
            "optimizer_seed": root,
            "branch_fit_seeds": list(c["branch_fit_seeds"]),
            "evaluation_seeds": seeds,
            "public": public,
        })
        job = deepcopy(old["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        job.update(
            project="Freq-HRL-RouteA",
            description=f"Stage9 unprivileged query ablation root {root}",
            signature=f"Freq-HRL/routeA-unprivileged/{RUN}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public),
            local_result_dir=str(ROOT / public),
            wait_for_files=[str(ROOT / PLAN), str(ROOT / MANIFEST)],
        )
        jobs.append(job)
    return {
        "run_name": RUN,
        "source_manifest": str(MANIFEST),
        "cells": cells,
        "jobs": jobs,
        "queries_per_path": 24,
        "labels_per_root": 192,
        "calls_per_episode": 24,
        "evaluation_episodes_per_root": len(METHODS) * 64,
        "evaluation_steps_per_root": len(METHODS) * 64 * 1200,
        "methods": list(METHODS),
        "minimum_relative_gain": 0.02,
        "scope": "label_free_unprivileged_branch_query_acquisition_on_shared_frozen_PointMaze_controllers",
        "original_confirmation_released": False,
    }


def load_frozen_sources(old: dict, root: int) -> tuple[dict, dict]:
    home = directory(old, root)
    if (home / "evaluate").exists():
        raise ValueError("original confirmation test directory must remain untouched")
    hist = json.loads((home / "build/private/history.json").read_text())
    dist = json.loads((home / "build/private/threshold.json").read_text())
    for name, rec in (("history", hist), ("threshold", dist)):
        if (
            rec["optimizer_seed"] != root
            or rec["method"] != name
            or rec["status"] != "validation_complete"
            or not Path(rec["controller_checkpoint"]).is_file()
        ):
            raise ValueError(f"frozen source build invalid for {name} on {root}")
    if hist["controller_checkpoint"] != dist["controller_checkpoint"]:
        raise ValueError("frozen sources do not share the same controller checkpoint")
    return hist, dist


def run_cell(plan: dict, cell: dict, old: dict, workdir: Path, ledger: ComputeLedger) -> dict:
    root = int(cell["optimizer_seed"])
    hist_rec, dist_rec = load_frozen_sources(old, root)
    controller, args = load_controller(hist_rec)
    seeds = list(cell["evaluation_seeds"])
    forbidden = set(
        args.train_seeds
        + args.selection_seeds
        + args.branch_fit_seeds
        + args.timing_selection_seeds
        + args.trigger_eval_seeds
    )
    if forbidden.intersection(seeds):
        raise ValueError("unprivileged evaluation overlaps training/selection/reserved seeds")

    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    scale = time_contract(args)
    predictor_name = "causal_validity_interactions"

    fitted_rules: dict[str, TimingRule] = {
        "periodic": TimingRule("periodic", "clock", threshold=0.0),
        "jitter": TimingRule("jitter", "jitter"),
        "distance_threshold": TimingRule(**dist_rec["selected_rule"]),
        "privileged_stage9": TimingRule(**hist_rec["selected_rule"]),
    }
    fit_summaries: dict[str, dict] = {}
    private_bundle: dict[str, dict] = {}

    for mode, acq in (
        ("unprivileged_uniform", "uniform"),
        ("unprivileged_observable", "observable"),
    ):
        with measured_phase(f"branch_supervision_{acq}"):
            labels = collect_unprivileged_renewal_labels(
                controller,
                acquisition=acq,
                task_spec=spec,
                args=args,
                time_scale=scale,
                queries_per_path=int(plan["queries_per_path"]),
            )
        if (
            len(labels) != int(plan["labels_per_root"])
            or any(r["query_selection_uses_privileged_events"] for r in labels)
            or any(r["candidate_feature_has_future_access"] for r in labels)
            or any(r["candidate_feature_has_regime_label"] for r in labels)
        ):
            raise RuntimeError(f"unprivileged label invariant failed for {mode}")
        with measured_phase(f"ridge_fit_{acq}"):
            fitted = fit_budgeted_trigger_predictors(
                labels,
                alpha_grid=args.ridge_alpha_grid,
                threshold_quantile=args.threshold_quantile,
                predictors=(predictor_name,),
            )[predictor_name]
            oof = out_of_fold_legacy_scores(labels, fitted)
        rule = TimingRule(
            mode,
            "legacy",
            view=predictor_name,
            threshold=float(fitted["threshold"]),
            model=fitted,
        )
        fitted_rules[mode] = rule
        fit_summaries[mode] = {
            "label_count": len(labels),
            "alpha": float(fitted["model"]["alpha"]),
            "threshold": float(fitted["threshold"]),
            "oof_mse": float(
                fitted["model"]["cross_validation_mse"][str(float(fitted["model"]["alpha"]))]
            ),
            "positive_label_fraction": float(
                np.mean([r["renew_ise_advantage"] > 0.0 for r in labels])
            ),
            "mean_label_advantage": float(
                np.mean([r["renew_ise_advantage"] for r in labels])
            ),
        }
        private_bundle[mode] = {
            "rule": rule.to_dict(),
            "oof_scores": oof.tolist(),
            "summary": fit_summaries[mode],
        }

    # Freeze unprivileged rules on disk BEFORE accessing any evaluation path.
    private_dir = workdir / "private"
    private_dir.mkdir(parents=True, exist_ok=False)
    emit(
        private_dir / "unprivileged_rules.json",
        {
            "optimizer_seed": root,
            "evaluation_accessed_at_freeze": False,
            "rules": private_bundle,
        },
    )

    rows: dict[str, list[dict]] = {}
    latencies: dict[str, dict] = {}
    early_calls: dict[str, float] = {}
    score_checks: dict[str, float] = {}
    return_means: dict[str, float] = {}
    for method in METHODS:
        latency = ControlLatency()
        with measured_phase(f"eval_{method}"):
            rows[method] = [
                run_episode(
                    controller,
                    task_spec=spec,
                    seed=s,
                    args=args,
                    time_scale=scale,
                    rule=fitted_rules[method],
                    latency_recorder=latency,
                    risk_spec={"distance": 1.0, "duration_steps": 25},
                )[1]
                for s in seeds
            ]
        latencies[method] = latency.summary(episodes=len(seeds), deadline_seconds=0.01)
        early_calls[method] = float(
            np.mean([
                sum(1 for step in r["decision_steps"][1:] if (step % 50) < 25)
                for r in rows[method]
            ])
        )
        score_checks[method] = float(
            np.mean([r["trigger_score_checks"] for r in rows[method]])
        )
        return_means[method] = float(
            np.mean([r["episode_return"] for r in rows[method]])
        )

    return {
        "evaluation_seeds": seeds,
        "fit_summaries": fit_summaries,
        "panels": panels(rows, methods=METHODS),
        "mean_return": return_means,
        "mean_early_calls": early_calls,
        "mean_score_checks": score_checks,
        "sustained_failure_counts": {
            m: int(sum(r["reliability"]["sustained_failure"] for r in rs))
            for m, rs in rows.items()
        },
        "latency": latencies,
    }


def readout(plan: dict, records: list[dict]) -> dict:
    if len(records) != len(plan["cells"]) or any(
        r["status"] != "complete"
        or r["optimizer_seed"] != c["optimizer_seed"]
        or r["evaluation_seeds"] != c["evaluation_seeds"]
        for c, r in zip(plan["cells"], records)
    ):
        raise ValueError("unprivileged readout requires all eight completed roots")

    def root_ise(panel: str, method: str) -> np.ndarray:
        return np.asarray([r["panels"][panel][method]["mean_ise"] for r in records], dtype=np.float64)

    def root_ret(method: str) -> np.ndarray:
        return np.asarray([r["mean_return"][method] for r in records], dtype=np.float64)

    comparisons: dict[str, dict] = {}
    for cand in CANDIDATES:
        c_all = root_ise("all", cand)
        c_ret = root_ret(cand)
        for ref in REFERENCES:
            r_all = root_ise("all", ref)
            r_ret = root_ret(ref)
            diff_ise = r_all - c_all
            ci_ise = paired_mean_interval(diff_ise)
            rel = float(diff_ise.mean() / r_all.mean())
            halves = [
                float(1.0 - root_ise(p, cand).mean() / root_ise(p, ref).mean())
                for p in ("first", "last")
            ]
            ci_ret = paired_mean_interval(c_ret - r_ret)
            comparisons[f"{cand}_vs_{ref}"] = {
                "paired_ise_gain": ci_ise,
                "relative_ise_gain": rel,
                "half_relative_gains": halves,
                "root_relative_gains": ((r_all - c_all) / r_all).tolist(),
                "favorable_roots": int(np.sum(diff_ise > 0.0)),
                "paired_return_gain": ci_ret,
                "supported": bool(
                    ci_ise["interval"][0] > 0.0
                    and rel >= plan["minimum_relative_gain"]
                    and min(halves) > 0.0
                ),
            }

    return {
        "run_name": plan["run_name"],
        "scope": plan["scope"],
        "root_count": len(records),
        "evaluation_paths_per_root": 64,
        "total_paired_paths": 512,
        "mean_ise": {m: float(root_ise("all", m).mean()) for m in METHODS},
        "mean_return": {m: float(root_ret(m).mean()) for m in METHODS},
        "mean_early_calls": {
            m: float(np.mean([r["mean_early_calls"][m] for r in records]))
            for m in METHODS
        },
        "mean_score_checks": {
            m: float(np.mean([r["mean_score_checks"][m] for r in records]))
            for m in METHODS
        },
        "comparisons": comparisons,
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
    workdir = ROOT / "results" / RUN / str(cell["optimizer_seed"])
    public_dir = workdir / "public"
    public_dir.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile, ledger.recording("unprivileged_ablation"):
            if profile != old["hardware_profile"]:
                raise ValueError("runtime differs from original frozen builds")
            result = {
                **run_cell(plan, cell, old, workdir, ledger),
                "status": "complete",
            }
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
