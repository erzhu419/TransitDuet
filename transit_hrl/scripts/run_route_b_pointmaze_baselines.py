"""Experiment B2: PointMaze strong baselines (Native-Periodic HRL, TEMPLE, Deliberation OC, Generic-MLP)."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

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
)
from freq_hrl.experiments.pointmaze_budgeted_trigger import (
    fit_budgeted_trigger_predictors,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from freq_hrl.experiments.route_b_baselines import (
    RouteBTimingRule,
    fit_deliberation_option_critic_rule,
    fit_generic_mlp_validity_predictor,
    fit_temple_gate_rule,
    train_native_periodic_controller,
)
from freq_hrl.rl import GoalConditionedActorCriticPPO, GoalConditionedPPOConfig
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE, panels
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_route_b_strong_baselines_20261118"
PLAN = Path("protocols/plan_validity_route_b_strong_baselines_2026-11-18.json")
RUNNER = "scripts/run_route_b_pointmaze_baselines.py"
METHODS = (
    "native_periodic_hrl",
    "shared_periodic",
    "shared_jitter",
    "distance_threshold",
    "temple_gate",
    "deliberation_option_critic",
    "generic_mlp_unprivileged",
    "ridge_unprivileged",
)
CANDIDATES = ("generic_mlp_unprivileged", "ridge_unprivileged")
REFERENCES = (
    "native_periodic_hrl",
    "shared_periodic",
    "shared_jitter",
    "distance_threshold",
    "temple_gate",
    "deliberation_option_critic",
)


def prepare(old: dict) -> dict:
    used = historical_path_seeds(old)
    cells = []
    jobs = {"train_periodic": [], "fit_baselines": [], "evaluate": []}
    template = old["jobs"]["audit"][0]

    for i, c in enumerate(old["confirmation_reservation"]):
        root = int(c["optimizer_seed"])
        base = 1_982_300_000 + i * 10_000
        seeds = list(range(base, base + 64))
        if used.intersection(seeds):
            raise ValueError("PointMaze Route B evaluation seeds overlap an existing role")
        used.update(seeds)
        cells.append({
            "optimizer_seed": root,
            "evaluation_seeds": seeds,
            "train_periodic_public": f"results/{RUN}/{root}/train_periodic/public",
            "fit_baselines_public": f"results/{RUN}/{root}/fit_baselines/public",
            "evaluate_public": f"results/{RUN}/{root}/evaluate/public",
        })
        for phase in ("train_periodic", "fit_baselines", "evaluate"):
            pub = cells[-1][f"{phase}_public"]
            job = deepcopy(template)
            cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, phase, "--cell", str(i)]
            job.update(
                project="Freq-HRL-RouteB",
                description=f"RouteB PointMaze {phase} root {root}",
                signature=f"Freq-HRL/routeB-pm-baselines/{RUN}/{phase}/{root}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
                result_dir=str(ROOT / pub),
                local_result_dir=str(ROOT / pub),
                wait_for_files=[str(ROOT / PLAN), str(ROOT / MANIFEST)],
            )
            jobs[phase].append(job)

    return {
        "run_name": RUN,
        "source_manifest": str(MANIFEST),
        "methods": list(METHODS),
        "cells": cells,
        "jobs": jobs,
        "minimum_relative_gain": 0.02,
        "scope": "pointmaze_unprivileged_generic_mlp_vs_native_periodic_temple_deliberation_oc_and_distance",
        "original_confirmation_released": False,
    }


def _load_periodic_controller(ckpt_path: Path, args: SimpleNamespace) -> GoalConditionedActorCriticPPO:
    scale = time_contract(args)
    dim = 6 + 6 * scale.history_steps
    model = GoalConditionedActorCriticPPO(
        GoalConditionedPPOConfig(
            upper_state_dim=dim,
            lower_state_dim=dim,
            goal_dim=2,
            action_dim=2,
            hidden_dim=args.reference_hidden_dim,
            learning_rate=args.learning_rate,
            epochs=4,
            minibatch_size=1024,
            init_log_std=-0.7,
        )
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model


def do_train_periodic(cell: dict, old: dict, root_dir: Path) -> dict:
    root = int(cell["optimizer_seed"])
    home = directory(old, root)
    hist_rec = json.loads((home / "build/private/history.json").read_text())
    args = SimpleNamespace(**hist_rec["configuration"])
    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    scale = time_contract(args)

    payload, model = train_native_periodic_controller(spec, args, scale)
    priv = root_dir / "train_periodic/private"
    priv.mkdir(parents=True, exist_ok=False)
    ckpt = priv / "native_periodic_controller.pt"
    torch.save(model.state_dict(), ckpt)
    return {
        "selected_checkpoint_iteration": int(payload["selected_checkpoint_iteration"]),
        "checkpoint_path": str(ckpt),
        "evaluation_accessed": False,
    }


def do_fit_baselines(cell: dict, old: dict, root_dir: Path) -> dict:
    root = int(cell["optimizer_seed"])
    home = directory(old, root)
    hist_rec = json.loads((home / "build/private/history.json").read_text())
    controller, args = load_controller(hist_rec)
    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    scale = time_contract(args)

    with measured_phase("unprivileged_labels"):
        labels = collect_unprivileged_renewal_labels(
            controller,
            acquisition="uniform",
            task_spec=spec,
            args=args,
            time_scale=scale,
            queries_per_path=24,
        )
    with measured_phase("fit_ridge_unprivileged"):
        ridge_fitted = fit_budgeted_trigger_predictors(
            labels,
            alpha_grid=args.ridge_alpha_grid,
            threshold_quantile=args.threshold_quantile,
            predictors=("causal_validity_interactions",),
        )["causal_validity_interactions"]
        ridge_rule = RouteBTimingRule(
            name="ridge_unprivileged",
            kind="legacy",
            view="causal_validity_interactions",
            threshold=float(ridge_fitted["threshold"]),
            model=ridge_fitted,
        )

    with measured_phase("fit_generic_mlp_unprivileged"):
        mlp_rule, _ = fit_generic_mlp_validity_predictor(
            labels,
            seed=root + 11,
            hidden_dim=32,
            epochs=220,
            threshold_quantile=args.threshold_quantile,
        )

    with measured_phase("fit_temple_gate"):
        temple_rule, temple_meta = fit_temple_gate_rule(
            controller,
            task_spec=spec,
            args=args,
            time_scale=scale,
            seed=root + 23,
            rounds=20,
            hidden_dim=32,
        )

    with measured_phase("fit_deliberation_oc"):
        oc_rule, oc_meta = fit_deliberation_option_critic_rule(
            controller,
            task_spec=spec,
            args=args,
            time_scale=scale,
            seed=root + 37,
            rollout_episodes=160,
            hidden_dim=32,
        )

    priv = root_dir / "fit_baselines/private"
    priv.mkdir(parents=True, exist_ok=False)
    emit(
        priv / "baselines.json",
        {
            "optimizer_seed": root,
            "evaluation_accessed": False,
            "rules": {
                "ridge_unprivileged": ridge_rule.to_dict(),
                "generic_mlp_unprivileged": mlp_rule.to_dict(),
                "temple_gate": temple_rule.to_dict(),
                "deliberation_option_critic": oc_rule.to_dict(),
            },
            "diagnostics": {
                "generic_mlp_oof_mse": mlp_rule.model["oof_mse"],
                "temple": temple_meta,
                "deliberation_oc": oc_meta,
            },
        },
    )
    return {
        "evaluation_accessed": False,
        "generic_mlp_oof_mse": float(mlp_rule.model["oof_mse"]),
        "temple_selected": temple_meta["selected_name"],
        "oc_selected": oc_meta["selected_name"],
    }


def do_evaluate(plan: dict, cell: dict, old: dict) -> dict:
    # Verify ALL 8 roots completed both build phases before ANY evaluation runs
    for c in plan["cells"]:
        r_seed = c["optimizer_seed"]
        p1 = ROOT / c["train_periodic_public"] / "result.json"
        p2 = ROOT / c["fit_baselines_public"] / "result.json"
        if not p1.is_file() or not p2.is_file():
            raise ValueError(f"global build freeze incomplete on root {r_seed}")
        if json.loads(p1.read_text())["status"] != "complete" or json.loads(p2.read_text())["status"] != "complete":
            raise ValueError(f"build status not complete on root {r_seed}")

    root = int(cell["optimizer_seed"])
    home = directory(old, root)
    hist_rec = json.loads((home / "build/private/history.json").read_text())
    dist_rec = json.loads((home / "build/private/threshold.json").read_text())
    shared_ctrl, args = load_controller(hist_rec)

    root_dir = ROOT / "results" / plan["run_name"] / str(root)
    periodic_ckpt = root_dir / "train_periodic/private/native_periodic_controller.pt"
    periodic_ctrl = _load_periodic_controller(periodic_ckpt, args)
    baseline_bundle = json.loads((root_dir / "fit_baselines/private/baselines.json").read_text())

    seeds = list(cell["evaluation_seeds"])
    forbidden = set(
        args.train_seeds
        + args.selection_seeds
        + args.branch_fit_seeds
        + args.timing_selection_seeds
        + args.trigger_eval_seeds
    )
    if forbidden.intersection(seeds):
        raise ValueError("Route B evaluation overlaps training/selection/reserved seeds")

    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    scale = time_contract(args)
    periodic_rule = TimingRule("periodic", "clock", threshold=0.0)

    dispatch = {
        "native_periodic_hrl": (periodic_ctrl, periodic_rule),
        "shared_periodic": (shared_ctrl, periodic_rule),
        "shared_jitter": (shared_ctrl, TimingRule("jitter", "jitter")),
        "distance_threshold": (shared_ctrl, TimingRule(**dist_rec["selected_rule"])),
        "temple_gate": (
            shared_ctrl,
            RouteBTimingRule.from_dict(baseline_bundle["rules"]["temple_gate"]),
        ),
        "deliberation_option_critic": (
            shared_ctrl,
            RouteBTimingRule.from_dict(baseline_bundle["rules"]["deliberation_option_critic"]),
        ),
        "generic_mlp_unprivileged": (
            shared_ctrl,
            RouteBTimingRule.from_dict(baseline_bundle["rules"]["generic_mlp_unprivileged"]),
        ),
        "ridge_unprivileged": (
            shared_ctrl,
            RouteBTimingRule.from_dict(baseline_bundle["rules"]["ridge_unprivileged"]),
        ),
    }

    rows: dict[str, list[dict]] = {}
    latencies: dict[str, dict] = {}
    returns: dict[str, float] = {}
    early_calls: dict[str, float] = {}
    score_checks: dict[str, float] = {}
    for method in METHODS:
        ctrl, rule = dispatch[method]
        latency = ControlLatency()
        with measured_phase(f"eval_{method}"):
            rows[method] = [
                run_episode(
                    ctrl,
                    task_spec=spec,
                    seed=s,
                    args=args,
                    time_scale=scale,
                    rule=rule,
                    latency_recorder=latency,
                    risk_spec={"distance": 1.0, "duration_steps": 25},
                )[1]
                for s in seeds
            ]
        latencies[method] = latency.summary(episodes=len(seeds), deadline_seconds=0.01)
        returns[method] = float(np.mean([r["episode_return"] for r in rows[method]]))
        early_calls[method] = float(
            np.mean([
                sum(1 for step in r["decision_steps"][1:] if (step % 50) < 25)
                for r in rows[method]
            ])
        )
        score_checks[method] = float(
            np.mean([r["trigger_score_checks"] for r in rows[method]])
        )

    return {
        "evaluation_seeds": seeds,
        "panels": panels(rows, methods=METHODS),
        "mean_return": returns,
        "mean_early_calls": early_calls,
        "mean_score_checks": score_checks,
        "sustained_failure_counts": {
            m: int(sum(r["reliability"]["sustained_failure"] for r in rs))
            for m, rs in rows.items()
        },
        "latency": latencies,
    }


def readout(plan: dict, records: list[dict]) -> dict:
    if len(records) != len(plan["cells"]) or any(r["status"] != "complete" for r in records):
        raise ValueError("Route B PointMaze readout requires all eight completed roots")

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
            diff = r_all - c_all
            ci = paired_mean_interval(diff)
            rel = float(diff.mean() / r_all.mean())
            halves = [
                float(1.0 - root_ise(p, cand).mean() / root_ise(p, ref).mean())
                for p in ("first", "last")
            ]
            comparisons[f"{cand}_vs_{ref}"] = {
                "paired_ise_gain": ci,
                "relative_ise_gain": rel,
                "half_relative_gains": halves,
                "favorable_roots": int(np.sum(diff > 0.0)),
                "paired_return_gain": paired_mean_interval(c_ret - r_ret),
                "supported": bool(
                    ci["interval"][0] > 0.0
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
        "comparisons": comparisons,
        "roots": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "train_periodic", "fit_baselines", "evaluate", "readout")
    )
    parser.add_argument("--cell", type=int)
    opts = parser.parse_args()
    old = json.loads((ROOT / MANIFEST).read_text())
    if opts.phase == "prepare":
        emit(ROOT / PLAN, prepare(old))
        return 0
    plan = json.loads((ROOT / PLAN).read_text())
    if opts.phase == "readout":
        records = [
            json.loads((ROOT / c["evaluate_public"] / "result.json").read_text())
            for c in plan["cells"]
        ]
        summary = readout(plan, records)
        summary_path = ROOT / "results" / RUN / "summary.json"
        if not summary_path.exists():
            emit(summary_path, summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "roots"}, indent=2))
        return 0

    cell = plan["cells"][opts.cell]
    root_dir = ROOT / "results" / RUN / str(cell["optimizer_seed"])
    pub_dir = ROOT / cell[f"{opts.phase}_public"]
    pub_dir.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile, ledger.recording(opts.phase):
            if profile != old["hardware_profile"]:
                raise ValueError("runtime differs from original frozen builds")
            if opts.phase == "train_periodic":
                body = do_train_periodic(cell, old, root_dir)
            elif opts.phase == "fit_baselines":
                body = do_fit_baselines(cell, old, root_dir)
            else:
                body = do_evaluate(plan, cell, old)
            result = {**body, "status": "complete"}
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(
        pub_dir / "result.json",
        {
            **result,
            "optimizer_seed": cell["optimizer_seed"],
            "measured_cost": ledger.summary(),
        },
    )
    print(cell["optimizer_seed"], opts.phase, result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
