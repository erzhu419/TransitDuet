"""Experiment B1: Independent 8-seed end-to-end confirmation on Reacher-v5 polygonal trajectory tracking."""

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

from freq_hrl.domains.mujoco.reacher_polygonal import (
    REACHER_POLYGONAL_SPEEDS,
    ReacherPolygonalSpec,
)
from freq_hrl.experiments.compute_budget import (
    ComputeLedger,
    ControlLatency,
    measured_phase,
    single_thread_runtime,
)
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import (
    collect_renewal_features,
    jitter_schedule,
    run_episode,
    time_contract,
    train_shared_controller,
)
from freq_hrl.experiments.plan_validity_unprivileged_and_pareto import (
    collect_unprivileged_renewal_labels,
)
from freq_hrl.experiments.pointmaze_budgeted_trigger import (
    fit_budgeted_trigger_predictors,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import (
    TimingRule,
    select_rule,
    threshold_candidates,
)
from freq_hrl.experiments.route_b_baselines import (
    RouteBTimingRule,
    fit_deliberation_option_critic_rule,
    fit_generic_mlp_validity_predictor,
    fit_temple_gate_rule,
    train_native_periodic_controller,
)
from freq_hrl.rl import GoalConditionedActorCriticPPO, GoalConditionedPPOConfig
from scripts.prepare_stage9_strong_timing import MANIFEST, training_seeds
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE, panels
from scripts.run_stage9_strong_timing_confirmation import emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "reacher_polygonal_route_b_confirmation_20261118"
PLAN = Path("protocols/plan_validity_reacher_polygonal_confirmation_2026-11-18.json")
RUNNER = "scripts/run_route_b_reacher_polygonal.py"
METHODS = (
    "native_periodic_hrl",
    "shared_periodic",
    "shared_jitter",
    "best_clock",
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
    "best_clock",
    "distance_threshold",
    "temple_gate",
    "deliberation_option_critic",
)


def base_config() -> dict:
    return {
        "env_id": "Reacher-v5",
        "iterations": 384,
        "horizon": 1200,
        "checkpoint_evaluation_interval": 8,
        "reference_hidden_dim": 128,
        "learning_rate": 3e-4,
        "upper_period_seconds": 0.50,
        "history_seconds": 0.64,
        "fast_period_seconds": 0.04,
        "maximum_subgoal_delta": 0.50,
        "branch_window_seconds": 0.50,
        "max_events_per_class": 4,
        "ridge_alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "threshold_quantile": 0.75,
        "max_offset_steps": 25,
        "check_stride_steps": 5,
        "regime_dwell_seconds": [0.80, 1.60],
        "target_speed_modes": list(REACHER_POLYGONAL_SPEEDS),
        "force_pulse_amplitude": 0.18,
        "force_pulse_duration_seconds": [0.04, 0.10],
        "force_pulse_gap_seconds": [0.45, 1.10],
        "distractor_amplitude": 0.50,
        "distractor_dwell_seconds": [0.35, 0.85],
    }


def prepare(old: dict) -> dict:
    used = historical_path_seeds(old)
    cfg = base_config()
    template = old["jobs"]["audit"][0]
    cells = []
    jobs = {"build_shared": [], "build_periodic": [], "evaluate": []}

    for i in range(8):
        root = 1_983_100_000 + i * 10_000
        roles = {
            "train_seeds": list(range(root + 1000, root + 1008)),
            "selection_seeds": list(range(root + 2000, root + 2008)),
            "branch_fit_seeds": list(range(root + 3000, root + 3008)),
            "timing_selection_seeds": list(range(root + 4000, root + 4008)),
            "evaluation_seeds": list(range(root + 5000, root + 5064)),
        }
        all_new = [
            root,
            *(s for bank in roles.values() for s in bank),
            *training_seeds(root, roles["train_seeds"], cfg["iterations"]),
        ]
        if len(set(all_new)) != len(all_new) or used.intersection(all_new):
            raise ValueError(f"Reacher polygonal seed collision on root {root}")
        used.update(all_new)

        row = {
            "optimizer_seed": root,
            **roles,
            "build_shared_public": f"results/{RUN}/{root}/build_shared/public",
            "build_periodic_public": f"results/{RUN}/{root}/build_periodic/public",
            "evaluate_public": f"results/{RUN}/{root}/evaluate/public",
        }
        cells.append(row)
        for phase in ("build_shared", "build_periodic", "evaluate"):
            pub = row[f"{phase}_public"]
            job = deepcopy(template)
            cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, phase, "--cell", str(i)]
            job.update(
                project="Freq-HRL-RouteB",
                description=f"RouteB ReacherPolygonal {phase} root {root}",
                signature=f"Freq-HRL/routeB-reacher-poly/{RUN}/{phase}/{root}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
                result_dir=str(ROOT / pub),
                local_result_dir=str(ROOT / pub),
                wait_for_files=[str(ROOT / PLAN), str(ROOT / MANIFEST)],
            )
            jobs[phase].append(job)

    return {
        "run_name": RUN,
        "source_manifest": str(MANIFEST),
        "hardware_profile": old["hardware_profile"],
        "config": cfg,
        "methods": list(METHODS),
        "cells": cells,
        "jobs": jobs,
        "minimum_relative_gain": 0.02,
        "scope": "independent_8_seed_end_to_end_second_plant_confirmation_reacher_polygonal_unprivileged",
    }


def _make_model(args: SimpleNamespace, seed: int | None = None) -> GoalConditionedActorCriticPPO:
    if seed is not None:
        torch.manual_seed(int(seed))
    scale = time_contract(args)
    dim = 6 + 6 * scale.history_steps
    return GoalConditionedActorCriticPPO(
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


def do_build_shared(plan: dict, cell: dict, root_dir: Path) -> dict:
    args = SimpleNamespace(**{**plan["config"], **cell})
    spec = ReacherPolygonalSpec(_task_options(args))
    scale = time_contract(args)

    with measured_phase("shared_controller"):
        payload, controller = train_shared_controller(spec, args, scale)

    # Waypoint authority check on timing_selection_seeds (no evaluation access!)
    periodic_sched = tuple(range(0, args.horizon, 50))
    with measured_phase("waypoint_authority_check"):
        cur_ises = [
            run_episode(
                controller,
                task_spec=spec,
                seed=s,
                args=args,
                time_scale=scale,
                schedule=periodic_sched,
                waypoint_audit="current",
            )[1][ISE]
            for s in args.timing_selection_seeds
        ]
        lag_ises = [
            run_episode(
                controller,
                task_spec=spec,
                seed=s,
                args=args,
                time_scale=scale,
                schedule=periodic_sched,
                waypoint_audit="lag_one_bin",
            )[1][ISE]
            for s in args.timing_selection_seeds
        ]
    waypoint_authority_gain = float(1.0 - np.mean(cur_ises) / np.mean(lag_ises))

    with measured_phase("unprivileged_labels"):
        labels = collect_unprivileged_renewal_labels(
            controller,
            acquisition="uniform",
            task_spec=spec,
            args=args,
            time_scale=scale,
            queries_per_path=24,
        )

    sel_cache: dict[str, list[dict]] = {}

    def eval_sel(rule: TimingRule) -> list[dict]:
        if rule.name not in sel_cache:
            sel_cache[rule.name] = [
                run_episode(
                    controller,
                    task_spec=spec,
                    seed=s,
                    args=args,
                    time_scale=scale,
                    rule=rule,
                )[1]
                for s in args.timing_selection_seeds
            ]
        return sel_cache[rule.name]

    with measured_phase("select_clock_and_distance"):
        best_clock, _ = select_rule(
            [TimingRule(f"clock_{off}", "clock", threshold=off / 25.0) for off in range(0, 26, 5)],
            eval_sel,
        )
        # Extract tracking_distance cut candidates from unprivileged fit rows directly (zero extra steps!)
        dist_cands = [
            r
            for r in threshold_candidates(labels)
            if r.kind == "threshold" and r.feature == "tracking_distance"
        ]
        best_dist, _ = select_rule(dist_cands, eval_sel)

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
            seed=args.optimizer_seed + 11,
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
            seed=args.optimizer_seed + 23,
            rounds=20,
            hidden_dim=32,
        )

    with measured_phase("fit_deliberation_oc"):
        oc_rule, oc_meta = fit_deliberation_option_critic_rule(
            controller,
            task_spec=spec,
            args=args,
            time_scale=scale,
            seed=args.optimizer_seed + 37,
            rollout_episodes=160,
            hidden_dim=32,
        )

    priv = root_dir / "build_shared/private"
    priv.mkdir(parents=True, exist_ok=False)
    ckpt = priv / "shared_controller.pt"
    torch.save(controller.state_dict(), ckpt)
    emit(
        priv / "rules.json",
        {
            "optimizer_seed": args.optimizer_seed,
            "evaluation_accessed": False,
            "waypoint_authority_relative_gain": waypoint_authority_gain,
            "rules": {
                "best_clock": best_clock.to_dict(),
                "distance_threshold": best_dist.to_dict(),
                "temple_gate": temple_rule.to_dict(),
                "deliberation_option_critic": oc_rule.to_dict(),
                "generic_mlp_unprivileged": mlp_rule.to_dict(),
                "ridge_unprivileged": ridge_rule.to_dict(),
            },
            "diagnostics": {
                "selected_checkpoint_iteration": int(payload["selected_checkpoint_iteration"]),
                "generic_mlp_oof_mse": float(mlp_rule.model["oof_mse"]),
                "temple": temple_meta,
                "deliberation_oc": oc_meta,
            },
        },
    )
    return {
        "selected_checkpoint_iteration": int(payload["selected_checkpoint_iteration"]),
        "waypoint_authority_relative_gain": waypoint_authority_gain,
        "evaluation_accessed": False,
    }


def do_build_periodic(plan: dict, cell: dict, root_dir: Path) -> dict:
    args = SimpleNamespace(**{**plan["config"], **cell})
    spec = ReacherPolygonalSpec(_task_options(args))
    scale = time_contract(args)

    payload, model = train_native_periodic_controller(spec, args, scale)
    priv = root_dir / "build_periodic/private"
    priv.mkdir(parents=True, exist_ok=False)
    ckpt = priv / "native_periodic_controller.pt"
    torch.save(model.state_dict(), ckpt)
    return {
        "selected_checkpoint_iteration": int(payload["selected_checkpoint_iteration"]),
        "evaluation_accessed": False,
    }


def do_evaluate(plan: dict, cell: dict) -> dict:
    # Enforce global freeze across ALL 8 roots before touching any evaluation seed!
    for c in plan["cells"]:
        p1 = ROOT / c["build_shared_public"] / "result.json"
        p2 = ROOT / c["build_periodic_public"] / "result.json"
        if not p1.is_file() or not p2.is_file():
            raise ValueError(f"Reacher global build freeze incomplete on {c['optimizer_seed']}")
        if json.loads(p1.read_text())["status"] != "complete" or json.loads(p2.read_text())["status"] != "complete":
            raise ValueError(f"Reacher build failed on {c['optimizer_seed']}")

    args = SimpleNamespace(**{**plan["config"], **cell})
    root = int(args.optimizer_seed)
    root_dir = ROOT / "results" / plan["run_name"] / str(root)
    shared_ctrl = _make_model(args)
    shared_ctrl.load_state_dict(
        torch.load(root_dir / "build_shared/private/shared_controller.pt", map_location="cpu", weights_only=True)
    )
    periodic_ctrl = _make_model(args)
    periodic_ctrl.load_state_dict(
        torch.load(root_dir / "build_periodic/private/native_periodic_controller.pt", map_location="cpu", weights_only=True)
    )
    untrained_ctrl = _make_model(args, seed=root)
    bundle = json.loads((root_dir / "build_shared/private/rules.json").read_text())

    seeds = list(cell["evaluation_seeds"])
    forbidden = set(
        args.train_seeds
        + args.selection_seeds
        + args.branch_fit_seeds
        + args.timing_selection_seeds
    )
    if forbidden.intersection(seeds):
        raise ValueError("Reacher evaluation overlaps training/selection/fit seeds")

    spec = ReacherPolygonalSpec(_task_options(args))
    scale = time_contract(args)
    periodic_rule = TimingRule("periodic", "clock", threshold=0.0)
    jitter_rule = TimingRule("jitter", "jitter")

    # Controller learning check
    untrained_ises = [
        run_episode(
            untrained_ctrl, task_spec=spec, seed=s, args=args, time_scale=scale, rule=jitter_rule
        )[1][ISE]
        for s in seeds
    ]

    dispatch = {
        "native_periodic_hrl": (periodic_ctrl, periodic_rule),
        "shared_periodic": (shared_ctrl, periodic_rule),
        "shared_jitter": (shared_ctrl, jitter_rule),
        "best_clock": (shared_ctrl, TimingRule(**bundle["rules"]["best_clock"])),
        "distance_threshold": (shared_ctrl, TimingRule(**bundle["rules"]["distance_threshold"])),
        "temple_gate": (shared_ctrl, RouteBTimingRule.from_dict(bundle["rules"]["temple_gate"])),
        "deliberation_option_critic": (
            shared_ctrl,
            RouteBTimingRule.from_dict(bundle["rules"]["deliberation_option_critic"]),
        ),
        "generic_mlp_unprivileged": (
            shared_ctrl,
            RouteBTimingRule.from_dict(bundle["rules"]["generic_mlp_unprivileged"]),
        ),
        "ridge_unprivileged": (
            shared_ctrl,
            RouteBTimingRule.from_dict(bundle["rules"]["ridge_unprivileged"]),
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
        "waypoint_authority_relative_gain": float(bundle["waypoint_authority_relative_gain"]),
        "untrained_mean_ise": float(np.mean(untrained_ises)),
        "panels": panels(rows, methods=METHODS),
        "mean_return": returns,
        "mean_early_calls": early_calls,
        "mean_score_checks": score_checks,
        "latency": latencies,
    }


def readout(plan: dict, records: list[dict]) -> dict:
    if len(records) != len(plan["cells"]) or any(r["status"] != "complete" for r in records):
        raise ValueError("Reacher polygonal readout requires all eight completed roots")

    def root_ise(panel: str, method: str) -> np.ndarray:
        return np.asarray([r["panels"][panel][method]["mean_ise"] for r in records], dtype=np.float64)

    def root_ret(method: str) -> np.ndarray:
        return np.asarray([r["mean_return"][method] for r in records], dtype=np.float64)

    untrained = np.asarray([r["untrained_mean_ise"] for r in records], dtype=np.float64)
    learning_gain = paired_mean_interval(untrained - root_ise("all", "shared_jitter"))
    wp_gains = np.asarray(
        [r["waypoint_authority_relative_gain"] for r in records], dtype=np.float64
    )

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
        "plant": "Gymnasium/Reacher-v5 (Polygonal 5-vertex trajectory)",
        "root_count": len(records),
        "evaluation_paths_per_root": 64,
        "total_paired_paths": 512,
        "controller_learning_ise_gain": learning_gain,
        "mean_waypoint_authority_relative_gain": float(wp_gains.mean()),
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
        "phase", choices=("prepare", "build_shared", "build_periodic", "evaluate", "readout")
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
            if profile != plan["hardware_profile"]:
                raise ValueError("runtime differs from registered cluster hardware")
            if opts.phase == "build_shared":
                body = do_build_shared(plan, cell, root_dir)
            elif opts.phase == "build_periodic":
                body = do_build_periodic(plan, cell, root_dir)
            else:
                body = do_evaluate(plan, cell)
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
