"""Frozen PointMaze controls for path-specific timing and paired supervision."""

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

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, jitter_schedule, run_episode, time_contract
from freq_hrl.experiments.plan_validity_unprivileged_and_pareto import unprivileged_uniform_opportunities
from freq_hrl.experiments.pointmaze_budgeted_trigger import fit_budgeted_trigger_predictors
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.route_b_baselines import RouteBTimingRule
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_route_b_pointmaze_baselines import PLAN as SOURCE_PLAN
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_review_controls_20261120"
PLAN = Path("protocols/plan_validity_pointmaze_review_controls_2026-11-20.json")
RUNNER = "scripts/run_pointmaze_review_controls.py"
VIEW = "causal_validity_interactions"
REFERENCES = ("yoked", "single_arm")


def read(path):
    return json.loads((ROOT / path).read_text())


def prepare():
    old = read(MANIFEST)
    used = historical_path_seeds(old)
    cells = []
    for i, source in enumerate(old["confirmation_reservation"]):
        start = 1_987_400_000 + i * 1000
        cells.append({
            "root": source["optimizer_seed"],
            "primary_fit_paths": source["branch_fit_seeds"],
            "shadow_fit_paths": list(range(start, start + 8)),
            "evaluation_paths": list(range(start + 100, start + 164)),
            "queries_per_path": 24,
            "assignment_seed": start + 200,
        })
    preflight = {**cells[0], "primary_fit_paths": cells[0]["primary_fit_paths"][:2],
                 "shadow_fit_paths": [1_987_499_000, 1_987_499_001],
                 "evaluation_paths": [1_987_499_100, 1_987_499_101],
                 "queries_per_path": 4, "assignment_seed": 1_987_499_200}
    fresh = [seed for cell in [preflight, *cells] for seed in (
        *cell["shadow_fit_paths"], *cell["evaluation_paths"], cell["assignment_seed"])]
    if len(fresh) != len(set(fresh)) or used.intersection(fresh):
        raise ValueError("review banks overlap historical or reserved roles")
    return {
        "run_name": RUN,
        "source_controller_manifest": str(MANIFEST),
        "source_ridge_plan": str(SOURCE_PLAN),
        "preflight": preflight,
        "cells": cells,
        "calls_per_episode": 24,
        "paired_fit": "Both arms per original clock-grid prefix; must reproduce frozen ridge.",
        "single_arm_fit": "One randomized arm per path at each shared query step; "
                          "complementary arm on a fresh independent path with the same clock.",
        "single_arm_target": "2 * (+1 keep, -1 renew) * (branch ISE - 0.5 * current squared distance)",
        "identical_fit": "39-feature ridge, same alpha grid and OOF quantile; "
                         "matched path pairs are held out together for the single-arm fit.",
        "yoked": "Swap full call sequences within consecutive fresh path pairs of the same root.",
        "inference": "Paired episode ISE averaged within eight roots; "
                     "two-sided Student-t 95% intervals (97.5% across the two primary contrasts), "
                     "plus two prespecified path halves. Evidence requires adjusted lower bound > 0 "
                     "and agreement in both halves.",
        "new_controller_training": False,
    }


def balanced_arms(rng, size):
    return rng.permutation([False, True] * (size // 2))


def single_arm_target(row, renew):
    names = tuple(row["feature_names"])
    errors = [row["features"][names.index(f"target_error_{j}")] for j in range(2)]
    center = 0.5 * sum(e * e for e in errors)
    return (-2.0 if renew else 2.0) * (row["branch_ise"] - center)


def fit_rows(cell, controller, args, spec, scale):
    rng = np.random.default_rng(cell["assignment_seed"])
    paired, single = [], []
    for primary, shadow in zip(cell["primary_fit_paths"], cell["shadow_fit_paths"], strict=True):
        schedule = jitter_schedule(args, primary)
        count = cell["queries_per_path"]
        queries = unprivileged_uniform_opportunities(
            seed=primary, horizon=args.horizon, schedule=schedule, count=count,
            period_steps=50, max_offset_steps=25, check_stride_steps=5, branch_window_steps=50,
        )
        for query, renew_on_primary in zip(queries, balanced_arms(rng, count), strict=True):
            def branch(seed, renew):
                return run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale,
                                   schedule=schedule, branch_step=query.step, renew=renew)[1]

            def add(rows, source, target):
                rows.append({"seed": primary, "causal_features": source["features"].tolist(),
                             "feature_names": list(source["feature_names"]),
                             "renew_ise_advantage": target,
                             "candidate_feature_has_future_access": False,
                             "candidate_feature_has_regime_label": False})

            with measured_phase("paired_branches"):
                keep, renew = branch(primary, False), branch(primary, True)
                if not (np.array_equal(keep["prefix"], renew["prefix"])
                        and np.array_equal(keep["features"], renew["features"])):
                    raise RuntimeError("paired branch changed the prefix")
                add(paired, keep, keep["branch_ise"] - renew["branch_ise"])
            with measured_phase("single_arm_branches"):
                for seed, act in ((primary, bool(renew_on_primary)), (shadow, not renew_on_primary)):
                    outcome = branch(seed, act)
                    add(single, outcome, single_arm_target(outcome, act))
    return paired, single


def fit(rows, args):
    return fit_budgeted_trigger_predictors(
        rows, alpha_grid=args.ridge_alpha_grid, threshold_quantile=args.threshold_quantile,
        predictors=(VIEW,),
    )[VIEW]


def run_cell(cell, preflight):
    root = cell["root"]
    source = directory(read(MANIFEST), root) / "build/private/history.json"
    controller, args = load_controller(read(source))
    source_cell, = [c for c in read(SOURCE_PLAN)["cells"] if c["optimizer_seed"] == root]
    source_path = Path(source_cell["fit_baselines_public"]).parent / "private/baselines.json"
    bundle = read(source_path)
    if (bundle["optimizer_seed"], bundle["evaluation_accessed"]) != (root, False):
        raise ValueError("expected a frozen ridge fit on this controller")
    frozen = RouteBTimingRule.from_dict(bundle["rules"]["ridge_unprivileged"])
    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    scale = time_contract(args)
    seeds = cell["evaluation_paths"]
    ledger = ComputeLedger()
    with single_thread_runtime(), ledger.recording():
        paired, single = fit_rows(cell, controller, args, spec, scale)
        counts = [ledger.phases[f"offline/{key}_branches"]["counts"]
                  for key in ("paired", "single_arm")]
        for field in ("environment_steps", "environment_resets", "planner_calls"):
            if counts[0][field] != counts[1][field]:
                raise RuntimeError(f"supervision budgets differ: {field}")
        with measured_phase("ridge_fit"):
            reproduced, fitted = fit(paired, args), fit(single, args)
        if not preflight:
            for field in ("weights", "feature_mean", "feature_scale", "alpha"):
                np.testing.assert_allclose(reproduced["model"][field], frozen.model["model"][field],
                                           atol=1e-12, rtol=1e-9)
            np.testing.assert_allclose(reproduced["threshold"], frozen.threshold, atol=1e-12, rtol=1e-9)
        single_rule = RouteBTimingRule("single_arm", "legacy", view=VIEW,
                                      threshold=fitted["threshold"], model=fitted)

        def episode(seed, **kwargs):
            row = run_episode(controller, task_spec=spec, seed=seed, args=args,
                              time_scale=scale, **kwargs)[1]
            if row["episode_length"] != 1200 or row["upper_decision_count"] != 24:
                raise RuntimeError("evaluation changed the episode or call budget")
            return row

        with measured_phase("online"):
            online = [episode(seed, rule=frozen) for seed in seeds]
        with measured_phase("yoked"):
            yoked = [episode(seed, schedule=online[i ^ 1]["decision_steps"]) for i, seed in enumerate(seeds)]
            for i, row in enumerate(yoked):
                if row["decision_steps"] != online[i ^ 1]["decision_steps"] or row["trigger_score_checks"]:
                    raise RuntimeError("yoked schedule did not replay its donor clock")
        with measured_phase("single_arm_evaluation"):
            independent = [episode(seed, rule=single_rule) for seed in seeds]

    return {"status": "complete", "root": root, "preflight": preflight,
            "supervision_steps_per_rule": counts[0]["environment_steps"],
            "supervision_planner_calls_per_rule": counts[0]["planner_calls"],
            "fit_rows": [len(paired), len(single)],
            "paths": [{"seed": seed, "donor": seeds[i ^ 1], "ridge": online[i][ISE], "yoked": yoked[i][ISE],
                       "single_arm": independent[i][ISE],
                       "yoked_schedule_differs": online[i]["decision_steps"] != yoked[i]["decision_steps"]}
                      for i, seed in enumerate(seeds)],
            "ledger": ledger.summary()}


def summarize(plan, records):
    if len(plan["cells"]) != 8 or len(records) != 8 or any(
        r["status"] != "complete" or (r["root"], r["preflight"], [p["seed"] for p in r["paths"]])
        != (c["root"], False, c["evaluation_paths"])
        for r, c in zip(records, plan["cells"], strict=True)
    ):
        raise ValueError("formal summary requires the complete frozen cohort")
    rows = {key: np.array([[p[key] for p in r["paths"]] for r in records]) for key in ("ridge", *REFERENCES)}
    contrasts = {}
    for key in REFERENCES:
        gain = rows[key] - rows["ridge"]
        roots = gain.mean(axis=1)
        ci = paired_mean_interval(roots)
        adjusted = paired_mean_interval(roots, alpha=0.025)
        halves = [float(half.mean()) for half in np.split(gain, 2, axis=1)]
        contrasts[key] = {"paired_gain": ci, "adjusted_interval": adjusted["interval"],
                          "relative_gain": float(gain.mean() / rows[key].mean()),
                          "half_gains": halves, "favorable_roots": int((roots > 0).sum()),
                          "supported": bool(adjusted["interval"][0] > 0 and min(halves) > 0)}
    return {"run_name": RUN, "root_count": len(records), "phase": "formal",
            "paths_per_root": rows["ridge"].shape[1], "calls_per_episode": plan["calls_per_episode"],
            "mean_ise": {key: float(value.mean()) for key, value in rows.items()},
            "contrasts": contrasts,
            "roots": [{"optimizer_seed": r["root"],
                       "supervision_steps_per_rule": r["supervision_steps_per_rule"],
                       "supervision_planner_calls_per_rule": r["supervision_planner_calls_per_rule"],
                       "fit_rows": r["fit_rows"],
                       "different_yoked_schedules": sum(p["yoked_schedule_differs"] for p in r["paths"]),
                       "mean_ise": {key: float(value[i].mean()) for key, value in rows.items()},
                       "cpu_seconds": r["ledger"]["cpu_seconds"]}
                      for i, r in enumerate(records)]}


def result_path(index):
    name = "preflight" if index == -1 else f"root_{index}"
    return Path("results") / RUN / name / "public/result.json"


def submit(plan):
    from scripts.run_route_a_scheduleurm import submit_jobs
    from scripts.run_route_b_scheduleurm import wait_paths, pull_summary

    old = read(MANIFEST)
    template = old["jobs"]["audit"][0]

    def job(i):
        pub = str(ROOT / result_path(i).parent)
        spec = deepcopy(template)
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        spec.update(project="Freq-HRL-Review", description=f"PointMaze matched review controls {i}",
                    signature=f"Freq-HRL/{RUN}/{i}",
                    cmd=template["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
                    result_dir=pub, local_result_dir=pub,
                    wait_for_files=[str(ROOT / p) for p in (PLAN, MANIFEST, SOURCE_PLAN)])
        return spec

    pre_ids, signatures = submit_jobs([job(-1)], RUN + "_preflight")
    wait_paths("Review controls preflight", [str(result_path(-1))], signatures)
    ids, signatures = submit_jobs([job(i) for i in range(len(plan["cells"]))], RUN)
    wait_paths("Review controls formal", [str(result_path(i)) for i in range(len(plan["cells"]))], signatures)
    pull_summary(RUN, RUNNER, pre_ids + ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "submit", "run", "readout"))
    parser.add_argument("--cell", type=int)
    opts = parser.parse_args()
    if opts.phase == "prepare":
        emit(ROOT / PLAN, prepare())
        return
    plan = read(PLAN)
    if opts.phase == "submit":
        submit(plan)
    elif opts.phase == "run":
        preflight = opts.cell == -1
        cell = plan["preflight"] if preflight else plan["cells"][opts.cell]
        path = ROOT / result_path(opts.cell)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = run_cell(cell, preflight)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(output, indent=2) + "\n")
        temp.replace(path)
    else:
        output = summarize(plan, [read(result_path(i)) for i in range(len(plan["cells"]))])
        path = ROOT / "results" / RUN / "summary.json"
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output["contrasts"], indent=2))


if __name__ == "__main__":
    main()
