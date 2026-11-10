#!/usr/bin/env python3
"""Whole-day out-of-fold residual shrinkage with frozen deployment and values."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v65_prior_residual as residual


expanded, warmup, deployment = residual.expanded, residual.warmup, residual.deployment
FOLDS = 4
VALIDATION_SEEDS = tuple(range(760001, 760033))
ITEMS = list(itertools.product(range(len(deployment.CASES)), range(len(VALIDATION_SEEDS))))
MODELS = dict(shrunk_linear="linear_residual", shrunk_mlp="mlp_residual")
CHANNELS = (*MODELS, *residual.CHANNELS)


def split_indices(case, fold):
    indices = expanded.training_indices(case, "large")
    return ([i for day, i in enumerate(indices) if day % FOLDS != fold],
            [i for day, i in enumerate(indices) if day % FOLDS == fold])


def specification(index):
    case, day = ITEMS[index]
    seed, actor_scenario = deployment.CASES[case]
    scenario, sampler = VALIDATION_SEEDS[day], 6600000 + case * 1000 + day
    return dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario,
                scenario_seed=scenario, sampler_seed=sampler, split="validation",
                label=f"{deployment.case_name(seed, actor_scenario)}_demand{scenario}_sample{sampler}")


def fit_fold_models(days, encoder, hidden, lr, seed):
    prepared = [residual.features(day, encoder) for day in days]
    prior = float(np.mean([day["final_cv_squared"] for day in days]))
    center, scale = residual.training_statistics(prepared)
    with torch.random.fork_rng():
        torch.manual_seed(seed + 2000)
        models = {name: residual.ResidualValue(name, center, scale, hidden) for name in residual.CANDIDATES}
    training = residual.fit_models(models, prepared, prior, lr, seed)
    return models, prior, training


def quadratic_stats(predicted, target):
    predicted = np.asarray(predicted, dtype=float)
    return dict(rr=float(np.square(predicted).mean()), ry=float((predicted * target).mean()),
                yy=float(target ** 2), decisions=len(predicted))


def estimate_shrinkage(rows):
    rr, ry, yy = (float(np.mean([row[key] for row in rows])) for key in ("rr", "ry", "yy"))
    alpha = float(np.clip(ry / rr, 0., 1.)) if rr > 0 else 0.
    return dict(alpha=alpha, rr=rr, ry=ry, yy=yy, prior_mse=yy,
                raw_mse=yy - 2 * ry + rr, calibrated_mse=yy - 2 * alpha * ry + alpha ** 2 * rr)


def fold_dir(root, case, fold):
    return root / "folds" / deployment.case_name(*deployment.CASES[case]) / f"fold{fold}"


def fit_fold(root, v59_root, v64_root, case, fold):
    label = deployment.case_name(*deployment.CASES[case])
    exported = root / "fold_exports" / label / f"fold{fold}"
    if (exported / "done.json").exists():
        return
    deployment.joint.optimized_runtime(root)
    source = torch.load(v59_root / "cells" / label / "learning_smoke.pt", map_location="cpu", weights_only=False)
    train, held = split_indices(case, fold)
    days = [expanded.read_day(v64_root, i) for i in train]
    hidden = source["trainer"]["cost_q_net"]["net.0.weight"].shape[0]
    lr = source["trainer"]["cost_q_optimizer"]["param_groups"][0]["lr"]
    seed, scenario = deployment.CASES[case]
    draw_seed = seed + scenario + 64000 + fold
    models, prior, training = fit_fold_models(days, source["encoder"], hidden, lr, draw_seed)
    rows = []
    for index in held:
        day = expanded.read_day(v64_root, index)
        inputs = residual.features(day, source["encoder"])
        target = float(day["final_cv_squared"] - prior)
        rows.append(dict(scenario_seed=day["scenario_seed"],
                         **{name: quadratic_stats(warmup.predict(model, inputs), target) for name, model in models.items()}))
    result = dict(case_index=case, fold=fold, prior=prior, training=training, fit_sampler_seed=draw_seed,
                  training_demand_seeds=[day["scenario_seed"] for day in days], days=rows,
                  actor_updates=0, dual_updates=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    destination = fold_dir(root, case, fold)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    exported.mkdir(parents=True, exist_ok=True)
    (exported / "done.json").write_text(json.dumps(dict(status="whole_day_oof_fold_complete", case_index=case, fold=fold)) + "\n")
    print("V66_FOLD_DONE " + json.dumps(dict(case_index=case, fold=fold, prior=prior, training=training)), flush=True)


def calibration_path(root, case):
    return root / "calibration" / deployment.case_name(*deployment.CASES[case]) / "done.json"


def calibrate(root, v65_root, case):
    folds = [json.loads((fold_dir(root, case, fold) / "done.json").read_text()) for fold in range(FOLDS)]
    days = [day for fold in folds for day in fold["days"]]
    if sorted(day["scenario_seed"] for day in days) != list(expanded.TRAIN_SEEDS):
        raise ValueError("cross-fitting must cover each of the 32 training days exactly once")
    source = residual.case_dir(v65_root, case)
    fitted = json.loads((source / "done.json").read_text())
    if fitted["fitting_demand_seeds"] != list(expanded.TRAIN_SEEDS):
        raise ValueError("production residual weights use a different fitting set")
    coefficients = {name: estimate_shrinkage([day[name] for day in days]) for name in residual.CANDIDATES}
    result = dict(case_index=case, coefficients=coefficients, oof_demand_days=len(days),
                  base_value_source=str(source / "residual_values.pt"), production_prior=fitted["prior"],
                  actor_updates=0, dual_updates=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    path = calibration_path(root, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    exported = root / "calibration_exports" / deployment.case_name(*deployment.CASES[case])
    exported.mkdir(parents=True, exist_ok=True)
    (exported / "done.json").write_text(json.dumps(dict(status="shrinkage_locked", case_index=case)) + "\n")
    print("V66_CALIBRATION_DONE " + json.dumps(result), flush=True)


def collect(root, v59_root, index):
    spec = specification(index)
    marker = root / "ready" / f"{spec['label']}.json"
    if marker.exists():
        return spec["label"]
    original = deployment.ITEMS.index((spec["case_index"], ("updated", "stochastic", 0)))
    audit = warmup.FullDayData(root, spec, None)
    label = deployment.collect(root, v59_root, None, original, decision_audit=audit,
                               scenario_seed=spec["scenario_seed"], private_sampler_seed=spec["sampler_seed"])
    row = json.loads((root / "ready" / f"{label}.json").read_text())
    marker.write_text(json.dumps(dict(**spec, **row["decision_audit"], actor_updates=0, dual_updates=0,
                                      diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")), indent=2) + "\n")
    print("V66_DATA_DONE " + json.dumps(spec), flush=True)
    return spec["label"]


def load_predictors(root, v59_root, v62_root, v64_root, case):
    calibration = json.loads(calibration_path(root, case).read_text())
    payload = torch.load(calibration["base_value_source"], map_location="cpu", weights_only=False)
    if payload["format"] != "freqduet-v65-prior-residual":
        raise ValueError("expected frozen terminal-residual value semantics")
    models = {}
    for name in residual.CANDIDATES:
        weights = payload["models"][name]
        model = residual.ResidualValue(name, weights["center"], weights["scale"], payload["hidden"])
        model.load_state_dict(weights)
        models[name] = model.eval()
    controls, means = residual.v64_models(v64_root, case)
    label = deployment.case_name(*deployment.CASES[case])
    frozen = torch.load(v59_root / "cells" / label / "learning_smoke.pt", map_location="cpu", weights_only=False)
    legacy = torch.load(expanded.terminal.case_dir(v62_root, case) / "cost_warmup.pt", map_location="cpu", weights_only=False)
    controls["frozen"] = warmup.load_cost_model(frozen["trainer"]["cost_q_net"]).eval()
    controls["legacy_direct"] = warmup.load_cost_model(legacy["cost_q_net"]).eval()
    return dict(models=models, controls=controls, means=means, payload=payload,
                alphas={name: value["alpha"] for name, value in calibration["coefficients"].items()})


def remaining_prediction(prior, prediction, prefix, alpha):
    return prior + alpha * np.asarray(prediction, dtype=float) - prefix


def score_day(root, index, predictors):
    spec = specification(index)
    marker = json.loads((root / "ready" / f"{spec['label']}.json").read_text())
    with np.load(marker["dataset"]) as data:
        day = {key: data[key] for key in data.files}
    payload, means = predictors["payload"], predictors["means"]
    inputs = residual.features(day, payload["encoder"])
    raw = {name: warmup.predict(model, inputs) for name, model in predictors["models"].items()}
    predictions = {name: remaining_prediction(payload["prior"], value, day["prefix"], 1.) for name, value in raw.items()}
    predictions.update({name: remaining_prediction(payload["prior"], raw[original], day["prefix"], predictors["alphas"][original])
                        for name, original in MODELS.items()})
    predictions.update({name: (expanded.terminal.predict_remaining(model, day) if name.endswith("_factored")
                               else warmup.predict(model, day)) for name, model in predictors["controls"].items()})
    predictions.update(zero=np.zeros_like(day["returns"]), small_prior=means["small"] - day["prefix"],
                       large_prior=means["large"] - day["prefix"])
    return dict(scenario_seed=spec["scenario_seed"], sampler_seed=spec["sampler_seed"],
                **{name: warmup.calibration_metrics(predictions[name], day["returns"]) for name in CHANNELS})


def strict_gate(days, candidate):
    return all(day[candidate][metric] < day[baseline][metric]
               for day in days for baseline in residual.CHANNELS for metric in ("mae", "rmse"))


def evaluate(root, v59_root, v62_root, v64_root, case):
    deployment.joint.optimized_runtime(root)
    predictors = load_predictors(root, v59_root, v62_root, v64_root, case)
    rows = [score_day(root, index, predictors) for index, (actor, _) in enumerate(ITEMS) if actor == case]
    gates = {name: strict_gate(rows, name) for name in MODELS}
    label = deployment.case_name(*deployment.CASES[case])
    output = root / "evaluation" / label
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(dict(case_index=case, days=rows, gate_pass=gates), indent=2) + "\n")
    exported = root / "evaluation_exports" / label
    exported.mkdir(parents=True, exist_ok=True)
    (exported / "done.json").write_text(json.dumps(dict(case_index=case, gate_pass=gates, validation_days=len(rows))) + "\n")
    print("V66_EVALUATE_DONE " + json.dumps(dict(case_index=case, gate_pass=gates)), flush=True)


def day_deltas(cases, left, right, metric):
    return [float(np.mean([next(d for d in case["days"] if d["scenario_seed"] == seed)[left][metric]
                           - next(d for d in case["days"] if d["scenario_seed"] == seed)[right][metric]
                           for case in cases])) for seed in VALIDATION_SEEDS]


def aggregate(root):
    cases = [json.loads((root / "evaluation" / deployment.case_name(*case) / "metrics.json").read_text())
             for case in deployment.CASES]
    rows = [day for case in cases for day in case["days"]]
    contrasts = dict(mlp_vs_prior=("shrunk_mlp", "large_prior"), mlp_shrinkage=("shrunk_mlp", "mlp_residual"),
                     linear_shrinkage=("shrunk_linear", "linear_residual"), linear_vs_prior=("shrunk_linear", "large_prior"),
                     mlp_vs_linear=("shrunk_mlp", "shrunk_linear"))
    coefficients = [json.loads(calibration_path(root, case).read_text()) for case in range(len(cases))]
    gates = {name: sum(case["gate_pass"][name] for case in cases) for name in MODELS}
    result = dict(status="whole_day_crossfit_shrinkage_complete", fresh_rollouts=len(rows), contexts=len(cases),
                  validation_demand_seeds=VALIDATION_SEEDS, heldout_decisions=sum(d["zero"]["count"] for d in rows),
                  oof_models=FOLDS * len(cases) * len(MODELS), steps_per_oof_model=warmup.FIT_STEPS, batch_size=warmup.FIT_BATCH,
                  coefficients=coefficients, production_value_updates=0,
                  strict_gate_cases=gates, action_audit_admitted={k: v == len(cases) for k, v in gates.items()},
                  control_admitted=False, actor_updates=0, dual_updates=0, main_policy_changed=False,
                  mean_metrics={name: {metric: float(np.mean([d[name][metric] for d in rows]))
                                       for metric in ("mae", "rmse", "bias")} for name in CHANNELS},
                  paired_effects={name: {metric: expanded.paired_interval(day_deltas(cases, *pair, metric), 66000 + i)
                                         for metric in ("mae", "rmse")} for i, (name, pair) in enumerate(contrasts.items())})
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V66_AGGREGATE " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "fold", "calibrate", "collection-smoke", "batch", "evaluate", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v62-root", type=Path)
    parser.add_argument("--v64-root", type=Path)
    parser.add_argument("--v65-root", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(deployment.CASES)), default=0)
    parser.add_argument("--fold", type=int, choices=range(FOLDS), default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    elif args.command == "calibrate":
        calibrate(args.out_root, args.v65_root, args.case_index)
    elif args.command in ("fold", "smoke"):
        if args.command == "smoke":
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                            str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v6[0-6]_*.py", "-v"], check=True)
        fit_fold(args.out_root, args.v59_root, args.v64_root, args.case_index, args.fold)
        if args.command == "smoke":
            output = args.out_root / "smoke_export"
            output.mkdir(exist_ok=True)
            (output / "done.json").write_text(json.dumps(dict(status="whole_day_crossfit_smoke_pass")) + "\n")
    elif args.command == "collection-smoke":
        collect(args.out_root, args.v59_root, 0)
        predictors = load_predictors(args.out_root, args.v59_root, args.v62_root, args.v64_root, 0)
        row = score_day(args.out_root, 0, predictors)
        output = args.out_root / "collection_smoke_export"
        output.mkdir(exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(status="locked_calibration_collection_smoke_pass",
                                                         scored_channels=len(CHANNELS), decisions=row["zero"]["count"])) + "\n")
    elif args.command == "evaluate":
        evaluate(args.out_root, args.v59_root, args.v62_root, args.v64_root, args.case_index)
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("range is outside the fixed 128 fresh full-day rollouts")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(collect, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                 range(args.start, args.end)))
        output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


if __name__ == "__main__":
    main()
