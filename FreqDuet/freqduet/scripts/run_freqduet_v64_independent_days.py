#!/usr/bin/env python3
"""Independent-demand-day quantity x value-representation experiment."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import torch

import run_freqduet_v63_terminal_value as terminal


warmup, deployment = terminal.warmup, terminal.deployment
TRAIN_SEEDS = tuple(range(640001, 640033))
VALIDATION_SEEDS = tuple(range(740001, 740009))
SMALL_DAYS = 4
DAYS = tuple(("train", s) for s in TRAIN_SEEDS) + tuple(("validation", s) for s in VALIDATION_SEEDS)
ITEMS = list(itertools.product(range(len(deployment.CASES)), range(len(DAYS))))
SIZES, REPRESENTATIONS = ("small", "large"), ("direct", "factored")
BASELINES = ("zero", "small_prior", "large_prior", "legacy_direct", "frozen")
CHANNELS = tuple(f"{size}_{rep}" for size in SIZES for rep in REPRESENTATIONS) + BASELINES
GATE_BASELINES = ("small_direct", "small_factored", *BASELINES)


def specification(index):
    case, day = ITEMS[index]
    seed, actor_scenario = deployment.CASES[case]
    split, scenario = DAYS[day]
    sample_seed = 6400000 + case * 1000 + day
    return dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario,
                scenario_seed=scenario, sampler_seed=sample_seed, split=split,
                label=f"{deployment.case_name(seed, actor_scenario)}_demand{scenario}_sample{sample_seed}")


def training_indices(case, size):
    count = SMALL_DAYS if size == "small" else len(TRAIN_SEEDS)
    return [ITEMS.index((case, day)) for day in range(count)]


def read_day(root, index):
    spec = specification(index)
    marker = json.loads((root / "ready" / f"{spec['label']}.json").read_text())
    with np.load(marker["dataset"]) as source:
        return dict(label=spec["label"], scenario_seed=spec["scenario_seed"],
                    sampler_seed=spec["sampler_seed"], **{key: source[key] for key in source.files})


def fit_pair(models, days, lr, seed, steps=warmup.FIT_STEPS, batch_size=warmup.FIT_BATCH):
    targets = dict(direct=days, factored=terminal.terminal_targets(days))
    return {rep: warmup.fit_cost(models[rep], targets[rep], lr, seed, steps=steps, batch_size=batch_size)
            for rep in REPRESENTATIONS}


def fit_dir(root, case, size):
    return root / "fits" / deployment.case_name(*deployment.CASES[case]) / size


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
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(dict(**spec, **row["decision_audit"], actor_updates=0, dual_updates=0,
                                      diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")), indent=2) + "\n")
    print("V64_DATA_DONE " + json.dumps(spec), flush=True)
    return spec["label"]


def fit(root, v59_root, case, size):
    destination = fit_dir(root, case, size)
    output = root / "fit_exports" / deployment.case_name(*deployment.CASES[case]) / size
    if (output / "done.json").exists():
        return
    deployment.joint.optimized_runtime(root)
    seed, scenario = deployment.CASES[case]
    source = v59_root / "cells" / deployment.case_name(seed, scenario) / "learning_smoke.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    weights = checkpoint["trainer"]["cost_q_net"]
    models = {rep: warmup.load_cost_model(weights) for rep in REPRESENTATIONS}
    days = [read_day(root, i) for i in training_indices(case, size)]
    lr = checkpoint["trainer"]["cost_q_optimizer"]["param_groups"][0]["lr"]
    sample_seed = seed + scenario + 64000
    training = fit_pair(models, days, lr, sample_seed)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format="freqduet-v64-data-size-representation",
                    direct_remaining_q_net=models["direct"].state_dict(),
                    terminal_cv_q_net=models["factored"].state_dict(), actor_source=str(source)),
               destination / "value_pair.pt")
    result = dict(case_index=case, size=size, training=training, fit_sampler_seed=sample_seed,
                  training_demand_seeds=[d["scenario_seed"] for d in days],
                  policy_sampler_seeds=[d["sampler_seed"] for d in days],
                  train_final_mean=float(np.mean([d["final_cv_squared"] for d in days])),
                  actor_updates=0, dual_updates=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (destination / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(status="paired_models_fit", case_index=case, size=size)) + "\n")
    print("V64_FIT_DONE " + json.dumps({k: v for k, v in result.items()
                                      if k not in ("training_demand_seeds", "policy_sampler_seeds")}), flush=True)


def strict_gate(days, representation):
    channel = f"large_{representation}"
    return all(day[channel][metric] < day[baseline][metric]
               for day in days for baseline in GATE_BASELINES for metric in ("mae", "rmse"))


def evaluate(root, v59_root, v62_root, case):
    deployment.joint.optimized_runtime(root)
    seed, scenario = deployment.CASES[case]
    label = deployment.case_name(seed, scenario)
    models, means = {}, {}
    for size in SIZES:
        path = fit_dir(root, case, size)
        fitted = json.loads((path / "done.json").read_text())
        payload = torch.load(path / "value_pair.pt", map_location="cpu", weights_only=False)
        if payload["format"] != "freqduet-v64-data-size-representation":
            raise ValueError("value pair has incompatible terminal/direct semantics")
        models[f"{size}_direct"] = warmup.load_cost_model(payload["direct_remaining_q_net"]).eval()
        models[f"{size}_factored"] = warmup.load_cost_model(payload["terminal_cv_q_net"]).eval()
        means[size] = fitted["train_final_mean"]
    original = torch.load(v59_root / "cells" / label / "learning_smoke.pt", map_location="cpu", weights_only=False)
    legacy = torch.load(terminal.case_dir(v62_root, case) / "cost_warmup.pt", map_location="cpu", weights_only=False)
    models["frozen"] = warmup.load_cost_model(original["trainer"]["cost_q_net"]).eval()
    models["legacy_direct"] = warmup.load_cost_model(legacy["cost_q_net"]).eval()
    rows = []
    for index, (actor, day_id) in enumerate(ITEMS):
        if actor != case or DAYS[day_id][0] != "validation":
            continue
        day = read_day(root, index)
        truth = day["returns"]
        predictions = {name: (terminal.predict_remaining(model, day) if name.endswith("_factored")
                              else warmup.predict(model, day)) for name, model in models.items()}
        predictions.update(zero=np.zeros_like(truth), small_prior=means["small"] - day["prefix"],
                           large_prior=means["large"] - day["prefix"])
        rows.append(dict(scenario_seed=day["scenario_seed"], sampler_seed=day["sampler_seed"],
                         **{name: warmup.calibration_metrics(predictions[name], truth) for name in CHANNELS}))
    gates = {rep: strict_gate(rows, rep) for rep in REPRESENTATIONS}
    result = dict(case_index=case, days=rows, gate_pass=gates, actor_updates=0, dual_updates=0,
                  main_policy_changed=False)
    output = root / "evaluation" / label
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    exported = root / "evaluation_exports" / label
    exported.mkdir(parents=True, exist_ok=True)
    (exported / "done.json").write_text(json.dumps(dict(case_index=case, gate_pass=gates, validation_days=len(rows))) + "\n")
    print("V64_EVALUATE_DONE " + json.dumps(dict(case_index=case, gate_pass=gates)), flush=True)


def day_deltas(cases, left, right, metric):
    return [float(np.mean([next(d for d in case["days"] if d["scenario_seed"] == seed)[left][metric]
                           - next(d for d in case["days"] if d["scenario_seed"] == seed)[right][metric]
                           for case in cases])) for seed in VALIDATION_SEEDS]


def paired_interval(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return dict(n_demand_days=len(values), mean=float(values.mean()),
                ci95=np.quantile(samples, [.025, .975]).tolist(), day_deltas=values.tolist())


def aggregate(root):
    cases = [json.loads((root / "evaluation" / deployment.case_name(*case) / "metrics.json").read_text())
             for case in deployment.CASES]
    rows = [day for case in cases for day in case["days"]]
    contrasts = dict(data_direct=("large_direct", "small_direct"),
                     data_factored=("large_factored", "small_factored"),
                     representation_small=("small_factored", "small_direct"),
                     representation_large=("large_factored", "large_direct"))
    effects = {name: {metric: paired_interval(day_deltas(cases, *pair, metric), 64000 + i)
                      for metric in ("mae", "rmse")} for i, (name, pair) in enumerate(contrasts.items())}
    gates = {rep: sum(case["gate_pass"][rep] for case in cases) for rep in REPRESENTATIONS}
    collected = [json.loads((root / "ready" / f"{specification(i)['label']}.json").read_text())
                 for i in range(len(ITEMS))]
    result = dict(status="independent_day_factorial_complete", collection_rollouts=len(collected),
                  collected_decisions=sum(d["decisions"] for d in collected), heldout_decisions=sum(d["zero"]["count"] for d in rows),
                  training_demand_days=dict(small=SMALL_DAYS, large=len(TRAIN_SEEDS)), validation_demand_seeds=VALIDATION_SEEDS,
                  contexts=len(cases), steps_per_model=warmup.FIT_STEPS, batch_size=warmup.FIT_BATCH,
                  strict_gate_cases=gates, action_audit_admitted={rep: gates[rep] == len(cases) for rep in REPRESENTATIONS},
                  control_admitted=False, actor_updates=0, dual_updates=0, main_policy_changed=False,
                  mean_metrics={name: {metric: float(np.mean([day[name][metric] for day in rows]))
                                       for metric in ("mae", "rmse", "bias")} for name in CHANNELS},
                  paired_effects=effects)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V64_AGGREGATE " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "fit", "evaluate", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v62-root", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(deployment.CASES)), default=0)
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    elif args.command == "fit":
        fit(args.out_root, args.v59_root, args.case_index, args.size)
    elif args.command == "evaluate":
        evaluate(args.out_root, args.v59_root, args.v62_root, args.case_index)
    elif args.command == "smoke":
        collect(args.out_root, args.v59_root, 0)
        output = args.out_root / "smoke_export"
        output.mkdir(exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(status="independent_day_collection_smoke_pass")) + "\n")
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("range is outside the fixed 160 independent-day rollouts")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(collect, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                 range(args.start, args.end)))
        output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


if __name__ == "__main__":
    main()
