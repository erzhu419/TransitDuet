#!/usr/bin/env python3
"""Analytic completed-prefix subtraction with a frozen-actor terminal CV predictor."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import torch

import run_freqduet_v62_cost_warmup as warmup


deployment = warmup.deployment
VALIDATION_SEEDS = (45207, 45307, 934211, 934311)
ITEMS = list(itertools.product(range(len(deployment.CASES)), VALIDATION_SEEDS))
BASELINES = ("direct_v62", "frozen", "zero", "causal_final_mean")


def specification(index):
    case, scenario = ITEMS[index]
    seed, actor_scenario = deployment.CASES[case]
    return dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario,
                scenario_seed=scenario, rep=2, split="validation",
                label=f"{deployment.case_name(seed, actor_scenario)}_demand{scenario}_r2")


def terminal_targets(days):
    return [dict(day, returns=np.full_like(day["returns"], day["final_cv_squared"])) for day in days]


def predict_remaining(model, day):
    return warmup.predict(model, day).astype(float) - day["prefix"]


def calibration_gate(days):
    return all(day["factored"][metric] < day[baseline][metric]
               for day in days for baseline in BASELINES for metric in ("mae", "rmse"))


def case_dir(root, case):
    return root / "fits" / deployment.case_name(*deployment.CASES[case])


def fit(root, v59_root, v62_root, case):
    destination = case_dir(root, case)
    if (destination / "done.json").exists():
        return
    deployment.joint.optimized_runtime(root)
    seed, scenario = deployment.CASES[case]
    cell = v59_root / "cells" / deployment.case_name(seed, scenario)
    checkpoint = torch.load(cell / "learning_smoke.pt", map_location="cpu", weights_only=False)
    model = warmup.load_cost_model(checkpoint["trainer"]["cost_q_net"])
    train = warmup.load_days(v62_root, case, "train")
    direct_fit = json.loads((case_dir(v62_root, case) / "done.json").read_text())
    lr = checkpoint["trainer"]["cost_q_optimizer"]["param_groups"][0]["lr"]
    if (direct_fit["training"]["steps"], direct_fit["training"]["batch_size"],
            direct_fit["training"]["learning_rate"], direct_fit["training_days"]) != (
            warmup.FIT_STEPS, warmup.FIT_BATCH, lr, [day["label"] for day in train]):
        raise ValueError("direct-return comparator has a different fitting protocol")
    # Keep V62's batch-draw seed: only the target/known-prefix representation changes.
    training = warmup.fit_cost(model, terminal_targets(train), lr, seed + scenario + 62000)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format="freqduet-v63-terminal-cv-factorization",
                    terminal_cv_q_net=model.state_dict(),
                    remaining_cost="terminal_prediction_minus_exact_completed_prefix_cv_squared",
                    actor_source=str(cell / "learning_smoke.pt"), **training), destination / "terminal_value.pt")
    result = dict(case_index=case, training=training, training_days=[d["label"] for d in train],
                  train_final_cv_squared_mean=float(np.mean([d["final_cv_squared"] for d in train])),
                  actor_updates=0, dual_updates=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (destination / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V63_FIT_DONE " + json.dumps(result), flush=True)


def collect(root, v59_root, index):
    spec = specification(index)
    marker = root / "ready" / f"{spec['label']}.json"
    if marker.exists():
        return spec["label"]
    original = deployment.ITEMS.index((spec["case_index"], ("updated", "stochastic", spec["rep"])))
    audit = warmup.FullDayData(root, spec, None)
    label = deployment.collect(root, v59_root, None, original, decision_audit=audit,
                               scenario_seed=spec["scenario_seed"],
                               private_sampler_seed=deployment.sampler_seed(spec["train_seed"], spec["rep"]))
    row = json.loads((root / "ready" / f"{label}.json").read_text())
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(dict(**spec, **row["decision_audit"], final_cv_squared=row["cv_squared"],
                                      training_updates=0), indent=2) + "\n")
    print("V63_DATA_DONE " + json.dumps(spec), flush=True)
    return spec["label"]


def action_spread(model, day):
    spreads = []
    with torch.no_grad():
        for start in range(0, len(day["returns"]), 512):
            states = torch.from_numpy(day["states"][start:start + 512])
            repeated = states.repeat_interleave(len(deployment.joint.ACTIONS), dim=0)
            actions = torch.tensor(deployment.joint.ACTIONS, dtype=torch.float32).repeat(len(states))[:, None]
            values = model(repeated, actions).reshape(len(states), -1)
            spreads.extend((values.max(dim=1).values - values.min(dim=1).values).tolist())
    return dict(mean=float(np.mean(spreads)), p95=float(np.quantile(spreads, .95)),
                max=float(np.max(spreads)), counterfactual_truth=False)


def evaluate(root, v59_root, v62_root, case):
    deployment.joint.optimized_runtime(root)
    fitted = json.loads((case_dir(root, case) / "done.json").read_text())
    payload = torch.load(case_dir(root, case) / "terminal_value.pt", map_location="cpu", weights_only=False)
    if payload["format"] != "freqduet-v63-terminal-cv-factorization":
        raise ValueError("terminal predictor must not be loaded as a direct remaining-cost critic")
    model = warmup.load_cost_model(payload["terminal_cv_q_net"]).eval()
    direct = torch.load(case_dir(v62_root, case) / "cost_warmup.pt", map_location="cpu", weights_only=False)
    direct_model = warmup.load_cost_model(direct["cost_q_net"]).eval()
    seed, scenario = deployment.CASES[case]
    original = torch.load(v59_root / "cells" / deployment.case_name(seed, scenario) / "learning_smoke.pt",
                          map_location="cpu", weights_only=False)
    frozen_model = warmup.load_cost_model(original["trainer"]["cost_q_net"]).eval()
    days = []
    for index, (actor, _) in enumerate(ITEMS):
        if actor != case:
            continue
        spec = specification(index)
        marker = json.loads((root / "ready" / f"{spec['label']}.json").read_text())
        with np.load(marker["dataset"]) as source:
            day = {name: source[name] for name in source.files}
        truth = day["returns"]
        predictions = dict(factored=predict_remaining(model, day), direct_v62=warmup.predict(direct_model, day),
                           frozen=warmup.predict(frozen_model, day), zero=np.zeros_like(truth),
                           causal_final_mean=fitted["train_final_cv_squared_mean"] - day["prefix"])
        days.append(dict(label=spec["label"],
                         **{name: warmup.calibration_metrics(values, truth) for name, values in predictions.items()},
                         action_spread=action_spread(model, day)))
    result = dict(case_index=case, validation=days, gate_pass=calibration_gate(days),
                  actor_updates=0, dual_updates=0, main_policy_changed=False)
    output = root / "evaluation" / deployment.case_name(seed, scenario)
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V63_EVALUATE_DONE " + json.dumps({k: v for k, v in result.items() if k != "validation"}), flush=True)


def aggregate(root):
    cases = [json.loads((root / "evaluation" / deployment.case_name(*case) / "done.json").read_text())
             for case in deployment.CASES]
    days = [day for case in cases for day in case["validation"]]
    result = dict(status="terminal_value_factorization_complete", fresh_rollouts=len(days),
                  validation_demand_seeds=VALIDATION_SEEDS, contexts=len(cases),
                  decisions=sum(d["factored"]["count"] for d in days),
                  passed_cases=sum(c["gate_pass"] for c in cases),
                  action_audit_admitted=all(c["gate_pass"] for c in cases), control_admitted=False,
                  actor_updates=0, dual_updates=0, main_policy_changed=False,
                  mean_metrics={name: {metric: float(np.mean([d[name][metric] for d in days]))
                                       for metric in ("mae", "rmse", "bias")}
                                for name in ("factored", *BASELINES)}, cases=cases)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V63_AGGREGATE " + json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "fit", "batch", "evaluate", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v62-root", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(deployment.CASES)), default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    elif args.command in ("smoke", "fit"):
        case = 0 if args.command == "smoke" else args.case_index
        fit(args.out_root, args.v59_root, args.v62_root, case)
        if args.command == "smoke":
            output = args.out_root / "smoke_export"
            output.mkdir(exist_ok=True)
            (output / "done.json").write_text(json.dumps(dict(status="matched_terminal_fit_smoke_pass")) + "\n")
    elif args.command == "evaluate":
        evaluate(args.out_root, args.v59_root, args.v62_root, args.case_index)
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("range is outside the fixed 16 fresh full-day rollouts")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(collect, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                 range(args.start, args.end)))
        output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


if __name__ == "__main__":
    main()
