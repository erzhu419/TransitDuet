#!/usr/bin/env python3
"""Fixed-budget causal prior/residual value calibration; actors remain frozen."""

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

import run_freqduet_v64_independent_days as expanded


warmup, deployment = expanded.warmup, expanded.deployment
VALIDATION_SEEDS = tuple(range(750001, 750017))
ITEMS = list(itertools.product(range(len(deployment.CASES)), range(len(VALIDATION_SEEDS))))
CANDIDATES = ("linear_residual", "mlp_residual")
CHANNELS = (*CANDIDATES, *expanded.CHANNELS)


def specification(index):
    case, day = ITEMS[index]
    seed, actor_scenario = deployment.CASES[case]
    scenario, sampler = VALIDATION_SEEDS[day], 6500000 + case * 1000 + day
    return dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario,
                scenario_seed=scenario, sampler_seed=sampler, split="validation",
                label=f"{deployment.case_name(seed, actor_scenario)}_demand{scenario}_sample{sampler}")


def features(day, encoder):
    states = day["states"]
    expected = encoder["local_dim"] + 11 + 6 * encoder["buses"]
    if states.shape[1] != expected:
        raise ValueError("state array does not match the frozen system encoder")
    completed = states[:, encoder["local_dim"] + 8]
    return dict(day, states=np.column_stack((states, day["prefix"], day["prefix"] * completed)).astype(np.float32))


def training_statistics(days):
    # Match fit_cost's uniform day sampling, rather than weighting long days more.
    means, squares = [], []
    for day in days:
        values = np.column_stack((day["states"], day["actions"])).astype(float)
        means.append(values.mean(axis=0))
        squares.append(np.square(values).mean(axis=0))
    center = np.mean(means, axis=0)
    scale = np.sqrt(np.maximum(np.mean(squares, axis=0) - center ** 2, 0))
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


class ResidualValue(torch.nn.Module):
    def __init__(self, kind, center, scale, hidden):
        super().__init__()
        self.register_buffer("center", torch.as_tensor(center).clone())
        self.register_buffer("scale", torch.as_tensor(scale).clone())
        if kind == "linear_residual":
            self.net = torch.nn.Linear(len(center), 1)
            output = self.net
        elif kind == "mlp_residual":
            self.net = torch.nn.Sequential(torch.nn.Linear(len(center), hidden), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden, 1))
            output = self.net[-1]
        else:
            raise ValueError("unknown residual representation")
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)

    def forward(self, states, actions):
        return self.net((torch.cat((states, actions), dim=-1) - self.center) / self.scale)


def fit_models(models, days, prior, lr, seed, steps=warmup.FIT_STEPS, batch_size=warmup.FIT_BATCH):
    targets = [dict(day, returns=np.full_like(day["returns"], day["final_cv_squared"] - prior)) for day in days]
    return {name: warmup.fit_cost(model, targets, lr, seed, steps=steps, batch_size=batch_size)
            for name, model in models.items()}


def case_dir(root, case):
    return root / "fits" / deployment.case_name(*deployment.CASES[case])


def v64_models(v64_root, case):
    models, means = {}, {}
    for size in expanded.SIZES:
        path = expanded.fit_dir(v64_root, case, size)
        metadata = json.loads((path / "done.json").read_text())
        payload = torch.load(path / "value_pair.pt", map_location="cpu", weights_only=False)
        if payload["format"] != "freqduet-v64-data-size-representation":
            raise ValueError("V64 comparator has incompatible value semantics")
        models[f"{size}_direct"] = warmup.load_cost_model(payload["direct_remaining_q_net"]).eval()
        models[f"{size}_factored"] = warmup.load_cost_model(payload["terminal_cv_q_net"]).eval()
        means[size] = metadata["train_final_mean"]
    return models, means


def fit(root, v59_root, v64_root, case):
    destination = case_dir(root, case)
    exported = root / "fit_exports" / deployment.case_name(*deployment.CASES[case])
    if (exported / "done.json").exists():
        return
    deployment.joint.optimized_runtime(root)
    actor_source = v59_root / "cells" / deployment.case_name(*deployment.CASES[case]) / "learning_smoke.pt"
    source = torch.load(actor_source, map_location="cpu", weights_only=False)
    encoder = source["encoder"]
    days = [expanded.read_day(v64_root, i) for i in expanded.training_indices(case, "large")]
    prepared = [features(day, encoder) for day in days]
    prior = float(np.mean([d["final_cv_squared"] for d in days]))
    center, scale = training_statistics(prepared)
    hidden = source["trainer"]["cost_q_net"]["net.0.weight"].shape[0]
    seed, scenario = deployment.CASES[case]
    draw_seed = seed + scenario + 64000
    with torch.random.fork_rng():
        torch.manual_seed(seed + scenario + 65000)
        models = {name: ResidualValue(name, center, scale, hidden) for name in CANDIDATES}
    lr = source["trainer"]["cost_q_optimizer"]["param_groups"][0]["lr"]
    training = fit_models(models, prepared, prior, lr, draw_seed)
    destination.mkdir(parents=True, exist_ok=True)
    # Freeze weights and training-only transforms before any new-day evaluation.
    torch.save(dict(format="freqduet-v65-prior-residual", prior=prior, encoder=encoder, hidden=hidden,
                    models={name: model.state_dict() for name, model in models.items()},
                    actor_source=str(actor_source)), destination / "residual_values.pt")
    old, old_means = v64_models(v64_root, case)
    if not np.isclose(prior, old_means["large"], rtol=0, atol=1e-12):
        raise ValueError("V64 and V65 do not share the same fitting-day prior")
    train_mse = {name: [] for name in (*CANDIDATES, "large_direct", "large_factored", "large_prior")}
    for day, inputs in zip(days, prepared):
        predictions = {name: prior + warmup.predict(model, inputs) - day["prefix"] for name, model in models.items()}
        predictions.update(large_direct=warmup.predict(old["large_direct"], day),
                           large_factored=expanded.terminal.predict_remaining(old["large_factored"], day),
                           large_prior=prior - day["prefix"])
        for name, values in predictions.items():
            train_mse[name].append(float(np.square(values - day["returns"]).mean()))
    result = dict(case_index=case, training=training, prior=prior,
                  train_mse={name: float(np.mean(values)) for name, values in train_mse.items()},
                  parameters={name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
                  fitting_demand_seeds=[d["scenario_seed"] for d in days], fit_sampler_seed=draw_seed,
                  actor_updates=0, dual_updates=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (destination / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    exported.mkdir(parents=True, exist_ok=True)
    (exported / "done.json").write_text(json.dumps(dict(status="residual_models_frozen", case_index=case)) + "\n")
    print("V65_FIT_DONE " + json.dumps({k: v for k, v in result.items() if k != "fitting_demand_seeds"}), flush=True)


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
    print("V65_DATA_DONE " + json.dumps(spec), flush=True)
    return spec["label"]


def strict_gate(days, candidate):
    return all(day[candidate][metric] < day[baseline][metric]
               for day in days for baseline in expanded.CHANNELS for metric in ("mae", "rmse"))


def evaluate(root, v59_root, v62_root, v64_root, case):
    deployment.joint.optimized_runtime(root)
    payload = torch.load(case_dir(root, case) / "residual_values.pt", map_location="cpu", weights_only=False)
    if payload["format"] != "freqduet-v65-prior-residual":
        raise ValueError("residual predictor must not be loaded as a direct cost critic")
    models = {}
    for name in CANDIDATES:
        weights = payload["models"][name]
        model = ResidualValue(name, weights["center"], weights["scale"], payload["hidden"])
        model.load_state_dict(weights)
        models[name] = model.eval()
    controls, means = v64_models(v64_root, case)
    label = deployment.case_name(*deployment.CASES[case])
    frozen = torch.load(v59_root / "cells" / label / "learning_smoke.pt", map_location="cpu", weights_only=False)
    legacy = torch.load(expanded.terminal.case_dir(v62_root, case) / "cost_warmup.pt", map_location="cpu", weights_only=False)
    controls["frozen"] = warmup.load_cost_model(frozen["trainer"]["cost_q_net"]).eval()
    controls["legacy_direct"] = warmup.load_cost_model(legacy["cost_q_net"]).eval()
    rows = []
    for index, (actor, _) in enumerate(ITEMS):
        if actor != case:
            continue
        spec = specification(index)
        marker = json.loads((root / "ready" / f"{spec['label']}.json").read_text())
        with np.load(marker["dataset"]) as data:
            day = {key: data[key] for key in data.files}
        inputs = features(day, payload["encoder"])
        predictions = {name: payload["prior"] + warmup.predict(model, inputs) - day["prefix"]
                       for name, model in models.items()}
        predictions.update({name: (expanded.terminal.predict_remaining(model, day) if name.endswith("_factored")
                                   else warmup.predict(model, day)) for name, model in controls.items()})
        predictions.update(zero=np.zeros_like(day["returns"]), small_prior=means["small"] - day["prefix"],
                           large_prior=means["large"] - day["prefix"])
        rows.append(dict(scenario_seed=spec["scenario_seed"], sampler_seed=spec["sampler_seed"],
                         **{name: warmup.calibration_metrics(predictions[name], day["returns"]) for name in CHANNELS}))
    gates = {name: strict_gate(rows, name) for name in CANDIDATES}
    output = root / "evaluation" / label
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(dict(case_index=case, days=rows, gate_pass=gates), indent=2) + "\n")
    exported = root / "evaluation_exports" / label
    exported.mkdir(parents=True, exist_ok=True)
    (exported / "done.json").write_text(json.dumps(dict(case_index=case, gate_pass=gates, validation_days=len(rows))) + "\n")
    print("V65_EVALUATE_DONE " + json.dumps(dict(case_index=case, gate_pass=gates)), flush=True)


def day_deltas(cases, left, right, metric):
    return [float(np.mean([next(d for d in case["days"] if d["scenario_seed"] == seed)[left][metric]
                           - next(d for d in case["days"] if d["scenario_seed"] == seed)[right][metric]
                           for case in cases])) for seed in VALIDATION_SEEDS]


def aggregate(root):
    cases = [json.loads((root / "evaluation" / deployment.case_name(*case) / "metrics.json").read_text())
             for case in deployment.CASES]
    rows = [day for case in cases for day in case["days"]]
    contrasts = dict(linear_vs_prior=("linear_residual", "large_prior"),
                     linear_vs_direct=("linear_residual", "large_direct"),
                     linear_vs_factored=("linear_residual", "large_factored"),
                     mlp_vs_prior=("mlp_residual", "large_prior"),
                     mlp_vs_linear=("mlp_residual", "linear_residual"))
    fits = [json.loads((case_dir(root, i) / "done.json").read_text()) for i in range(len(cases))]
    gates = {name: sum(case["gate_pass"][name] for case in cases) for name in CANDIDATES}
    result = dict(status="prior_residual_calibration_complete", fresh_rollouts=len(rows), contexts=len(cases),
                  validation_demand_seeds=VALIDATION_SEEDS, heldout_decisions=sum(d["zero"]["count"] for d in rows),
                  steps_per_model=warmup.FIT_STEPS, batch_size=warmup.FIT_BATCH,
                  strict_gate_cases=gates, action_audit_admitted={k: v == len(cases) for k, v in gates.items()},
                  control_admitted=False, actor_updates=0, dual_updates=0, main_policy_changed=False,
                  parameters=[f["parameters"] for f in fits],
                  train_mse={name: float(np.mean([f["train_mse"][name] for f in fits])) for name in fits[0]["train_mse"]},
                  mean_metrics={name: {metric: float(np.mean([d[name][metric] for d in rows]))
                                       for metric in ("mae", "rmse", "bias")} for name in CHANNELS},
                  paired_effects={name: {metric: expanded.paired_interval(day_deltas(cases, *pair, metric), 65000 + i)
                                         for metric in ("mae", "rmse")} for i, (name, pair) in enumerate(contrasts.items())})
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V65_AGGREGATE " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "fit", "batch", "evaluate", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v62-root", type=Path)
    parser.add_argument("--v64-root", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(deployment.CASES)), default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    elif args.command in ("fit", "smoke"):
        if args.command == "smoke":
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                            str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v6[0-5]_*.py", "-v"], check=True)
        fit(args.out_root, args.v59_root, args.v64_root, args.case_index)
        if args.command == "smoke":
            collect(args.out_root, args.v59_root, 0)
            output = args.out_root / "smoke_export"
            output.mkdir(exist_ok=True)
            (output / "done.json").write_text(json.dumps(dict(status="prior_residual_fit_collection_smoke_pass")) + "\n")
    elif args.command == "evaluate":
        evaluate(args.out_root, args.v59_root, args.v62_root, args.v64_root, args.case_index)
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("range is outside the fixed 64 fresh full-day rollouts")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(collect, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                 range(args.start, args.end)))
        output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


if __name__ == "__main__":
    main()
