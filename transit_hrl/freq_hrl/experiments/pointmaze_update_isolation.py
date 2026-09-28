"""Isolate PPO updates while retaining the native joint-renewal control loop."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import load_controller, raw_directory, write_json
from scripts import pointmaze_update_isolation_stage36_spec as spec


def update_components(model, batch, components):
    metrics = {}
    for level in components:
        metrics.update(model._update_level(
            level=level, batch=getattr(batch, level), actor=getattr(model, level + "_actor"),
            value_net=getattr(model, level + "_value"),
            actor_optimizer=getattr(model, level + "_actor_optimizer"),
            value_optimizer=getattr(model, level + "_value_optimizer")))
    return metrics


def change_norms(weights, initial):
    return {name: float(np.sqrt(sum(float(torch.sum((state[key] - initial[name][key]) ** 2))
                                   for key in state))) for name, state in weights.items()}


def train(root, method, *, preflight, output, specification=spec):
    spec = specification
    native = spec.native_method(method)
    args = spec.source.arguments(root, preflight=preflight)
    opt, roles = spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    controller, source_cell, replay = load_controller(args, spec.source_result(root, preflight=preflight))
    model = joint.make_model(controller, native, root=root)
    if model.upper_cost_value is not None or model.lower_cost_value is not None or model.hf_actor is not None:
        raise ValueError("Stage-36 requires the registered unconstrained Stage-33 controller")
    initial = joint.inference_weights(model)
    raw = raw_directory(output)
    history, costs, updates, evaluation = [], [], {}, {}
    best_rank, best_iteration = None, None
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
                             initializer=joint.init_worker, initargs=(model.config, args, native)) as pool:
        def episodes(seeds, *, phase, sample=False):
            capture = phase in spec.COHORTS
            directory = raw / phase
            if capture:
                directory.mkdir(exist_ok=True)
            weights = joint.inference_weights(model)
            jobs = [(weights, seed, sample, str(directory / f"episode_{seed}.npz") if capture else None) for seed in seeds]
            pairs = list(pool.map(joint.worker_rollout, jobs))
            costs.extend({"phase": phase, **{k: row[k] for k in
                          ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}} for _, row in pairs)
            return pairs

        def select(iteration):
            nonlocal best_rank, best_iteration
            rows = [row for _, row in episodes(roles["selection"], phase="selection")]
            rank = (float(np.mean([row["charged_utility"] for row in rows])),
                    -float(np.mean([row["tracking_squared_error_integral"] for row in rows])))
            history.append({"iteration": iteration, "utility": rank[0], "ise": -rank[1],
                            "return": float(np.mean([row["episode_return"] for row in rows])),
                            "calls": float(np.mean([row["upper_inference_calls"] for row in rows]))})
            if best_rank is None or rank > best_rank:
                best_rank, best_iteration = rank, iteration
                torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "method": method,
                            "iteration": iteration, "state_dict": model.state_dict()}, raw / "selected.pt")
            print(f"root {root} {method}: selection iteration {iteration}; elapsed {time.monotonic() - started:.1f}s", flush=True)

        select(0)
        for iteration in range(1, opt["iterations"] + 1):
            offset = (iteration - 1) * opt["rollouts_per_iteration"]
            pairs = episodes(roles["training"][offset:offset + opt["rollouts_per_iteration"]], phase="train", sample=True)
            np.random.seed(np.random.SeedSequence([spec.SHUFFLE_SEED_NAMESPACE, root, iteration]).generate_state(1)[0])
            metrics = update_components(model, concat_hierarchical_batches([batch for batch, _ in pairs]), spec.COMPONENTS[method])
            for key, value in metrics.items():
                if "optimizer_steps" in key:
                    updates[key] = updates.get(key, 0) + int(value)
            if iteration % opt["selection_interval"] == 0:
                select(iteration)
        trained_changes = change_norms(joint.inference_weights(model), initial)
        torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "method": method,
                    "iteration": opt["iterations"], "state_dict": model.state_dict()}, raw / "final.pt")
        evaluation["final"] = [row for _, row in episodes(roles["evaluation"], phase="final")]
        selected = torch.load(raw / "selected.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["state_dict"])
        selected_changes = change_norms(joint.inference_weights(model), initial)
        evaluation["selected"] = [row for _, row in episodes(roles["evaluation"], phase="selected")]
    totals = {phase: {key: sum(row[key] for row in costs if row["phase"] == phase)
                      for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}
              for phase in ("train", "selection", *spec.COHORTS)}
    totals["factual_replay"] = {"upper_inference_calls": len(source_cell["factual_row"]["decision_steps"]),
                                "lower_inference_calls": args.horizon, "gate_inference_calls": 0}
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
              "root": root, "method": method, "preflight": preflight, "options": opt, "seed_roles": roles,
              "budget": spec.budget(preflight=preflight), "inference_counts": totals,
              "source_selected_iteration": source_cell["selected_checkpoint_iteration"], "source_replay": replay,
              "checkpoint": str(raw / "selected.pt"), "final_checkpoint": str(raw / "final.pt"),
              "selected_iteration": best_iteration, "selection_history": history, "optimizer_steps": updates,
              "trained_parameter_change_norms": trained_changes, "selected_parameter_change_norms": selected_changes,
              "evaluation_rows": evaluation, "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def audit_result(result, *, raw_path, specification=spec):
    spec = specification
    root, method, preflight = result["root"], result["method"], result["preflight"]
    native = spec.native_method(method)
    if (result["status"] != "complete" or result["protocol"] != spec.EXPERIMENT_PROTOCOL
            or result["contract"] != spec.contract() or result["options"] != spec.options(preflight=preflight)
            or result["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or result["budget"] != spec.budget(preflight=preflight)):
        raise ValueError("update-isolation result violates the frozen protocol")
    args, opt = spec.source.arguments(root, preflight=preflight), result["options"]
    for phase, expected in (("train", result["budget"]["training_primitive_steps"]),
                            ("selection", result["budget"]["selection_primitive_steps"]),
                            ("final", opt["evaluation_paths"] * args.horizon),
                            ("selected", opt["evaluation_paths"] * args.horizon), ("factual_replay", args.horizon)):
        if result["inference_counts"][phase]["lower_inference_calls"] != expected:
            raise ValueError("component-isolation primitive accounting changed")
    levels = ("upper", "lower") if method == "fixed50" else ("upper", "lower", "promotion")
    for level in levels:
        enabled = level in spec.COMPONENTS[method]
        for suffix in ("actor", "value"):
            name = level + "_" + suffix
            count = result["optimizer_steps"].get(name + "_optimizer_steps", 0)
            delta = result["trained_parameter_change_norms"][name]
            if (enabled and (count < 1 or delta <= 0)) or (not enabled and (count != 0 or delta != 0)):
                raise ValueError("component freeze/update contract violated")
            if not enabled and result["selected_parameter_change_norms"][name] != 0:
                raise ValueError("selected frozen component changed")
    if [row["iteration"] for row in result["selection_history"]] != [0, *range(opt["selection_interval"], opt["iterations"] + 1, opt["selection_interval"])]:
        raise ValueError("checkpoint selection budget changed")
    best = max(result["selection_history"], key=lambda row: (row["utility"], -row["ise"]))
    if result["selected_iteration"] != best["iteration"] or set(result["evaluation_rows"]) != set(spec.COHORTS):
        raise ValueError("checkpoint selection or evaluation cohorts changed")
    for cohort in spec.COHORTS:
        rows = result["evaluation_rows"][cohort]
        if [row["seed"] for row in rows] != result["seed_roles"]["evaluation"]:
            raise ValueError("component-isolation evaluation roster incomplete")
        joint.audit_trajectories(rows, args=args, method=native, raw_path=Path(raw_path) / cohort)
        for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
            if sum(row[key] for row in rows) != result["inference_counts"][cohort][key]:
                raise ValueError("evaluation inference accounting changed")
    return {"status": "passed", "root": root, "method": method,
            "episodes": sum(len(rows) for rows in result["evaluation_rows"].values()),
            "checks": "frozen_components_native_metrics_causal_gate_execution_and_accounting"}


def aggregate(results, *, specification=spec):
    spec = specification
    cells = {(r["root"], r["method"]): r for r in results}
    if len(results) != len(spec.OPTIMIZER_ROOTS) * len(spec.METHODS) or set(cells) != {
            (root, method) for root in spec.OPTIMIZER_ROOTS for method in spec.METHODS}:
        raise ValueError("full component-isolation root/method roster incomplete")
    keys = ("episode_return", "tracking_squared_error_integral", "upper_inference_calls", "charged_utility")
    roots = []
    for root in spec.OPTIMIZER_ROOTS:
        means = {cohort: {m: {k: float(np.mean([row[k] for row in cells[root, m]["evaluation_rows"][cohort]]))
                              for k in keys} for m in spec.METHODS} for cohort in spec.COHORTS}
        vector = spec.contrasts(means["final"])
        roots.append({"root": root, "means": means, "endpoints": dict(zip(spec.ENDPOINTS, vector))})
    x = np.asarray([[row["endpoints"][k] for k in spec.ENDPOINTS] for row in roots])
    rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
    draws = x[rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))].mean(axis=1)
    tail = .05 / (2 * spec.CI_FAMILY_SIZE)
    bounds = np.quantile(draws, [tail, 1 - tail], axis=0)
    endpoints = {name: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
                        "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
                 for i, name in enumerate(spec.ENDPOINTS)}
    means = {cohort: {m: {k: float(np.mean([r["means"][cohort][m][k] for r in roots])) for k in keys}
                      for m in spec.METHODS} for cohort in spec.COHORTS}
    return {"status": spec.AGGREGATE_STATUS, "primary_cohort": "final",
            "root_count": len(roots), "primary_endpoints": endpoints, "means": means, "root_rows": roots}
