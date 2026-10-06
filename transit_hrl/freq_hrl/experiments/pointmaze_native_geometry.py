"""Compare native mean directions with matched full-policy calibration."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch
from torch import nn

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from freq_hrl.rl.native_return_step import fit_native_return_step
from . import pointmaze_policy_curvature as source
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_native_geometry_stage126_spec as spec

joint = source.joint


def natural_candidates(actor, states, labels):
    if len(actor.net) != 1 or not isinstance(actor.net[0], nn.Linear):
        raise ValueError("Stage126 requires the existing linear Gaussian upper mean")
    signals = {panel: np.asarray([r["gradients"][panel] for r in labels]) for panel in spec.PANELS}
    directions, geometry = native_mean_directions(states, signals, actor.log_std.detach().exp().numpy(), damping=spec.DAMPING)
    vectors = {}
    for panel, direction in directions.items():
        mapped = {"net.0.weight": direction["weight"], "net.0.bias": direction["bias"]}
        vectors[panel] = np.concatenate([mapped[name].ravel() if name in mapped else np.zeros(p.numel())
            for name, p in actor.named_parameters()])
    candidates, radius, cost = matched_perturbations(actor, np.asarray(states, dtype=np.float32),
        -np.mean(list(vectors.values()), axis=0), delta=spec.FISHER_RADIUS, chunk_size=1024)
    geometry.update(preconditioned_panel_cosine=cosine(vectors["A"], vectors["B"]), radius=radius)
    return {k: a.state_dict() for k, a in candidates.items()}, geometry, cost


def direction_summary(actor, upper, states, labels):
    actor.load_state_dict(upper)
    with torch.no_grad():
        mean = actor.distribution(torch.as_tensor(np.asarray(states, dtype=np.float32))).mean.double().numpy()
    signal = np.mean([np.asarray([r["gradients"][p] for r in labels]) for p in spec.PANELS], axis=0)
    alignment = np.sum(mean * signal, axis=1)
    return {"query_first_order_gains": alignment.tolist(), "mean_first_order_gain": float(alignment.mean()),
        "action_mean_rms": float(np.sqrt(np.square(mean).mean())),
        "action_state_variation_rms": float(np.sqrt(np.square(mean - mean.mean(0)).mean()))}


def fit_method(rows, method):
    return fit_native_return_step(*[[r[k]["episode_return"] for r in rows]
        for k in ("zero", method + "_plus", method + "_minus")])


def evaluation_summary(rows):
    metrics = ("episode_return", "reference_correction_rms", "reference_correction_peak", "plan_delta_rms", "upper_mean_rms")
    means = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in metrics} for v in spec.VARIANTS}
    effects = {}
    for a, b in spec.CONTRASTS:
        difference = [r[a]["episode_return"] - r[b]["episode_return"] for r in rows]
        effects[f"{a}_minus_{b}"] = {"mean": float(np.mean(difference)), "paired_differences": difference}
    return {"mean_metrics": means, "effects": effects}


def run(root, output):
    output = output.resolve()
    cached = json.loads(spec.label_result(root).read_text())
    labels_spec = spec.source.source.source
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"]) != (
            "complete", labels_spec.PROTOCOL, root, labels_spec.budget()):
        raise ValueError("Stage126 requires completed Stage123 native labels")
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, cost, groups = spec.arguments(root), dict.fromkeys(spec.budget(), 0), {}
    cost["label_cache_loads"] = 1
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            p, model = str(period), models[str(period)]
            before = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            initial = copy.deepcopy(trainer.upper_actor.state_dict())
            labels, envelope = cached["groups"][p]["queries"], calibrations[p]["envelope"]
            if [r["query"] for r in labels] != labels_spec.queries(root):
                raise ValueError("Stage126 query roster changed")
            jobs = [(joint.weights(model), teacher, r["query"], r["coordinate_suffix_returns"]["A"]["zero"],
                period, predictor, envelope) for r in labels]
            replay = list(pool.map(source.source.replay_query, jobs))
            for row in replay:
                cost["training_state_replays"] += 1
                for key in ("native_episodes", "native_steps", "native_donor_response_calls", "native_upper_calls"):
                    cost[key] += row[key]
            states = [r["state"] for r in replay]
            directions = {"euclidean": source.load_directions(root, period)}
            cost["upper_checkpoint_loads"] += 2
            directions["natural"], geometry, fit_cost = natural_candidates(trainer.upper_actor, states, labels)
            cost["empirical_fisher_solves"] += 1
            for key, value in fit_cost.items():
                cost[key] += value
            summaries = {m: direction_summary(trainer.upper_actor, d["plus"], states, labels) for m, d in directions.items()}
            cost["training_direction_forward_batches"] += 2
            variants = [("zero", initial, "forecast"), *[(m + "_" + sign, directions[m][sign], "joint")
                for m in spec.METHODS for sign in ("plus", "minus")]]
            roles, training = spec.training_roles(root), {}
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel],
                    period, predictor, envelope) for r in roles]
                training[panel] = list(pool.map(source.worker_group, jobs))
                for rows in training[panel]:
                    source.count_group(cost, rows, "training")
            fits = {m: {panel: fit_method(training[panel], m) for panel in spec.PANELS} for m in spec.METHODS}
            for m in spec.METHODS:
                fits[m]["pooled"] = fit_method(training["A"] + training["B"], m)
            cost["native_return_fits"] += 6
            crossfit = {}
            for train_panel, test_panel in (("A", "B"), ("B", "A")):
                variants = [("zero", initial, "forecast"), *[(m, source.scaled_upper(initial, directions[m]["plus"],
                    fits[m][train_panel]["scale"]), "joint") for m in spec.METHODS]]
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][test_panel],
                    period, predictor, envelope) for r in roles]
                gains = {m: [] for m in spec.METHODS}
                for index, rows in enumerate(pool.map(source.worker_group, jobs)):
                    source.count_group(cost, rows, "crossfit")
                    if rows["zero"]["episode_return"] != training[test_panel][index]["zero"]["episode_return"]:
                        raise ValueError("Stage126 crossfit zero replay changed")
                    for m in spec.METHODS:
                        gains[m].append(rows[m]["episode_return"] - rows["zero"]["episode_return"])
                crossfit[f"{train_panel}_to_{test_panel}"] = {m: {"mean_gain": float(np.mean(v)), "paired_differences": v} for m, v in gains.items()}
            plus = {m: source.scaled_upper(initial, directions[m]["plus"], fits[m]["pooled"]["scale"]) for m in spec.METHODS}
            minus = source.scaled_upper(initial, directions["natural"]["minus"], fits["natural"]["pooled"]["scale"])
            cost["upper_candidate_weight_steps"] += 9
            variants = [("source_flat", initial, "flat"), ("source_forecast", initial, "forecast"),
                ("euclidean", plus["euclidean"], "joint"), ("natural", plus["natural"], "joint"),
                ("natural_descent", minus, "joint"), ("natural_blinded", plus["natural"], "forecast")]
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, variants, seed, seed, period, predictor, envelope) for seed in seeds]
            evaluation = list(pool.map(source.worker_group, jobs))
            for rows in evaluation:
                source.count_group(cost, rows, "evaluation")
                if rows["natural_blinded"]["episode_return"] != rows["source_forecast"]["episode_return"]:
                    raise ValueError("Stage126 blinded upper must exactly execute forecast")
            path = output.parent / "final_weights" / f"period_{period}_natural_upper.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "geometry": geometry,
                "fit": fits["natural"]["pooled"], "weights": plus["natural"]}, path)
            cost["checkpoint_writes"] += 1
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[p] = {"geometry": geometry, "direction_summaries": summaries, "training_roles": roles,
                "training_native_return_fits": fits, "training_crossfit": crossfit, "evaluation_seeds": seeds,
                "lower_and_critics_frozen": "passed", **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: matched native geometry evaluation complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage126 measured budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "inherited_Stage123_cost": cached["cost"], "inherited_Stage124_cost": spec.source.source.budget(),
        "minimum_episode_gain": spec.MINIMUM_GAIN, "kind": "matched_native_mean_geometry_development_not_joint_HRL_confirmation"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: native mean-geometry result written", flush=True)
    return result
