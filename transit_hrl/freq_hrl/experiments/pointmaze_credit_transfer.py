"""Compare unrestricted raw-history means with a causal summary subspace."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from . import pointmaze_credit_coverage as source
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_credit_transfer_stage128_spec as spec

joint, curvature = source.joint, source.curvature


def causal_summary_projection():
    projection = np.zeros((26, 392), dtype=np.float64)
    projection[:6, :6] = np.eye(6)
    projection[6:8, -2:] = np.eye(2)
    time = np.arange(64, dtype=np.float64)
    time -= time.mean()
    trend = time / np.dot(time, time)
    for channel in range(6):
        columns = 6 + channel + 6 * np.arange(64)
        projection[8 + channel, columns[-1]] = 1.
        projection[14 + channel, columns] = 1. / 64
        projection[20 + channel, columns] = trend
    return projection


def worker_replay(job):
    weights, teacher, label, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(weights)
    trainer = joint.make_trainer(model, teacher, args)
    q = label["query"]
    query = {"scenario_seed": q["scenario_seed"], "start": q["start"], "prefix_noise_seed": q["noise_seed"],
        "suffix_noise_seeds": {"A": q["noise_seed"]}}
    row, audit = source.credit.intervention_episode(trainer, args=args, query=query, panel="A",
        action_delta=np.zeros(spec.source.ACTION_DIM), period=period, predictor=predictor, envelope=envelope)
    for key in ("episode_return", "suffix_return"):
        np.testing.assert_allclose(row[key], label["zero_" + key], atol=1e-9, rtol=0)
    if row["reference_correction_peak"] != 0.:
        raise ValueError("Stage128 replay changed the zero reference policy")
    return {**label, "state": audit["state"], "native_episodes": 1, "native_steps": args.horizon,
        "native_upper_calls": args.horizon // period, "native_donor_response_calls": 2 * args.horizon}


def prediction(actor, upper, labels):
    actor = copy.deepcopy(actor)
    actor.load_state_dict(upper)
    with torch.no_grad():
        means = actor.distribution(torch.as_tensor(np.asarray([r["state"] for r in labels], dtype=np.float32))).mean.double().numpy()
    values = np.sum(means * np.asarray([r["gradient"] for r in labels]), axis=1)
    trajectories = {}
    for row, value in zip(labels, values):
        key = (row["query"]["scenario_seed"], row["query"]["panel"])
        trajectories[key] = trajectories.get(key, 0.) + float(value)
    return {"mean_episode_derivative": float(np.mean(list(trajectories.values()))),
        "trajectory_derivatives": [{"scenario_seed": seed, "panel": panel, "derivative": value}
            for (seed, panel), value in trajectories.items()]}


def learn_directions(actor, labels):
    states = np.asarray([r["state"] for r in labels], dtype=np.float32)
    gradients = np.asarray([r["gradient"] for r in labels], dtype=np.float64)
    signals = {panel: gradients * (2 * np.asarray([r["query"]["panel"] == panel for r in labels]))[:, None]
        for panel in spec.PANELS}
    projections = {"raw": np.eye(392), "compact": causal_summary_projection()}
    candidates, summaries = {}, {}
    cost = {"empirical_fisher_solves": 0, "fisher_jvp_batches": 0, "exact_kl_forward_batches": 0}
    for method, projection in projections.items():
        directions, geometry = native_mean_directions(states @ projection.T, signals,
            actor.log_std.detach().exp().numpy(), damping=spec.DAMPING)
        cost["empirical_fisher_solves"] += 1
        vectors = {}
        for panel, direction in directions.items():
            mapped = {"net.0.weight": direction["weight"] @ projection, "net.0.bias": direction["bias"]}
            vectors[panel] = np.concatenate([mapped[name].ravel() if name in mapped else np.zeros(p.numel())
                for name, p in actor.named_parameters()])
        actors, radius, work = matched_perturbations(actor, states, -np.mean(list(vectors.values()), axis=0),
            delta=spec.FISHER_RADIUS, chunk_size=1024)
        candidates[method] = {sign: a.state_dict() for sign, a in actors.items()}
        summaries[method] = {"geometry": geometry, "radius": radius,
            "preconditioned_panel_cosine": cosine(vectors["A"], vectors["B"]),
            "training_prediction": prediction(actor, candidates[method]["plus"], labels)}
        for key, value in work.items():
            cost[key] += value
    return candidates, summaries, cost


def run(root, output):
    output = output.resolve()
    cached = json.loads(spec.source_result(root).read_text())
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"]) != (
            "complete", spec.source.PROTOCOL, root, spec.source.budget()):
        raise ValueError("Stage128 requires completed Stage127 native credit")
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, cost, groups = spec.arguments(root), dict.fromkeys(spec.budget(), 0), {}
    cost["source_cell_loads"] = 1
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            p, model = str(period), models[str(period)]
            before = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            initial = copy.deepcopy(trainer.upper_actor.state_dict())
            envelope, cached_labels = calibrations[p]["envelope"], cached["groups"][p]["native_labels"]
            if [r["query"] for r in cached_labels] != spec.source.queries(root, period):
                raise ValueError("Stage128 source label roster changed")
            jobs = [(joint.weights(model), teacher, r, period, predictor, envelope) for r in cached_labels]
            labels = list(pool.map(worker_replay, jobs))
            for row in labels:
                cost["training_state_replays"] += 1
                for key in ("native_episodes", "native_steps", "native_upper_calls", "native_donor_response_calls"):
                    cost[key] += row[key]
            folds = []
            for role in spec.source.label_roles(root):
                train = [r for r in labels if r["query"]["scenario_seed"] != role["scenario_seed"]]
                test = [r for r in labels if r["query"]["scenario_seed"] == role["scenario_seed"]]
                directions, learning, work = learn_directions(trainer.upper_actor, train)
                cost["leave_scene_out_fits"] += 2
                for key, value in work.items():
                    cost[key] += value
                variants = [(m + "_" + sign, curvature.scaled_upper(initial, directions[m][sign], spec.AUDIT_SCALE), "joint")
                    for m in spec.METHODS for sign in ("plus", "minus")]
                jobs = [(joint.weights(model), teacher, variants, role["scenario_seed"], role["noise_seeds"][panel],
                    period, predictor, envelope) for panel in spec.PANELS]
                audit = list(pool.map(curvature.worker_group, jobs))
                transfer = {}
                for rows in audit:
                    curvature.count_group(cost, rows, "transfer_audit")
                for method in spec.METHODS:
                    predicted = prediction(trainer.upper_actor, directions[method]["plus"], test)
                    finite = [(r[method + "_plus"]["episode_return"] - r[method + "_minus"]["episode_return"])
                        / (2 * spec.AUDIT_SCALE) for r in audit]
                    transfer[method] = {"training": learning[method], "held_out_prediction": predicted,
                        "held_out_policy_secant": float(np.mean(finite)), "paired_secants": finite}
                folds.append({"held_out_scenario": role["scenario_seed"], "training_scenarios": sorted({r["query"]["scenario_seed"] for r in train}),
                    "methods": transfer})
            directions, learning, work = learn_directions(trainer.upper_actor, labels)
            for key, value in work.items():
                cost[key] += value
            variants = [("zero", initial, "forecast"), *[(m + "_" + sign, directions[m][sign], "joint")
                for m in spec.METHODS for sign in ("plus", "minus")]]
            roles, training = spec.training_roles(root), {}
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel],
                    period, predictor, envelope) for r in roles]
                training[panel] = list(pool.map(curvature.worker_group, jobs))
                for rows in training[panel]:
                    curvature.count_group(cost, rows, "training")
            fits = {m: {panel: source.source.fit_method(training[panel], m) for panel in spec.PANELS} for m in spec.METHODS}
            for m in spec.METHODS:
                fits[m]["pooled"] = source.source.fit_method(training["A"] + training["B"], m)
            cost["native_return_fits"] += 6
            crossfit = {}
            for train_panel, test_panel in (("A", "B"), ("B", "A")):
                variants = [("zero", initial, "forecast"), *[(m, curvature.scaled_upper(initial, directions[m]["plus"],
                    fits[m][train_panel]["scale"]), "joint") for m in spec.METHODS]]
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][test_panel],
                    period, predictor, envelope) for r in roles]
                gains = {m: [] for m in spec.METHODS}
                for index, rows in enumerate(pool.map(curvature.worker_group, jobs)):
                    curvature.count_group(cost, rows, "crossfit")
                    if rows["zero"]["episode_return"] != training[test_panel][index]["zero"]["episode_return"]:
                        raise ValueError("Stage128 crossfit zero replay changed")
                    for m in spec.METHODS:
                        gains[m].append(rows[m]["episode_return"] - rows["zero"]["episode_return"])
                crossfit[f"{train_panel}_to_{test_panel}"] = {m: {"mean_gain": float(np.mean(v)), "paired_differences": v} for m, v in gains.items()}
            plus = {m: curvature.scaled_upper(initial, directions[m]["plus"], fits[m]["pooled"]["scale"]) for m in spec.METHODS}
            minus = curvature.scaled_upper(initial, directions["compact"]["minus"], fits["compact"]["pooled"]["scale"])
            cost["upper_candidate_weight_steps"] += 8 * spec.source.LABEL_SCENARIOS + 11
            variants = [("source_flat", initial, "flat"), ("source_forecast", initial, "forecast"),
                ("raw", plus["raw"], "joint"), ("compact", plus["compact"], "joint"),
                ("compact_descent", minus, "joint"), ("compact_blinded", plus["compact"], "forecast")]
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, variants, seed, seed, period, predictor, envelope) for seed in seeds]
            evaluation = list(pool.map(curvature.worker_group, jobs))
            for rows in evaluation:
                curvature.count_group(cost, rows, "evaluation")
                if rows["compact_blinded"]["episode_return"] != rows["source_forecast"]["episode_return"]:
                    raise ValueError("Stage128 blinded upper must exactly execute forecast")
            for method in spec.METHODS:
                path = output.parent / "final_weights" / f"period_{period}_{method}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "method": method,
                    "fit": fits[method]["pooled"], "weights": plus[method]}, path)
                cost["checkpoint_writes"] += 1
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[p] = {"leave_scene_out": folds, "learning": learning, "training_roles": roles,
                "training_native_return_fits": fits, "training_crossfit": crossfit, "evaluation_seeds": seeds,
                "lower_and_critics_frozen": "passed", **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: scene credit transfer evaluation complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage128 measured budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "inherited_Stage127_cost": cached["cost"], "minimum_episode_gain": spec.MINIMUM_GAIN,
        "kind": "native_scene_credit_transfer_development_not_joint_HRL_confirmation"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: native scene credit transfer result written", flush=True)
    return result


def evaluation_summary(rows):
    metrics = ("episode_return", "reference_correction_rms", "reference_correction_peak", "plan_delta_rms", "upper_mean_rms")
    means = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in metrics} for v in spec.VARIANTS}
    effects = {}
    for a, b in spec.CONTRASTS:
        difference = [r[a]["episode_return"] - r[b]["episode_return"] for r in rows]
        effects[f"{a}_minus_{b}"] = {"mean": float(np.mean(difference)), "paired_differences": difference}
    return {"mean_metrics": means, "effects": effects}
