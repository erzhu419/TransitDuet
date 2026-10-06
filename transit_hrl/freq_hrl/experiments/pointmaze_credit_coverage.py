"""Decision-complete native derivatives and sparse-time control on shared states."""

from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from . import pointmaze_native_geometry as source
from . import pointmaze_reference_counterfactual as credit
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_credit_coverage_stage127_spec as spec

joint, curvature = source.joint, source.source


def worker_label(job):
    weights, teacher, query, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(weights)
    trainer = joint.make_trainer(model, teacher, args)
    before = joint.weights(trainer)
    # A label follows one complete noise trajectory, including its prefix.
    intervention = {"scenario_seed": query["scenario_seed"], "start": query["start"],
        "prefix_noise_seed": query["noise_seed"], "suffix_noise_seeds": {"A": query["noise_seed"]}}
    rows, common, max_error, max_peak = [], None, 0., 0.
    deltas = [np.zeros(spec.ACTION_DIM, dtype=np.float32)]
    for axis in range(spec.ACTION_DIM):
        for sign in (1, -1):
            delta = np.zeros(spec.ACTION_DIM, dtype=np.float32)
            delta[axis] = sign * spec.EPSILON
            deltas.append(delta)
    for delta in deltas:
        row, audit = credit.intervention_episode(trainer, args=args, query=intervention, panel="A",
            action_delta=delta, period=period, predictor=predictor, envelope=envelope)
        if common is None:
            common = audit
        else:
            for key in ("state", "query_state", "prefix_rewards", "measurements"):
                np.testing.assert_array_equal(audit[key], common[key])
            np.testing.assert_array_equal(audit["commands"][:query["start"]], common["commands"][:query["start"]])
            error = float(np.abs(audit["innovations"] - common["innovations"]).max())
            np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
            max_error = max(max_error, error)
        np.testing.assert_allclose(row["episode_return"], row["prefix_return"] + row["suffix_return"], atol=1e-9, rtol=0)
        if row["reference_correction_peak"] > joint.spec.REFERENCE_LIMIT + 1e-8:
            raise ValueError("Stage127 changed the bounded native reference response")
        max_peak = max(max_peak, row["reference_correction_peak"])
        rows.append(row)
    gradient = [(rows[1 + 2*i]["suffix_return"] - rows[2 + 2*i]["suffix_return"]) / (2 * spec.EPSILON)
        for i in range(spec.ACTION_DIM)]
    torch.testing.assert_close(joint.weights(trainer), before, atol=0, rtol=0)
    return {"query": query, "state": common["state"], "gradient": gradient,
        "zero_episode_return": rows[0]["episode_return"], "zero_suffix_return": rows[0]["suffix_return"],
        "max_innovation_error": max_error, "max_reference_peak": max_peak,
        "pairing_and_freeze": "passed", "native_episodes": len(rows), "native_steps": len(rows) * args.horizon,
        "native_upper_calls": len(rows) * (args.horizon // period), "native_donor_response_calls": 2 * len(rows) * args.horizon}


def learn_directions(actor, labels, period):
    states = np.asarray([r["state"] for r in labels], dtype=np.float32)
    gradients = np.asarray([r["gradient"] for r in labels], dtype=np.float64)
    starts = [s for s in spec.COARSE_STARTS if s < spec.arguments(spec.ROOTS[0]).horizon]
    coarse = np.asarray([r["query"]["start"] in starts for r in labels])
    inverse_probability = (spec.arguments(spec.ROOTS[0]).horizon // period) / len(starts)
    signals = {}
    for method in spec.METHODS:
        coverage = np.ones(len(labels)) if method == "complete" else coarse * inverse_probability
        for panel in spec.PANELS:
            mask = np.asarray([r["query"]["panel"] == panel for r in labels])
            signals[f"{method}_{panel}"] = gradients * (2 * coverage * mask)[:, None]
    # Both controls use all the same states and the same damped Fisher matrix.
    directions, geometry = native_mean_directions(states, signals, actor.log_std.detach().exp().numpy(), damping=spec.DAMPING)
    candidates, summaries, cost = {}, {}, {"empirical_fisher_solves": 1, "fisher_jvp_batches": 0, "exact_kl_forward_batches": 0}
    for method in spec.METHODS:
        vectors = {}
        for panel in spec.PANELS:
            direction = directions[f"{method}_{panel}"]
            mapped = {"net.0.weight": direction["weight"], "net.0.bias": direction["bias"]}
            vectors[panel] = np.concatenate([mapped[name].ravel() if name in mapped else np.zeros(p.numel())
                for name, p in actor.named_parameters()])
        actors, radius, work = matched_perturbations(actor, states, -np.mean(list(vectors.values()), axis=0),
            delta=spec.FISHER_RADIUS, chunk_size=1024)
        candidates[method] = {sign: a.state_dict() for sign, a in actors.items()}
        with torch.no_grad():
            means = actors["plus"].distribution(torch.as_tensor(states)).mean.double().numpy()
        alignment = np.sum(means * gradients, axis=1)
        by_trajectory = {}
        for row, gain in zip(labels, alignment):
            q = row["query"]
            key = (q["scenario_seed"], q["panel"])
            by_trajectory[key] = by_trajectory.get(key, 0.) + float(gain)
        summaries[method] = {"preconditioned_panel_cosine": cosine(vectors["A"], vectors["B"]),
            "radius": radius, "credit_rows_used": len(labels) if method == "complete" else int(coarse.sum()),
            "all_query_negative_alignments": int((alignment < 0).sum()),
            "predicted_episode_directional_derivative": float(np.mean(list(by_trajectory.values()))),
            "trajectory_predictions": [{"scenario_seed": seed, "panel": panel, "derivative": value}
                for (seed, panel), value in by_trajectory.items()]}
        for key, value in work.items():
            cost[key] += value
    return candidates, {"shared_geometry": geometry, "directions": summaries}, cost


def audit_summary(rows, learning):
    summary = {}
    for method in spec.METHODS:
        finite = [(r[method + "_plus"]["episode_return"] - r[method + "_minus"]["episode_return"])
            / (2 * spec.AUDIT_SCALE) for r in rows]
        predicted = [r["derivative"] for r in learning["directions"][method]["trajectory_predictions"]]
        summary[method] = {"local_sum_mean": float(np.mean(predicted)), "full_policy_secant_mean": float(np.mean(finite)),
            "local_sums": predicted, "full_policy_secants": finite, "cosine": cosine(predicted, finite)}
    return summary


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
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, cost, groups = spec.arguments(root), dict.fromkeys(spec.budget(), 0), {}
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            p, model = str(period), models[str(period)]
            before = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            initial = copy.deepcopy(trainer.upper_actor.state_dict())
            envelope, roster = calibrations[p]["envelope"], spec.queries(root, period)
            jobs = [(joint.weights(model), teacher, q, period, predictor, envelope) for q in roster]
            labels = []
            for index, row in enumerate(pool.map(worker_label, jobs), 1):
                labels.append(row)
                cost["label_queries"] += 1
                cost["label_episodes"] += row["native_episodes"]
                for key in ("native_episodes", "native_steps", "native_upper_calls", "native_donor_response_calls"):
                    cost[key] += row[key]
                if index % 16 == 0 or index == len(roster):
                    print(f"root={root} period={period}: native labels {index}/{len(roster)}", flush=True)
            if [r["query"] for r in labels] != roster:
                raise ValueError("Stage127 native credit roster changed")
            directions, learning, work = learn_directions(trainer.upper_actor, labels, period)
            for key, value in work.items():
                cost[key] += value
            label_roles = [(r, panel) for r in spec.label_roles(root) for panel in spec.PANELS]
            variants = [(m + "_" + sign, curvature.scaled_upper(initial, directions[m][sign], spec.AUDIT_SCALE), "joint")
                for m in spec.METHODS for sign in ("plus", "minus")]
            jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel],
                period, predictor, envelope) for r, panel in label_roles]
            audits = list(pool.map(curvature.worker_group, jobs))
            for rows in audits:
                curvature.count_group(cost, rows, "gradient_audit")
            variants = [("zero", initial, "forecast"), *[(m + "_" + sign, directions[m][sign], "joint")
                for m in spec.METHODS for sign in ("plus", "minus")]]
            roles, training = spec.training_roles(root), {}
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel],
                    period, predictor, envelope) for r in roles]
                training[panel] = list(pool.map(curvature.worker_group, jobs))
                for rows in training[panel]:
                    curvature.count_group(cost, rows, "training")
            fits = {m: {panel: source.fit_method(training[panel], m) for panel in spec.PANELS} for m in spec.METHODS}
            for m in spec.METHODS:
                fits[m]["pooled"] = source.fit_method(training["A"] + training["B"], m)
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
                        raise ValueError("Stage127 crossfit zero replay changed")
                    for m in spec.METHODS:
                        gains[m].append(rows[m]["episode_return"] - rows["zero"]["episode_return"])
                crossfit[f"{train_panel}_to_{test_panel}"] = {m: {"mean_gain": float(np.mean(v)), "paired_differences": v}
                    for m, v in gains.items()}
            plus = {m: curvature.scaled_upper(initial, directions[m]["plus"], fits[m]["pooled"]["scale"]) for m in spec.METHODS}
            minus = curvature.scaled_upper(initial, directions["complete"]["minus"], fits["complete"]["pooled"]["scale"])
            cost["upper_candidate_weight_steps"] += 15
            variants = [("source_flat", initial, "flat"), ("source_forecast", initial, "forecast"),
                ("coarse", plus["coarse"], "joint"), ("complete", plus["complete"], "joint"),
                ("complete_descent", minus, "joint"), ("complete_blinded", plus["complete"], "forecast")]
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, variants, seed, seed, period, predictor, envelope) for seed in seeds]
            evaluation = list(pool.map(curvature.worker_group, jobs))
            for rows in evaluation:
                curvature.count_group(cost, rows, "evaluation")
                if rows["complete_blinded"]["episode_return"] != rows["source_forecast"]["episode_return"]:
                    raise ValueError("Stage127 blinded upper must exactly execute forecast")
            for method in spec.METHODS:
                path = output.parent / "final_weights" / f"period_{period}_{method}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "method": method,
                    "fit": fits[method]["pooled"], "weights": plus[method]}, path)
                cost["checkpoint_writes"] += 1
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[p] = {"learning": learning, "gradient_consistency_audit": audit_summary(audits, learning),
                "native_labels": [{k: v for k, v in r.items() if k != "state"} for r in labels],
                "training_roles": roles, "training_native_return_fits": fits, "training_crossfit": crossfit,
                "evaluation_seeds": seeds, "lower_and_critics_frozen": "passed", **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: credit coverage evaluation complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage127 measured budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "minimum_episode_gain": spec.MINIMUM_GAIN, "kind": "native_temporal_credit_coverage_development_not_joint_HRL_confirmation"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: native temporal credit coverage result written", flush=True)
    return result
