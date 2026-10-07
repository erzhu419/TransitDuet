"""Refresh native credit at a learned upper versus an equal-radius stale step."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from . import pointmaze_credit_transfer as source
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_native_geometry import fit_method
from .pointmaze_root_response import write_json
from scripts import pointmaze_compact_continuation_stage130_spec as spec

joint, curvature, credit = source.joint, source.curvature, source.source.credit


def worker_label(job):
    weights, teacher, upper, query, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(weights)
    trainer = joint.make_trainer(model, teacher, args)
    trainer.upper_actor.load_state_dict(upper)
    before = joint.weights(trainer)
    intervention = {"scenario_seed": query["scenario_seed"], "start": query["start"],
        "prefix_noise_seed": query["noise_seed"], "suffix_noise_seeds": {"A": query["noise_seed"]}}
    deltas = [np.zeros(spec.ACTION_DIM, dtype=np.float32)]
    for axis in range(spec.ACTION_DIM):
        for sign in (1, -1):
            delta = np.zeros(spec.ACTION_DIM, dtype=np.float32)
            delta[axis] = sign * spec.EPSILON
            deltas.append(delta)
    rows, common, error, peak = [], None, 0., 0.
    for delta in deltas:
        row, audit = credit.intervention_episode(trainer, args=args, query=intervention, panel="A",
            action_delta=delta, period=period, predictor=predictor, envelope=envelope)
        if common is None:
            common = audit
        else:
            for key in ("state", "query_state", "prefix_rewards", "measurements"):
                np.testing.assert_array_equal(audit[key], common[key])
            np.testing.assert_array_equal(audit["commands"][:query["start"]], common["commands"][:query["start"]])
            np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
            error = max(error, float(np.abs(audit["innovations"] - common["innovations"]).max()))
        np.testing.assert_allclose(row["episode_return"], row["prefix_return"] + row["suffix_return"], atol=1e-9, rtol=0)
        if row["reference_correction_peak"] > joint.spec.REFERENCE_LIMIT + 1e-8:
            raise ValueError("Stage130 changed native reference authority")
        peak = max(peak, row["reference_correction_peak"])
        rows.append(row)
    torch.testing.assert_close(joint.weights(trainer), before, atol=0, rtol=0)
    return {"query": query, "state": common["state"],
        "gradient": [(rows[1+2*i]["suffix_return"] - rows[2+2*i]["suffix_return"]) / (2*spec.EPSILON) for i in range(spec.ACTION_DIM)],
        "baseline_episode_return": rows[0]["episode_return"], "baseline_suffix_return": rows[0]["suffix_return"],
        "max_innovation_error": error, "max_reference_peak": peak, "pairing_and_freeze": "passed",
        "native_episodes": len(rows), "native_steps": len(rows)*args.horizon,
        "native_upper_calls": len(rows)*(args.horizon//period), "native_donor_response_calls": 2*len(rows)*args.horizon}


def learn_continuations(actor, origin, labels):
    states = np.asarray([r["state"] for r in labels], dtype=np.float32)
    gradients = np.asarray([r["gradient"] for r in labels], dtype=np.float64)
    signals = {panel: gradients * (2*np.asarray([r["query"]["panel"] == panel for r in labels]))[:, None] for panel in spec.PANELS}
    projection = source.causal_summary_projection()
    directions, geometry = native_mean_directions(states @ projection.T, signals,
        actor.log_std.detach().exp().numpy(), damping=spec.DAMPING)
    vectors = {}
    for panel, direction in directions.items():
        mapped = {"net.0.weight": direction["weight"] @ projection, "net.0.bias": direction["bias"]}
        vectors[panel] = np.concatenate([mapped[name].ravel() if name in mapped else np.zeros(p.numel()) for name, p in actor.named_parameters()])
    stale = np.concatenate([(p.detach() - origin[name]).numpy().ravel() for name, p in actor.named_parameters()])
    candidates, learning = {}, {"geometry": geometry, "preconditioned_panel_cosine": cosine(vectors["A"], vectors["B"])}
    cost = {"empirical_fisher_solves": 1, "fisher_jvp_batches": 0, "exact_kl_forward_batches": 0}
    with torch.no_grad():
        baseline = actor.distribution(torch.as_tensor(states)).mean.double().numpy()
    for method, vector in (("refresh", np.mean(list(vectors.values()), axis=0)), ("stale", stale)):
        actors, radius, work = matched_perturbations(actor, states, -vector, delta=spec.FISHER_RADIUS, chunk_size=1024)
        candidates[method] = {sign: a.state_dict() for sign, a in actors.items()}
        with torch.no_grad():
            mean = actors["plus"].distribution(torch.as_tensor(states)).mean.double().numpy()
        by_trajectory = {}
        for r, value in zip(labels, np.sum((mean-baseline)*gradients, axis=1)):
            q = r["query"]
            key = (q["scenario_seed"], q["panel"])
            by_trajectory[key] = by_trajectory.get(key, 0.) + float(value)
        learning[method] = {"radius": radius, "predicted_incremental_episode_derivative": float(np.mean(list(by_trajectory.values()))),
            "policy_mean_rms": float(np.sqrt(np.square(mean).mean())),
            "incremental_mean_rms": float(np.sqrt(np.square(mean-baseline).mean()))}
        for k, v in work.items():
            cost[k] += v
    return candidates, learning, cost


def evaluation_summary(rows):
    metrics = ("episode_return", "reference_correction_rms", "reference_correction_peak", "plan_delta_rms", "upper_mean_rms")
    means = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in metrics} for v in spec.VARIANTS}
    effects = {}
    for a, b in spec.CONTRASTS:
        paired = [r[a]["episode_return"]-r[b]["episode_return"] for r in rows]
        effects[f"{a}_minus_{b}"] = {"mean": float(np.mean(paired)), "paired_differences": paired}
    return {"mean_metrics": means, "effects": effects}


def run(root, output):
    output = output.resolve()
    cached = json.loads(spec.source_result(root).read_text())
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"]) != ("complete", spec.source.PROTOCOL, root, spec.source.budget()):
        raise ValueError("Stage130 requires the completed Stage128 compact source")
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
            origin = copy.deepcopy(trainer.upper_actor.state_dict())
            path = spec.source_result(root).parent/"final_weights"/f"period_{period}_compact_upper.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if (saved["protocol"], saved["root"], saved["period"], saved["method"], saved["fit"]) != (
                    spec.source.PROTOCOL, root, period, "compact", cached["groups"][p]["training_native_return_fits"]["compact"]["pooled"]):
                raise ValueError("Stage130 learned source upper changed")
            current = saved["weights"]
            trainer.upper_actor.load_state_dict(current)
            torch.testing.assert_close(current["log_std"], origin["log_std"], atol=0, rtol=0)
            cost["upper_checkpoint_loads"] += 1
            envelope, roster = calibrations[p]["envelope"], spec.queries(root, period)
            jobs = [(joint.weights(model), teacher, current, q, period, predictor, envelope) for q in roster]
            labels = []
            for index, row in enumerate(pool.map(worker_label, jobs), 1):
                labels.append(row)
                cost["label_queries"] += 1
                cost["label_episodes"] += row["native_episodes"]
                for k in ("native_episodes", "native_steps", "native_upper_calls", "native_donor_response_calls"):
                    cost[k] += row[k]
                if index % 16 == 0 or index == len(roster):
                    print(f"root={root} period={period}: current-policy labels {index}/{len(roster)}", flush=True)
            directions, learning, work = learn_continuations(trainer.upper_actor, origin, labels)
            for k, v in work.items():
                cost[k] += v
            variants = [("zero", current, "joint"), *[(m+"_"+sign, directions[m][sign], "joint") for m in spec.METHODS for sign in ("plus", "minus")]]
            roles, training = spec.training_roles(root), {}
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel], period, predictor, envelope) for r in roles]
                training[panel] = list(pool.map(curvature.worker_group, jobs))
                for rows in training[panel]:
                    curvature.count_group(cost, rows, "training")
            fits = {m: {panel: fit_method(training[panel], m) for panel in spec.PANELS} for m in spec.METHODS}
            for m in spec.METHODS:
                fits[m]["pooled"] = fit_method(training["A"]+training["B"], m)
            cost["native_return_fits"] += 6
            crossfit = {}
            for train_panel, test_panel in (("A", "B"), ("B", "A")):
                variants = [("zero", current, "joint"), *[(m, curvature.scaled_upper(current, directions[m]["plus"], fits[m][train_panel]["scale"]), "joint") for m in spec.METHODS]]
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][test_panel], period, predictor, envelope) for r in roles]
                gains = {m: [] for m in spec.METHODS}
                for i, rows in enumerate(pool.map(curvature.worker_group, jobs)):
                    curvature.count_group(cost, rows, "crossfit")
                    if rows["zero"]["episode_return"] != training[test_panel][i]["zero"]["episode_return"]:
                        raise ValueError("Stage130 current-policy crossfit replay changed")
                    for m in spec.METHODS:
                        gains[m].append(rows[m]["episode_return"]-rows["zero"]["episode_return"])
                crossfit[f"{train_panel}_to_{test_panel}"] = {m: {"mean_gain": float(np.mean(v)), "paired_differences": v} for m, v in gains.items()}
            plus = {m: curvature.scaled_upper(current, directions[m]["plus"], fits[m]["pooled"]["scale"]) for m in spec.METHODS}
            minus = curvature.scaled_upper(current, directions["refresh"]["minus"], fits["refresh"]["pooled"]["scale"])
            cost["upper_candidate_weight_steps"] += 11
            variants = [("source_flat", origin, "flat"), ("source_forecast", origin, "forecast"), ("single", current, "joint"),
                ("refresh", plus["refresh"], "joint"), ("stale", plus["stale"], "joint"),
                ("refresh_descent", minus, "joint"), ("refresh_blinded", plus["refresh"], "forecast")]
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, variants, seed, seed, period, predictor, envelope) for seed in seeds]
            evaluation = list(pool.map(curvature.worker_group, jobs))
            for rows in evaluation:
                curvature.count_group(cost, rows, "evaluation")
                if rows["refresh_blinded"]["episode_return"] != rows["source_forecast"]["episode_return"]:
                    raise ValueError("Stage130 blinded upper must exactly execute forecast")
            for method in spec.METHODS:
                path = output.parent/"final_weights"/f"period_{period}_{method}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "method": method,
                    "fit": fits[method]["pooled"], "weights": plus[method]}, path)
                cost["checkpoint_writes"] += 1
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[p] = {"learning": learning, "native_labels": [{k:v for k,v in r.items() if k != "state"} for r in labels],
                "training_roles": roles, "training_native_return_fits": fits, "training_crossfit": crossfit,
                "evaluation_seeds": seeds, "lower_and_critics_frozen": "passed", **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: compact continuation complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage130 measured budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "inherited_Stage128_cost": cached["cost"], "minimum_episode_gain": spec.MINIMUM_GAIN,
        "kind": "two_development_root_current_policy_credit_not_independent_confirmation"}
    write_json(output, result)
    write_json(output.parent/"completion"/"ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: compact continuation result written", flush=True)
    return result
