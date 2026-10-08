"""Keep a harmful warm-start PPO direction, change only radius and subspace."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_warm_start_joint as warm
from .pointmaze_credit_transfer import causal_summary_projection
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_bounded_ppo_stage137_spec as spec

joint = warm.joint


def project_displacement(displacement):
    projection = causal_summary_projection()
    result = copy.deepcopy(displacement)
    weight = displacement["net.0.weight"].double().numpy()
    projected = weight @ projection.T @ np.linalg.solve(projection @ projection.T, projection)
    result["net.0.weight"] = torch.as_tensor(projected, dtype=displacement["net.0.weight"].dtype)
    energy = float(np.square(weight).sum())
    return result, float(np.square(weight-projected).sum()/energy) if energy else 0.


def candidates(actor, updated, states):
    before = copy.deepcopy(actor.state_dict())
    delta = {k: updated[k]-before[k] for k in before}
    compact, outside = project_displacement(delta)
    weights, geometry = {"warm_start": before, "raw_adam": updated}, {"raw_weight_step_outside_compact_fraction": outside}
    cost = {"fisher_jvp_batches": 0, "exact_kl_forward_batches": 0}
    for method, direction in (("bounded_adam", delta), ("compact_adam", compact)):
        vector = np.concatenate([direction[n].numpy().ravel() for n, _ in actor.named_parameters()])
        models, radius, work = matched_perturbations(actor, states, -vector,
            delta=spec.FISHER_RADIUS, chunk_size=spec.source.ppo.MINIBATCH)
        for sign, model in models.items():
            weights[method+"_"+sign] = copy.deepcopy(model.state_dict())
        geometry[method] = radius
        for k, v in work.items():
            cost[k] += v
    tensor = torch.as_tensor(states)
    with torch.no_grad():
        old = actor.distribution(tensor)
        for variant, state in weights.items():
            candidate = copy.deepcopy(actor)
            candidate.load_state_dict(state)
            new = candidate.distribution(tensor)
            geometry[variant + "_policy"] = {
                "empirical_KL": float(torch.distributions.kl_divergence(
                    torch.distributions.Normal(old.mean.double(), old.stddev.double()),
                    torch.distributions.Normal(new.mean.double(), new.stddev.double())).sum(-1).mean()),
                "mean_step_rms": float((new.mean.double()-old.mean.double()).square().mean().sqrt()),
                "mean_rms": float(new.mean.double().square().mean().sqrt())}
    cost["policy_geometry_forward_batches"] = 1+len(weights)
    return weights, geometry, cost


def worker_collect(job):
    source_weights, teacher, initial, seed, noises, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    rows = []
    for noise in noises:
        batch, row, _ = joint.native_episode(trainer, args=args, seed=seed, noise_seed=noise,
            arm="joint", period=period, predictor=predictor, envelope=envelope, collect=True)
        rows.append((batch, row))
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return rows


def worker_evaluate(job):
    source_weights, teacher, initial, upper_weights, seed, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    rows, common = {}, None
    for variant in spec.VARIANTS:
        trainer.upper_actor.load_state_dict(upper_weights["warm_start" if variant == "source_forecast" else variant])
        _, row, audit = joint.native_episode(trainer, args=args, seed=seed, noise_seed=seed,
            arm="forecast" if variant == "source_forecast" else "joint", period=period,
            predictor=predictor, envelope=envelope, collect=False)
        if common is not None:
            np.testing.assert_array_equal(audit["measurements"], common["measurements"])
            np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
        else:
            common = audit
        rows[variant] = {**row, "variant": variant}
    for name in ("lower_actor", "lower_value", "upper_value"):
        torch.testing.assert_close(getattr(trainer, name).state_dict(), initial[name], atol=0, rtol=0)
    return rows


def evaluation_summary(rows):
    effects, tracking = {}, {}
    for a, b in spec.CONTRASTS:
        for metric, output, sign in (("episode_return", effects, 1), ("tracking_squared_error_integral", tracking, -1)):
            paired = [sign*(r[a][metric]-r[b][metric]) for r in rows]
            output[f"{a}_minus_{b}"] = {"mean": float(np.mean(paired)), "paired_differences": paired}
    return {"effects": effects, "tracking_error_reduction_positive_is_better": tracking,
        "mean_metrics": {v: {k: float(np.mean([r[v][k] for r in rows])) for k in spec.METRICS} for v in spec.VARIANTS}}


def run(root, output):
    cached = json.loads(spec.source_result(root).read_text())
    warm_cached = json.loads(spec.source.source_result(root).read_text())
    if (root not in spec.ROOTS or cached["protocol"] != spec.source.PROTOCOL or cached["status"] != "complete"
            or cached["root"] != root or cached["cost"] != spec.source.budget()
            or cached["seed_roles"] != spec.source.seed_roles(root)):
        raise ValueError("Stage137 requires the completed Stage136 training roster")
    if (warm_cached["protocol"], warm_cached["status"], warm_cached["root"], warm_cached["cost"]) != (
            spec.source.source.PROTOCOL, "complete", root, spec.source.source.budget()):
        raise ValueError("Stage137 warm source differs from Stage135")
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, roles = spec.arguments(root), spec.seed_roles(root)
    groups, cost = {}, dict.fromkeys(spec.budget(), 0)
    cost.update(source_cell_loads=2, lower_checkpoint_loads=len(spec.PERIODS), upper_checkpoint_loads=len(spec.PERIODS))
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            snapshot = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            upper, provenance = warm.load_selected_upper(warm_cached, root, period)
            torch.testing.assert_close(upper["log_std"], trainer.upper_actor.log_std, atol=0, rtol=0)
            trainer.upper_actor.load_state_dict(upper)
            initial, source_weights = joint.weights(trainer), joint.weights(model)
            envelope = calibrations[str(period)]["envelope"]
            jobs = [(source_weights, teacher, initial, r["scenario_seed"], r["noise_seeds"], period, predictor, envelope)
                for r in roles["replayed_training"]]
            batches, returns = [], []
            for registered, group in zip(roles["replayed_training"], pool.map(worker_collect, jobs)):
                if [(r["seed"], r["noise_seed"]) for _, r in group] != [(registered["scenario_seed"], n) for n in registered["noise_seeds"]]:
                    raise ValueError("Stage137 replay scenario/noise order changed")
                for batch, row in group:
                    batches.append(batch); returns.append(row["episode_return"])
                    warm.count_row(cost, row, "replay")
                    cost["credit_checks"] += 1
            prior = cached["groups"][str(period)]
            np.testing.assert_allclose(returns, prior["training_returns"]["sampled"], atol=1e-8, rtol=0)
            report = warm.update_intervention(trainer, batches, root=root, period=period, method="upper_only")["upper"]
            previous = prior["updates"]["upper_only"]["upper"]
            if report["optimizer_seed"] != previous["optimizer_seed"] or report["optimizer_steps"] != previous["optimizer_steps"]:
                raise ValueError("Stage137 original PPO update changed")
            np.testing.assert_allclose([report["parameter_delta_rms"][n] for n in joint.NETWORKS],
                [previous["parameter_delta_rms"][n] for n in joint.NETWORKS], atol=1e-10, rtol=0)
            for k in ("upper_actor_optimizer_steps", "upper_value_optimizer_steps", "lower_actor_optimizer_steps", "lower_value_optimizer_steps"):
                cost[k] += report["optimizer_steps"][k]
            base_actor = copy.deepcopy(trainer.upper_actor)
            base_actor.load_state_dict(upper)
            states = np.concatenate([b.upper.state for b in batches])
            upper_weights, geometry, work = candidates(base_actor, copy.deepcopy(trainer.upper_actor.state_dict()), states)
            for k, v in work.items():
                cost[k] += v
            jobs = [(source_weights, teacher, initial, upper_weights, seed, period, predictor, envelope) for seed in roles["evaluation"]]
            evaluation = []
            for seed, rows in zip(roles["evaluation"], pool.map(worker_evaluate, jobs)):
                for row in rows.values():
                    if row["seed"] != seed or row["noise_seed"] != seed:
                        raise ValueError("Stage137 evaluation roster changed")
                    warm.count_row(cost, row, "evaluation")
                evaluation.append(rows)
            joint.source.native.curves.support.assert_frozen(model, snapshot)
            groups[str(period)] = {"selected_source": provenance, "training_replay_and_original_update": "passed",
                "deployed_lower_critics_teacher_and_std_frozen": "passed", "original_update": report,
                "geometry": geometry, **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: bounded PPO signs complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage137 measured native/optimizer budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "seed_roles": roles,
        "contract": spec.contract(), "groups": groups, "cost": cost,
        "kind": "same_bad_PPO_step_scale_subspace_diagnosis_not_joint_HRL_confirmation",
        "inherited_Stage136_cost": cached["cost"], "inherited_Stage135_cost": warm_cached["cost"],
        "inherited_earlier_source_cost": warm_cached["inherited_source_cost"]}
    write_json(output, result)
    write_json(output.parent/"completion"/"ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    return result
