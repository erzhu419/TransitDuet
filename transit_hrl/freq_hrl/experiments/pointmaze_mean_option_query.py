"""Fit upper mean steps from paired native option queries on mean trajectories."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_exploration_match as previous
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_mean_option_query_stage141_spec as spec

joint, warm, credit = previous.joint, previous.warm, previous.credit


def paired_query_gradient(plus, minus, innovations, std):
    return ((np.asarray(plus)-np.asarray(minus))/2)[:, None]*np.asarray(innovations)/std


def query_innovations(actor, states, steps, policy_seed):
    innovations = []
    with torch.no_grad():
        for state, step in zip(states, steps):
            torch.manual_seed(policy_seed+step)
            distribution = actor.distribution(torch.as_tensor(state).view(1, -1))
            innovations.append(((distribution.sample()[0]-distribution.mean[0])/distribution.stddev[0]).numpy())
    return np.stack(innovations)


def worker_mean_query(job, *, probe_scale=1.):
    source_weights, teacher, initial, role, panel, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    call = dict(args=args, seed=role["scenario_seed"], noise_seed=role["noise_seeds"][panel],
        arm="joint", period=period, predictor=predictor, envelope=envelope, collect=True, sample_upper=False)
    batch, row, audit = joint.native_episode(trainer, **call)
    means = credit.scalar_means(trainer.upper_actor, batch.upper.state)
    np.testing.assert_array_equal(batch.upper.action, means)
    policy_seed, _ = joint.source.scenario.spec.noise_seeds(args.optimizer_seed, role["scenario_seed"], role["noise_seeds"][panel])
    counts = dict.fromkeys(spec.budget(), 0)
    warm.count_row(counts, row, "collection")
    returns = {"plus": [], "minus": []}
    innovations = query_innovations(trainer.upper_actor, batch.upper.state, row["decision_steps"], policy_seed)
    std = trainer.upper_actor.log_std.detach().exp().numpy()*probe_scale
    for index, step in enumerate(row["decision_steps"]):
        innovation = innovations[index]
        for sign, direction in (("plus", 1), ("minus", -1)):
            action = means[index]+direction*std*innovation
            probe, control, control_audit = joint.native_episode(trainer, **call, upper_override={"step": step, "action": action})
            np.testing.assert_array_equal(probe.upper.state[:index+1], batch.upper.state[:index+1])
            np.testing.assert_array_equal(probe.upper.action[:index], batch.upper.action[:index])
            np.testing.assert_array_equal(probe.upper.action[index], action)
            for key in ("state", "action", "reward"):
                np.testing.assert_array_equal(getattr(probe.lower, key)[:step], getattr(batch.lower, key)[:step])
            keep = np.arange(batch.upper.size) != index
            np.testing.assert_array_equal(probe.upper.action[keep], credit.scalar_means(trainer.upper_actor, probe.upper.state)[keep])
            np.testing.assert_array_equal(control_audit["measurements"], audit["measurements"])
            np.testing.assert_allclose(control_audit["innovations"], audit["innovations"], atol=3e-5, rtol=0)
            returns[sign].append(control["episode_return"])
            warm.count_row(counts, control, "counterfactual")
            counts["credit_checks"] += 1
        counts["mean_query_label_pairs"] += 1
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return {"batch": batch.upper, "gradient": paired_query_gradient(returns["plus"], returns["minus"], innovations, std),
        "cost": counts, "path": {"scenario_seed": role["scenario_seed"], "noise_seed": role["noise_seeds"][panel],
            "panel": panel, "mean_return": row["episode_return"], "query_returns": returns,
            "decision_steps": row["decision_steps"], "prefix_future_mean_and_noise_pairing": "passed"}}


def mean_candidates(trainer, rows):
    initial = joint.weights(trainer)
    batch = concat_level_batches([r["batch"] for r in rows])
    signal = np.concatenate([r["gradient"] for r in rows])
    std = trainer.upper_actor.log_std.detach().exp().numpy()
    directions, geometry = native_mean_directions(batch.state, {"pooled": signal}, std, damping=1.)
    mapped = {"net.0.weight": directions["pooled"]["weight"], "net.0.bias": directions["pooled"]["bias"]}
    vector = np.concatenate([mapped[n].ravel() if n in mapped else np.zeros(p.numel()) for n, p in trainer.upper_actor.named_parameters()])
    actors, radius, work = matched_perturbations(trainer.upper_actor, batch.state, -vector,
        delta=spec.RADIUS, chunk_size=joint.spec.MINIBATCH)
    weights, rms = {}, {}
    with torch.no_grad():
        mean = trainer.upper_actor.distribution(torch.as_tensor(batch.state)).mean.double()
        for sign, actor in actors.items():
            rms[sign] = float((actor.distribution(torch.as_tensor(batch.state)).mean.double()-mean).square().mean().sqrt())
            np.testing.assert_allclose(rms[sign], spec.MEAN_STEP_RMS, atol=1e-8, rtol=0)
            torch.testing.assert_close(actor.log_std, trainer.upper_actor.log_std, atol=0, rtol=0)
            weights[sign] = copy.deepcopy(actor.state_dict())
    folds = []
    for fold in (0, 1):
        states = np.concatenate([r["batch"].state for r in rows[fold::2]])
        targets = np.concatenate([r["gradient"] for r in rows[fold::2]])
        folds.append(np.r_[(targets.T@states/len(states)).ravel(), targets.mean(0)])
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return weights, {"geometry": geometry, "radius": radius, "mean_step_RMS": rms,
        "noise_fold_gradient_cosine": cosine(*folds)}, {**work, "empirical_fisher_solves": 1,
        "policy_geometry_forward_batches": 3, "upper_candidate_weight_steps": 2}


def worker_evaluate(job, *, variants=spec.VARIANTS):
    source_weights, teacher, initial, uppers, seed, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    output, common, upper_common = {}, None, None
    for mode in spec.MODES:
        rows = {}
        for variant in variants:
            if mode == "sampled" and variant == "source_forecast":
                rows[variant] = output["mean"][variant]
                continue
            trainer.upper_actor.load_state_dict(uppers["warm_start" if variant == "source_forecast" else variant])
            batch, row, audit = joint.native_episode(trainer, args=args, seed=seed, noise_seed=seed,
                arm="forecast" if variant == "source_forecast" else "joint", period=period,
                predictor=predictor, envelope=envelope, collect=False, sample_upper=mode == "sampled")
            assert batch is None
            if common is None: common = audit
            else:
                np.testing.assert_array_equal(audit["measurements"], common["measurements"])
                np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
            if mode == "sampled":
                if upper_common is None: upper_common = audit["upper_innovations"]
                else: np.testing.assert_allclose(audit["upper_innovations"], upper_common, atol=3e-5, rtol=0)
            rows[variant] = {**row, "variant": variant}
        output[mode] = rows
    for name in joint.NETWORKS[1:]:
        torch.testing.assert_close(getattr(trainer, name).state_dict(), initial[name], atol=0, rtol=0)
    torch.testing.assert_close(trainer.upper_actor.log_std, initial["upper_actor"]["log_std"], atol=0, rtol=0)
    return output


def evaluation_summary(scenes, *, variants=spec.VARIANTS, contrasts=spec.CONTRASTS):
    modes = {}
    for mode in spec.MODES:
        modes[mode] = {name: {f"{a}_minus_{b}": previous.paired_effect([sign*(r[mode][a][metric]-r[mode][b][metric]) for r in scenes])
            for a, b in contrasts} for name, metric, sign in (
                ("effects", "episode_return", 1), ("tracking_error_reduction_positive_is_better", "tracking_squared_error_integral", -1))}
        modes[mode]["mean_metrics"] = {v: {k: float(np.mean([r[mode][v][k] for r in scenes])) for k in spec.METRICS} for v in variants}
    return {"evaluation": modes, "sampled_minus_mean_return": {v: previous.paired_effect([
        r["sampled"][v]["episode_return"]-r["mean"][v]["episode_return"] for r in scenes]) for v in variants}}


def run(root, output):
    cached = json.loads(spec.source_result(root).read_text())
    warm_cached = json.loads(spec.warm_result(root).read_text())
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"], cached["seed_roles"]) != (
            "complete", spec.source.PROTOCOL, root, spec.source.budget(), spec.source.seed_roles(root)):
        raise ValueError("Stage141 requires the completed Stage140 sampling contract")
    if (warm_cached["status"], warm_cached["protocol"], warm_cached["root"], warm_cached["cost"]) != (
            "complete", spec.source.warm_source.PROTOCOL, root, spec.source.warm_source.budget()):
        raise ValueError("Stage141 warm source differs from Stage135")
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, roles = spec.arguments(root), spec.seed_roles(root)
    cost, groups = dict.fromkeys(spec.budget(), 0), {}
    cost.update(source_cell_loads=2, lower_checkpoint_loads=len(spec.PERIODS), upper_checkpoint_loads=len(spec.PERIODS))
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            snapshot = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            upper, provenance = warm.load_selected_upper(warm_cached, root, period)
            trainer.upper_actor.load_state_dict(upper)
            initial = previous.scale_initial(joint.weights(trainer), "reduced")
            joint.load_weights(trainer, initial)
            source_weights, envelope = joint.weights(model), calibrations[str(period)]["envelope"]
            jobs = [(source_weights, teacher, initial, r, p, period, predictor, envelope)
                for r in roles["replayed_training"] for p in spec.PANELS]
            paths = cached["groups"][str(period)]["training"]["reduced"]["counterfactual_paths"]
            sampled = list(pool.map(previous.previous.worker_replay, [j+(p,) for j, p in zip(jobs, paths)]))
            queried = list(pool.map(worker_mean_query, jobs))
            for row in sampled:
                warm.count_row(cost, row["row"], "replay"); cost["credit_checks"] += 1
            for job, row in zip(jobs, queried):
                if (row["path"]["scenario_seed"], row["path"]["panel"], row["path"]["noise_seed"]) != (
                        job[3]["scenario_seed"], job[4], job[3]["noise_seeds"][job[4]]):
                    raise ValueError("Stage141 query roster changed")
                for k, v in row["cost"].items(): cost[k] += v
            restored, sampled_fit, work = previous.natural_candidates(trainer, sampled, "reduced")
            for k, v in work.items(): cost[k] += v
            chunks = int(np.ceil(len(sampled)*(args.horizon//period)/joint.spec.MINIBATCH))
            cost["score_gradient_batches"] += chunks; cost["mean_score_forward_batches"] += chunks
            fitted, mean_fit, work = mean_candidates(trainer, queried)
            for k, v in work.items(): cost[k] += v
            uppers = {"warm_start": initial["upper_actor"],
                **{f"sampled_credit_{s}": restored[f"natural_{s}"] for s in ("plus", "minus")},
                **{f"mean_query_{s}": fitted[s] for s in ("plus", "minus")}}
            scenes = list(pool.map(worker_evaluate, [(source_weights, teacher, initial, uppers, seed, period,
                predictor, envelope) for seed in roles["evaluation"]]))
            for seed, scene in zip(roles["evaluation"], scenes):
                for mode in spec.MODES:
                    for variant, row in scene[mode].items():
                        if (row["seed"], row["noise_seed"], row["upper_sample"]) != (
                                seed, seed, mode == "sampled" and variant != "source_forecast"):
                            raise ValueError("Stage141 evaluation roster or sampling changed")
                        if mode == "sampled" and variant == "source_forecast": cost["evaluation_alias_assignments"] += 1
                        else: warm.count_row(cost, row, "evaluation")
            joint.source.native.curves.support.assert_frozen(model, snapshot)
            groups[str(period)] = {"selected_source": provenance, "sampled_credit_replay": "passed",
                "learning": {"sampled_credit": sampled_fit, "mean_query": mean_fit},
                "mean_query_paths": [r["path"] for r in queried], "frozen_deployment_and_noise_pairing": "passed",
                **evaluation_summary(scenes)}
            print(f"root={root} period={period}: mean-option query evaluation complete", flush=True)
    if cost != spec.budget(): raise ValueError(f"Stage141 measured query and replay budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "contract": spec.contract(),
        "seed_roles": roles, "cost": cost, "groups": groups, "kind": "query_augmented_mean_objective_development_not_joint_HRL",
        "inherited_Stage140_cost": cached["cost"], "inherited_Stage135_cost": warm_cached["cost"],
        "inherited_earlier_source_cost": {k: v for k, v in cached.items() if k.startswith("inherited_") and k != "inherited_Stage135_cost"}}
    write_json(output, result)
    write_json(output.parent/"completion"/"ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    return result
