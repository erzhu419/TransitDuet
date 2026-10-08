"""Replay wide mean-option labels and test narrower probes at the same policy."""

from concurrent.futures import ProcessPoolExecutor
import copy
from functools import partial
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_mean_option_query as previous
from .pointmaze_actor_credit import cosine
from .pointmaze_root_response import write_json
from scripts import pointmaze_local_option_probe_stage142_spec as spec

joint, warm = previous.joint, previous.warm


def worker_replay(job):
    source_weights, teacher, initial, role, panel, period, predictor, envelope, cached = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    batch, row, _ = joint.native_episode(trainer, args=args, seed=role["scenario_seed"],
        noise_seed=role["noise_seeds"][panel], arm="joint", period=period, predictor=predictor,
        envelope=envelope, collect=True, sample_upper=False)
    if (cached["scenario_seed"], cached["noise_seed"], cached["panel"], cached["decision_steps"],
            cached["prefix_future_mean_and_noise_pairing"]) != (
            row["seed"], row["noise_seed"], panel, row["decision_steps"], "passed"):
        raise ValueError("Stage142 cached wide-query roster changed")
    np.testing.assert_allclose(row["episode_return"], cached["mean_return"], atol=1e-8, rtol=0)
    means = previous.credit.scalar_means(trainer.upper_actor, batch.upper.state)
    np.testing.assert_array_equal(batch.upper.action, means)
    policy_seed, _ = joint.source.scenario.spec.noise_seeds(args.optimizer_seed, role["scenario_seed"], role["noise_seeds"][panel])
    innovations = previous.query_innovations(trainer.upper_actor, batch.upper.state, row["decision_steps"], policy_seed)
    gradient = previous.paired_query_gradient(cached["query_returns"]["plus"], cached["query_returns"]["minus"],
        innovations, trainer.upper_actor.log_std.detach().exp().numpy())
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return {"batch": batch.upper, "gradient": gradient, "row": row, "path": cached}


def query_response(paths):
    odd, even = [], []
    for path in paths:
        plus, minus = [np.asarray(path["query_returns"][s]) for s in ("plus", "minus")]
        odd.extend((plus-minus)/2)
        even.extend((plus+minus)/2-path["mean_return"])
    odd, even = np.asarray(odd), np.asarray(even)
    return {"half_difference_rms": float(np.sqrt(np.mean(odd**2))),
        "even_response_rms": float(np.sqrt(np.mean(even**2))), "even_response_mean": float(even.mean()),
        "even_response_negative_fraction": float((even < 0).mean())}


def run(root, output):
    cached = json.loads(spec.source_result(root).read_text())
    warm_cached = json.loads(spec.warm_result(root).read_text())
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"], cached["seed_roles"], cached["contract"]) != (
            "complete", spec.source.PROTOCOL, root, spec.source.budget(), spec.source.seed_roles(root), spec.source.contract()):
        raise ValueError("Stage142 requires completed Stage141 mean-query credit")
    warm_spec = spec.source.source.warm_source
    if (warm_cached["status"], warm_cached["protocol"], warm_cached["root"], warm_cached["cost"]) != (
            "complete", warm_spec.PROTOCOL, root, warm_spec.budget()):
        raise ValueError("Stage142 warm source differs from Stage135")
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
            initial = previous.previous.scale_initial(joint.weights(trainer), "reduced")
            joint.load_weights(trainer, initial)
            source_weights, envelope = joint.weights(model), calibrations[str(period)]["envelope"]
            jobs = [(source_weights, teacher, initial, r, p, period, predictor, envelope)
                for r in roles["replayed_training"] for p in spec.PANELS]
            paths = cached["groups"][str(period)]["mean_query_paths"]
            rows = {"wide": list(pool.map(worker_replay, [j+(p,) for j, p in zip(jobs, paths)])),
                "local": list(pool.map(partial(previous.worker_mean_query, probe_scale=spec.PROBE_SCALES["local"]), jobs))}
            for row in rows["wide"]:
                warm.count_row(cost, row["row"], "replay"); cost["credit_checks"] += 1
            for old, local in zip(rows["wide"], rows["local"]):
                np.testing.assert_array_equal(old["batch"].state, local["batch"].state)
                np.testing.assert_array_equal(old["batch"].action, local["batch"].action)
                np.testing.assert_allclose(old["path"]["mean_return"], local["path"]["mean_return"], atol=1e-8, rtol=0)
                for k, v in local["cost"].items(): cost[k] += v
            uppers, learning = {"warm_start": initial["upper_actor"]}, {}
            for method in spec.METHODS:
                actors, learning[method], work = previous.mean_candidates(trainer, rows[method])
                uppers.update({method+"_"+sign: state for sign, state in actors.items()})
                for k, v in work.items(): cost[k] += v
            np.testing.assert_allclose(learning["wide"]["noise_fold_gradient_cosine"],
                cached["groups"][str(period)]["learning"]["mean_query"]["noise_fold_gradient_cosine"], atol=1e-9, rtol=0)
            gradient_cosine = cosine(np.concatenate([r["gradient"] for r in rows["wide"]]).ravel(),
                np.concatenate([r["gradient"] for r in rows["local"]]).ravel())
            evaluation = list(pool.map(partial(previous.worker_evaluate, variants=spec.VARIANTS),
                [(source_weights, teacher, initial, uppers, seed, period, predictor, envelope) for seed in roles["evaluation"]]))
            for seed, scene in zip(roles["evaluation"], evaluation):
                for mode in spec.MODES:
                    for variant, row in scene[mode].items():
                        if (row["seed"], row["noise_seed"], row["upper_sample"]) != (
                                seed, seed, mode == "sampled" and variant != "source_forecast"):
                            raise ValueError("Stage142 evaluation roster or sampling changed")
                        if mode == "sampled" and variant == "source_forecast": cost["evaluation_alias_assignments"] += 1
                        else: warm.count_row(cost, row, "evaluation")
            joint.source.native.curves.support.assert_frozen(model, snapshot)
            groups[str(period)] = {"selected_source": provenance, "wide_label_and_policy_replay": "passed",
                "shared_mean_path_and_probe_innovations": "passed", "learning": learning,
                "query_responses": {m: query_response([r["path"] for r in rows[m]]) for m in spec.METHODS},
                "query_gradient_wide_to_local_cosine": gradient_cosine,
                "local_query_paths": [r["path"] for r in rows["local"]], "frozen_deployment_and_noise_pairing": "passed",
                **previous.evaluation_summary(evaluation, variants=spec.VARIANTS, contrasts=spec.CONTRASTS)}
            print(f"root={root} period={period}: local-option probe evaluation complete", flush=True)
    if cost != spec.budget(): raise ValueError(f"Stage142 measured query/replay budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "contract": spec.contract(),
        "seed_roles": roles, "cost": cost, "groups": groups, "kind": "local_probe_radius_development_not_joint_HRL",
        "inherited_Stage141_cost": cached["cost"], "inherited_Stage135_cost": warm_cached["cost"],
        "inherited_earlier_source_cost": {k: v for k, v in cached.items() if k.startswith("inherited_") and k != "inherited_Stage135_cost"}}
    write_json(output, result)
    write_json(output.parent/"completion"/"ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    return result
