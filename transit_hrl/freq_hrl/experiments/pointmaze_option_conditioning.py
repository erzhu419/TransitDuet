"""Fit identical cached native option credit in raw and causal-summary spaces."""

from concurrent.futures import ProcessPoolExecutor
import copy
from functools import partial
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_local_option_probe as previous
from .pointmaze_actor_credit import cosine
from .pointmaze_credit_transfer import causal_summary_projection
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_option_conditioning_stage143_spec as spec

joint, warm, query = previous.joint, previous.warm, previous.previous


def worker_replay(job):
    source_weights, teacher, initial, role, panel, period, predictor, envelope, paths = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    batch, row, _ = joint.native_episode(trainer, args=args, seed=role["scenario_seed"],
        noise_seed=role["noise_seeds"][panel], arm="joint", period=period, predictor=predictor,
        envelope=envelope, collect=True, sample_upper=False)
    np.testing.assert_array_equal(batch.upper.action, query.credit.scalar_means(trainer.upper_actor, batch.upper.state))
    policy_seed, _ = joint.source.scenario.spec.noise_seeds(args.optimizer_seed, role["scenario_seed"], role["noise_seeds"][panel])
    innovations = query.query_innovations(trainer.upper_actor, batch.upper.state, row["decision_steps"], policy_seed)
    std = trainer.upper_actor.log_std.detach().exp().numpy()
    gradients = {}
    for probe in spec.PROBES:
        cached = paths[probe]
        if (cached["scenario_seed"], cached["noise_seed"], cached["panel"], cached["decision_steps"],
                cached["prefix_future_mean_and_noise_pairing"]) != (
                row["seed"], row["noise_seed"], panel, row["decision_steps"], "passed"):
            raise ValueError("Stage143 archived option label roster changed")
        np.testing.assert_allclose(row["episode_return"], cached["mean_return"], atol=1e-8, rtol=0)
        gradients[probe] = query.paired_query_gradient(cached["query_returns"]["plus"], cached["query_returns"]["minus"],
            innovations, std*spec.source.PROBE_SCALES[probe])
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return {"batch": batch.upper, "gradients": gradients, "row": row, "panel": panel}


def fit_candidates(trainer, rows):
    initial = joint.weights(trainer)
    batch = concat_level_batches([r["batch"] for r in rows])
    gradients = {p: np.concatenate([r["gradients"][p] for r in rows]) for p in spec.PROBES}
    panels = np.concatenate([np.full(r["batch"].size, r["panel"]) for r in rows])
    signals = {p+"_"+fold: g if fold == "pooled" else g*(2*(panels == fold))[:, None]
        for p, g in gradients.items() for fold in ("pooled", *spec.PANELS)}
    std = trainer.upper_actor.log_std.detach().exp().numpy()
    projections = {"raw": np.eye(batch.state.shape[1]), "compact": causal_summary_projection()}
    weights, learning = {"warm_start": initial["upper_actor"]}, {}
    cost = dict.fromkeys(("empirical_fisher_solves", "fisher_jvp_batches", "exact_kl_forward_batches",
        "policy_geometry_forward_batches", "upper_candidate_weight_steps"), 0)
    with torch.no_grad():
        baseline = trainer.upper_actor.distribution(torch.as_tensor(batch.state)).mean.double().numpy()
    cost["policy_geometry_forward_batches"] += 1
    mean_steps = {}
    for representation, projection in projections.items():
        directions, geometry = native_mean_directions(batch.state@projection.T, signals, std, damping=1.)
        cost["empirical_fisher_solves"] += 1
        for probe in spec.PROBES:
            vectors = {}
            for fold in ("pooled", *spec.PANELS):
                direction = directions[probe+"_"+fold]
                mapped = {"net.0.weight": direction["weight"]@projection, "net.0.bias": direction["bias"]}
                vectors[fold] = np.concatenate([mapped[n].ravel() if n in mapped else np.zeros(p.numel())
                    for n, p in trainer.upper_actor.named_parameters()])
            method = probe+"_"+representation
            actors, radius, work = matched_perturbations(trainer.upper_actor, batch.state, -vectors["pooled"],
                delta=spec.RADIUS, chunk_size=joint.spec.MINIBATCH)
            rms, predictions = {}, {}
            for sign, actor in actors.items():
                torch.testing.assert_close(actor.log_std, trainer.upper_actor.log_std, atol=0, rtol=0)
                with torch.no_grad():
                    change = actor.distribution(torch.as_tensor(batch.state)).mean.double().numpy()-baseline
                rms[sign] = float(np.sqrt(np.mean(change**2)))
                np.testing.assert_allclose(rms[sign], spec.MEAN_STEP_RMS, atol=1e-7, rtol=0)
                derivative, start = [], 0
                for row in rows:
                    end = start+row["batch"].size
                    derivative.append(float(np.sum(change[start:end]*gradients[probe][start:end])))
                    start = end
                predictions[sign] = {panel: float(np.mean([v for v, r in zip(derivative, rows) if r["panel"] == panel])) for panel in spec.PANELS}
                weights[method+"_"+sign] = copy.deepcopy(actor.state_dict())
                cost["policy_geometry_forward_batches"] += 1
                if sign == "plus": mean_steps[method] = change
            for k, v in work.items(): cost[k] += v
            cost["upper_candidate_weight_steps"] += 2
            learning[method] = {"geometry": geometry, "radius": radius, "mean_step_RMS": rms,
                "preconditioned_noise_panel_cosine": cosine(vectors["A"], vectors["B"]),
                "training_gradient_predicted_increment_by_panel": predictions}
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return weights, learning, {p: cosine(mean_steps[p+"_raw"].ravel(), mean_steps[p+"_compact"].ravel()) for p in spec.PROBES}, cost


def run(root, output):
    cached, wide = [json.loads(spec.source_result(root, wide=w).read_text()) for w in (False, True)]
    warm_cached = json.loads(spec.warm_result(root).read_text())
    for cell, protocol in ((cached, spec.source), (wide, spec.source.source)):
        if (cell["status"], cell["protocol"], cell["root"], cell["cost"], cell["seed_roles"], cell["contract"]) != (
                "complete", protocol.PROTOCOL, root, protocol.budget(), protocol.seed_roles(root), protocol.contract()):
            raise ValueError("Stage143 requires completed fixed-policy wide/local label caches")
    warm_spec = spec.source.source.source.warm_source
    if (warm_cached["status"], warm_cached["protocol"], warm_cached["root"], warm_cached["cost"]) != (
            "complete", warm_spec.PROTOCOL, root, warm_spec.budget()):
        raise ValueError("Stage143 warm source differs from Stage135")
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, roles = spec.arguments(root), spec.seed_roles(root)
    cost, groups = dict.fromkeys(spec.budget(), 0), {}
    cost.update(source_cell_loads=3, lower_checkpoint_loads=len(spec.PERIODS), upper_checkpoint_loads=len(spec.PERIODS))
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            snapshot = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            upper, provenance = warm.load_selected_upper(warm_cached, root, period)
            trainer.upper_actor.load_state_dict(upper)
            initial = query.previous.scale_initial(joint.weights(trainer), "reduced")
            joint.load_weights(trainer, initial)
            source_weights, envelope = joint.weights(model), calibrations[str(period)]["envelope"]
            paths = [dict(zip(spec.PROBES, values)) for values in zip(
                wide["groups"][str(period)]["mean_query_paths"], cached["groups"][str(period)]["local_query_paths"])]
            jobs = [(source_weights, teacher, initial, r, p, period, predictor, envelope, label)
                for (r, p), label in zip(((r, p) for r in roles["replayed_training"] for p in spec.PANELS), paths)]
            rows = list(pool.map(worker_replay, jobs))
            for row in rows:
                warm.count_row(cost, row["row"], "replay"); cost["credit_checks"] += 1
            uppers, learning, similarities, work = fit_candidates(trainer, rows)
            for k, v in work.items(): cost[k] += v
            evaluation = list(pool.map(partial(query.worker_evaluate, variants=spec.VARIANTS),
                [(source_weights, teacher, initial, uppers, seed, period, predictor, envelope) for seed in roles["evaluation"]]))
            for seed, scene in zip(roles["evaluation"], evaluation):
                for mode in spec.MODES:
                    for variant, row in scene[mode].items():
                        if (row["seed"], row["noise_seed"], row["upper_sample"]) != (
                                seed, seed, mode == "sampled" and variant != "source_forecast"):
                            raise ValueError("Stage143 evaluation roster or sampling changed")
                        if mode == "sampled" and variant == "source_forecast": cost["evaluation_alias_assignments"] += 1
                        else: warm.count_row(cost, row, "evaluation")
            joint.source.native.curves.support.assert_frozen(model, snapshot)
            groups[str(period)] = {"selected_source": provenance, "both_probe_label_replays": "passed",
                "learning": learning, "raw_to_compact_training_mean_step_cosine": similarities,
                "frozen_deployment_and_noise_pairing": "passed",
                **query.evaluation_summary(evaluation, variants=spec.VARIANTS, contrasts=spec.CONTRASTS)}
            print(f"root={root} period={period}: cached-option conditioning evaluation complete", flush=True)
    if cost != spec.budget(): raise ValueError(f"Stage143 measured replay/evaluation budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "contract": spec.contract(),
        "seed_roles": roles, "cost": cost, "groups": groups, "kind": "cached_gradient_update_conditioning_development_not_joint_HRL",
        "inherited_Stage142_cost": cached["cost"], "inherited_Stage141_cost": wide["cost"],
        "inherited_Stage135_cost": warm_cached["cost"], "inherited_earlier_source_cost": cached["inherited_earlier_source_cost"]}
    write_json(output, result)
    write_json(output.parent/"completion"/"ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    return result
