"""Learn the upper mean from native return derivatives, with the lower frozen."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_reference_counterfactual as source
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_native_upper_step_stage124_spec as spec

joint = source.joint


def replay_query(job):
    weights, teacher, query, expected_return, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(weights)
    trainer = joint.make_trainer(model, teacher, args)
    row, audit = source.intervention_episode(trainer, args=args, query=query, panel="A",
        action_delta=np.zeros(spec.source.ACTION_DIM), period=period, predictor=predictor, envelope=envelope)
    np.testing.assert_allclose(row["suffix_return"], expected_return, atol=1e-9, rtol=0)
    if row["reference_correction_peak"] != 0.:
        raise ValueError("Stage124 state reconstruction changed the zero reference policy")
    return {"query": query, "state": audit["state"], "source_zero_return_reproduced": "passed",
        "native_episodes": 1, "native_steps": row["episode_length"],
        "native_donor_response_calls": 2 * row["episode_length"], "native_upper_calls": row["upper_calls"]}


def learn_candidates(trainer, states, labels):
    actor = trainer.upper_actor
    named = list(actor.named_parameters())
    active = [(name, p) for name, p in named if p.requires_grad]
    state_tensor = torch.as_tensor(np.asarray(states, dtype=np.float32))
    gradients = {}
    for panel in spec.source.PANELS:
        signal = torch.as_tensor(np.asarray([r["gradients"][panel] for r in labels], dtype=np.float32))
        loss = -(actor.distribution(state_tensor).mean * signal).sum() / len(labels)
        values = torch.autograd.grad(loss, [p for _, p in active])
        by_name = {name: value.detach().double().numpy().ravel() for (name, _), value in zip(active, values)}
        gradients[panel] = np.concatenate([by_name[name] if name in by_name else np.zeros(p.numel()) for name, p in named])
    pooled = np.mean(list(gradients.values()), axis=0)
    candidates, geometry, cost = matched_perturbations(actor, np.asarray(states, dtype=np.float32), pooled,
        delta=spec.FISHER_RADIUS, chunk_size=spec.CHUNK_SIZE)
    for candidate in candidates.values():
        torch.testing.assert_close(candidate.log_std, actor.log_std, atol=0, rtol=0)
    return candidates, {"actor_gradient_cosine": cosine(gradients["A"], gradients["B"]),
        "geometry": geometry}, cost


def evaluation_group(job):
    source_weights, teacher, states, seed, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    rows, common = {}, None
    mapping = {"source_flat": ("initial", "flat"), "source_forecast": ("initial", "forecast"),
        "native_ascent": ("plus", "joint"), "native_descent": ("minus", "joint"),
        "native_blinded": ("plus", "forecast")}
    for variant in spec.VARIANTS:
        key, arm = mapping[variant]
        joint.load_weights(trainer, states[key])
        _, row, audit = joint.native_episode(trainer, args=args, seed=seed, noise_seed=seed,
            arm=arm, period=period, predictor=predictor, envelope=envelope, collect=False)
        if common is None:
            common = audit
        else:
            np.testing.assert_array_equal(audit["measurements"], common["measurements"])
            np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
        row["variant"] = variant
        rows[variant] = row
    if rows["source_forecast"]["episode_return"] != rows["native_blinded"]["episode_return"]:
        raise ValueError("Stage124 blinded learned upper must exactly execute source forecast")
    torch.testing.assert_close(trainer.lower_actor.teacher.state_dict(), teacher, atol=0, rtol=0)
    return rows


def evaluation_summary(evaluation):
    metrics = ("episode_return", "reference_correction_rms", "reference_correction_peak", "plan_delta_rms", "upper_mean_rms")
    means = {v: {k: float(np.mean([r[k] for r in rows])) for k in metrics} for v, rows in evaluation.items()}
    effects = {}
    for a, b in spec.CONTRASTS:
        difference = [x["episode_return"] - y["episode_return"] for x, y in zip(evaluation[a], evaluation[b])]
        effects[f"{a}_minus_{b}"] = {"mean": float(np.mean(difference)), "paired_differences": difference}
    return {"mean_metrics": means, "effects": effects}


def run(root, output):
    output = output.resolve()
    cached = json.loads(spec.source_result(root).read_text())
    if (cached["status"] != "complete" or cached["protocol"] != spec.source.PROTOCOL or cached["root"] != root
            or cached["epsilon"] != spec.source.EPSILON or cached["cost"] != spec.source.budget()
            or cached["teacher_upper_lower_and_source_frozen"] != "passed"):
        raise ValueError("Stage124 needs the completed frozen Stage123 native credit cache")
    args = spec.source.arguments(root)
    models, predictor, _, calibrations = joint.source.load_source(root)
    groups, cost = {}, dict.fromkeys(spec.budget(), 0)
    cost["label_cache_loads"] = 1
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            before = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=spec.source.source)
            labels = cached["groups"][str(period)]["queries"]
            if [r["query"] for r in labels] != spec.source.queries(root):
                raise ValueError("Stage124 cached query roster changed")
            jobs = [(joint.weights(model), teacher, r["query"], r["coordinate_suffix_returns"]["A"]["zero"],
                period, predictor, calibrations[str(period)]["envelope"]) for r in labels]
            replay = list(pool.map(replay_query, jobs))
            for row in replay:
                cost["training_state_replays"] += 1
                for key in ("native_episodes", "native_steps", "native_donor_response_calls", "native_upper_calls"):
                    cost[key] += row[key]
            trainer = joint.make_trainer(model, teacher, args)
            initial = joint.weights(trainer)
            candidates, learning, fit_cost = learn_candidates(trainer, [r["state"] for r in replay], labels)
            states = {"initial": initial, **{k: {**initial, "upper_actor": actor.state_dict()} for k, actor in candidates.items()}}
            torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
            for key, value in fit_cost.items():
                cost[key] += value
            cost["actor_pullback_forward_batches"] += 2
            cost["actor_pullback_backward_batches"] += 2
            cost["upper_candidate_weight_steps"] += 2
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, states, seed, period, predictor, calibrations[str(period)]["envelope"]) for seed in seeds]
            evaluation = {v: [] for v in spec.VARIANTS}
            for seed, rows in zip(seeds, pool.map(evaluation_group, jobs)):
                cost["evaluation_pair_groups"] += 1
                for variant, row in rows.items():
                    if row["seed"] != seed or row["noise_seed"] != seed:
                        raise ValueError("Stage124 evaluation roster changed")
                    for key, field in (("native_steps", "episode_length"), ("native_upper_calls", "upper_calls"),
                            ("native_donor_response_calls", "reference_donor_calls")):
                        cost[key] += row[field]
                    cost["evaluation_episodes"] += 1
                    cost["native_episodes"] += 1
                    evaluation[variant].append(row)
            learning["upper_weight_delta_rms"] = {k: joint.parameter_delta(initial["upper_actor"], actor.state_dict()) for k, actor in candidates.items()}
            for sign, actor in candidates.items():
                path = output.parent / "final_weights" / f"period_{period}_{sign}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "sign": sign,
                    "source_run": spec.SOURCE_RUN, "fisher_radius": spec.FISHER_RADIUS, "weights": actor.state_dict()}, path)
                cost["checkpoint_writes"] += 1
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[str(period)] = {"learning": learning, "evaluation_seeds": seeds,
                "state_replays": len(replay), "source_replay_and_lower_freeze": "passed", **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: native upper step evaluated", flush=True)
    if cost != spec.budget():
        raise ValueError("Stage124 measured new-work budget changed")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "inherited_Stage123_cost": cached["cost"], "minimum_episode_gain": spec.MINIMUM_GAIN,
        "kind": "two_root_cached_native_upper_learning_development_not_joint_HRL_or_independent_confirmation"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: learned native upper step result written", flush=True)
    return result
