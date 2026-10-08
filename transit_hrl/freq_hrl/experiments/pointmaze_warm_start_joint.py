"""Isolate the first native joint PPO update after validated upper training."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch, HierarchicalTrajectoryBatch
from . import pointmaze_joint_reference as joint
from .pointmaze_root_response import write_json
from scripts.diagnose_pointmaze_joint_reference_credit_stage122 import diagnose_level
from scripts import pointmaze_warm_start_joint_stage136_spec as spec


def empty_level(batch):
    return LevelTrajectoryBatch(**{k: getattr(batch, k)[:0] for k in
        ("state", "action", "reward", "duration", "done", "old_logp", "old_value")})


def update_intervention(trainer, batches, *, root, period, method):
    reports = {}
    for level in ("upper", "lower"):
        if method not in (level + "_only", "joint"):
            continue
        partial = [HierarchicalTrajectoryBatch(
            upper=b.upper if level == "upper" else empty_level(b.upper),
            lower=b.lower if level == "lower" else empty_level(b.lower)) for b in batches]
        reports[level] = joint.update(trainer, partial,
            optimizer_seed=spec.optimizer_seed(root, period, level))
        inactive = "lower" if level == "upper" else "upper"
        if any(reports[level]["parameter_delta_rms"][inactive + suffix] != 0
                for suffix in ("_actor", "_value")):
            raise ValueError("Stage136 changed the inactive level")
    return reports


def load_selected_upper(cached, root, period):
    group = cached["groups"][str(period)]
    method = group["selection"]["method"]
    if method == "two_step":
        protocol, saved_method = spec.source.source.PROTOCOL, "refresh"
        source_cell = json.loads(spec.source.source_result(root).read_text())
        path = spec.source.source_result(root).parent / "final_weights" / f"period_{period}_refresh_upper.pt"
        fit = source_cell["groups"][str(period)]["training_native_return_fits"]["refresh"]["pooled"]
    else:
        protocol, saved_method = spec.source.PROTOCOL, method
        path = spec.source_result(root).parent / "final_weights" / f"period_{period}_{method}_upper.pt"
        fit = group["training_native_return_fits"][method]["pooled"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["protocol"], saved["root"], saved["period"], saved["method"], saved["fit"]) != (
            protocol, root, period, saved_method, fit):
        raise ValueError("Stage136 selected upper checkpoint differs from its source fit")
    return saved["weights"], {"method": method, "protocol": protocol, "fit": fit}


def worker_group(job):
    source_weights, teacher, states, seed, noises, period, predictor, envelope, collect = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    outputs = []
    variants = ("sampled", "warm_start", "source_forecast") if collect else spec.VARIANTS
    for noise in noises:
        common = None
        for variant in variants:
            key = "warm_start" if collect else "joint" if variant == "joint_blinded" else variant
            joint.load_weights(trainer, states[key])
            arm = "forecast" if variant in ("source_forecast", "joint_blinded") else "joint"
            batch, row, audit = joint.native_episode(trainer, args=args, seed=seed, noise_seed=noise,
                arm=arm, period=period, predictor=predictor, envelope=envelope,
                collect=variant == "sampled")
            if common is not None:
                np.testing.assert_array_equal(audit["measurements"], common["measurements"])
                np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
            else:
                common = audit
            outputs.append((variant, batch, {**row, "variant": variant}))
    return outputs


def count_row(cost, row, phase):
    cost[phase + "_episodes"] += 1
    cost["native_episodes"] += 1
    for key, row_key in (("native_steps", "episode_length"), ("native_lower_calls", "lower_calls"),
            ("native_upper_calls", "upper_calls"), ("native_donor_response_calls", "reference_donor_calls"),
            ("planning_renewals", "plan_renewals"), ("planning_fits", "plan_fits"),
            ("planning_reference_calls", "reference_calls")):
        cost[key] += row[row_key]
    cost["credit_checks"] += int(phase == "collection")


def evaluation_summary(rows):
    effects = {}
    for a, b in spec.CONTRASTS:
        paired = [r[a]["episode_return"] - r[b]["episode_return"] for r in rows]
        effects[f"{a}_minus_{b}"] = {"mean": float(np.mean(paired)), "paired_differences": paired}
    interaction = [r["joint"]["episode_return"] - r["upper_only"]["episode_return"]
        - r["lower_only"]["episode_return"] + r["warm_start"]["episode_return"] for r in rows]
    effects["joint_interaction"] = {"mean": float(np.mean(interaction)), "paired_differences": interaction}
    metrics = ("episode_return", "reference_correction_rms", "reference_correction_peak",
        "learned_residual_rms", "learned_residual_peak", "plan_delta_rms", "upper_mean_rms")
    return {"effects": effects, "mean_metrics": {v: {k: float(np.mean([r[v][k] for r in rows]))
        for k in metrics} for v in spec.VARIANTS}}


def run(root, output):
    if root not in spec.ROOTS:
        raise ValueError("Stage136 uses the first two Stage135 roots, not selected winners")
    cached = json.loads(spec.source_result(root).read_text())
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"]) != (
            "complete", spec.source.PROTOCOL, root, spec.source.budget()):
        raise ValueError("Stage136 requires the completed Stage135 source")
    args, roles = spec.arguments(root), spec.seed_roles(root)
    models, predictor, _, calibrations = joint.source.load_source(root)
    cost, groups = dict.fromkeys(spec.budget(), 0), {}
    cost.update(source_cell_loads=1, source_clone_loads=len(spec.PERIODS),
        lower_checkpoint_loads=len(spec.PERIODS), upper_checkpoint_loads=len(spec.PERIODS))
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            source_snapshot = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            upper, provenance = load_selected_upper(cached, root, period)
            torch.testing.assert_close(upper["log_std"], trainer.upper_actor.log_std, atol=0, rtol=0)
            trainer.upper_actor.load_state_dict(upper)
            initial = joint.weights(trainer)
            source_weights = joint.weights(model)
            envelope = calibrations[str(period)]["envelope"]
            jobs = [(source_weights, teacher, {"warm_start": initial}, r["scenario_seed"],
                r["noise_seeds"], period, predictor, envelope, True) for r in roles["training"]]
            batches, training = [], {v: [] for v in ("sampled", "warm_start", "source_forecast")}
            for registered, outputs in zip(roles["training"], pool.map(worker_group, jobs)):
                for variant, batch, row in outputs:
                    if row["seed"] != registered["scenario_seed"] or row["noise_seed"] not in registered["noise_seeds"]:
                        raise ValueError("Stage136 training roster changed")
                    if batch is not None:
                        batches.append(batch)
                    training[variant].append(row)
                    count_row(cost, row, "collection" if variant == "sampled" else "training_comparator")
            diagnostic = {level: diagnose_level(trainer, batches, level) for level in ("upper", "lower")}
            for level in ("upper", "lower"):
                per_episode = args.horizon//period if level == "upper" else args.horizon
                cost[level + "_diagnostic_gradient_batches"] += 4 * int(np.ceil(spec.SCENARIOS*per_episode/joint.spec.MINIBATCH))
            states, updates = {"warm_start": initial, "source_forecast": initial}, {}
            for method in spec.METHODS:
                current = joint.make_trainer(model, teacher, args)
                joint.load_weights(current, initial)
                updates[method] = update_intervention(current, batches, root=root, period=period, method=method)
                for report in updates[method].values():
                    cost["ppo_update_calls"] += 1
                    for k in ("upper_actor_optimizer_steps", "upper_value_optimizer_steps",
                            "lower_actor_optimizer_steps", "lower_value_optimizer_steps"):
                        cost[k] += report["optimizer_steps"][k]
                torch.testing.assert_close(current.lower_actor.teacher.state_dict(), teacher, atol=0, rtol=0)
                torch.testing.assert_close(current.upper_actor.log_std, upper["log_std"], atol=0, rtol=0)
                states[method] = joint.weights(current)
            for level in ("upper", "lower"):
                for suffix in ("_actor", "_value"):
                    name = level + suffix
                    torch.testing.assert_close(states["joint"][name], states[level + "_only"][name], atol=0, rtol=0)
            jobs = [(source_weights, teacher, states, seed, [seed], period, predictor, envelope, False)
                for seed in roles["evaluation"]]
            evaluation = []
            for seed, outputs in zip(roles["evaluation"], pool.map(worker_group, jobs)):
                rows = {}
                for variant, _, row in outputs:
                    if row["seed"] != seed or row["noise_seed"] != seed:
                        raise ValueError("Stage136 evaluation roster changed")
                    if (row["reference_correction_peak"] > joint.spec.REFERENCE_LIMIT + 1e-8
                            or row["learned_residual_peak"] > joint.spec.RESIDUAL_LIMIT + 1e-8):
                        raise ValueError("Stage136 exceeded original control authority")
                    rows[variant] = row
                    count_row(cost, row, "evaluation")
                evaluation.append(rows)
            joint.source.native.curves.support.assert_frozen(model, source_snapshot)
            groups[str(period)] = {"selected_source": provenance, "diagnostic": diagnostic,
                "updates": updates, "layer_parameter_isolation": "passed", "source_teacher_and_std_frozen": "passed",
                "training_returns": {v: [r["episode_return"] for r in rows] for v, rows in training.items()},
                "initial_sampled_minus_mean": float(np.mean([a["episode_return"]-b["episode_return"]
                    for a, b in zip(training["sampled"], training["warm_start"])])), **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: one-round warm-start isolation complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage136 measured native/optimizer budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "contract": spec.contract(),
        "seed_roles": roles, "cost": cost, "groups": groups, "kind": "warm_start_joint_credit_development_diagnosis",
        "source_protocol": spec.source.PROTOCOL, "inherited_source_cost": cached["cost"],
        "inherited_earlier_source_cost": cached["inherited_source_cost"]}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    return result
