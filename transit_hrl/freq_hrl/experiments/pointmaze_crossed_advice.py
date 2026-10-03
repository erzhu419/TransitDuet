"""Cross plan inputs while holding each advice-trained lower policy fixed."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from . import pointmaze_optional_plan as source
from .pointmaze_root_response import write_json
from scripts import pointmaze_crossed_advice_stage108_spec as spec


def load_source(root):
    originals, predictor, _, calibration = source.load_source(root)
    cell = json.loads(spec.source_result(root).read_text())
    source.qualify(cell, preflight=False)
    models = {}
    for p in spec.PERIODS:
        models[str(p)] = {}
        for method in spec.LOWERS:
            path = spec.donor_checkpoint(root, p, method)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if ((saved["protocol"], saved["root"], saved["period"], saved["method"], saved["updates"]) !=
                    (spec.source.EXPERIMENT_PROTOCOL, root, p, method, 8)
                    or cell["groups"][str(p)]["trained"][method]["checkpoint"] != str(path)):
                raise ValueError("Crossed execution requires all registered final Stage107 donors")
            model = copy.deepcopy(originals[str(p)])
            snapshot = copy.deepcopy(model.state_dict())
            model.load_state_dict(saved["weights"])
            source.learning.check_training_freeze(model, snapshot, ("lower",))
            models[str(p)][method] = model
    return models, predictor, spec.source_record(root), calibration


def execution_weights(model, plan):
    weights = source.native.joint.inference_weights(model)
    if plan == "noise":
        weights = copy.deepcopy(weights)
        last = len(model.upper_actor.net)-1
        for suffix in ("weight", "bias"):
            weights["upper_actor"][f"net.{last}.{suffix}"].zero_()
    return weights


def production_variant(plan):
    return "learned_hint" if plan in ("learned", "noise") else "forecast_hint" if plan == "forecast" else "blind"


def worker_episode(job):
    weights, seed, variant, period, predictor, alpha, envelope = job
    lower, plan = spec.VARIANTS[variant]
    batch, row = source.worker_episode((weights, seed, seed, production_variant(plan), period, predictor, alpha, envelope, False))
    if batch is not None:
        raise ValueError("Fixed-policy execution must not create a training batch")
    row.update(variant=variant, lower_donor=lower, execution_mode=plan)
    return row


def check_row(row, period, horizon):
    lower, plan = spec.VARIANTS[row["variant"]]
    if (row["lower_donor"], row["execution_mode"]) != (lower, plan):
        raise ValueError("Crossed lower donor or plan intervention changed")
    source.check_row({**row, "variant": production_variant(plan)}, period, horizon)


def paired_effects(period, evaluation, seeds, root):
    if set(evaluation) != set(spec.VARIANTS) or any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("Crossed execution changed its eight-policy paired roster")
    for rows in evaluation.values():
        for row in rows:
            expected = source.scenario.spec.noise_seeds(root, row["seed"], row["seed"])
            if (row["noise_seed"], row["policy_seed"], row["lower_seed"]) != (row["seed"], *expected):
                raise ValueError("Crossed execution changed paired independent noise")
    for lower in spec.LOWERS:
        for learned, noise in zip(evaluation[f"{lower}_learned"], evaluation[f"{lower}_noise"]):
            np.testing.assert_allclose(learned["upper_standard_noise"], noise["upper_standard_noise"], atol=3e-5, rtol=0)
    return {f"{period}/{lower}/{a}_minus_{b}": float(np.mean([x["episode_return"]-y["episode_return"]
        for x, y in zip(evaluation[f"{lower}_{a}"], evaluation[f"{lower}_{b}"])]))
        for lower in spec.LOWERS for a, b in spec.CONTRAST_PAIRS}


def run(root, *, preflight, output):
    models, predictor, initialization, calibrations = load_source(root)
    args, roles, o = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    cost, planning, groups = dict.fromkeys(spec.budget(preflight=preflight), 0), dict.fromkeys(spec.planning_budget(preflight=preflight), 0), {}
    cost.update(source_clone_loads=2, forecaster_loads=1, decoder_loads=2, source_cell_loads=2,
        donor_checkpoint_loads=8, expanded_model_initializations=2, frozen_lower_models_initialized=4, donor_freeze_checks=4)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=o["workers"], mp_context=mp.get_context("spawn"), initializer=source.native.init_worker,
            initargs=(models[str(spec.PERIODS[0])][spec.LOWERS[0]].config, args)) as pool:
        for period in spec.PERIODS:
            calibration = calibrations[str(period)]
            evaluation = {}
            for lower in spec.LOWERS:
                model = models[str(period)][lower]
                snapshot = copy.deepcopy(model.state_dict())
                for plan in spec.PLAN_MODES:
                    variant = f"{lower}_{plan}"
                    weights = execution_weights(model, plan)
                    cost["zero_mean_upper_interventions"] += int(plan == "noise")
                    jobs = [(weights, s, variant, period, predictor, calibration["alpha"], calibration["envelope"])
                        for s in roles["native_evaluation"]]
                    rows = list(pool.map(worker_episode, jobs))
                    for row in rows:
                        check_row(row, period, args.horizon)
                        for k, v in (("native_episodes", 1), ("native_steps", row["episode_length"]),
                                ("native_lower_calls", row["lower_calls"]), ("native_upper_calls", row["upper_calls"]),
                                ("native_network_checks", 1)):
                            cost[k] += v
                        for k in planning:
                            planning[k] += row[k]
                    evaluation[variant] = rows
                source.native.curves.support.assert_frozen(model, snapshot)
                cost["frozen_model_checks"] += 1
            effects = paired_effects(period, evaluation, roles["native_evaluation"], root)
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            cost["upper_noise_pair_checks"] += len(spec.LOWERS)*len(roles["native_evaluation"])
            groups[str(period)] = {"evaluation": evaluation, "effects": effects,
                "task_options": spec.task_options(root, preflight=preflight), "source_and_Adam_unchanged": "passed"}
            print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}: all8 fixed-policy execution variants complete", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "source_initialization": initialization, "cost": cost,
        "native_planning_cost": planning, "groups": groups, "optimizer_steps": 0, "policy_updates": 0,
        "critic_fits": 0, "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0,
        "wall_seconds": time.monotonic()-started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent/"completion"/"ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["source_initialization"] != spec.source_record(cell["root"])
            or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight) or cell["native_planning_cost"] != spec.planning_budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps", "policy_updates", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Fixed-lower source, execution protocol or exact budget changed")
    h = spec.arguments(cell["root"], preflight=preflight).horizon
    for p, g in cell["groups"].items():
        if g["source_and_Adam_unchanged"] != "passed" or g["task_options"] != spec.task_options(cell["root"], preflight=preflight):
            raise ValueError("Fixed-policy source freeze or task changed")
        for variant, rows in g["evaluation"].items():
            for r in rows:
                if r["variant"] != variant:
                    raise ValueError("Crossed execution variant accounting changed")
                check_row(r, int(p), h)
        effects = paired_effects(p, g["evaluation"], cell["seed_roles"]["native_evaluation"], cell["root"])
        if effects != g["effects"] or not np.isfinite(list(effects.values())).all():
            raise ValueError("Crossed execution reward contrasts changed")
    return cell


def aggregate(cells, *, preflight):
    result = source.learning.native.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(primary_endpoints=list(spec.PRIMARY_ENDPOINTS),
        learned_residual_confirmation="mechanical_only" if preflight else (
            "supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"),
        performance_claim="fixed_lower_learned_upper_content_not_training_or_frequency_superiority",
        native_trial_prerequisite="Stage67_critic_HOLD_unchanged")
    return result
