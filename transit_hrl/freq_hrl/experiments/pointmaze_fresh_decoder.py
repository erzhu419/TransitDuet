"""Rebuild coherent bounded decoders without any original-cohort artifact."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from . import pointmaze_fresh_teachers as teachers
from . import pointmaze_calibrated_residual as curves
from . import pointmaze_bounded_residual as bounded
from . import pointmaze_feasible_credit as native
from .pointmaze_root_response import write_json
from .pointmaze_goal_validation import _json_ready
from scripts import pointmaze_fresh_decoder_stage97_spec as spec


def load_source(root):
    path = spec.source_result(root)
    cell = json.loads(path.read_text())
    teachers.qualify(cell, preflight=False)
    if cell["root"] != root:
        raise ValueError("fresh decoder source root changed")
    models = {}
    for period in spec.PERIODS:
        payload = torch.load(cell["checkpoints"][f"clone_{period}"], map_location="cpu", weights_only=False)
        if ((payload["protocol"], payload["root"], payload["preflight"], payload["period"], payload["bc_epochs"]) !=
                (spec.source.EXPERIMENT_PROTOCOL, root, False, period, 64) or payload["frozen_std_and_upper"] != "passed"):
            raise ValueError("fresh decoder requires the fixed-final new cohort BC clone")
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**payload["config"]))
        if _json_ready(payload["config"]) != cell["config"]:
            raise ValueError("fresh decoder source architecture differs from recorded teacher")
        for name, weights in payload["weights"].items():
            getattr(model, name).load_state_dict(weights)
        for tensor in (model.upper_actor.net[-1].weight, model.upper_actor.net[-1].bias):
            torch.testing.assert_close(tensor, torch.zeros_like(tensor), atol=0, rtol=0)
        models[str(period)] = model
    with np.load(cell["forecaster"]) as archive:
        predictor = {k: archive[k] for k in archive.files}
    return models, predictor, cell


def calibrate(root, period, *, model, predictor, source, bounds, cost):
    args, roles = spec.source.arguments(root, preflight=False), {"calibration_labels": source["seed_roles"]["labels"]}
    saved = source["groups"][str(period)]["cloning"]["final_action_mse"]
    cache, target, base_command, responses = curves.prepare_calibration(root, period, model=model,
        predictor=predictor, args=args, roles=roles, bounds=bounds, saved_bc_mse=saved, cost=cost,
        preflight=False, label_archive=spec.label_archive, proposal_seed=spec.proposal_seed)
    states = np.concatenate([c[0] for c in cache])
    error = np.concatenate([c[3][:, :-1].reshape(-1, 2) - c[1] for c in cache])
    envelope = curves.support.fit_support(states[:, -2:], error)
    ratio_alpha = curves.calibration_alpha(responses["zero"]["bc_command_mse"], responses["original"]["command_change_rms"])

    def evaluate(alpha):
        cost["constraint_response_evaluations"] += 1
        return curves.response(curves.evaluate_curve(model.lower_actor, cache, alpha, cost), base_command, target)

    responses["ratio"] = evaluate(ratio_alpha)
    target_rms = float(np.sqrt(responses["zero"]["bc_command_mse"]))
    trace = bounded.contract_response(ratio_alpha, target_rms, evaluate, responses["ratio"])
    responses["bounded"] = trace[-1]["response"]
    result = {"alpha": trace[-1]["alpha"], "ratio_alpha": ratio_alpha, "target_rms": target_rms,
        "responses": responses, "solver_trace": trace, "saved_bc_mse": saved, "envelope": envelope,
        "bounded_to_bc_rmse_ratio": responses["bounded"]["command_change_rms"] / target_rms,
        "data_role": "new_cohort_BC_labels_only", "bc_mse_reproduction": "passed", "label_plan_checks": "passed",
        "constraint": "passed"}
    check_calibration(result)
    return result


def check_calibration(c):
    bounded.check_response_constraint(c)
    np.testing.assert_allclose(c["responses"]["zero"]["bc_command_mse"], c["saved_bc_mse"], atol=1e-7, rtol=1e-5)
    if (c["data_role"] != "new_cohort_BC_labels_only" or c["bc_mse_reproduction"] != "passed"
            or c["label_plan_checks"] != "passed" or not 0 <= c["alpha"] <= c["ratio_alpha"] <= 1):
        raise ValueError("fresh decoder calibration or label role changed")


def worker_probe(job):
    weights, seed, mode, period, predictor, alpha, envelope = job
    _, args = native._WORKER
    policy_seed, lower_seed = spec.noise_seeds(args.optimizer_seed, seed)
    batch, row = native.native_episode(weights, seed=seed, variant=mode, period=period, predictor=predictor,
        alpha=alpha, envelope=envelope, collect=False, policy_seed=policy_seed, lower_seed=lower_seed)
    if batch is not None:
        raise ValueError("decoder probe unexpectedly collected training data")
    return row


def check_probes(root, period, evaluation, seeds, horizon, alpha):
    if set(evaluation) != set(spec.MODES) or any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("fresh decoder native probe roster changed")
    for mode, rows in evaluation.items():
        for row, zero in zip(rows, evaluation["zero"]):
            native.curves.paths.check_row(row, period, horizon)
            expected = spec.noise_seeds(root, row["seed"])
            if ((row["policy_seed"], row["lower_seed"]) != expected or row["variant"] != mode
                    or row["alpha"] != (0. if mode == "zero" else alpha) or row["upper_replay_forward_calls"] != 0):
                raise ValueError("fresh decoder probe sampling or frozen alpha changed")
            np.testing.assert_allclose(row["upper_standard_noise"], zero["upper_standard_noise"], atol=2e-6, rtol=0)


def run(root, *, preflight, output):
    started = time.monotonic()
    models, predictor, source = load_source(root)
    args, roles = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost = dict.fromkeys(spec.budget(preflight=preflight), 0)
    cost.update(source_clone_loads=len(models), forecaster_loads=1, layout_loads=1)
    bounds = curves.support.layout_bounds(args, roles["calibration_labels"][0])
    snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    groups = {}
    # Both decoders are fixed before querying any native reward.
    for period in spec.PERIODS:
        groups[str(period)] = {"calibration": calibrate(root, period, model=models[str(period)], predictor=predictor,
            source=source, bounds=bounds, cost=cost)}
        print(f"root{root}/period{period}: fresh alpha={groups[str(period)]['calibration']['alpha']:.8f} frozen", flush=True)
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS, 0)
    with ProcessPoolExecutor(max_workers=spec.options(preflight=preflight)["workers"], mp_context=mp.get_context("spawn"),
            initializer=native.init_worker, initargs=(models[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, model = str(period), models[str(period)]
            c, weights = groups[p]["calibration"], native.joint.inference_weights(model)
            ev = {mode: list(pool.map(worker_probe, [(weights, seed, mode, period, predictor,
                0. if mode == "zero" else c["alpha"], c["envelope"]) for seed in roles["native_probe"]])) for mode in spec.MODES}
            check_probes(root, period, ev, roles["native_probe"], args.horizon, c["alpha"])
            cost["native_pair_checks"] += len(roles["native_probe"])
            for rows in ev.values():
                for row in rows:
                    cost["native_episodes"] += 1
                    cost["native_steps"] += row["episode_length"]
                    cost["native_lower_calls"] += row["lower_calls"]
                    cost["native_upper_calls"] += row["upper_calls"]
                    cost["native_network_checks"] += 1
                    for key in planning: planning[key] += row[key]
            curves.support.assert_frozen(model, snapshots[p])
            cost["frozen_model_checks"] += 1
            groups[p].update(evaluation=ev, source_and_Adam_unchanged="passed", pairing="passed")
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "groups": groups, "native_planning_cost": planning,
        "source_result": str(spec.source_result(root)), "source_checkpoints": source["checkpoints"],
        "wall_seconds": time.monotonic() - started, "performance_confirmation": "not_tested"}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "root": root})
    return cell


def qualify(cell, *, preflight):
    root, roles = cell["root"], spec.seed_roles(cell["root"], preflight=preflight)
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or root not in spec.roots(preflight=preflight) or cell["preflight"] != preflight or cell["seed_roles"] != roles
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or cell["cost"] != spec.realized_budget(cell, preflight=preflight)
            or cell["source_result"] != str(spec.source_result(root)) or cell["performance_confirmation"] != "not_tested"):
        raise ValueError("fresh decoder source, budget or protocol changed")
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS, 0)
    for period in spec.PERIODS:
        g = cell["groups"][str(period)]
        check_calibration(g["calibration"])
        if (g["calibration"]["envelope"]["rows"] != len(roles["calibration_labels"]) * spec.source.arguments(root, preflight=False).horizon
                or g["pairing"] != "passed" or g["source_and_Adam_unchanged"] != "passed"):
            raise ValueError("fresh decoder full-label envelope or source freeze changed")
        check_probes(root, period, g["evaluation"], roles["native_probe"], spec.arguments(root, preflight=preflight).horizon, g["calibration"]["alpha"])
        for rows in g["evaluation"].values():
            for row in rows:
                for key in planning: planning[key] += row[key]
    if cell["native_planning_cost"] != planning:
        raise ValueError("fresh decoder native planning budget changed")
    return cell


def aggregate(cells, *, preflight):
    if len(cells) != len(spec.roots(preflight=preflight)) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("fresh decoder cohort incomplete")
    for cell in cells: qualify(cell, preflight=preflight)
    return {"status": "preflight_passed" if preflight else "decoder_artifacts_ready", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "roots": sorted(c["root"] for c in cells),
        "cost": {k: sum(c["cost"][k] for c in cells) for k in spec.budget(preflight=preflight)},
        "performance_confirmation": "not_tested", "next": "fresh_joint_and_lower_donors_then_unchanged_staged_upper_rule"}
