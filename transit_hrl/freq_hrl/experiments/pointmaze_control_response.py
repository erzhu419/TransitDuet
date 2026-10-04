"""Measure the plan-to-command channel on intact current-lower states."""
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp
import time

import numpy as np
import torch
from . import pointmaze_crossed_advice as crossed
from .pointmaze_root_response import write_json
from scripts import pointmaze_control_response_stage109_spec as spec

source = crossed.source
_BOUNDS = None


def init_worker(config, args):
    global _BOUNDS
    source.native.init_worker(config, args)
    task = source.native.joint._make_task(env_id=args.env_id, seed=args.optimizer_seed,
        horizon=args.horizon, **source.native.joint._task_options(args))
    try:
        _BOUNDS = source.native.joint.pointmaze_goal_bounds(task.environment)
    finally:
        task.environment.close()


def decode_points(base, plan, action, bounds, alpha):
    coefficients = plan.mapper.residual_coefficients(action)
    points = np.clip(base.astype(np.float64)+plan.basis@coefficients.reshape(2, plan.mapper.curve.basis_dim).T, *bounds).astype(np.float32)
    return source.native.curves.blend(base, points, alpha)


def counterfactual_states(model, states, *, predictor, period, scale, alpha, bounds, root, seed):
    indices = np.arange(0, len(states), spec.STRIDE)
    feedback = states[indices, :392]
    candidates = {k: np.r_[feedback.T, np.zeros((4, len(indices)), dtype=np.float32)].T.copy() for k in spec.PROBES}
    candidates["forecast"] = states[indices].copy()
    plan = source.native.curves.CalibratedPlan(predictor, period, scale, alpha, {})
    rng = np.random.default_rng(np.random.SeedSequence([109, root, seed, period]))
    upper_means, upper_std = [], None
    for start in range(0, len(states), period):
        targets = states[start, 6:390].reshape(64, 6)[-min(64, start+1):, :2]
        base = source.baseline.forecast.plan_points(targets, "ridge_velocity", predictor, period, bounds)
        with torch.inference_mode():
            dist = model.upper_actor.distribution(torch.as_tensor(states[start:start+1, :390]))
            mean, std = dist.mean[0].numpy(), dist.stddev[0].numpy()
        z = rng.normal(size=4).astype(np.float32)
        upper_means.append(mean.copy())
        upper_std = std.copy()
        actions = {"mean": mean, "sample": mean+std*z, "noise": std*z}
        for axis in range(4):
            offset = np.eye(4, dtype=np.float32)[axis]*spec.EPSILON
            actions.update({f"axis{axis}_plus": mean+offset, f"axis{axis}_minus": mean-offset})
        points = {"forecast": base, "zero": decode_points(base, plan, np.zeros(4), bounds, 1.)}
        points.update({f"{s}_{m}": decode_points(base, plan, action, bounds, alpha if s == "legacy" else 1.)
            for s in spec.SCALES for m, action in actions.items()})
        selected = np.flatnonzero((indices >= start)&(indices < start+period))
        ages = indices[selected]-start
        target = feedback[selected, 6:390].reshape(-1, 64, 6)[:, -1, :2]
        for name, curve in points.items():
            velocity = (curve[ages+1]-curve[ages])/source.baseline.forecast.spec.DT_SECONDS
            candidates[name][selected, 392:] = np.c_[curve[ages]-target, velocity-feedback[selected, -2:]]
    np.testing.assert_array_equal(candidates["forecast"], states[indices], err_msg="Causal forecast replay changed")
    np.testing.assert_array_equal(candidates["zero"], candidates["forecast"], err_msg="Zero residual changed forecast")
    for candidate in candidates.values():
        np.testing.assert_array_equal(candidate[:, :392], feedback)
    return candidates, np.asarray(upper_means), upper_std


def responses(actor, candidates):
    means = {}
    with torch.inference_mode():
        for name, states in candidates.items():
            dist = actor.distribution(torch.as_tensor(states))
            means[name] = dist.mean.numpy().astype(np.float64)
            std = dist.stddev[0].numpy().astype(np.float64)
    base = means["forecast"]
    metrics = {}
    for name, mean in means.items():
        difference = mean-base
        kl = .5*np.square(difference/std).sum(axis=1)
        metrics[name] = {"raw_mean_delta_rms": float(np.sqrt(np.square(difference).mean())),
            "command_delta_rms": float(np.sqrt(np.square(np.tanh(mean)-np.tanh(base)).mean())),
            "same_covariance_KL_mean": float(kl.mean()), "same_covariance_KL_q99": float(np.quantile(kl, .99)),
            "position_hint_rms": float(np.sqrt(np.square(candidates[name][:, 392:394]).mean())),
            "velocity_hint_rms": float(np.sqrt(np.square(candidates[name][:, 394:]).mean()))}
    for s in spec.SCALES:
        jacobian = np.stack([(np.tanh(means[f"{s}_axis{i}_plus"])-np.tanh(means[f"{s}_axis{i}_minus"]))/(2*spec.EPSILON)
            for i in range(4)], axis=-1)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        metrics[s+"_secant"] = {"command_Jacobian_frobenius_rms": float(np.sqrt(np.square(jacobian).sum(axis=(1,2)).mean())),
            "largest_singular_mean": float(singular[:, 0].mean()), "smallest_singular_mean": float(singular[:, 1].mean()),
            "sample_minus_noise_command_rms": float(np.sqrt(np.square(np.tanh(means[s+"_sample"])-np.tanh(means[s+"_noise"])).mean()))}
    return metrics


def worker_episode(job):
    weights, seed, period, predictor, alpha, envelope = job
    batch, native_row = source.worker_episode((weights, seed, seed, "forecast_hint", period, predictor, alpha, envelope, True))
    model, args = source.native._WORKER
    source.check_row(native_row, period, args.horizon)
    candidates, upper_mean, upper_std = counterfactual_states(model, batch.state, predictor=predictor, period=period,
        scale=args.maximum_subgoal_delta, alpha=alpha, bounds=_BOUNDS, root=args.optimizer_seed, seed=seed)
    metrics = responses(model.lower_actor, candidates)
    torch.testing.assert_close(source.native.joint.inference_weights(model), weights, atol=0, rtol=0)
    return {"seed": seed, "trajectory": native_row, "responses": metrics,
        "upper_mean_rms": float(np.sqrt(np.square(upper_mean.astype(np.float64)).mean())),
        "upper_std_rms": float(np.sqrt(np.square(upper_std.astype(np.float64)).mean())),
        "upper_tanh_mean_rms": float(np.sqrt(np.square(np.tanh(upper_mean.astype(np.float64))).mean())),
        "probe_states": len(candidates["forecast"]), "feedback_forecast_zero_checks": "passed", "network_check": "passed"}


def run(root, *, preflight, output):
    models, predictor, record, calibrations = crossed.load_source(root)
    args, roles, o = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    started, groups = time.monotonic(), {}
    with ProcessPoolExecutor(max_workers=o["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(models["50"]["learned_hint"].config, args)) as pool:
        for period in spec.PERIODS:
            model, cal = models[str(period)]["learned_hint"], calibrations[str(period)]
            snapshot = copy.deepcopy(model.state_dict())
            weights = source.native.joint.inference_weights(model)
            jobs = [(weights, s, period, predictor, cal["alpha"], cal["envelope"]) for s in roles["native_evaluation"]]
            groups[str(period)] = {"alpha": cal["alpha"], "rows": list(pool.map(worker_episode, jobs))}
            source.native.curves.support.assert_frozen(model, snapshot)
            print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}: fixed-state control responses complete", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "source_record": record, "seed_roles": roles, "cost": spec.budget(preflight=preflight),
        "source_loads": {"donor_checkpoints": 8, "source_cells": 2, "source_clones": 2, "forecasters": 1,
            "decoders": 2, "loaded_advice_models": 4, "probed_advice_models": 2},
        "policy_updates": 0, "critic_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0,
        "groups": groups, "wall_seconds": time.monotonic()-started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent/"completion"/"ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root, h = cell["root"], spec.arguments(cell["root"], preflight=preflight).horizon
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or root not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["source_record"] != spec.source.source_record(root) or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or cell["cost"] != spec.budget(preflight=preflight)
            or any(cell[k] for k in ("policy_updates", "critic_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Current-lower response changed protocol, donor, roster, budget or freeze")
    observed = dict.fromkeys(cell["cost"], 0)
    observed.update(worker_bounds_environment_constructions=spec.options(preflight=preflight)["workers"], source_freeze_checks=2)
    for p, group in cell["groups"].items():
        if not 0 < group["alpha"] <= 1 or [r["seed"] for r in group["rows"]] != cell["seed_roles"]["native_evaluation"]:
            raise ValueError("Control response alpha or scenario roster changed")
        for row in group["rows"]:
            trajectory = row["trajectory"]
            source.check_row(trajectory, int(p), h)
            expected_noise = source.scenario.spec.noise_seeds(root, row["seed"], row["seed"])
            if ((trajectory["seed"], trajectory["noise_seed"], trajectory["policy_seed"], trajectory["lower_seed"])
                    != (row["seed"], row["seed"], *expected_noise) or trajectory["variant"] != "forecast_hint"
                    or row["feedback_forecast_zero_checks"] != "passed" or row["network_check"] != "passed"
                    or row["probe_states"] != len(range(0, h, spec.STRIDE))
                    or set(row["responses"]) != set(spec.PROBES)|{s+"_secant" for s in spec.SCALES}
                    or not np.isfinite([v for m in row["responses"].values() for v in m.values()]).all()
                    or any(row["responses"][k]["command_delta_rms"] != 0 for k in ("forecast", "zero"))):
                raise ValueError("Response probes changed feedback, zero-action identity or independent trajectory noise")
            n, renewals = row["probe_states"], h//int(p)
            for k, v in {"native_episodes": 1, "native_steps": h, "native_lower_calls": h,
                    "probe_states": n, "probe_lower_mean_rows": len(spec.PROBES)*n,
                    "probe_upper_distribution_rows": renewals, "probe_forecast_reconstructions": renewals,
                    "probe_residual_curve_decodes": (len(spec.PROBES)-2)*renewals,
                    "forecast_ols_fits": trajectory["plan_ols_fits"], "forecast_ridge_predictions": trajectory["plan_ridge_predictions"],
                    "native_network_checks": 1, "probe_network_checks": 1, "counterfactual_feedback_checks": 1, "forecast_replay_checks": 1,
                    "zero_action_identity_checks": 1}.items():
                observed[k] += v
    if observed != cell["cost"]:
        raise ValueError("Control response measured episode or forward-row count changed")
    return cell


def aggregate(cells, *, preflight):
    if [c["root"] for c in cells] != list(spec.roots(preflight=preflight)):
        raise ValueError("Response diagnostic requires the complete ordered root cohort")
    for cell in cells:
        qualify(cell, preflight=preflight)
    root_rows = []
    for cell in cells:
        groups = {}
        for p, g in cell["groups"].items():
            responses_mean = {name: {metric: float(np.mean([r["responses"][name][metric] for r in g["rows"]]))
                for metric in g["rows"][0]["responses"][name]} for name in g["rows"][0]["responses"]}
            groups[p] = {"alpha": g["alpha"], "responses": responses_mean,
                **{k: float(np.mean([r[k] for r in g["rows"]])) for k in ("upper_mean_rms", "upper_std_rms", "upper_tanh_mean_rms")}}
        root_rows.append({"root": cell["root"], "groups": groups, "wall_seconds": cell["wall_seconds"]})
    periods = {str(p): {name: {metric: float(np.mean([r["groups"][str(p)]["responses"][name][metric] for r in root_rows]))
        for metric in root_rows[0]["groups"][str(p)]["responses"][name]} for name in spec.PROBES+tuple(s+"_secant" for s in spec.SCALES)}
        for p in spec.PERIODS}
    return {"status": "qualified", "protocol": spec.EXPERIMENT_PROTOCOL, "preflight": preflight,
        "root_rows": root_rows, "equal_root_response_means": periods,
        "cost": {k: sum(c["cost"][k] for c in cells) for k in cells[0]["cost"]},
        "performance_claim": "not_tested_same_state_response_only", "KL_reference": spec.KL_REFERENCE}
