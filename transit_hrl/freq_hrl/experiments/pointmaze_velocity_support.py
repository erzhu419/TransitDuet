"""Audit teacher-clone velocity support without fitting on native rewards."""

import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from . import pointmaze_learned_plan as learned
from . import pointmaze_matched_upper as native
from .pointmaze_independent_credit import assert_frozen
from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder
from .pointmaze_root_response import write_json
from freq_hrl.domains.mujoco.pointmaze_regime import PointMazeRegimeDriver, PointMazeRegimeObservation
from scripts import pointmaze_velocity_support_stage76_spec as spec


RAW_KEYS = ("measurement", "physical", "achieved_before", "target_before", "lower_reference",
    "lower_actor_context", "upper_plan_action", "action", "decision_steps")


def replay_plan(measurement, actions, *, predictor, period, scale, bounds):
    measurement, actions = np.asarray(measurement, dtype=np.float32), np.asarray(actions, dtype=np.float32)
    h = len(measurement)
    if h % period or actions.shape != (h // period, 4):
        raise ValueError("Stage76 curve replay option roster changed")
    base_reference, reference, base_velocity, velocity = [np.empty((h, 2), dtype=np.float32) for _ in range(4)]
    plan = learned.ResidualPlan(predictor, period, scale)
    for i, start in enumerate(range(0, h, period)):
        prefix = measurement[max(0, start - 63):start + 1]
        plan.decode(action=actions[i], observation=SimpleNamespace(task_measurement=measurement[start]),
            history=SimpleNamespace(history=prefix.reshape(-1)), step=start, world_low=bounds[0], world_high=bounds[1])
        base_reference[start:start + period], reference[start:start + period] = plan.base_points[:-1], plan.points[:-1]
        base_velocity[start:start + period] = np.diff(plan.base_points, axis=0) / learned.forecast.spec.DT_SECONDS
        velocity[start:start + period] = np.diff(plan.points, axis=0) / learned.forecast.spec.DT_SECONDS
    return base_reference, reference, base_velocity, velocity, {
        "plan_decodes": len(actions), "plan_ols_fits": plan.ols_fits, "plan_ridge_predictions": plan.ridge_predictions,
        "bernstein_basis_evaluations": period + 1}


def label_states(raw, args):
    history = PointMazeRegimeFeatureBuilder(time_scale=native.joint.scale_for(args))
    states = []
    for t in range(args.horizon):
        m = raw["measurement"][t]
        obs = PointMazeRegimeObservation(physical=raw["physical"][t], achieved_goal=raw["achieved_before"][t],
            target=raw["target_before"][t], task_measurement=m, force=m[2:4], distractor=m[4:6])
        history.reset(obs) if t == 0 else history.update(obs)
        states.append(np.r_[history.lower_state(obs, subgoal=raw["lower_reference"][t]), raw["lower_actor_context"][t]])
    return np.asarray(states, dtype=np.float32)


def fit_support(velocity, position_error):
    v, error = np.asarray(velocity, dtype=np.float64), np.asarray(position_error, dtype=np.float64)
    speed = np.linalg.norm(v, axis=1)
    return {"rows": len(v), "axis_min": v.min(0).tolist(), "axis_max": v.max(0).tolist(),
        "axis_mean": v.mean(0).tolist(), "axis_std": v.std(0).tolist(),
        "velocity_speed_q99": float(np.quantile(speed, spec.SUPPORT_QUANTILE)),
        "velocity_speed_rms": float(np.sqrt(np.square(speed).mean())),
        "position_error_q99": float(np.quantile(np.linalg.norm(error, axis=1), spec.SUPPORT_QUANTILE)),
        "position_error_rms": float(np.sqrt(np.square(error).sum(1).mean())),
        "data_role": "historical_BC_labels_only"}


def profile(velocity, support):
    v = np.asarray(velocity, dtype=np.float64)
    speed = np.linalg.norm(v, axis=1)
    outside = np.any((v < support["axis_min"]) | (v > support["axis_max"]), axis=1)
    return {"rows": len(v), "axis_min": v.min(0).tolist(), "axis_max": v.max(0).tolist(),
        "speed_quantiles": np.quantile(speed, [.5, .95, .99, 1.]).tolist(),
        "speed_rms": float(np.sqrt(np.square(speed).mean())),
        "outside_label_axis_range_rate": float(outside.mean()),
        "above_label_speed_q99_rate": float((speed > support["velocity_speed_q99"]).mean())}


def actor_responses(actor, states, target_commands, sampled_velocity):
    totals = {v: dict.fromkeys(("command_error_squared_sum", "command_change_squared_sum", "gaussian_kl_sum"), 0.) for v in spec.VARIANTS}
    batches = 0
    with torch.inference_mode():
        for start in range(0, len(states), spec.CHUNK_SIZE):
            x = torch.as_tensor(states[start:start + spec.CHUNK_SIZE], dtype=torch.float32)
            y = torch.as_tensor(target_commands[start:start + len(x)], dtype=torch.float32)
            old = actor.distribution(x)
            for variant in spec.VARIANTS:
                changed = x.clone()
                if variant == "zero_velocity":changed[:, -2:] = 0.
                if variant == "sampled_velocity":
                    changed[:, -2:] = torch.as_tensor(sampled_velocity[start:start + len(x)], dtype=torch.float32)
                current = old if variant == "as_label" else actor.distribution(changed)
                totals[variant]["command_error_squared_sum"] += float((current.mean.tanh().double() - y.double()).square().sum())
                totals[variant]["command_change_squared_sum"] += float((current.mean.tanh().double() - old.mean.tanh().double()).square().sum())
                totals[variant]["gaussian_kl_sum"] += float(torch.distributions.kl_divergence(old, current).double().sum())
                batches += 1
    return totals, {"offline_actor_rows": len(spec.VARIANTS) * len(states), "offline_actor_forward_batches": batches}


def response_summary(totals, rows):
    return {v: {"bc_command_mse": d["command_error_squared_sum"] / (2 * rows),
        "command_change_rms": float(np.sqrt(d["command_change_squared_sum"] / (2 * rows))),
        "conditional_gaussian_kl_mean": d["gaussian_kl_sum"] / rows} for v, d in totals.items()}


def check_native_frame(base_reference, reference, base_velocity, velocity, measurement, row):
    dt = learned.forecast.spec.DT_SECONDS
    expected = {"reference_target_squared_error_integral": float(np.square(reference.astype(np.float64) - measurement[:, :2]).sum() * dt),
        "reference_residual_squared_integral": float(np.square(reference.astype(np.float64) - base_reference).sum() * dt),
        "velocity_residual_squared_integral": float(np.square(velocity.astype(np.float64) - base_velocity).sum() * dt)}
    for k, value in expected.items():
        np.testing.assert_allclose(value, row[k], atol=1e-9, rtol=1e-9, err_msg="Stage76 native plan replay differs from Stage75")


def effects(period, coverage):
    return {f"{period}/{k}/residual_minus_base": coverage["residual"][k] - coverage["base"][k] for k in spec.COVERAGE_METRICS}


def layout_bounds(args, seed):
    task = native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **native.joint._task_options(args))
    try:return native.joint.pointmaze_goal_bounds(task.environment)
    finally:task.environment.close()


def run(root, *, preflight, output):
    old = json.loads(spec.source_result(root, preflight=preflight).read_text())
    if (old["status"], old["protocol"], old["root"], old["preflight"], old["contract"]) != (
            "complete", spec.source.EXPERIMENT_PROTOCOL, root, preflight, spec.source.contract()):
        raise ValueError("Stage76 requires completed frozen Stage75 source")
    clones, predictor, initialization = native.load_source(root, preflight=preflight)
    labels_file = spec.clones.source_result(root, preflight=preflight)
    labels = json.loads(labels_file.read_text())
    roles, args = spec.seed_roles(root, preflight=preflight), spec.arguments(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    bounds = layout_bounds(args, roles["calibration_labels"][0])
    cost["layout_loads"] += 1
    for period in spec.PERIODS:
        p, model = str(period), clones[str(period)]
        snapshot = copy.deepcopy(model.state_dict())
        label_velocity, position_error = [], []
        total = {v: dict.fromkeys(("command_error_squared_sum", "command_change_squared_sum", "gaussian_kl_sum"), 0.) for v in spec.VARIANTS}
        for seed in roles["calibration_labels"]:
            with np.load(spec.label_archive(root, period, seed, preflight=preflight)) as a:
                raw = {k: a[k] for k in RAW_KEYS}
            np.testing.assert_array_equal(raw["upper_plan_action"], np.zeros((args.horizon // period, 4)))
            np.testing.assert_array_equal(raw["decision_steps"], np.arange(0, args.horizon, period))
            states = label_states(raw, args)
            torch.manual_seed(spec.counterfactual_seed(root, seed, period))
            std = model.upper_actor.log_std.detach().exp().clamp(1e-4, 3.)
            proposed = torch.distributions.Normal(torch.zeros(args.horizon // period, 4), std).sample().numpy()
            br, _, bv, v, planning = replay_plan(raw["measurement"], proposed, predictor=predictor, period=period,
                scale=args.maximum_subgoal_delta, bounds=bounds)
            np.testing.assert_array_equal(br, raw["lower_reference"])
            np.testing.assert_array_equal(bv, raw["lower_actor_context"])
            value, actor_cost = actor_responses(model.lower_actor, states, raw["action"], v)
            for variant, d in value.items():
                for k in d:total[variant][k] += d[k]
            for k, n in {**actor_cost, **planning}.items():cost[k] += n
            cost["counterfactual_upper_proposals"] += len(proposed)
            cost["label_archive_loads"] += 1
            cost["label_state_rows"] += len(states)
            cost["label_plan_checks"] += 1
            cost["replayed_velocity_rows"] += 2 * len(v)
            label_velocity.append(bv)
            position_error.append(br.astype(np.float64) - raw["measurement"][:, :2])
        support = fit_support(np.concatenate(label_velocity), np.concatenate(position_error))
        responses = response_summary(total, support["rows"])
        saved = labels["cloning"][p]["clone"]["final_action_mse"]
        observed = responses["as_label"]["bc_command_mse"]
        np.testing.assert_allclose(observed, saved, atol=1e-7, rtol=1e-5, err_msg="Stage76 BC state reconstruction changes saved clone command MSE")
        cost["bc_mse_reproductions"] += 1
        arrays = {"base": [], "residual": []}
        ev = old["groups"][p]["evaluation"]
        seeds = roles["native_plan_replay"]
        if any([r["seed"] for r in ev[m]] != seeds for m in ("R0V0", "R1V1")):
            raise ValueError("Stage76 native plan replay roster changed")
        for z, row in zip(ev["R0V0"], ev["R1V1"]):
            if any(z[k] != row[k] for k in ("upper_proposed_actions", "decision_steps", "policy_seed", "lower_seed")):
                raise ValueError("Stage76 Stage75 proposal pairing changed")
            if row["decision_steps"] != list(range(0, args.horizon, period)):
                raise ValueError("Stage76 native plan decision schedule changed")
            driver = PointMazeRegimeDriver(seed=row["seed"], horizon=args.horizon,
                dt_seconds=learned.forecast.spec.DT_SECONDS, **native.joint._task_options(args))
            measurement = np.asarray([np.concatenate(driver.sample(t)) for t in range(args.horizon)], dtype=np.float32)
            br, r, bv, v, planning = replay_plan(measurement, row["upper_proposed_actions"], predictor=predictor,
                period=period, scale=args.maximum_subgoal_delta, bounds=bounds)
            check_native_frame(br, r, bv, v, measurement, row)
            for k, n in planning.items():cost[k] += n
            cost["native_driver_paths"] += 1
            cost["native_plan_frame_checks"] += 1
            cost["replayed_velocity_rows"] += 2 * len(v)
            arrays["base"].append(bv);arrays["residual"].append(v)
        coverage = {k: profile(np.concatenate(v), support) for k, v in arrays.items()}
        assert_frozen(model, snapshot)
        cost["frozen_model_checks"] += 1
        groups[p] = {"calibration": support, "label_plan_check": "passed",
            "bc_mse_reproduction": {"saved": saved, "observed": observed, "status": "passed"},
            "actor_responses": responses, "native_coverage": coverage,
            "native_plan_frame_check": "passed", "source_and_Adam_unchanged": "passed", "effects": effects(p, coverage)}
        print(f"velocity support {root}/{period}: labels, clone commands and native plan frames reproduced", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "groups": groups, "cost": cost, "source_initialization": initialization,
        "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight) or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage76 frozen audit protocol or budget changed")
    roles, h = cell["seed_roles"], spec.arguments(cell["root"], preflight=preflight).horizon
    for p, g in cell["groups"].items():
        if (any(g[k] != "passed" for k in ("label_plan_check", "native_plan_frame_check", "source_and_Adam_unchanged"))
                or g["bc_mse_reproduction"]["status"] != "passed" or g["calibration"]["data_role"] != "historical_BC_labels_only"
                or g["calibration"]["rows"] != len(roles["calibration_labels"]) * h
                or set(g["actor_responses"]) != set(spec.VARIANTS) or set(g["native_coverage"]) != {"base", "residual"}
                or g["effects"] != effects(p, g["native_coverage"])):
            raise ValueError("Stage76 input calibration, replay or actor response changed")
        for c in g["native_coverage"].values():
            if c["rows"] != len(roles["native_plan_replay"]) * h or any(not 0 <= c[k] <= 1 for k in spec.COVERAGE_METRICS):
                raise ValueError("Stage76 native input coverage accounting changed")
        np.testing.assert_allclose(g["bc_mse_reproduction"]["observed"], g["bc_mse_reproduction"]["saved"], atol=1e-7, rtol=1e-5)
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage76 requires every frozen root")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    values = [{k: v for g in c["groups"].values() for k, v in g["effects"].items()} for c in rows]
    x = np.asarray([[v[k] for k in spec.ENDPOINTS] for v in values])
    endpoints = {k: {"mean": float(x[:, i].mean())} for i, k in enumerate(spec.ENDPOINTS)}
    if not preflight:
        idx = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(rows), (spec.BOOTSTRAP_DRAWS, len(rows)))
        tail = .05 / (2 * len(spec.ENDPOINTS))
        bounds = np.quantile(x[idx].mean(1), [tail, 1-tail], axis=0)
        for i, k in enumerate(spec.ENDPOINTS):
            endpoints[k].update(ci=bounds[:, i].tolist(), effect="positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive")
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows, "endpoints": endpoints,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_offline_input_support_and_calibration_only"}
