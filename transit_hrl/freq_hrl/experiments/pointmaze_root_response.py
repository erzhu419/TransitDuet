"""From-scratch controllers and fit-before-query linear response qualification."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.core.plan_response import PlanResponseCritic
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from .pointmaze_budgeted_trigger import balanced_jitter_schedule
from .pointmaze_goal_validation import _json_ready, _training_seed
from .pointmaze_plan_validity_branching import _task_options
from .pointmaze_plan_value_qualification import (
    build_pointmaze_plan_value_model, pointmaze_plan_value_dimensions,
    rollout_hrl_pointmaze_plan_value, train_pointmaze_plan_value_cell,
)
from . import pointmaze_forecast_response as response
from . import pointmaze_plan_hold as hold
from . import pointmaze_separate_motion as motion
from scripts import pointmaze_root_response_stage33_spec as spec


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def raw_directory(output):
    directory = output.resolve().parent.with_name(output.parent.name + "_raw")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def scale_for(args):
    return PhysicalTimeScaleContract(dt_seconds=.01, upper_period_seconds=args.upper_period_seconds,
                                    history_seconds=args.history_seconds, fast_period_seconds=args.fast_period_seconds)


def schedule_for(args, seed):
    return balanced_jitter_schedule(seed=seed, horizon=args.horizon, period_steps=50, max_offset_steps=25)


def train_controller(args, output):
    started = time.monotonic()
    roles = spec.seed_roles(args.optimizer_seed, preflight=args.preflight)
    logged_seeds = {_training_seed(optimizer_seed=args.optimizer_seed, rollout_root=args.train_seeds[0], iteration=i): i
                    for i in range(args.iterations) if i % args.checkpoint_evaluation_interval == 0}

    def training_schedule(seed):
        if seed in logged_seeds:
            print(f"root {args.optimizer_seed}: starting iteration {logged_seeds[seed] + 1}/{args.iterations}; "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
        return schedule_for(args, seed)

    payload, controller = train_pointmaze_plan_value_cell(
        method=spec.POLICY, env_id=args.env_id, train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds, eval_seeds=(*args.branch_fit_seeds, *args.trigger_eval_seeds),
        iterations=args.iterations, horizon=args.horizon, optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds, history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds, maximum_subgoal_delta=args.maximum_subgoal_delta,
        reference_hidden_dim=args.reference_hidden_dim, learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        waypoint_perturbation=.25, event_window_seconds=1., task_options=_task_options(args),
        diagnostic_schedules=("fixed",), training_decision_steps_fn=training_schedule,
        training_schedule_name="balanced_jitter")
    raw = raw_directory(output)
    checkpoint = raw / "controller.pt"
    torch.save({"optimizer_seed": args.optimizer_seed, "options": spec.options(args.optimizer_seed, preflight=args.preflight),
                "selected_iteration": payload["selected_checkpoint_iteration"],
                "state_dict": controller.state_dict()}, checkpoint)
    compact = {key: payload[key] for key in (
        "selected_checkpoint_iteration", "history", "summary", "capacity", "dimensions", "config",
        "runtime_versions", "world_low", "world_high", "actor_optimizer_steps_train",
        "critic_optimizer_steps_train", "gradient_updates_train", "checkpoint_rank_contract")}
    compact.update(controller_checkpoint=str(checkpoint),
                   factual_row=payload["canonical_evaluation_rows"][0],
                   seed_roles=roles, budget=spec.budget(args.optimizer_seed, preflight=args.preflight),
                   wall_seconds=time.monotonic() - started)
    result = {"status": "complete", "protocol": {"protocol_version": spec.EXPERIMENT_PROTOCOL,
              "phase": "train", "optimizer_seed": args.optimizer_seed,
              "options": spec.options(args.optimizer_seed, preflight=args.preflight)}, "cells": [compact]}
    write_json(output, result)
    print(f"controller frozen at iteration {compact['selected_checkpoint_iteration']}; "
          f"{compact['budget']['controller_total_primitive_steps']} controller steps", flush=True)
    return result


def load_controller(args, source):
    result = json.loads(source.read_text())
    expected = _json_ready(spec.options(args.optimizer_seed, preflight=args.preflight))
    if (result["status"] != "complete" or result["protocol"]["phase"] != "train"
            or result["protocol"]["protocol_version"] != spec.EXPERIMENT_PROTOCOL
            or result["protocol"]["optimizer_seed"] != args.optimizer_seed
            or result["protocol"]["options"] != expected or len(result["cells"]) != 1):
        raise ValueError("Stage-33 controller is not the frozen fresh-root training result")
    cell = result["cells"][0]
    checkpoint = torch.load(cell["controller_checkpoint"], map_location="cpu", weights_only=False)
    if (checkpoint["optimizer_seed"] != args.optimizer_seed or _json_ready(checkpoint["options"]) != expected
            or checkpoint["selected_iteration"] != cell["selected_checkpoint_iteration"]):
        raise ValueError("Stage-33 controller checkpoint differs from its training result")
    scale = scale_for(args)
    dimensions = pointmaze_plan_value_dimensions(env_id=args.env_id, horizon=args.horizon,
                                                time_scale=scale, task_options=_task_options(args))
    controller, _ = build_pointmaze_plan_value_model(method=spec.POLICY, dimensions=dimensions,
        reference_hidden_dim=args.reference_hidden_dim, learning_rate=args.learning_rate,
        optimizer_seed=args.optimizer_seed)
    controller.load_state_dict(checkpoint["state_dict"])
    factual = cell["factual_row"]
    _, replay = rollout_hrl_pointmaze_plan_value(controller, method=spec.POLICY, env_id=args.env_id,
        seed=factual["seed"], sample=False, horizon=args.horizon,
        parameter_budget=cell["capacity"]["reference_parameter_budget"], time_scale=scale,
        maximum_subgoal_delta=args.maximum_subgoal_delta, waypoint_perturbation=.25, event_window_seconds=1.,
        task_options=_task_options(args), schedule_mode="balanced_jitter",
        decision_steps_override=schedule_for(args, factual["seed"]))
    errors = {k: abs(replay[k] - factual[k]) for k in ("episode_return", "tracking_squared_error_integral")}
    if max(errors.values()) > 1e-6 or replay["decision_steps"] != factual["decision_steps"]:
        raise RuntimeError("Stage-33 frozen controller does not replay the selected factual episode")
    return controller, cell, {"seed": factual["seed"], "absolute_errors": errors}


def sample_pairs(args, controller, scale, cases, *, label):
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=hold.init_worker, initargs=(controller, args, scale)) as pool:
        for future in as_completed([pool.submit(hold.sample_case, case) for case in cases]):
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == len(cases):
                print(f"{label} pairs: {len(rows)}/{len(cases)}", flush=True)
    return sorted(rows, key=lambda row: (row["seed"], row["check_step"]))


def fit_response(train, models, *, root):
    truth = np.stack([row["curve"] for row in train])
    critics, designs = {}, {}
    for method in response.METHODS:
        x = response.design(train, models, method=method, root=root)
        critics[method] = PlanResponseCritic(durations_seconds=np.asarray(hold.HORIZONS) * .01,
                                            ridge_alpha=1.).fit(x, truth)
        designs[method] = x
    return critics, designs


def predict_response(query, models, critics, *, root):
    designs = {method: response.design(query, models, method=method, root=root) for method in response.METHODS}
    predictions = {method: critics[method].predict_rates(x) for method, x in designs.items()}
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("Stage-33 response predictions are non-finite")
    return predictions, designs


def qualify(args, source, output):
    started = time.monotonic()
    controller, trained, factual = load_controller(args, source)
    scale = scale_for(args)
    roles = spec.seed_roles(args.optimizer_seed, preflight=args.preflight)
    hold.validate_paths(args, {"fit": roles["response_fit"], "evaluation": roles["response_eval"]},
                        {"temporal_seed_roles": {k: roles[k] for k in ("motion_fit", "motion_eval")}})
    protocol = {"horizon": args.horizon, "task_options": _task_options(args)}
    motion_rows = {role: motion.sample_rows(roles[role], horizon=args.horizon) for role in ("motion_fit", "motion_eval")}
    tapes = motion.tapes_for(roles["motion_fit"] + roles["motion_eval"], protocol=protocol)
    train_x, train_y = motion.labeled_motion(tapes, motion_rows["motion_fit"])
    query_x, query_y = motion.labeled_motion(tapes, motion_rows["motion_eval"])
    models, predictions = motion.fit_models(train_x, train_y, motion_rows["motion_fit"],
        query_x, motion_rows["motion_eval"], root=args.optimizer_seed)
    raw = raw_directory(output)
    write_json(raw / "motion_fits.json", {m: model.fitted for m, model in models.items()})
    np.savez_compressed(raw / "motion.npz", train_x=train_x, train_y=train_y, query_x=query_x, query_y=query_y,
        train_seeds=[r["seed"] for r in motion_rows["motion_fit"]], train_steps=[r["step"] for r in motion_rows["motion_fit"]],
        query_seeds=[r["seed"] for r in motion_rows["motion_eval"]], query_steps=[r["step"] for r in motion_rows["motion_eval"]],
        **{m + "_prediction": p for m, p in predictions.items()})
    motion_scores = motion.motion_metrics(query_y, predictions)
    print(f"motion frozen; diagnostic gate={motion_scores['motion_gate_passed']}; root retained", flush=True)
    fit_cases, query_cases, excluded = spec.response_cases(args.optimizer_seed, preflight=args.preflight)
    train = sample_pairs(args, controller, scale, fit_cases, label="fit")
    adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2, args.maximum_subgoal_delta, dtype=np.float32),
        world_low=np.asarray(trained["world_low"]), world_high=np.asarray(trained["world_high"]))
    for row in train:
        row["candidate_plan"], row["proposal_state"] = response.proposal(row, controller, adapter, history_steps=64)
    critics, designs = fit_response(train, models, root=args.optimizer_seed)
    write_json(raw / "response_fits.json", {m: critic.fitted for m, critic in critics.items()})
    print("all seven linear response heads frozen before query replay", flush=True)
    query = sample_pairs(args, controller, scale, query_cases, label="query")
    for row in query:
        row["candidate_plan"], row["proposal_state"] = response.proposal(row, controller, adapter, history_steps=64)
    predictions, query_designs = predict_response(query, models, critics, root=args.optimizer_seed)
    scores = [{**{k: v for k, v in row.items() if k not in ("sequence", "step_ise", "proposal_state")},
               "predicted_rates": {m: p[i] for m, p in predictions.items()}} for i, row in enumerate(query)]
    arrays = {f"{role}_{key}": np.stack([row[key] for row in rows])
              for role, rows in (("train", train), ("query", query))
              for key in ("sequence", "curve", "step_ise", "seed", "check_step", "candidate_plan", "proposal_state")}
    np.savez_compressed(raw / "response.npz", **arrays, feature_names=train[0]["feature_names"],
        **{m + "_" + k: v for m in response.METHODS for k, v in
           (("train_design", designs[m]), ("query_design", query_designs[m]), ("prediction", predictions[m]))})
    metrics = response.summarize(scores)
    budget = spec.budget(args.optimizer_seed, preflight=args.preflight)
    actual = {"fit_pair_primitive_steps": sum(row["primitive_steps"] for row in train),
              "query_pair_primitive_steps": sum(row["primitive_steps"] for row in query),
              "candidate_proposal_inference_calls": len(train) + len(query)}
    if any(actual[k] != budget[k] for k in actual):
        raise RuntimeError("Stage-33 measured branch cost differs from the frozen budget")
    cell = {"optimizer_seed": args.optimizer_seed, "controller_result": str(source),
            "controller_selected_iteration": trained["selected_checkpoint_iteration"], "seed_roles": roles,
            "factual_replay": factual, "budget": budget, "training_pairs": len(train), "evaluation_pairs": len(query),
            "excluded_incomplete_fit_prefixes": excluded, "controller_updates": 0,
            "motion": {"metrics": motion_scores, "fit_rows": len(train_x), "evaluation_rows": len(query_x),
                       "generated_exogenous_tape_points": sum(len(t) for t in tapes.values())},
            "rows": scores, "metrics": metrics, "raw_server_directory": str(raw),
            "raw_server_bytes": {p.name: p.stat().st_size for p in raw.iterdir() if p.is_file()},
            "root_point_gates_passed": metrics["prediction_gate_passed"] and metrics["decision_gate_passed"],
            "wall_seconds": time.monotonic() - started}
    write_json(output, {"status": "complete", "protocol": {"protocol_version": spec.EXPERIMENT_PROTOCOL,
        "phase": "response", "optimizer_seed": args.optimizer_seed,
        "options": spec.options(args.optimizer_seed, preflight=args.preflight),
        "qualification": spec.qualification_contract()}, "cells": [cell]})
    return cell


def root_aggregate(results):
    by_root = {result["protocol"]["optimizer_seed"]: result for result in results}
    if len(results) != len(spec.OPTIMIZER_ROOTS) or set(by_root) != set(spec.OPTIMIZER_ROOTS):
        raise ValueError("Stage-33 aggregation requires the entire eight-root roster without duplicates")
    vectors = []
    for root in spec.OPTIMIZER_ROOTS:
        result = by_root[root]
        if (result["status"] != "complete" or result["protocol"]["phase"] != "response"
                or result["protocol"]["protocol_version"] != spec.EXPERIMENT_PROTOCOL
                or result["protocol"]["options"] != _json_ready(spec.options(root, preflight=False))
                or result["protocol"]["qualification"] != spec.qualification_contract()):
            raise ValueError("Stage-33 root result differs from the frozen protocol")
        cell = result["cells"][0]
        fit_cases, query_cases, _ = spec.response_cases(root, preflight=False)
        expected = sorted((c["seed"], c["check_step"]) for c in query_cases)
        if (cell["seed_roles"] != spec.seed_roles(root, preflight=False)
                or cell["training_pairs"] != len(fit_cases) or len(cell["rows"]) != len(expected)
                or sorted((r["seed"], r["check_step"]) for r in cell["rows"]) != expected):
            raise ValueError("Stage-33 result has missing or substituted fit/query paths")
        scores = response.summarize(cell["rows"])
        mse = scores["settled_rate_mse"]
        vectors.append([scores["settled_ise_benefit_vs_control"][c] for c in spec.DECISION_CONTROLS]
                       + [mse[c] - mse["history"] for c in spec.PREDICTION_CONTROLS])
    values = np.asarray(vectors)
    rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
    draws = rng.integers(0, len(values), size=(spec.BOOTSTRAP_DRAWS, len(values)))
    tail = 100 * .05 / (2 * len(spec.ENDPOINTS))
    intervals = np.percentile(values[draws].mean(axis=1), [tail, 100 - tail], axis=0)
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "qualification": spec.qualification_contract(),
            "optimizer_roots": list(spec.OPTIMIZER_ROOTS), "root_endpoint_values": values.tolist(),
            "endpoints": {name: {"mean": float(values[:, i].mean()), "simultaneous_ci95": intervals[:, i].tolist()}
                          for i, name in enumerate(spec.ENDPOINTS)},
            "qualification_passed": bool(np.all(intervals[0] > 0))}
