"""Rebuild native initializers and causal BC teachers on an unseen cohort."""

from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from freq_hrl.rl.goal_conditioned_actor_critic import GoalConditionedActorCriticPPO
from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_plan_value_qualification as task_core
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from . import pointmaze_learned_plan as learned
from .pointmaze_root_response import raw_directory, scale_for, schedule_for, write_json
from scripts import pointmaze_fresh_teachers_stage96_spec as spec

_WORKER = None


def init_worker(goal_config, args, parameter_budget):
    global _WORKER
    torch.set_num_threads(1)
    controller = GoalConditionedActorCriticPPO(goal_config)
    clock = clocks.make_model(controller, "learned_history", root=args.optimizer_seed)
    _WORKER = controller, clock, learned.make_model(clock), args, parameter_budget


def worker_rollout(job):
    weights, seed, phase, period, predictor, gain, path, sample = job
    controller, clock, template, args, capacity = _WORKER
    model = {"controller": controller, "warmup": clock, "labels": template}[phase]
    for name, state in weights.items():
        getattr(model, name).load_state_dict(state)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed, phase))
    if phase == "controller":
        batch, row = task_core.rollout_hrl_pointmaze_plan_value(model, method="hrl_regime_history",
            schedule_mode="balanced_jitter", env_id=args.env_id, seed=seed, horizon=args.horizon, sample=sample,
            parameter_budget=capacity, time_scale=scale_for(args), maximum_subgoal_delta=args.maximum_subgoal_delta,
            waypoint_perturbation=.25, event_window_seconds=1., task_options=joint._task_options(args),
            decision_steps_override=schedule_for(args, seed))
        if row["protocol_valid"] != 1 or row["episode_length"] != args.horizon:
            raise ValueError("fresh controller native episode violates fixed horizon or schedule")
        row = {**{k: row[k] for k in ("seed", "episode_return", "episode_length", "decision_steps")},
            "upper_inference_calls": row["upper_decision_count"], "lower_inference_calls": row["episode_length"],
            "gate_inference_calls": 0}
    else:
        original, reference = model.lower_actor, None
        kwargs = {"lower_value_context_builder": clocks.context_builder("task_clock")}
        if phase == "labels":
            reference = learned.ResidualPlan(predictor, period, args.maximum_subgoal_delta)
            model.lower_actor = learned.forecast.VelocityFeedbackActor(original, gain, "waypoint")
            kwargs = {"upper_plan_decoder": reference.decode, "lower_reference_builder": reference,
                "lower_actor_context_builder": reference.actor_context, "lower_value_context_builder": reference.value_context}
        try:
            batch, row, raw = joint.rollout(model, args, "learned_history" if phase == "warmup" else f"fixed{period}",
                seed=seed, sample=True, capture=phase == "labels", upper_sample=False, gate_sample=False,
                lower_sample=phase == "warmup", lower_seed=spec.lower_seed(args.optimizer_seed, seed) if phase == "warmup" else None,
                lower_credit="task_option", **kwargs)
        finally:
            model.lower_actor = original
        if phase == "labels":
            raw.update(upper_plan_action=np.asarray(reference.actions), upper_plan_coefficients=np.asarray(reference.coefficients))
            row.update(**learned.audit_plan(raw, row, predictor=predictor, period=period,
                scale=args.maximum_subgoal_delta, bounds=reference.bounds, batch=batch))
            np.savez_compressed(path, **raw)
        row = {k: row[k] for k in ("seed", "episode_return", "episode_length", "decision_steps",
            "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}
    torch.testing.assert_close(joint.inference_weights(model), weights, atol=0, rtol=0)
    return batch, row


def make_controller(args):
    dimensions = task_core.pointmaze_plan_value_dimensions(env_id=args.env_id, horizon=args.horizon,
        time_scale=scale_for(args), task_options=joint._task_options(args))
    return task_core.build_pointmaze_plan_value_model(method="hrl_regime_history", dimensions=dimensions,
        reference_hidden_dim=args.reference_hidden_dim, learning_rate=args.learning_rate, optimizer_seed=args.optimizer_seed)


def run(root, *, preflight, output):
    args, opt, roles = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    controller, capacity = make_controller(args)
    raw, started, counts = raw_directory(output), time.monotonic(), {"episodes": 0, "steps": 0, "upper_calls": 0, "lower_calls": 0, "gate_calls": 0}
    records, checkpoints, groups = {}, {}, {}
    initial = joint.inference_weights(controller)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(controller.goal_config, args, capacity["reference_parameter_budget"])) as pool:
        def episodes(model, seeds, phase, *, period=0, predictor=None, gain=None, sample=True):
            directory = raw / str(period) / "teacher" / "labels" / "0" / "deterministic"
            if phase == "labels":
                directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(model)
            outputs = list(pool.map(worker_rollout, [(weights, seed, phase, period, predictor, gain,
                str(directory / f"episode_{seed}.npz") if phase == "labels" else None, sample) for seed in seeds]))
            for _, row in outputs:
                if row["seed"] not in seeds or row["episode_length"] != args.horizon:
                    raise ValueError("teacher build returned a different native seed or horizon")
                counts["episodes"] += 1
                counts["steps"] += row["episode_length"]
                for name in ("upper", "lower", "gate"):
                    counts[name + "_calls"] += row[name + "_inference_calls"]
            return outputs

        def checkpoint(name, model, **metadata):
            path = raw / f"{name}_final.pt"
            torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight,
                "config": model.config.__dict__, "weights": joint.inference_weights(model), **metadata}, path)
            checkpoints[name] = str(path)

        diagnostics = {"initial": [row for _, row in episodes(controller, roles["controller_diagnostic"], "controller", sample=False)]}
        history = []
        for iteration in range(1, opt["controller_iterations"] + 1):
            offset = (iteration - 1) * opt["controller_paths_per_iteration"]
            seeds = roles["controller_training"][offset:offset + opt["controller_paths_per_iteration"]]
            pairs = episodes(controller, seeds, "controller")
            np.random.seed(spec.shuffle_seed(root, iteration, "controller"))
            metrics = controller.update(concat_hierarchical_batches([b for b, _ in pairs]))
            history.append({"iteration": iteration, "seeds": seeds,
                "native_steps": sum(row["episode_length"] for _, row in pairs),
                "optimizer_steps": {k: int(v) for k, v in metrics.items() if "optimizer_steps" in k}})
            if iteration % 16 == 0 or iteration == opt["controller_iterations"]:
                print(f"root{root}: native initializer {iteration}/{opt['controller_iterations']}", flush=True)
        diagnostics["final"] = [row for _, row in episodes(controller, roles["controller_diagnostic"], "controller", sample=False)]
        checkpoint("controller", controller, iterations=opt["controller_iterations"], checkpoint_selection="fixed_final")
        records["controller"] = {"history": history, "diagnostics": diagnostics, "final_iteration": opt["controller_iterations"],
            "actor_change_norms": {
                name: float(np.sqrt(sum(float(torch.sum((value - initial[name][k]) ** 2)) for k, value in state.items())))
                for name, state in joint.inference_weights(controller).items() if name.endswith("actor")}}

        clock = clocks.make_model(controller, "learned_history", root=root)
        before, warm_history = joint.inference_weights(clock), []
        for iteration in range(1, opt["critic_warmup_iterations"] + 1):
            offset = (iteration - 1) * opt["warmup_paths_per_iteration"]
            seeds = roles["critic_warmup"][offset:offset + opt["warmup_paths_per_iteration"]]
            pairs = episodes(clock, seeds, "warmup")
            np.random.seed(spec.shuffle_seed(root, iteration, "warmup"))
            metrics = clock._update_level(level="lower", batch=concat_hierarchical_batches([b for b, _ in pairs]).lower,
                actor=clock.lower_actor, value_net=clock.lower_value, actor_optimizer=clock.lower_actor_optimizer,
                value_optimizer=clock.lower_value_optimizer, actor_updates_enabled=False)
            warm_history.append({"iteration": iteration, "seeds": seeds,
                "optimizer_steps": {k: int(v) for k, v in metrics.items() if "optimizer_steps" in k}})
        after = joint.inference_weights(clock)
        for name in before:
            if name != "lower_value":
                torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)
        checkpoint("clock", clock, iterations=opt["critic_warmup_iterations"], actor_freeze="passed")
        records["warmup"] = {"history": warm_history, "actor_and_other_value_freeze": "passed"}
        template = learned.make_model(clock)
        predictor, fitting = learned.forecast.fit_forecaster(args, roles["fitting"])
        np.savez_compressed(raw / "forecaster.npz", **predictor)
        gain = learned.feedback_gain()
        for period in spec.PERIODS:
            pairs = episodes(template, roles["labels"], "labels", period=period, predictor=predictor, gain=gain)
            labelled = concat_hierarchical_batches([batch for batch, _ in pairs]).lower
            model, cloning = learned.clone(template, labelled, root=root, period=period, epochs=opt["bc_epochs"], sham=False)
            checkpoint(f"clone_{period}", model, period=period, bc_epochs=opt["bc_epochs"], frozen_std_and_upper="passed")
            groups[str(period)] = {"cloning": cloning, "label_rows": [row for _, row in pairs],
                "clone_checkpoint": checkpoints[f"clone_{period}"], "label_archive_directory": str(raw / str(period) / "teacher" / "labels" / "0" / "deterministic")}
            print(f"root{root}: fresh teacher and BC complete at period{period}", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": spec.budget(preflight=preflight),
        "native_counts": counts, "records": records, "groups": groups, "fitting": fitting,
        "checkpoints": checkpoints, "forecaster": str(raw / "forecaster.npz"), "config": template.config.__dict__,
        "historical_artifact_loads": 0, "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "root": root, "protocol": spec.EXPERIMENT_PROTOCOL})
    return result


def qualify(cell, *, preflight):
    root, b = cell["root"], spec.budget(preflight=preflight)
    opt, roles = spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["preflight"] != preflight or cell["options"] != opt or cell["seed_roles"] != roles or cell["budget"] != b
            or cell["historical_artifact_loads"] != 0 or cell["native_counts"]["episodes"] != b["native_episodes"]
            or cell["native_counts"]["steps"] != b["native_steps"] or cell["native_counts"]["lower_calls"] != b["native_steps"]
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS} or len(cell["checkpoints"]) != b["checkpoint_writes"]):
        raise ValueError("fresh teacher roster or native build budget changed")
    for phase, role, iterations, fields in (("controller", "controller_training", opt["controller_iterations"],
            {k: b["controller_" + k.replace("optimizer_steps", "steps")] for k in (
                "upper_actor_optimizer_steps", "upper_value_optimizer_steps", "lower_actor_optimizer_steps", "lower_value_optimizer_steps")}),
            ("warmup", "critic_warmup", opt["critic_warmup_iterations"],
             {"lower_actor_optimizer_steps": 0, "lower_value_optimizer_steps": b["warmup_lower_value_steps"]})):
        history = cell["records"][phase]["history"]
        if ([r["iteration"] for r in history] != list(range(1, iterations + 1))
                or [seed for row in history for seed in row["seeds"]] != roles[role]
                or any(sum(row["optimizer_steps"][k] for row in history) != v for k, v in fields.items())):
            raise ValueError("fresh teacher optimizer or training path budget changed")
    controller = cell["records"]["controller"]
    if controller["final_iteration"] != opt["controller_iterations"] or cell["records"]["warmup"]["actor_and_other_value_freeze"] != "passed":
        raise ValueError("fresh teacher final checkpoint or warmup actor freeze changed")
    for rows in controller["diagnostics"].values():
        if [row["seed"] for row in rows] != roles["controller_diagnostic"]:
            raise ValueError("fresh teacher diagnostic paths changed")
    fit = cell["fitting"]
    if (fit["driver_paths"] != b["forecaster_driver_paths"] or fit["observations"] != b["forecaster_observations"]
            or fit["rows"] != b["forecaster_rows"] or fit["native_steps"] != 0 or fit["ridge_solves"] != 1):
        raise ValueError("fresh forecaster fitting budget changed")
    for group in cell["groups"].values():
        if ([row["seed"] for row in group["label_rows"]] != roles["labels"] or group["cloning"]["sham"]
                or group["cloning"]["steps"] != b["bc_optimizer_steps"] // len(spec.PERIODS)):
            raise ValueError("fresh teacher labels or fixed-final BC budget changed")
    return cell


def aggregate(cells, *, preflight):
    if len(cells) != len(spec.roots(preflight=preflight)) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("fresh teacher cohort incomplete")
    for cell in cells:
        qualify(cell, preflight=preflight)
        for file in (*cell["checkpoints"].values(), cell["forecaster"]):
            if not Path(file).is_file():
                raise ValueError("completed fresh teacher artifact missing")
        for group in cell["groups"].values():
            for seed in cell["seed_roles"]["labels"]:
                if not (Path(group["label_archive_directory"]) / f"episode_{seed}.npz").is_file():
                    raise ValueError("fresh decoder prerequisite label archive missing")
    return {"status": "preflight_passed" if preflight else "teacher_artifacts_ready",
        "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "roots": sorted(c["root"] for c in cells),
        "native_episodes": sum(c["budget"]["native_episodes"] for c in cells),
        "native_steps": sum(c["budget"]["native_steps"] for c in cells),
        "historical_artifact_loads": 0, "performance_confirmation": "not_tested",
        "next": "fresh_non_reward_decoder_calibration_then_full_fresh_staged_training"}
