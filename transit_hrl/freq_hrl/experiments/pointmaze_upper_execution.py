"""Freeze native learned policies and ablate only upper residual execution."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from . import pointmaze_joint_renewal as joint
from . import pointmaze_learned_plan as learned
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_upper_execution_stage56_spec as spec


class ExecutedPlan(learned.ResidualPlan):
    def __init__(self, predictor, period, scale, intervention):
        super().__init__(predictor, period, scale)
        if intervention not in spec.POLICIES:
            raise ValueError("unregistered upper execution intervention")
        self.intervention, self.proposed_actions = intervention, []

    def decode(self, *, action, **kwargs):
        self.proposed_actions.append(action.copy())
        executed = action if self.intervention == "normal" else np.zeros_like(action)
        return super().decode(action=executed, **kwargs)


def audit_intervention(raw, row, *, predictor, period, scale, bounds):
    proposed, executed = raw["upper_proposed_action"], raw["upper_plan_action"]
    np.testing.assert_equal(proposed.shape, (len(row["decision_steps"]), 4))
    np.testing.assert_equal(executed.shape, proposed.shape)
    expected = proposed if row["policy"] == "normal" else np.zeros_like(proposed)
    np.testing.assert_array_equal(executed, expected, err_msg="upper execution intervention changed")
    return learned.audit_plan(raw, row, predictor=predictor, period=period, scale=scale, bounds=bounds)


_WORKER = None


def init_worker(config, args):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = FrequencySeparatedActorCriticPPO(config), args


def worker_rollout(job):
    weights, seed, intervention, period, mode, predictor, path = job
    model, args = _WORKER
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    reference = ExecutedPlan(predictor, period, args.maximum_subgoal_delta, intervention)
    _, row, raw = joint.rollout(model, args, f"fixed{period}", seed=seed, capture=True,
        lower_credit="task_option", upper_plan_decoder=reference.decode, lower_reference_builder=reference,
        lower_actor_context_builder=reference.actor_context, lower_value_context_builder=reference.value_context,
        **spec.rollout_arguments(args.optimizer_seed, seed, mode=mode))
    torch.testing.assert_close(joint.inference_weights(model), weights, atol=0, rtol=0)
    raw.update(upper_proposed_action=np.asarray(reference.proposed_actions), upper_plan_action=np.asarray(reference.actions),
               upper_plan_coefficients=np.asarray(reference.coefficients))
    row.update(policy=intervention, period=period, deployment_mode=mode,
               policy_seed=spec.policy_seed(args.optimizer_seed, seed), lower_actor_type=type(model.lower_actor).__name__)
    row.update(
        frozen_network_check="passed", initial_upper_action=reference.proposed_actions[0].tolist(),
        proposed_action_rms=float(np.sqrt(np.square(raw["upper_proposed_action"].astype(np.float64)).mean())),
        executed_action_rms=float(np.sqrt(np.square(raw["upper_plan_action"].astype(np.float64)).mean())),
        executed_plan_delta_squared_sum=reference.executed_delta_squared_sum,
        plan_ols_fits=reference.ols_fits, plan_ridge_predictions=reference.ridge_predictions,
        reference_evaluations=reference.calls, actor_context_evaluations=reference.context_calls,
        upper_plan_decodes=len(reference.actions), bernstein_basis_evaluations=period + 1,
        audit_bernstein_basis_evaluations=period + 1,
        **audit_intervention(raw, row, predictor=predictor, period=period,
                             scale=args.maximum_subgoal_delta, bounds=reference.bounds))
    np.savez_compressed(path, **raw)
    return row


def load_source(root, *, preflight):
    path = spec.source_result(root, preflight=preflight)
    c = json.loads(path.read_text())
    if (c["status"] != "complete" or c["protocol"] != spec.previous.EXPERIMENT_PROTOCOL
            or c["contract"] != spec.previous.contract() or (c["root"], c["preflight"]) != (root, preflight)
            or c["options"] != spec.previous.options(preflight=preflight)
            or c["seed_roles"] != spec.previous.seed_roles(root, preflight=preflight)):
        raise ValueError("Stage56 source differs from frozen Stage55")
    forecaster = path.parent.with_name(path.parent.name + "_raw") / "forecaster.npz"
    with np.load(forecaster) as archive:
        predictor = {k: archive[k] for k in archive.files}
    models, checkpoints = {}, {}
    for period in spec.PERIODS:
        p = str(period)
        checkpoint = c["checkpoints"][p]["joint_ppo"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload["protocol"], payload["root"], payload["period"], payload["policy"]) != (
                spec.previous.EXPERIMENT_PROTOCOL, root, period, "joint_ppo"):
            raise ValueError("Stage56 requires fixed final joint checkpoint")
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**payload["state_dict"]["config"]))
        model.load_state_dict(payload["state_dict"])
        if model.config.__dict__ != c["config"] or (model.config.lower_state_dim, model.config.lower_value_state_dim,
                model.config.upper_action_dim, model.config.promotion_state_dim) != (392, 394, 4, 0):
            raise ValueError("Stage56 source learned architecture changed")
        models[p], checkpoints[p] = model, checkpoint
    return models, predictor, {"result": str(path), "checkpoints": checkpoints, "forecaster": str(forecaster),
        "config": c["config"], "upstream_budget_per_root": c["budget"]}


def evaluate(root, *, preflight, output):
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles, budget = spec.seed_roles(root, preflight=preflight), spec.budget(preflight=preflight)
    models, predictor, source = load_source(root, preflight=preflight)
    raw, started = raw_directory(output), time.monotonic()
    config = models[str(spec.PERIODS[0])].config
    rows = {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(config, args)) as pool:
        for period in spec.PERIODS:
            p = str(period)
            weights, rows[p] = joint.inference_weights(models[p]), {}
            for policy in spec.POLICIES:
                rows[p][policy] = {}
                for mode in spec.MODES:
                    directory = raw / p / policy / mode
                    directory.mkdir(parents=True, exist_ok=True)
                    stage = list(pool.map(worker_rollout, [(weights, seed, policy, period, mode, predictor,
                        str(directory / f"episode_{seed}.npz")) for seed in roles["evaluation"]]))
                    joint.audit_trajectories(stage, args=args, method=f"fixed{period}", raw_path=directory)
                    rows[p][policy][mode] = stage
            print(f"evaluated {root}/period{period}/both_execution_conditions", flush=True)
    counts = dict.fromkeys(learned.COUNT_KEYS, 0)
    for stage in rows.values():
        for modes in stage.values():
            for episodes in modes.values():
                for row in episodes:
                    counts["primitive_steps"] += row["episode_length"]
                    for key in learned.COUNT_KEYS[1:]:
                        counts[key] += row[key]
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "source": source,
        "evaluation_rows": rows, "inference_counts": counts, "native_trace_audits": budget["native_trace_audits"],
        "checkpoint_loads": len(models), "forecaster_loads": 1, "new_forecaster_fits": 0,
        "optimizer_steps": 0, "verification_primitive_steps": 0, "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def qualify(c, *, preflight):
    root, budget = c["root"], spec.budget(preflight=preflight)
    roles, horizon = spec.seed_roles(root, preflight=preflight), spec.arguments(root, preflight=preflight).horizon
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["preflight"] != preflight or c["options"] != spec.options(preflight=preflight)
            or c["seed_roles"] != roles or c["budget"] != budget
            or any(c[k] != budget[k] for k in ("native_trace_audits", "checkpoint_loads", "forecaster_loads",
                                              "new_forecaster_fits", "optimizer_steps", "verification_primitive_steps"))
            or set(c["evaluation_rows"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage56 frozen protocol or cost changed")
    observed, means = dict.fromkeys(learned.COUNT_KEYS, 0), {}
    for period in spec.PERIODS:
        p, stage = str(period), c["evaluation_rows"][str(period)]
        if set(stage) != set(spec.POLICIES) or any(set(modes) != set(spec.MODES) for modes in stage.values()):
            raise ValueError("Stage56 execution-condition roster incomplete")
        means[p] = {mode: {} for mode in spec.MODES}
        for mode in spec.MODES:
            normal, zero = stage["normal"][mode], stage["zero_residual"][mode]
            if [r["initial_upper_action"] for r in normal] != [r["initial_upper_action"] for r in zero]:
                raise ValueError("Stage56 paired initial upper actions differ")
            for policy in spec.POLICIES:
                rows = stage[policy][mode]
                if [r["seed"] for r in rows] != roles["evaluation"]:
                    raise ValueError("Stage56 paired fresh path roster changed")
                for row in rows:
                    kwargs = spec.rollout_arguments(root, row["seed"], mode=mode)
                    fits = horizon // period - 1
                    extras = {"plan_ols_fits": fits, "audit_ols_fits": fits, "plan_ridge_predictions": fits,
                        "audit_ridge_predictions": fits, "reference_evaluations": horizon, "actor_context_evaluations": horizon,
                        "upper_plan_decodes": horizon // period, "bernstein_basis_evaluations": period + 1,
                        "audit_bernstein_basis_evaluations": period + 1}
                    if (row["policy"] != policy or row["period"] != period or row["deployment_mode"] != mode
                            or row["method"] != f"fixed{period}" or row["episode_length"] != horizon
                            or row["decision_steps"] != list(range(0, horizon, period))
                            or row["upper_inference_calls"] != horizon // period or row["lower_inference_calls"] != horizon
                            or row["gate_inference_calls"] or row["candidate_preview_calls"]
                            or row["policy_seed"] != spec.policy_seed(root, row["seed"])
                            or row["lower_actor_type"] != "GaussianActor" or row["frozen_network_check"] != "passed"
                            or any(row[k] != v for k, v in kwargs.items() if k != "sample")
                            or any(row[k] != v for k, v in extras.items())):
                        raise ValueError("Stage56 learned execution or inference accounting changed")
                    if policy == "zero_residual":
                        if row["executed_action_rms"] != 0 or row["executed_plan_delta_squared_sum"] != 0:
                            raise ValueError("Stage56 zero residual executed a nonzero plan")
                    elif row["executed_action_rms"] != row["proposed_action_rms"]:
                        raise ValueError("Stage56 normal plan did not execute proposed action")
                    observed["primitive_steps"] += horizon
                    for key in learned.COUNT_KEYS[1:]:
                        observed[key] += row[key]
                if sum(r["proposed_action_rms"] for r in rows) <= 0 or (policy == "normal" and
                        sum(r["executed_plan_delta_squared_sum"] for r in rows) <= 0):
                    raise ValueError("Stage56 learned upper intervention was null")
                means[p][mode][policy] = {k: float(np.mean([r[k] for r in rows])) for k in
                    (*spec.METRICS, "proposed_action_rms", "executed_action_rms", "executed_plan_delta_squared_sum")}
    expected = {k: budget["total_primitive_steps"] if k == "primitive_steps" else budget[k] for k in learned.COUNT_KEYS}
    if observed != c["inference_counts"] or observed != expected:
        raise ValueError("Stage56 total native budget changed")
    return {"root": root, "means": means, "endpoints": spec.contrasts(means)}, observed


def bootstrap(rows):
    x = np.asarray([[r["endpoints"][k] for k in spec.ENDPOINTS] for r in rows])
    indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
    tail = .05 / (2 * len(spec.ENDPOINTS))
    bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
    return {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
        "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
        for i, k in enumerate(spec.ENDPOINTS)}


def aggregate(results, *, preflight):
    cells = {c["root"]: c for c in results}
    roots = spec.roots(preflight=preflight)
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("Stage56 root roster incomplete")
    rows, totals = [], dict.fromkeys(learned.COUNT_KEYS, 0)
    for root in roots:
        row, cost = qualify(cells[root], preflight=preflight)
        rows.append(row)
        for key in totals:
            totals[key] += cost[key]
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "method_cost": totals,
        **{k: sum(c[k] for c in results) for k in ("native_trace_audits", "checkpoint_loads", "forecaster_loads",
                                                 "new_forecaster_fits", "optimizer_steps", "verification_primitive_steps")}}
    if not preflight:
        endpoints = bootstrap(rows)
        summary.update(primary_endpoints=endpoints, upper_execution_gate="passed" if all(
            e["effect"] == "positive" for e in endpoints.values()) else "failed")
    return summary
