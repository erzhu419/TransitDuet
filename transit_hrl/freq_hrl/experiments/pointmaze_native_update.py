"""Evaluate exact archived first-update policies in the native control loop."""

import json
import time

import numpy as np
import torch
from . import pointmaze_first_update as archive
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as native
from . import pointmaze_joint_renewal as joint
from . import pointmaze_learned_plan as learned
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_native_update_stage61_spec as spec


def init_worker(config, args):
    diagnostics.init_worker(config, args)
    native.init_worker(config, args)


def evaluate(pool, model, predictor, args, *, arm, period, seeds, directory):
    directory.mkdir(parents=True, exist_ok=True)
    weights = joint.inference_weights(model)
    outputs = list(pool.map(native.worker_rollout, [(weights, seed, arm, period, "eval", spec.MODE,
        predictor, str(directory / f"episode_{seed}.npz")) for seed in seeds]))
    rows = [row for _, row in outputs]
    joint.audit_trajectories(rows, args=args, method=f"fixed{period}", raw_path=directory)
    return rows


def train(root, *, preflight, output):
    reference = json.loads(spec.source_result(root, preflight=preflight).read_text())
    _, _, _, frozen = archive.qualify(reference, preflight=preflight, specification=spec.source)
    if frozen or reference["root"] != root:
        raise ValueError("Stage61 requires nonzero Stage60 actors for this root")
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles, budget = spec.seed_roles(root, preflight=preflight), spec.budget(preflight=preflight)
    for origin in (spec.deployment, spec.deployment.SOURCE_SPEC):
        used = {seed for values in origin.seed_roles(root, preflight=preflight).values() for seed in values}
        if used.intersection(roles["evaluation"]):
            raise ValueError("Stage61 evaluation paths overlap source fitting or training")
    raw, started = raw_directory(output), time.monotonic()
    evaluations, checkpoints = {}, {}

    def observer(pool, period, arm, clone, treatments, predictor):
        p = str(period)
        if p not in evaluations:
            evaluations[p] = {"clone": evaluate(pool, clone, predictor, args, arm="clone", period=period,
                seeds=roles["evaluation"], directory=raw / p / "clone")}
            checkpoints[p] = {}
        evaluations[p][arm], checkpoints[p][arm] = {}, {}
        for treatment, model in treatments.items():
            directory = raw / p / arm / treatment
            rows = evaluate(pool, model, predictor, args, arm=arm, period=period,
                seeds=roles["evaluation"], directory=directory)
            checkpoint = directory / "policy.pt"
            torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period,
                "arm": arm, "treatment": treatment, "state_dict": model.state_dict()}, checkpoint)
            evaluations[p][arm][treatment] = rows
            checkpoints[p][arm][treatment] = str(checkpoint)
        print(f"native one-update evaluation {root}/period{period}/{arm} complete", flush=True)

    replay = archive.replay(root, preflight=preflight, output=raw / "first_update.json",
        specification=spec.source, model_observer=observer, worker_initializer=init_worker)
    if replay["comparisons"] != reference["comparisons"] or replay["cost"] != reference["cost"]:
        raise ValueError("Stage61 replay differs from the exact Stage60 first updates")
    native_counts = dict.fromkeys(learned.COUNT_KEYS, 0)
    for stage in evaluations.values():
        groups = [stage["clone"], *[rows for arm in spec.TRAIN_POLICIES for rows in stage[arm].values()]]
        for rows in groups:
            for row in rows:
                native_counts["primitive_steps"] += row["episode_length"]
                for key in learned.COUNT_KEYS[1:]:
                    native_counts[key] += row[key]
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget,
        "archive_replay": replay, "archive_reproduction_check": "passed", "source_result": str(spec.source_result(root, preflight=preflight)),
        "evaluation_rows": evaluations, "checkpoints": checkpoints, "native_evaluation_counts": native_counts,
        "native_trace_audits": budget["native_trace_audits"],
        "candidate_checkpoint_writes": sum(len(t) for arms in checkpoints.values() for t in arms.values()),
        "evaluation_forecaster_loads": 0, "new_forecaster_fits": 0, "supervised_steps": 0,
        "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def qualify(c, *, preflight):
    root, budget = c["root"], spec.budget(preflight=preflight)
    roles = spec.seed_roles(root, preflight=preflight)
    replay = c["archive_replay"]
    _, _, _, frozen = archive.qualify(replay, preflight=preflight, specification=spec.source)
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["preflight"] != preflight or c["options"] != spec.options(preflight=preflight)
            or c["seed_roles"] != roles or c["budget"] != budget or frozen or replay["root"] != root
            or c["archive_reproduction_check"] != "passed" or replay["cost"] != budget["archive"]
            or any(c[k] != budget[k] for k in ("native_trace_audits", "candidate_checkpoint_writes", "new_forecaster_fits", "supervised_steps"))
            or c["evaluation_forecaster_loads"] != 0
            or any(set(c[k]) != {str(p) for p in spec.PERIODS} for k in ("evaluation_rows", "checkpoints"))):
        raise ValueError("Stage61 frozen source, roster or cost changed")
    args = spec.arguments(root, preflight=preflight)
    counts, means = dict.fromkeys(learned.COUNT_KEYS, 0), {}

    def rows_check(rows, arm, period):
        if [r["seed"] for r in rows] != roles["evaluation"]:
            raise ValueError("Stage61 paired native seed roster changed")
        for row in rows:
            calls = args.horizon // period
            wanted = {"episode_length": args.horizon, "upper_inference_calls": calls,
                "lower_inference_calls": args.horizon, "gate_inference_calls": 0, "candidate_preview_calls": 0,
                "plan_ols_fits": calls - 1, "audit_ols_fits": calls - 1,
                "plan_ridge_predictions": calls - 1, "audit_ridge_predictions": calls - 1,
                "reference_evaluations": args.horizon, "actor_context_evaluations": args.horizon,
                "upper_plan_decodes": calls, "bernstein_basis_evaluations": period + 1,
                "audit_bernstein_basis_evaluations": period + 1}
            if (row["arm"] != arm or row["period"] != period or row["phase"] != "eval"
                    or row["deployment_mode"] != spec.MODE or row["policy"] != spec.deployment.execution(arm)
                    or row["policy_seed"] != spec.deployment.policy_seed(root, row["seed"])
                    or row["method"] != f"fixed{period}" or row["rollout_network_check"] != "passed"
                    or row["lower_actor_type"] != "GaussianActor"
                    or row["decision_steps"] != list(range(0, args.horizon, period))
                    or any(row[k] != v for k, v in wanted.items())):
                raise ValueError("Stage61 native execution or inference counts changed")
            if (arm != "joint_ppo" and (row["executed_action_rms"] != 0 or row["executed_plan_delta_squared_sum"] != 0)
                    or arm == "joint_ppo" and row["executed_action_rms"] != row["proposed_action_rms"]):
                raise ValueError("Stage61 native upper execution rule changed")
            counts["primitive_steps"] += row["episode_length"]
            for key in learned.COUNT_KEYS[1:]:
                counts[key] += row[key]
        return {k: float(np.mean([r[k] for r in rows])) for k in
            (*spec.METRICS, "proposed_action_rms", "executed_action_rms", "executed_plan_delta_squared_sum")}

    for period in spec.PERIODS:
        p, stage = str(period), c["evaluation_rows"][str(period)]
        if set(stage) != {"clone", *spec.TRAIN_POLICIES} or set(c["checkpoints"][p]) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage61 clone or training-arm roster changed")
        clone = rows_check(stage["clone"], "clone", period)
        means[p] = {}
        for arm in spec.TRAIN_POLICIES:
            if set(stage[arm]) != set(spec.TREATMENTS) or set(c["checkpoints"][p][arm]) != set(spec.TREATMENTS):
                raise ValueError("Stage61 treatment roster changed")
            means[p][arm] = {"clone": clone, **{t: rows_check(rows, arm, period) for t, rows in stage[arm].items()}}
    if counts != c["native_evaluation_counts"] or counts != budget["native_evaluation"]:
        raise ValueError("Stage61 actual native cost differs from frozen budget")
    return {"root": root, "means": means, "endpoints": spec.contrasts(means)}


def bootstrap(rows):
    x = np.asarray([[r["endpoints"][key] for key in spec.ENDPOINTS] for r in rows])
    indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(
        0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
    tail = .05 / (2 * len(spec.ENDPOINTS))
    ci = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
    return {key: {"mean": float(x[:, i].mean()), "ci": ci[:, i].tolist(),
        "effect": "positive" if ci[0, i] > 0 else "negative" if ci[1, i] < 0 else "inconclusive"}
        for i, key in enumerate(spec.ENDPOINTS)}


def aggregate(cells, *, preflight):
    by_root = {c["root"]: c for c in cells}
    if len(by_root) != len(cells) or set(by_root) != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage61 complete root roster required")
    rows = [qualify(by_root[root], preflight=preflight) for root in spec.roots(preflight=preflight)]
    replay = archive.aggregate([c["archive_replay"] for c in cells], preflight=preflight, specification=spec.source)
    replay.pop("root_rows")
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "archive_mechanics": replay,
        "native_evaluation_counts": {k: sum(c["native_evaluation_counts"][k] for c in cells) for k in learned.COUNT_KEYS},
        **{k: sum(c[k] for c in cells) for k in ("native_trace_audits", "candidate_checkpoint_writes",
            "evaluation_forecaster_loads", "new_forecaster_fits", "supervised_steps")}}
    if not preflight:
        endpoints = bootstrap(rows)
        def positive_against(comparator):
            return all(endpoints[f"period{p}:{a}:backtracking_kl_minus_{comparator}"]["effect"] == "positive"
                for p in spec.PERIODS for a in spec.TRAIN_POLICIES)
        summary.update(primary_endpoints=endpoints,
            repair_gain_gate="passed" if positive_against("plain") else "failed",
            training_gain_gate="passed" if positive_against("clone") else "failed")
    return summary
