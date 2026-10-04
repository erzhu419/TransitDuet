"""Learn the upper mean from paired native plan effects, not noisy score credit."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_local_plan_gain as probe
from . import pointmaze_upper_suffix_credit as previous
from . import pointmaze_upper_residual_train as base
from . import pointmaze_upper_full_plan_train as full
from . import pointmaze_feasible_credit as statistics
from .pointmaze_root_response import write_json
from scripts import pointmaze_native_plan_gradient_stage119_spec as spec


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm else 0.


def worker_query(job):
    weights, lower_state, upper_state, query, period, predictor, calibration = job
    model, _ = base.source.native._WORKER
    upper = full.upper_branch(model)
    upper.load_state_dict(upper_state)
    panels, common, state, largest = {}, None, None, 0.
    for panel in spec.PANELS:
        rows, reference = {}, None
        for variant in spec.VARIANTS:
            row, audit = probe.episode(weights, lower_state, query=query, panel=panel, variant=variant,
                period=period, predictor=predictor, calibration=calibration, upper=upper)
            if common is None:
                common, state = audit, audit["upper_state"].copy()
            else:
                for key in ("query_state", "upper_state", "prefix_rewards", "measurements"):
                    np.testing.assert_array_equal(audit[key], common[key])
                np.testing.assert_array_equal(audit["commands"][:query["start"]], common["commands"][:query["start"]])
            if reference is None:
                reference = audit
            else:
                largest = max(largest, float(np.max(np.abs(audit["innovations"] - reference["innovations"]))))
                np.testing.assert_allclose(audit["innovations"], reference["innovations"], atol=3e-5, rtol=0)
                np.testing.assert_allclose(row["episode_return"] - rows["zero"]["episode_return"],
                    row["suffix_return"] - rows["zero"]["suffix_return"], atol=1e-9, rtol=0)
            rows[variant] = row
        panels[panel] = rows
    torch.testing.assert_close(upper.state_dict(), upper_state, atol=0, rtol=0)
    torch.testing.assert_close(base.source.native.joint.inference_weights(model), weights, atol=0, rtol=0)
    return {"query": query, "state": state, "panels": panels, "gradients": probe.gradients(panels),
        "pairing": "passed", "policy_freeze": "passed", "innovation_max_error": largest}


def pullback(actor, queries, cost):
    state = np.asarray([q["state"] for q in queries], dtype=np.float32)
    gradients, score_costs, signal_rms = {}, {}, {}
    native = {}
    for panel in spec.PANELS:
        native[panel] = np.asarray([q["gradients"][panel] for q in queries], dtype=np.float64)
        mean = actor.distribution(torch.as_tensor(state)).mean
        loss = -(mean * torch.as_tensor(native[panel], dtype=mean.dtype)).sum() / len(queries)
        values = torch.autograd.grad(loss, (actor.readout.weight, actor.readout.bias))
        gradients[panel] = np.concatenate([value.detach().double().numpy().ravel() for value in values])
        score_costs[panel] = {"actor_pullback_forward_batches": 1, "actor_pullback_backward_batches": 1}
        for key in score_costs[panel]:
            cost[key] += 1
        signal_rms[panel] = float(np.sqrt(np.square(native[panel]).mean()))
    return {"gradients": gradients, "states": np.concatenate((state, state)),
        "score_costs": score_costs, "signal_rms": signal_rms,
        "native_gradient_cosine": cosine(native["A"], native["B"]),
        "readout_gradient_cosine": cosine(gradients["A"], gradients["B"])}


def evaluation_group(job):
    weights, lower_state, upper_state, reference_state, seed, period, predictor, calibration, args = job
    rows, innovations = {}, {}
    for variant in spec.ARMS:
        arm = "learned" if variant in ("native_fd", "stage118_suffix") else "forecast"
        state = reference_state if variant == "stage118_suffix" else upper_state if variant != "forecast" else None
        _, _, row, innovation = base.native_episode(weights, lower_state, state, seed=seed, noise_seed=seed,
            arm=arm, period=period, predictor=predictor, calibration=calibration, args=args,
            collect=False, upper_sample=False, upper_factory=full.upper_branch, plan_factory=full.plan_for_arm)
        row.update(variant=variant, state_arm=arm)
        rows[variant], innovations[variant] = row, innovation
    for innovation in innovations.values():
        np.testing.assert_allclose(innovation, innovations["forecast"], atol=3e-5, rtol=0)
    if rows["forecast"]["episode_return"] != rows["native_fd_blinded"]["episode_return"]:
        raise ValueError("Stage119 blinded upper must execute forecast")
    return {"evaluation": rows, "pairing": "passed"}


def count_query(cost, result):
    cost["training_queries"] += 1
    cost["native_pair_checks"] += 1
    cost["prefix_pair_checks"] += len(spec.PANELS) * len(spec.VARIANTS) - 1
    cost["external_pair_checks"] += len(spec.PANELS) * len(spec.VARIANTS) - 1
    cost["innovation_pair_checks"] += len(spec.PANELS) * (len(spec.VARIANTS) - 1)
    cost["training_policy_freeze_checks"] += 1
    for rows in result["panels"].values():
        for row in rows.values():
            cost["training_episodes"] += 1
            cost["native_episodes"] += 1
            cost["native_steps"] += row["episode_length"]
            cost["native_lower_calls"] += row["lower_calls"]
            cost["native_upper_calls"] += row["upper_actor_calls"]
            cost["suffix_credit_checks"] += 1
            for key, field in (("planning_renewals", "plan_renewals"), ("planning_fits", "plan_fits"),
                    ("planning_predictions", "plan_fits"), ("planning_reference_calls", "reference_evaluations"),
                    ("planning_context_calls", "actor_context_evaluations")):
                cost[key] += row[field]


def load_reference(root, period, model):
    saved = torch.load(spec.reference_checkpoint(root, period), map_location="cpu", weights_only=False)
    if (saved["protocol"], saved["root"], saved["period"], saved["method"]) != (previous.spec.EXPERIMENT_PROTOCOL, root, period, "suffix"):
        raise ValueError("Stage119 reference must be the final Stage118 suffix upper")
    actor = full.upper_branch(model)
    actor.load_state_dict(saved["weights"])
    torch.testing.assert_close(actor.base.state_dict(), full.upper_branch(model).base.state_dict(), atol=0, rtol=0)
    return saved["weights"]


def run(root, *, preflight, output):
    previous.check_prerequisite()
    base.source.qualify(json.loads(spec.source_result(root).read_text()), preflight=False)
    models, predictor, _, calibrations = base.source.load_source(root)
    args, roles, options = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    cost, groups = dict.fromkeys(spec.budget(preflight=preflight), 0), {}
    cost.update(source_cell_loads=1, source_clone_loads=len(models), lower_checkpoint_loads=len(models),
        reference_checkpoint_loads=len(models), upper_branch_initializations=len(models))
    with ProcessPoolExecutor(max_workers=options["workers"], mp_context=mp.get_context("spawn"),
            initializer=base.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            before = copy.deepcopy(model.state_dict())
            weights = base.source.native.joint.inference_weights(model)
            lower_state = base.load_lower_state(root, period, protocol=spec)
            lower_before = copy.deepcopy(lower_state)
            reference = load_reference(root, period, model)
            reference_before = copy.deepcopy(reference)
            actor, history = full.upper_branch(model), []
            for iteration, round_roles in enumerate(roles["training_rounds"], 1):
                roster = round_roles[str(period)]
                jobs = [(weights, lower_state, copy.deepcopy(actor.state_dict()), query, period, predictor, calibrations[str(period)])
                    for query in roster]
                results = list(pool.map(worker_query, jobs))
                if [q["query"] for q in results] != roster:
                    raise ValueError("Stage119 native query roster changed")
                for result in results:
                    count_query(cost, result)
                score = pullback(actor, results, cost)
                update = base.lower_training.residual_update(actor, score, cost=cost)
                update.update(iteration=iteration, native_gradient_cosine=score["native_gradient_cosine"],
                    readout_gradient_cosine=score["readout_gradient_cosine"],
                    queries=[{key: value for key, value in q.items() if key != "state"} for q in results])
                history.append(update)
                print(f"Stage119 root{root} period{period} update{iteration}: native plan gradients complete", flush=True)
            jobs = [(weights, lower_state, copy.deepcopy(actor.state_dict()), reference, seed, period, predictor, calibrations[str(period)], args)
                for seed in roles["native_evaluation"]]
            evaluation = {variant: [] for variant in spec.ARMS}
            for result in pool.map(evaluation_group, jobs):
                if result["pairing"] != "passed":
                    raise ValueError("Stage119 evaluation common-noise check failed")
                for variant, row in result["evaluation"].items():
                    previous.count_episode(cost, row, training=False)
                    evaluation[variant].append(row)
                cost["native_pair_checks"] += 1
            base.source.native.curves.support.assert_frozen(model, before)
            torch.testing.assert_close(lower_state, lower_before, atol=0, rtol=0)
            torch.testing.assert_close(reference, reference_before, atol=0, rtol=0)
            torch.testing.assert_close(actor.base.state_dict(), full.upper_branch(model).base.state_dict(), atol=0, rtol=0)
            cost["frozen_model_checks"] += 1
            if not preflight:
                path = output.parent / "final_weights" / f"period_{period}_native_fd_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period, "weights": actor.state_dict()}, path)
                cost["checkpoint_writes"] += 1
            groups[str(period)] = {"history": history, "evaluation": evaluation,
                "effects": base.paired_effects(period, evaluation, roles["native_evaluation"], protocol=spec),
                "source_and_reference_unchanged": "passed"}
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "seed_roles": roles, "groups": groups, "cost": cost}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root = cell["root"]
    if (cell["status"] != "complete" or root not in spec.roots(preflight=preflight) or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
            or cell["contract"] != spec.contract() or cell["preflight"] != preflight
            or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight) or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage119 frozen protocol, roster or measured budget changed")
    horizon = spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        if group["source_and_reference_unchanged"] != "passed" or len(group["history"]) != spec.options(preflight=preflight)["updates"]:
            raise ValueError("Stage119 source freeze or final update count changed")
        for iteration, update in enumerate(group["history"], 1):
            roster = cell["seed_roles"]["training_rounds"][iteration - 1][period]
            if update["iteration"] != iteration or [q["query"] for q in update["queries"]] != roster or update["geometry"]["radius_check"] != "passed":
                raise ValueError("Stage119 native credit roster or Fisher update changed")
            for q in update["queries"]:
                if q["pairing"] != "passed" or q["policy_freeze"] != "passed" or not 0 <= q["innovation_max_error"] <= 3e-5:
                    raise ValueError("Stage119 current-policy prefix or noise pairing failed")
                if set(q["panels"]) != set(spec.PANELS) or any(set(rows) != set(spec.VARIANTS) for rows in q["panels"].values()):
                    raise ValueError("Stage119 native coordinate intervention roster changed")
                for panel, rows in q["panels"].items():
                    for variant, row in rows.items():
                        if (row["episode_length"] != horizon or row["upper_actor_calls"] != horizon // int(period)
                                or row["lower_calls"] != horizon or row["intervention_decisions"] != int(variant != "zero")):
                            raise ValueError("Stage119 current-policy closed-loop schedule changed")
                        for role, noise in (("prefix_seed", q["query"]["prefix_noise_seed"]), ("suffix_seed", q["query"]["suffix_noise_seeds"][panel])):
                            if row[role] != base.source.scenario.spec.noise_seeds(root, q["query"]["scenario_seed"], noise)[1]:
                                raise ValueError("Stage119 query noise roles changed")
                        np.testing.assert_allclose(row["episode_return"], row["prefix_return"] + row["suffix_return"], atol=1e-9, rtol=0)
                        np.testing.assert_allclose(row["suffix_return"], row["option_return"] + row["tail_return"], atol=1e-9, rtol=0)
                if q["gradients"] != probe.gradients(q["panels"]):
                    raise ValueError("Stage119 native full-suffix derivative changed")
        expected = base.paired_effects(int(period), group["evaluation"], cell["seed_roles"]["native_evaluation"], protocol=spec)
        if group["effects"] != expected or not np.isfinite(list(expected.values())).all():
            raise ValueError("Stage119 fresh final-policy effects changed")
        for variant, rows in group["evaluation"].items():
            for row in rows:
                base.check_row(row, period=int(period), horizon=horizon, variant=variant)
                if row["upper_sample"] or not row["lower_sample"]:
                    raise ValueError("Stage119 deployed policy sampling changed")
        if any(a["episode_return"] != b["episode_return"] for a, b in zip(group["evaluation"]["forecast"], group["evaluation"]["native_fd_blinded"])):
            raise ValueError("Stage119 blinded upper differs from forecast")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(p): all(result["endpoints"][f"{p}/{a}_minus_{b}"]["ci"][0] > 0
        for a, b in spec.CONTRASTS) for p in spec.PERIODS}
    result.update(native_plan_gradient_gain_gate="mechanical_only" if preflight else
        "supported_both_periods" if all(passed.values()) else "partial" if any(passed.values()) else "not_supported",
        period_native_plan_gradient_gain_gate=passed,
        performance_claim="native_plan_gradient_upper_learning_more_simulation_not_equal_budget_or_joint_HRL")
    return result
