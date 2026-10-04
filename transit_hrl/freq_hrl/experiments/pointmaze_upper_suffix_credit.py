"""Isolate decision-to-episode-end upper credit from architecture and budget."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_upper_residual_train as base
from . import pointmaze_upper_full_plan_train as full
from . import pointmaze_feasible_credit as statistics
from .pointmaze_root_response import write_json
from scripts import pointmaze_upper_suffix_credit_stage118_spec as spec


def score_upper(actor, pair_groups, *, horizon, period, cost, method):
    gradients, states, score_costs, signal_rms = {}, [], {}, {}
    for name, pairs in pair_groups.items():
        option_returns = np.asarray([[batch.reward for batch in pair["upper_batches"]]
            for pair in pairs], dtype=np.float64)
        if option_returns.shape[-1] != horizon // period:
            raise ValueError("Stage118 credit is not aligned to upper decisions")
        for pair_returns, pair in zip(option_returns, pairs):
            for values, batch, row in zip(pair_returns, pair["lower_batches"], pair["rows"]):
                np.testing.assert_allclose(values.sum(), row["episode_return"], atol=.002, rtol=0)
                np.testing.assert_allclose(base.independent.exact_returns(batch, 1.)[0],
                    row["episode_return"], atol=.002, rtol=0)
                cost["objective_checks"] += 1
                cost["mc_calls"] += 1
        if method == "suffix":
            credit = np.cumsum(option_returns[..., ::-1], axis=-1)[..., ::-1]
        elif method == "option":
            credit = option_returns
        else:
            raise ValueError("Stage118 credit method changed")
        signal = base.source.scenario.leave_other_out(credit).reshape(-1)
        upper = base.concat_level_batches(batch for pair in pairs for batch in pair["upper_batches"])
        scored, score_cost = base.lower_training.residual_actor_gradients(actor, upper, {"scenario": signal},
            clip_ratio=.2, chunk_size=spec.CHUNK_SIZE)
        gradients[name], score_costs[name] = scored["scenario"], score_cost
        states.append(upper.state)
        signal_rms[name] = float(np.sqrt(np.square(signal).mean()))
        for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
            cost[key] += score_cost[key]
    a, b = (gradients[name] for name in ("A", "B"))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"gradients": gradients, "states": np.concatenate(states), "score_costs": score_costs,
        "signal_rms": signal_rms, "cross_batch_gradient_cosine": float(a @ b / norm) if norm else 0.}


def evaluation_group(job):
    weights, lower_state, upper_states, seed, period, predictor, calibration, args = job
    rows, innovations = {}, {}
    for variant in spec.ARMS:
        arm = "learned" if variant in spec.METHODS else "forecast"
        state = upper_states.get("suffix" if variant == "suffix_blinded" else variant)
        _, _, row, innovation = base.native_episode(weights, lower_state, state, seed=seed, noise_seed=seed,
            arm=arm, period=period, predictor=predictor, calibration=calibration, args=args,
            collect=False, upper_sample=False, upper_factory=full.upper_branch, plan_factory=full.plan_for_arm)
        row.update(variant=variant, state_arm=arm)
        rows[variant], innovations[variant] = row, innovation
    for innovation in innovations.values():
        np.testing.assert_allclose(innovation, innovations["forecast"], atol=3e-5, rtol=0)
    for key in ("episode_return", "tracking_squared_error_integral", "upper_calls", "decision_steps"):
        if rows["forecast"][key] != rows["suffix_blinded"][key]:
            raise ValueError("Stage118 blinded plan is not the forecast execution")
    return {"seed": seed, "evaluation": rows, "pairing": "passed"}


def check_initial_credit_pairing(pair_groups, cost):
    for name in ("A", "B"):
        for option, suffix in zip(pair_groups["option"][name], pair_groups["suffix"][name]):
            for a, b in zip(option["upper_batches"], suffix["upper_batches"]):
                for key in ("state", "action", "reward", "old_logp"):
                    np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            if option["rows"] != suffix["rows"]:
                raise ValueError("Stage118 initial credit comparison changed native episodes")
            cost["initial_credit_pair_checks"] += 1


def count_episode(cost, row, *, training):
    cost["training_episodes" if training else "evaluation_episodes"] += 1
    cost["native_episodes"] += 1
    cost["native_steps"] += row["episode_length"]
    cost["native_lower_calls"] += row["lower_calls"]
    cost["native_upper_calls"] += row["upper_calls"]
    cost["native_network_checks"] += 1
    for key, field in (("planning_renewals", "plan_renewals"), ("planning_fits", "plan_ols_fits"),
            ("planning_predictions", "plan_ridge_predictions"), ("planning_reference_calls", "reference_evaluations"),
            ("planning_context_calls", "actor_context_evaluations")):
        cost[key] += row[field]


def check_prerequisite():
    result = json.loads(spec.prerequisite_summary().read_text())
    if (result["protocol"] != "pointmaze_local_plan_gain_stage117_v1" or result["status"] != "complete"
            or result["local_plan_gain_gate"] != "supported_both_periods"):
        raise ValueError("Stage118 requires the full Stage117 local plan gain gate")


def run(root, *, preflight, output):
    check_prerequisite()
    base.source.qualify(json.loads(spec.source_result(root).read_text()), preflight=False)
    models, predictor, _, calibrations = base.source.load_source(root)
    args, roles, options = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    cost, groups = dict.fromkeys(spec.budget(preflight=preflight), 0), {}
    cost.update(source_cell_loads=1, source_clone_loads=len(models), lower_checkpoint_loads=len(models),
        upper_branch_initializations=len(models) * len(spec.METHODS))
    with ProcessPoolExecutor(max_workers=options["workers"], mp_context=mp.get_context("spawn"),
            initializer=base.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            before = copy.deepcopy(model.state_dict())
            weights = base.source.native.joint.inference_weights(model)
            lower_state = base.load_lower_state(root, period, protocol=spec)
            lower_before = copy.deepcopy(lower_state)
            actors = {method: full.upper_branch(model) for method in spec.METHODS}
            histories = {method: [] for method in spec.METHODS}
            for iteration, round_roles in enumerate(roles["training_rounds"], 1):
                round_pairs = {}
                for method, actor in actors.items():
                    pair_groups = {}
                    for name in ("credit_A", "credit_B"):
                        roster = round_roles[name]
                        jobs = [(weights, lower_state, copy.deepcopy(actor.state_dict()), row["scenario_seed"], row["noise_seeds"],
                            period, predictor, calibrations[str(period)], args) for row in roster]
                        pairs = list(pool.map(full.training_pair, jobs))
                        pair_groups[name[-1]] = pairs
                        for pair, registered in zip(pairs, roster):
                            if (pair["pairing"] != "passed" or [row["noise_seed"] for row in pair["rows"]] != registered["noise_seeds"]
                                    or any(row["seed"] != registered["scenario_seed"] for row in pair["rows"])):
                                raise ValueError("Stage118 paired training seed roster changed")
                            for row in pair["rows"]:
                                base.check_row(row, period=period, horizon=args.horizon)
                                count_episode(cost, row, training=True)
                            cost["scenario_pair_checks"] += 1
                    if iteration == 1:
                        round_pairs[method] = pair_groups
                    score = score_upper(actor, pair_groups, horizon=args.horizon, period=period, cost=cost, method=method)
                    history = base.lower_training.residual_update(actor, score, cost=cost)
                    history.update(iteration=iteration, cross_batch_gradient_cosine=score["cross_batch_gradient_cosine"])
                    histories[method].append(history)
                if iteration == 1:
                    check_initial_credit_pairing(round_pairs, cost)
                print(f"Stage118 root{root} period{period} update{iteration}: both credit learners complete", flush=True)
            upper_states = {method: copy.deepcopy(actor.state_dict()) for method, actor in actors.items()}
            jobs = [(weights, lower_state, upper_states, seed, period, predictor, calibrations[str(period)], args)
                for seed in roles["native_evaluation"]]
            evaluation = {variant: [] for variant in spec.ARMS}
            for result in pool.map(evaluation_group, jobs):
                if result["pairing"] != "passed":
                    raise ValueError("Stage118 evaluation action-noise pairing failed")
                for variant, row in result["evaluation"].items():
                    base.check_row(row, period=period, horizon=args.horizon, variant=variant)
                    count_episode(cost, row, training=False)
                    evaluation[variant].append(row)
                cost["native_pair_checks"] += 1
            base.source.native.curves.support.assert_frozen(model, before)
            torch.testing.assert_close(lower_state, lower_before, atol=0, rtol=0)
            cost["frozen_model_checks"] += 1
            for method, actor in actors.items():
                torch.testing.assert_close(actor.base.state_dict(), full.upper_branch(model).base.state_dict(), atol=0, rtol=0)
                if not preflight:
                    path = output.parent / "final_weights" / f"period_{period}_{method}_upper.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period,
                        "method": method, "weights": actor.state_dict()}, path)
                    cost["checkpoint_writes"] += 1
            groups[str(period)] = {"evaluation": evaluation, "histories": histories,
                "effects": base.paired_effects(period, evaluation, roles["native_evaluation"], protocol=spec),
                "source_and_lower_unchanged": "passed"}
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "seed_roles": roles, "cost": cost, "groups": groups}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root = cell["root"]
    if (root not in spec.roots(preflight=preflight) or cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
            or cell["contract"] != spec.contract() or cell["preflight"] != preflight
            or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight) or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage118 source, pairing roster or fixed budget changed")
    horizon = spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        if group["source_and_lower_unchanged"] != "passed" or set(group["histories"]) != set(spec.METHODS):
            raise ValueError("Stage118 frozen models or matched learners changed")
        for history in group["histories"].values():
            if [row["iteration"] for row in history] != list(range(1, spec.options(preflight=preflight)["updates"] + 1)):
                raise ValueError("Stage118 selected a non-final iteration")
            if any(row["geometry"]["radius_check"] != "passed" for row in history):
                raise ValueError("Stage118 Fisher update changed")
        effects = base.paired_effects(int(period), group["evaluation"], cell["seed_roles"]["native_evaluation"], protocol=spec)
        if group["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage118 fresh paired evaluation changed")
        for variant, rows in group["evaluation"].items():
            for row in rows:
                base.check_row(row, period=int(period), horizon=horizon, variant=variant)
                if row["upper_sample"] or not row["lower_sample"]:
                    raise ValueError("Stage118 evaluation sampling changed")
        for forecast, blinded in zip(group["evaluation"]["forecast"], group["evaluation"]["suffix_blinded"]):
            if forecast["episode_return"] != blinded["episode_return"] or forecast["tracking_squared_error_integral"] != blinded["tracking_squared_error_integral"]:
                raise ValueError("Stage118 blinded plan differs from forecast")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(p): all(result["endpoints"][f"{p}/{a}_minus_{b}"]["ci"][0] > 0
        for a, b in spec.CONTRASTS) for p in spec.PERIODS}
    result.update(upper_credit_gain_gate="mechanical_only" if preflight else
        "supported_both_periods" if all(passed.values()) else "partial" if any(passed.values()) else "not_supported",
        period_upper_credit_gain_gate=passed,
        performance_claim="matched_learned_upper_suffix_credit_not_joint_actor_critic_or_promotion")
    return result
