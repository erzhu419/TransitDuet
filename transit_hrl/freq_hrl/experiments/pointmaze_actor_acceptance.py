"""A paired native training intervention, accepting by the training objective only."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from .pointmaze_critic_calibration import lower_update, policy_drift
from .pointmaze_update_direction import load_pair, surrogate_terms
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_actor_acceptance_stage45_spec as spec


def actor_update(model, batch, treatment):
    if treatment not in spec.TREATMENTS:
        raise ValueError("unregistered actor acceptance treatment")
    actor, optimizer = model.lower_actor, model.lower_actor_optimizer
    state, adam = copy.deepcopy(actor.state_dict()), copy.deepcopy(optimizer.state_dict())
    before = surrogate_terms(model, actor, batch)
    update = lower_update(model, batch, "actor_critic")
    tentative = surrogate_terms(model, actor, batch)
    accepted = treatment == "vanilla" or tentative["actor_objective"] >= before["actor_objective"]
    if not accepted:
        actor.load_state_dict(state)
        optimizer.load_state_dict(adam)
    deployed = tentative if accepted else surrogate_terms(model, actor, batch)
    attempted = int(update["lower_actor_optimizer_steps"])
    return {"accepted": accepted, "objective_before": before["actor_objective"],
            "objective_tentative": tentative["actor_objective"], "objective_deployed": deployed["actor_objective"],
            "clipped_gain_tentative": tentative["clipped_surrogate"] - before["clipped_surrogate"],
            "objective_drop": tentative["actor_objective"] < before["actor_objective"],
            "attempted_actor_steps": attempted, "retained_actor_steps": attempted if accepted else 0,
            "value_steps": int(update["lower_value_optimizer_steps"])}


def worker_rollout(job):
    weights, seed, method, phase, mode, path = job
    model, args, _, _ = joint._WORKER
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    batch, row, raw = joint.rollout(model, args, "learned_history", seed=seed, capture=True,
        lower_credit=spec.source.source.LOWER_CREDIT[method], lower_value_context_builder=clocks.context_builder(method),
        **spec.rollout_arguments(args.optimizer_seed, seed, phase=phase, mode=mode))
    row.update(policy_seed=spec.policy_seed(args.optimizer_seed, seed), deployment_mode=mode)
    clocks.audit_context(None if batch is None else batch.lower, row, raw["lower_value_context"],
                         clock=spec.source.source.VALUE_CLOCK[method])
    np.savez_compressed(path, **raw)
    return None if batch is None else batch.lower, row


def first_batch_pair(left, right):
    fields = ("state", "value_state", "action", "reward", "done", "duration", "old_logp", "old_value", "next_value", "terminal")
    for field in fields:
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field), err_msg=f"paired first batch {field} differs")
    return {"status": "passed", "transitions": left.size, "fields": list(fields)}


def train(root, *, preflight, output):
    args, opt, roles = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    sources = {m: json.loads(spec.source_result(root, m, preflight=preflight).read_text()) for m in spec.METHODS}
    before = {m: load_pair(sources[m], root=root, method=m, preflight=preflight)[0] for m in spec.METHODS}
    reference = before[spec.METHODS[0]]
    frozen = joint.inference_weights(reference)
    for model in before.values():
        for name, weights in frozen.items():
            if name != "lower_value":
                torch.testing.assert_close(getattr(model, name).state_dict(), weights, atol=0, rtol=0)
    raw, started = raw_directory(output), time.monotonic()
    counts = {phase: {k: 0 for k in ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}
              for phase in ("train", "eval")}
    evaluation, training, pairs, checkpoints, expected_steps = {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=joint.init_worker,
                             initargs=(reference.config, args, "learned_history", "intrinsic_option")) as pool:
        def episodes(model, method, policy, phase, mode, seeds, iteration):
            directory = raw / policy.replace(":", "/") / str(iteration) / mode
            directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(model)
            output_rows = list(pool.map(worker_rollout, [(weights, seed, method, phase, mode,
                               str(directory / f"episode_{seed}.npz")) for seed in seeds]))
            rows = [row for _, row in output_rows]
            joint.audit_trajectories(rows, args=args, method="learned_history", raw_path=directory)
            for row in rows:
                counts[phase]["primitive_steps"] += row["episode_length"]
                for key in counts[phase]:
                    if key != "primitive_steps":
                        counts[phase][key] += row[key]
            return output_rows

        def snapshot(model, method, policy, iteration):
            for name, weights in frozen.items():
                if name not in ("lower_actor", "lower_value"):
                    torch.testing.assert_close(getattr(model, name).state_dict(), weights, atol=0, rtol=0)
            directory = raw / policy.replace(":", "/")
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = directory / f"iteration_{iteration}.pt"
            torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "policy": policy,
                        "iteration": iteration, "state_dict": model.state_dict()}, checkpoint)
            checkpoints.setdefault(policy, {})[str(iteration)] = str(checkpoint)
            evaluation.setdefault(policy, {})[str(iteration)] = {mode: [row for _, row in episodes(
                model, method, policy, "eval", mode, roles["evaluation"], iteration)] for mode in spec.MODES}

        snapshot(reference, "frozen", "frozen", 0)
        for method in spec.METHODS:
            training[method] = {}
            first = None
            config = before[method].config
            size = opt["rollouts_per_iteration"] * args.horizon
            expected_steps[method] = max(1, int(config.epochs)) * math.ceil(size / max(1, min(int(config.minibatch_size), size)))
            for treatment in spec.TREATMENTS:
                model = copy.deepcopy(before[method])
                policy = f"{method}:{treatment}"
                history = training[method][treatment] = []
                for iteration in range(1, opt["learning_iterations"] + 1):
                    start = (iteration - 1) * opt["rollouts_per_iteration"]
                    rollouts = episodes(model, method, policy, "train", "training",
                                        roles["training"][start:start + opt["rollouts_per_iteration"]], iteration)
                    batch = concat_level_batches([b for b, _ in rollouts])
                    if iteration == 1:
                        if treatment == "vanilla":
                            first = batch
                        else:
                            pairs[method] = first_batch_pair(first, batch)
                    old_actor = copy.deepcopy(model.lower_actor)
                    np.random.seed(spec.shuffle_seed(root, iteration))
                    step = actor_update(model, batch, treatment)
                    history.append({"iteration": iteration, "primitive_steps": batch.size, **step,
                        "policy_drift": policy_drift(model.lower_actor, old_actor, batch.state),
                        "rollout_sampling": [{k: row[k] for k in ("seed", "policy_seed", "upper_sample", "gate_sample", "lower_sample", "lower_seed", "gate_seed")}
                                             for _, row in rollouts],
                        "inference_counts": {k: sum(row[k] for _, row in rollouts) for k in
                                             ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}})
                    if iteration in spec.snapshots(preflight=preflight):
                        snapshot(model, method, policy, iteration)
                        print(f"trained {root}/{policy} iteration {iteration}: accepted {sum(r['accepted'] for r in history)}/{len(history)}", flush=True)
    budget = spec.budget(preflight=preflight)
    if counts["train"]["primitive_steps"] != budget["training_primitive_steps"] or counts["eval"]["primitive_steps"] != budget["evaluation_primitive_steps"]:
        raise ValueError("actor acceptance native accounting changed")
    warm = spec.source.options(preflight=preflight)["warmup_iterations"]
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
              "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "inference_counts": counts,
              "training": training, "evaluation_rows": evaluation, "first_batch_pairs": pairs,
              "source_checkpoints": {m: sources[m]["snapshots"][str(warm)]["checkpoint"] for m in spec.METHODS},
              "checkpoints": checkpoints, "expected_steps_per_update": expected_steps,
              "fixed_upper_gate_networks": "passed", "native_trace_audits": budget["native_trace_audits"],
              "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def aggregate(results, *, preflight):
    roots = spec.roots(preflight=preflight)
    cells = {r["root"]: r for r in results}
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("actor acceptance root roster incomplete")
    opt, budget, root_rows = spec.options(preflight=preflight), spec.budget(preflight=preflight), []
    totals = {k: 0 for k in ("attempted_actor_steps", "retained_actor_steps", "value_steps")}
    for root in roots:
        cell = cells[root]
        if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
                or cell["preflight"] != preflight or cell["options"] != opt or cell["budget"] != budget
                or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
                or cell["fixed_upper_gate_networks"] != "passed" or cell["native_trace_audits"] != budget["native_trace_audits"]):
            raise ValueError("actor acceptance result violates the frozen protocol")
        if set(cell["training"]) != set(spec.METHODS) or set(cell["first_batch_pairs"]) != set(spec.METHODS) or set(cell["evaluation_rows"]) != set(spec.POLICIES):
            raise ValueError("actor acceptance policy roster incomplete")
        decisions, train_counts = {}, {k: 0 for k in cell["inference_counts"]["train"]}
        for method in spec.METHODS:
            if cell["first_batch_pairs"][method]["status"] != "passed" or cell["first_batch_pairs"][method]["transitions"] != opt["rollouts_per_iteration"] * spec.arguments(root, preflight=preflight).horizon:
                raise ValueError("actor acceptance first batch pair changed")
            if set(cell["training"][method]) != set(spec.TREATMENTS):
                raise ValueError("actor acceptance treatment roster incomplete")
            decisions[method] = {}
            for treatment in spec.TREATMENTS:
                history = cell["training"][method][treatment]
                if [r["iteration"] for r in history] != list(range(1, opt["learning_iterations"] + 1)):
                    raise ValueError("actor acceptance training roster incomplete")
                for row in history:
                    accept = treatment == "vanilla" or row["objective_tentative"] >= row["objective_before"]
                    steps = cell["expected_steps_per_update"][method]
                    if (row["accepted"] != accept or row["objective_drop"] != (row["objective_tentative"] < row["objective_before"])
                            or row["attempted_actor_steps"] != steps or row["value_steps"] != steps
                            or row["retained_actor_steps"] != (steps if accept else 0)
                            or row["objective_deployed"] != (row["objective_tentative"] if accept else row["objective_before"])
                            or (not accept and row["policy_drift"]["gaussian_kl"] != 0.)):
                        raise ValueError("actor acceptance decision or optimizer accounting changed")
                    begin = (row["iteration"] - 1) * opt["rollouts_per_iteration"]
                    seeds = cell["seed_roles"]["training"][begin:begin + opt["rollouts_per_iteration"]]
                    check_sampling(row["rollout_sampling"], root, seeds, phase="train", mode="training")
                    if row["primitive_steps"] != len(seeds) * spec.arguments(root, preflight=preflight).horizon or row["inference_counts"]["lower_inference_calls"] != row["primitive_steps"]:
                        raise ValueError("actor acceptance training step accounting changed")
                    train_counts["primitive_steps"] += row["primitive_steps"]
                    for key in row["inference_counts"]:
                        train_counts[key] += row["inference_counts"][key]
                    for key in totals:
                        totals[key] += row[key]
                decisions[method][treatment] = {**{k: sum(r[k] for r in history) for k in totals},
                    "accepted_updates": sum(r["accepted"] for r in history), "objective_drops": sum(r["objective_drop"] for r in history)}
        means = {"0": {mode: {} for mode in spec.MODES}, **{str(i): {mode: {} for mode in spec.MODES} for i in spec.snapshots(preflight=preflight)}}
        eval_counts = {k: 0 for k in cell["inference_counts"]["eval"]}
        for policy in spec.POLICIES:
            snapshots = (0,) if policy == "frozen" else spec.snapshots(preflight=preflight)
            if set(cell["evaluation_rows"][policy]) != {str(i) for i in snapshots}:
                raise ValueError("actor acceptance snapshot roster incomplete")
            for iteration in snapshots:
                stages = cell["evaluation_rows"][policy][str(iteration)]
                if set(stages) != set(spec.MODES):
                    raise ValueError("actor acceptance deployment roster incomplete")
                for mode, rows in stages.items():
                    check_sampling(rows, root, cell["seed_roles"]["evaluation"], phase="eval", mode=mode)
                    for row in rows:
                        if row["episode_length"] != spec.arguments(root, preflight=preflight).horizon:
                            raise ValueError("actor acceptance evaluation horizon changed")
                        eval_counts["primitive_steps"] += row["episode_length"]
                        for key in eval_counts:
                            if key != "primitive_steps":
                                eval_counts[key] += row[key]
                    means[str(iteration)][mode][policy] = {k: float(np.mean([r[k] for r in rows])) for k in spec.METRICS}
        if (train_counts != cell["inference_counts"]["train"] or eval_counts != cell["inference_counts"]["eval"]
                or train_counts["primitive_steps"] != budget["training_primitive_steps"] or eval_counts["primitive_steps"] != budget["evaluation_primitive_steps"]):
            raise ValueError("actor acceptance inference accounting changed")
        root_rows.append({"root": root, "means": means, "decisions": decisions,
                          "endpoints": spec.contrasts(means, preflight=preflight)})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
               "contract": spec.contract(), "root_rows": root_rows, "optimizer_steps": totals,
               "native_trace_audits": sum(r["native_trace_audits"] for r in results), "verification_primitive_steps": 0,
               "method_cost": {k: sum(r["inference_counts"][p][k] for r in results for p in ("train", "eval"))
                               for k in ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}}
    if not preflight:
        x = np.asarray([[row["endpoints"][k] for k in spec.ENDPOINTS] for row in root_rows])
        indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
        summary["primary_endpoints"] = {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
            "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"} for i, k in enumerate(spec.ENDPOINTS)}
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        np.testing.assert_allclose(np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0), bounds, atol=1e-10, rtol=0)
        summary["independent_statistics"] = {"status": "passed", "endpoints": len(spec.ENDPOINTS)}
    return summary


def check_sampling(rows, root, seeds, *, phase, mode):
    if [r["seed"] for r in rows] != seeds:
        raise ValueError("actor acceptance paired seed roster changed")
    for row in rows:
        kwargs = spec.rollout_arguments(root, row["seed"], phase=phase, mode=mode)
        if (row["policy_seed"] != spec.policy_seed(root, row["seed"])
                or (phase == "eval" and row["deployment_mode"] != mode)
                or any(row[k] != v for k, v in kwargs.items() if k != "sample")):
            raise ValueError("actor acceptance paired sampling changed")
