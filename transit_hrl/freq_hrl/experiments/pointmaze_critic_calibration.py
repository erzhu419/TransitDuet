"""Compare delayed actor learning with and without critic-only calibration."""

import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import load_controller, raw_directory, write_json
from .pointmaze_update_isolation import change_norms
from scripts import pointmaze_critic_calibration_stage39_spec as spec


def monte_carlo_returns(batch, gamma):
    target = np.empty(batch.size, dtype=np.float64)
    successor = 0.
    for i in range(batch.size - 1, -1, -1):
        successor = float(batch.reward[i]) + gamma ** int(batch.duration[i]) * (1. - batch.done[i]) * successor
        target[i] = successor
    return target


def policy_drift(actor, reference, state):
    with torch.no_grad():
        tensor = torch.as_tensor(state, dtype=torch.float32, device=next(actor.parameters()).device)
        before, after = reference.distribution(tensor), actor.distribution(tensor)
        kl = torch.distributions.kl_divergence(before, after).sum(dim=-1)
        shift = torch.tanh(after.mean) - torch.tanh(before.mean)
        return {"gaussian_kl": float(kl.mean()), "squashed_mean_action_rmse": float(shift.square().mean().sqrt()),
                "mean_std": float(after.stddev.mean())}


def probe_diagnostics(model, batch, reference):
    with torch.no_grad():
        value_state = batch.state if batch.value_state is None else batch.value_state
        value = model.lower_value(torch.as_tensor(value_state, dtype=torch.float32, device=model.device)).cpu().numpy()
    target = monte_carlo_returns(batch, model.config.gamma)
    return {"value_mean": float(value.mean()), "mc_return_mean": float(target.mean()),
            "mc_value_mse": float(np.mean((value - target) ** 2)),
            **policy_drift(model.lower_actor, reference, batch.state)}


def worker_rollout(job):
    weights, seed, phase, mode, path, _ = job
    model, args, native, credit = joint._WORKER
    model.load_state_dict(weights)
    sampling = phase in ("train", "probe")
    policy_seed = int(seed) + args.optimizer_seed if sampling else spec.policy_seed(args.optimizer_seed, seed)
    torch.manual_seed(policy_seed)
    batch, row, raw = joint.rollout(model, args, native, seed=seed, sample=sampling,
                                  capture=path is not None, lower_credit=credit,
                                  lower_sample=None if sampling else mode == "lower_sampled")
    row.update(policy_seed=policy_seed, deployment_mode=mode)
    if path is not None:
        np.savez_compressed(path, **raw)
    return batch, row


def lower_update(model, batch, kind):
    if kind == "none":
        return {"lower_actor_optimizer_steps": 0., "lower_value_optimizer_steps": 0.}
    if kind not in ("critic", "actor_critic"):
        raise ValueError("unknown lower update kind")
    return model._update_level(level="lower", batch=batch, actor=model.lower_actor,
                               value_net=model.lower_value, actor_optimizer=model.lower_actor_optimizer,
                               value_optimizer=model.lower_value_optimizer,
                               actor_updates_enabled=kind == "actor_critic")


def train(root, method, *, preflight, output, specification=spec, rollout_worker=worker_rollout,
          model_factory=joint.make_model):
    spec = specification
    if method not in spec.METHODS:
        raise ValueError("unregistered calibration method")
    args, opt = spec.source.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles = spec.seed_roles(root, preflight=preflight)
    controller, source_cell, replay = load_controller(args, spec.source_result(root, preflight=preflight))
    model = model_factory(controller, "learned_history", root=root)
    if model.lower_cost_value is not None or model.upper_cost_value is not None or model.hf_actor is not None:
        raise ValueError("Stage-39 requires the unconstrained Stage-33 controller")
    initial, anchor = joint.inference_weights(model), copy.deepcopy(model.lower_actor)
    raw, started = raw_directory(output), time.monotonic()
    costs, stages, training = [], {}, []
    initial_training_credit, first_learning_credit = None, None
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
                             initializer=joint.init_worker,
                             initargs=(model.config, args, "learned_history", spec.LOWER_CREDIT[method])) as pool:
        def episodes(seeds, *, phase, mode="training", directory=None):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(model)
            pairs = list(pool.map(rollout_worker, [(weights, seed, phase, mode,
                         str(directory / f"episode_{seed}.npz") if directory is not None else None, method) for seed in seeds]))
            costs.extend({"phase": phase, **{k: row[k] for k in
                          ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}} for _, row in pairs)
            return pairs

        probe, probe_row = episodes(roles["probe"], phase="probe", directory=raw / "probe")[0]

        def snapshot(iteration):
            path = raw / f"iteration_{iteration}.pt"
            torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "method": method,
                        "iteration": iteration, "state_dict": model.state_dict()}, path)
            stage = {"checkpoint": str(path), "diagnostics": probe_diagnostics(model, probe.lower, anchor),
                     "parameter_change_norms": change_norms(joint.inference_weights(model), initial),
                     "evaluation_rows": {}}
            for mode in spec.MODES:
                directory = raw / f"iteration_{iteration}" / mode
                stage["evaluation_rows"][mode] = [row for _, row in episodes(
                    roles["evaluation"], phase="eval", mode=mode, directory=directory)]
            stages[str(iteration)] = stage
            print(f"root {root} {method}: snapshot {iteration}; elapsed {time.monotonic() - started:.1f}s", flush=True)

        snapshot(0)
        for iteration in range(1, spec.iterations(preflight=preflight) + 1):
            start = (iteration - 1) * opt["rollouts_per_iteration"]
            mode = "warmup" if iteration <= opt["warmup_iterations"] else "learning"
            pairs = episodes(roles["training"][start:start + opt["rollouts_per_iteration"]], phase="train", mode=mode)
            if initial_training_credit is None:
                initial_training_credit = {"seed": pairs[0][1]["seed"], **pairs[0][1]["lower_training_credit"]}
            if first_learning_credit is None and iteration > opt["warmup_iterations"]:
                first_learning_credit = {"seed": pairs[0][1]["seed"], **pairs[0][1]["lower_training_credit"]}
            batch = concat_hierarchical_batches([b for b, _ in pairs]).lower
            before = copy.deepcopy(model.lower_actor)
            advantage, target = model._gae(batch.reward, batch.done, batch.duration, batch.old_value)
            np.random.seed(spec.shuffle_seed(root, iteration))
            kind = spec.update_kind(method, iteration, preflight=preflight)
            update = lower_update(model, batch, kind)
            training.append({"iteration": iteration, "update_kind": kind, "primitive_steps": batch.size,
                             "reward_sum": float(np.sum(batch.reward, dtype=np.float64)),
                             "task_reward_sum": sum(row["lower_training_credit"]["task_reward_sum"] for _, row in pairs),
                             "done_count": int(batch.done.sum()),
                             "option_count": sum(row["lower_training_credit"]["option_count"] for _, row in pairs),
                             "old_value_mean": float(batch.old_value.mean()), "gae_target_mean": float(target.mean()),
                             "advantage_mean": float(advantage.mean()), "advantage_std": float(advantage.std()),
                             "actor_optimizer_steps": int(update["lower_actor_optimizer_steps"]),
                             "value_optimizer_steps": int(update["lower_value_optimizer_steps"]),
                             "rollout_sampling": [{key: row[key] for key in
                                 ("seed", "policy_seed", "upper_sample", "gate_sample", "lower_sample", "lower_seed", "gate_seed")}
                                 for _, row in pairs],
                             "policy_drift": policy_drift(model.lower_actor, before, batch.state)})
            if iteration in spec.snapshots(preflight=preflight):
                snapshot(iteration)
    counts = {phase: {key: sum(row[key] for row in costs if row["phase"] == phase)
                     for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}
              for phase in ("train", "probe", "eval")}
    counts["factual_replay"] = {"upper_inference_calls": len(source_cell["factual_row"]["decision_steps"]),
                                "lower_inference_calls": args.horizon, "gate_inference_calls": 0}
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
              "root": root, "method": method, "preflight": preflight, "options": opt, "seed_roles": roles,
              "budget": spec.budget(preflight=preflight), "inference_counts": counts,
              "source_selected_iteration": source_cell["selected_checkpoint_iteration"], "source_replay": replay,
              "probe_credit": {"seed": probe_row["seed"], **probe_row["lower_training_credit"]},
              "initial_training_credit": initial_training_credit, "training": training, "snapshots": stages,
              "first_learning_credit": first_learning_credit,
              "optimizer_steps": {name: sum(row[name] for row in training)
                                  for name in ("actor_optimizer_steps", "value_optimizer_steps")},
              "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def audit_result(result, *, raw_path, specification=spec):
    spec = specification
    root, method, preflight = result["root"], result["method"], result["preflight"]
    if (method not in spec.METHODS or root not in spec.roots(preflight=preflight)
            or result["status"] != "complete" or result["protocol"] != spec.EXPERIMENT_PROTOCOL
            or result["contract"] != spec.contract() or result["options"] != spec.options(preflight=preflight)
            or result["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or result["budget"] != spec.budget(preflight=preflight)):
        raise ValueError("calibration result violates the frozen protocol")
    args, opt = spec.source.arguments(root, preflight=preflight), result["options"]
    for phase, field in (("train", "training_primitive_steps"), ("eval", "evaluation_primitive_steps"),
                         ("probe", "probe_primitive_steps"), ("factual_replay", "factual_replay_primitive_steps")):
        if result["inference_counts"][phase]["lower_inference_calls"] != result["budget"][field]:
            raise ValueError("calibration primitive accounting changed")
    if [row["iteration"] for row in result["training"]] != list(range(1, spec.iterations(preflight=preflight) + 1)):
        raise ValueError("calibration training roster incomplete")
    for row in result["training"]:
        kind = spec.update_kind(method, row["iteration"], preflight=preflight)
        if (row["update_kind"] != kind or row["primitive_steps"] != opt["rollouts_per_iteration"] * args.horizon
                or row["done_count"] != row["option_count"]
                or (row["actor_optimizer_steps"] > 0) != (kind == "actor_critic")
                or (row["value_optimizer_steps"] > 0) != (kind != "none")):
            raise ValueError("calibration actor/critic update contract violated")
        if kind != "actor_critic" and row["policy_drift"]["gaussian_kl"] != 0.:
            raise ValueError("frozen warmup actor distribution changed")
        if spec.LOWER_CREDIT[method] == "task_option":
            np.testing.assert_allclose(row["reward_sum"], row["task_reward_sum"], atol=1e-10, rtol=0)
    for key, count in result["optimizer_steps"].items():
        if count != sum(row[key] for row in result["training"]):
            raise ValueError("calibration optimizer accounting changed")
    expected_stages = [str(i) for i in spec.snapshots(preflight=preflight)]
    if set(result["snapshots"]) != set(expected_stages):
        raise ValueError("calibration snapshot roster incomplete")
    warm = opt["warmup_iterations"]
    for text, stage in result["snapshots"].items():
        iteration = int(text)
        for name, delta in stage["parameter_change_norms"].items():
            allowed = (name == "lower_value" and method != "frozen" and
                       (iteration > warm or (iteration > 0 and spec.update_kind(method, iteration, preflight=preflight) == "critic"))) or (
                       name == "lower_actor" and method != "frozen" and iteration > warm)
            if (not allowed and delta != 0.) or (allowed and delta <= 0.):
                raise ValueError("calibration snapshot component freeze violated")
        if set(stage["evaluation_rows"]) != set(spec.MODES):
            raise ValueError("calibration deployment modes incomplete")
        for mode, rows in stage["evaluation_rows"].items():
            if [row["seed"] for row in rows] != result["seed_roles"]["evaluation"]:
                raise ValueError("calibration evaluation roster incomplete")
            if any(row["lower_sample"] != (mode == "lower_sampled") or row["gate_sample"]
                   or row["deployment_mode"] != mode or row["policy_seed"] != spec.policy_seed(root, row["seed"]) for row in rows):
                raise ValueError("lower-only deployment sampling changed")
            joint.audit_trajectories(rows, args=args, method="learned_history",
                                     raw_path=Path(raw_path) / f"iteration_{text}" / mode)
    rows = [row for stage in result["snapshots"].values() for values in stage["evaluation_rows"].values() for row in values]
    for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
        if sum(row[key] for row in rows) != result["inference_counts"]["eval"][key]:
            raise ValueError("calibration evaluation inference accounting changed")
    return {"root": root, "method": method, "status": "passed", "episodes": len(rows)}


def aggregate(results, *, preflight, specification=spec):
    spec = specification
    roots = spec.roots(preflight=preflight)
    cells = {(r["root"], r["method"]): r for r in results}
    if len(cells) != len(results) or set(cells) != {(r, m) for r in roots for m in spec.METHODS}:
        raise ValueError("calibration root/method roster incomplete")
    root_rows = []
    for root in roots:
        means = {str(i): {mode: {m: {k: float(np.mean([row[k] for row in cells[root, m]["snapshots"][str(i)]["evaluation_rows"][mode]]))
                                   for k in spec.METRICS} for m in spec.METHODS} for mode in spec.MODES}
                 for i in spec.snapshots(preflight=preflight)}
        root_rows.append({"root": root, "means": means,
                          "endpoints": dict(zip(spec.ENDPOINTS, spec.contrasts(means, preflight=preflight)))})
    summary = {"root_count": len(roots), "root_rows": root_rows, "checkpoint_selection": "none"}
    if not preflight:
        x = np.asarray([[row["endpoints"][key] for key in spec.ENDPOINTS] for row in root_rows])
        rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
        draws = x[rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))].mean(axis=1)
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(draws, [tail, 1 - tail], axis=0)
        summary["primary_endpoints"] = {key: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
                    "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
                    for i, key in enumerate(spec.ENDPOINTS)}
    return summary
