"""Evaluate signed displacements without training or changing the update direction."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from .pointmaze_update_direction import load_pair
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_signed_microstep_stage44_spec as spec


def interpolated_weights(before, after, scale):
    weights = joint.inference_weights(before)
    old, new = weights["lower_actor"], after.lower_actor.state_dict()
    if scale == 0.:
        return weights
    weights["lower_actor"] = {key: value.detach().cpu().clone() if scale == 1.
                              else old[key] + scale * (value.detach().cpu() - old[key]) for key, value in new.items()}
    return weights


def policy_weights(pairs):
    reference = pairs[spec.METHODS[0]][0]
    frozen = joint.inference_weights(reference)
    weights = {"frozen": frozen}
    for method, (before, after) in pairs.items():
        for name, state in frozen.items():
            if name != "lower_value":
                torch.testing.assert_close(getattr(before, name).state_dict(), state, rtol=0, atol=0)
            if name not in ("lower_actor", "lower_value"):
                torch.testing.assert_close(getattr(after, name).state_dict(), state, rtol=0, atol=0)
        zero = interpolated_weights(before, after, 0.)
        torch.testing.assert_close(zero, joint.inference_weights(before), rtol=0, atol=0)
        for variant, scale in spec.SCALES.items():
            candidate = interpolated_weights(before, after, scale)
            if scale == 1.:
                torch.testing.assert_close(candidate["lower_actor"], after.lower_actor.state_dict(), rtol=0, atol=0)
            for name in candidate:
                if name != "lower_actor":
                    torch.testing.assert_close(candidate[name], zero[name], rtol=0, atol=0)
            weights[f"{method}:{variant}"] = candidate
    return weights


def worker_rollout(job):
    weights, seed, policy, mode, path = job
    model, args, _, _ = joint._WORKER
    model.load_state_dict(weights)
    method = policy.split(":")[0]
    original = spec.source.source
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    _, row, raw = joint.rollout(model, args, "learned_history", seed=seed, capture=True,
        lower_credit=original.LOWER_CREDIT[method], lower_value_context_builder=clocks.context_builder(method),
        **spec.rollout_arguments(args.optimizer_seed, seed, mode=mode))
    row.update(policy_seed=spec.policy_seed(args.optimizer_seed, seed), deployment_mode=mode)
    clocks.audit_context(None, row, raw["lower_value_context"], clock=original.VALUE_CLOCK[method])
    np.savez_compressed(path, **raw)
    return row


def diagnose(root, *, preflight, output):
    opt, roles = spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    args = spec.source.source.source.arguments(root, preflight=preflight)
    sources = {m: json.loads(spec.source_result(root, m, preflight=preflight).read_text()) for m in spec.METHODS}
    pairs = {m: load_pair(sources[m], root=root, method=m, preflight=preflight) for m in spec.METHODS}
    reference = pairs[spec.METHODS[0]][0]
    weights = policy_weights(pairs)
    raw, started = raw_directory(output), time.monotonic()
    counts = {key: 0 for key in ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}
    evaluation = {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=joint.init_worker,
                             initargs=(reference.config, args, "learned_history", "intrinsic_option")) as pool:
        for policy in spec.POLICIES:
            evaluation[policy] = {}
            for mode in spec.MODES:
                directory = raw / policy.replace(":", "/") / mode
                directory.mkdir(parents=True, exist_ok=True)
                rows = list(pool.map(worker_rollout, [(weights[policy], seed, policy, mode,
                        str(directory / f"episode_{seed}.npz")) for seed in roles["evaluation"]]))
                joint.audit_trajectories(rows, args=args, method="learned_history", raw_path=directory)
                evaluation[policy][mode] = rows
                for row in rows:
                    counts["primitive_steps"] += row["episode_length"]
                    for key in counts:
                        if key != "primitive_steps":
                            counts[key] += row[key]
            print(f"evaluated {root}/{policy}: signed displacement, both native modes", flush=True)
    budget = spec.budget(preflight=preflight)
    if counts["primitive_steps"] != budget["evaluation_primitive_steps"]:
        raise ValueError("signed-microstep native accounting changed")
    warm = spec.source.options(preflight=preflight)["warmup_iterations"]
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
              "root": root, "preflight": preflight, "seed_roles": roles, "budget": budget,
              "inference_counts": {"evaluation": counts}, "optimizer_steps": 0,
              "weight_controls": "shared_before_actors_exact_zero_full_exact_other_networks_unchanged",
              "source_checkpoints": {m: [sources[m]["snapshots"][str(i)]["checkpoint"] for i in (warm, warm + 1)] for m in spec.METHODS},
              "evaluation_rows": evaluation, "native_trace_audits": budget["native_trace_audits"],
              "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def aggregate(results, *, preflight):
    roots = spec.roots(preflight=preflight)
    cells = {r["root"]: r for r in results}
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("signed-microstep root roster incomplete")
    root_rows = []
    opt, budget = spec.options(preflight=preflight), spec.budget(preflight=preflight)
    for root in roots:
        cell = cells[root]
        if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
                or cell["preflight"] != preflight or cell["contract"] != spec.contract()
                or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight) or cell["budget"] != budget
                or cell["optimizer_steps"] != 0 or cell["native_trace_audits"] != budget["native_trace_audits"]
                or cell["weight_controls"] != "shared_before_actors_exact_zero_full_exact_other_networks_unchanged"):
            raise ValueError("signed-microstep result violates the frozen protocol")
        if set(cell["evaluation_rows"]) != set(spec.POLICIES):
            raise ValueError("signed-microstep policy roster incomplete")
        counted = {key: 0 for key in cell["inference_counts"]["evaluation"]}
        for policy in spec.POLICIES:
            if set(cell["evaluation_rows"][policy]) != set(spec.MODES):
                raise ValueError("signed-microstep mode roster incomplete")
            for mode in spec.MODES:
                rows = cell["evaluation_rows"][policy][mode]
                if [r["seed"] for r in rows] != cell["seed_roles"]["evaluation"]:
                    raise ValueError("signed-microstep fresh paired roster changed")
                for row in rows:
                    kwargs = spec.rollout_arguments(root, row["seed"], mode=mode)
                    if (row["policy_seed"] != spec.policy_seed(root, row["seed"]) or row["deployment_mode"] != mode
                            or any(row[k] != v for k, v in kwargs.items() if k != "sample")):
                        raise ValueError("signed-microstep sampling changed")
                    if row["episode_length"] != opt["horizon"]:
                        raise ValueError("signed-microstep native horizon changed")
                    counted["primitive_steps"] += row["episode_length"]
                    for key in counted:
                        if key != "primitive_steps":
                            counted[key] += row[key]
        if counted != cell["inference_counts"]["evaluation"] or counted["primitive_steps"] != budget["total_primitive_steps"]:
            raise ValueError("signed-microstep inference accounting changed")
        means = {mode: {p: {k: float(np.mean([r[k] for r in cell["evaluation_rows"][p][mode]])) for k in spec.METRICS}
                        for p in spec.POLICIES} for mode in spec.MODES}
        full_vs_zero = {mode: {m: means[mode][m + ":full"]["episode_return"] - means[mode]["frozen"]["episode_return"]
                             for m in spec.METHODS} for mode in spec.MODES}
        root_rows.append({"root": root, "means": means, "endpoints": spec.contrasts(means["lower_sampled"]),
                          "descriptive_full_vs_zero": full_vs_zero, "inference_counts": cell["inference_counts"],
                          "native_trace_audits": cell["native_trace_audits"]})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
               "contract": copy.deepcopy(spec.contract()), "root_rows": root_rows, "optimizer_steps": 0,
               "native_trace_audits": sum(r["native_trace_audits"] for r in results),
               "method_cost": {k: sum(r["inference_counts"]["evaluation"][k] for r in results) for k in
                               ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")},
               "verification_primitive_steps": 0}
    if not preflight:
        x = np.asarray([[r["endpoints"][key] for key in spec.ENDPOINTS] for r in root_rows])
        rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
        indices = rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
        summary["primary_endpoints"] = {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
            "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"} for i, k in enumerate(spec.ENDPOINTS)}
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        np.testing.assert_allclose(np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0), bounds, atol=1e-10, rtol=0)
        summary["independent_statistics"] = {"status": "passed", "endpoints": len(spec.ENDPOINTS)}
    return summary
