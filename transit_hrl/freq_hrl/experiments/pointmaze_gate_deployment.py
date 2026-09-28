"""Compare cached final gates under threshold and Bernoulli deployment."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_level_deployment_stage37_spec as spec


def load_cached(root, method, *, preflight):
    result = json.loads(spec.gate_source_result(root, method, preflight=preflight).read_text())
    if (result["status"], result["protocol"], result["root"], result["method"], result["preflight"]) != (
            "complete", spec.previous.EXPERIMENT_PROTOCOL, root, method, preflight):
        raise ValueError("cached gate source identity changed")
    checkpoint = torch.load(result["final_checkpoint"], map_location="cpu", weights_only=False)
    if (checkpoint["protocol"], checkpoint["root"], checkpoint["method"], checkpoint["iteration"]) != (
            spec.previous.EXPERIMENT_PROTOCOL, root, method, spec.previous.options(preflight=preflight)["iterations"]):
        raise ValueError("gate diagnosis requires the registered final checkpoint")
    model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**checkpoint["state_dict"]["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model, result["final_checkpoint"], checkpoint["iteration"]


_WORKER = None


def init_worker(config, args):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = FrequencySeparatedActorCriticPPO(config), args


def worker_rollout(job):
    weights, seed, mode, stream, gate_seed, path = job
    model, args = _WORKER
    model.load_state_dict(weights)
    _, row, trace = joint.rollout(model, args, "learned_history", seed=seed, sample=False,
                                 capture=True, gate_sample=mode == "sampled", gate_seed=gate_seed)
    np.savez_compressed(path, **trace)
    return {**row, "mode": mode, "stream": stream}


def evaluate(root, *, preflight, output):
    args = spec.source.arguments(root, preflight=preflight)
    opt, roles = spec.gate_options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    raw = raw_directory(output)
    rows, checkpoints = {}, {}
    started = time.monotonic()
    first, _, _ = load_cached(root, spec.GATE_SOURCES[0], preflight=preflight)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(first.config, args)) as pool:
        for method in spec.GATE_SOURCES:
            model, checkpoint, iteration = load_cached(root, method, preflight=preflight)
            checkpoints[method] = {"path": checkpoint, "iteration": iteration}
            weights = joint.inference_weights(model)
            rows[method] = {}
            for mode in spec.GATE_MODES:
                streams = range(opt["sample_streams"]) if mode == "sampled" else range(1)
                jobs = []
                for stream in streams:
                    directory = raw / method / mode / f"stream_{stream}"
                    directory.mkdir(parents=True, exist_ok=True)
                    jobs.extend((weights, seed, mode, stream,
                                 spec.gate_seed(root, seed, stream) if mode == "sampled" else None,
                                 str(directory / f"episode_{seed}.npz")) for seed in roles["gate_evaluation"])
                rows[method][mode] = list(pool.map(worker_rollout, jobs))
            print(f"root {root} cached {method}: both gate modes complete; elapsed {time.monotonic() - started:.1f}s", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "method": spec.GATE_TASK,
              "phase": "cached_gate_deployment", "root": root, "preflight": preflight,
              "contract": spec.gate_contract(), "options": opt, "seed_roles": roles,
              "budget": spec.gate_budget(preflight=preflight), "checkpoints": checkpoints,
              "evaluation_rows": rows, "wall_seconds": time.monotonic() - started,
              "inference_counts": {k: sum(row[k] for modes in rows.values() for group in modes.values() for row in group)
                                   for k in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}}
    write_json(output, result)
    return result


def audit_probabilities(model, row, path):
    with np.load(path) as trace:
        states, probabilities = trace["gate_states"], trace["gate_probabilities"]
    with torch.no_grad():
        measured = model.promotion_actor.distribution(torch.as_tensor(states, dtype=torch.float32)).probs.numpy().reshape(-1)
    np.testing.assert_allclose(probabilities, measured, atol=2e-6, rtol=0)
    actions = []
    for step, p in zip(row["gate_steps"], probabilities):
        if row["gate_sample"]:
            torch.manual_seed(row["gate_seed"] + step)
            action = int(torch.bernoulli(torch.tensor([[p]], dtype=torch.float32)).item())
        else:
            action = int(p >= .5)
        actions.append(action)
    np.testing.assert_array_equal(actions, row["gate_actions"])


def audit_result(result, *, raw_path):
    root, preflight = result["root"], result["preflight"]
    if (result["status"] != "complete" or result["protocol"] != spec.EXPERIMENT_PROTOCOL
            or result["method"] != spec.GATE_TASK or result["phase"] != "cached_gate_deployment"
            or result["contract"] != spec.gate_contract() or result["options"] != spec.gate_options(preflight=preflight)
            or result["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or result["budget"] != spec.gate_budget(preflight=preflight)):
        raise ValueError("gate deployment result violates the frozen protocol")
    args, replays = spec.source.arguments(root, preflight=preflight), []
    verification = {"primitive_steps": 0, "upper_inference_calls": 0, "lower_inference_calls": 0, "gate_inference_calls": 0}
    for method in spec.GATE_SOURCES:
        model, checkpoint, iteration = load_cached(root, method, preflight=preflight)
        if result["checkpoints"][method] != {"path": checkpoint, "iteration": iteration}:
            raise ValueError("gate evaluation did not use the cached final checkpoint")
        for mode in spec.GATE_MODES:
            rows = result["evaluation_rows"][method][mode]
            streams = range(result["options"]["sample_streams"]) if mode == "sampled" else range(1)
            expected = [(seed, stream) for stream in streams for seed in result["seed_roles"]["gate_evaluation"]]
            if [(row["seed"], row["stream"]) for row in rows] != expected:
                raise ValueError("gate deployment path/stream roster incomplete")
            for stream in streams:
                group = [row for row in rows if row["stream"] == stream]
                directory = Path(raw_path) / method / mode / f"stream_{stream}"
                joint.audit_trajectories(group, args=args, method="learned_history", raw_path=directory)
                for row in group:
                    expected_seed = spec.gate_seed(root, row["seed"], stream) if mode == "sampled" else None
                    if row["mode"] != mode or row["gate_sample"] != (mode == "sampled") or row["gate_seed"] != expected_seed:
                        raise ValueError("gate sampling mode or seed changed")
                    audit_probabilities(model, row, directory / f"episode_{row['seed']}.npz")
            expected_row = rows[0]
            _, observed, _ = joint.rollout(model, args, "learned_history", seed=expected_row["seed"], sample=False,
                                          gate_sample=mode == "sampled", gate_seed=expected_row["gate_seed"])
            for key in ("episode_return", "tracking_squared_error_integral", "charged_utility"):
                np.testing.assert_allclose(observed[key], expected_row[key], atol=1e-6, rtol=0)
            for key in ("decision_steps", "gate_steps", "gate_actions"):
                np.testing.assert_array_equal(observed[key], expected_row[key])
            verification["primitive_steps"] += args.horizon
            for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
                verification[key] += observed[key]
            replays.append({"root": root, "source": method, "mode": mode, "status": "passed"})
    all_rows = [row for modes in result["evaluation_rows"].values() for group in modes.values() for row in group]
    if len(all_rows) != result["budget"]["evaluation_episodes"] or result["inference_counts"]["lower_inference_calls"] != result["budget"]["total_primitive_steps"]:
        raise ValueError("cached gate primitive accounting changed")
    for key in result["inference_counts"]:
        if sum(row[key] for row in all_rows) != result["inference_counts"][key]:
            raise ValueError("cached gate inference accounting changed")
    return {"status": "passed", "root": root, "episodes": len(all_rows),
            "replays": replays, "verification_cost": verification}


def aggregate(results):
    cells = {r["root"]: r for r in results}
    if len(results) != len(spec.OPTIMIZER_ROOTS) or set(cells) != set(spec.OPTIMIZER_ROOTS):
        raise ValueError("full cached gate root roster incomplete")
    keys = ("episode_return", "tracking_squared_error_integral", "upper_inference_calls", "charged_utility")
    roots = []
    for root in spec.OPTIMIZER_ROOTS:
        means = {m: {mode: {k: float(np.mean([row[k] for row in cells[root]["evaluation_rows"][m][mode]]))
                           for k in keys} for mode in spec.GATE_MODES} for m in spec.GATE_SOURCES}
        roots.append({"root": root, "means": means, "endpoints": dict(zip(spec.GATE_ENDPOINTS, spec.gate_contrasts(means)))})
    x = np.array([[r["endpoints"][k] for k in spec.GATE_ENDPOINTS] for r in roots])
    rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
    draws = x[rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))].mean(axis=1)
    tail = .05 / (2 * spec.CI_FAMILY_SIZE)
    bounds = np.quantile(draws, [tail, 1 - tail], axis=0)
    endpoints = {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
                     "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
                 for i, k in enumerate(spec.GATE_ENDPOINTS)}
    means = {m: {mode: {k: float(np.mean([r["means"][m][mode][k] for r in roots])) for k in keys}
                 for mode in spec.GATE_MODES} for m in spec.GATE_SOURCES}
    return {"status": "stage37_gate_deployment_diagnosis_complete", "root_count": len(roots),
            "primary_endpoints": endpoints, "means": means, "root_rows": roots}
