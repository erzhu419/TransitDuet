#!/usr/bin/env python3
"""Verify matched warmup, actual learning execution and every deployment snapshot."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments.pointmaze_critic_calibration import aggregate, probe_diagnostics
from freq_hrl.experiments.pointmaze_frozen_execution import audit_result
from freq_hrl.experiments.pointmaze_lower_credit import audit_credit_batch
from freq_hrl.experiments.pointmaze_root_response import write_json
from freq_hrl.experiments.pointmaze_update_isolation import change_norms
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts.analyze_pointmaze_critic_calibration_stage39 import compare_trace
from scripts import pointmaze_frozen_execution_stage40_spec as spec


def audit_warmup_pair(left, right):
    for name in ("upper_actor", "lower_actor", "upper_value", "lower_value", "promotion_actor", "promotion_value"):
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    a, b = left["lower_value_optimizer"], right["lower_value_optimizer"]
    if a["param_groups"] != b["param_groups"]:
        raise ValueError("paired warmup critic Adam configuration differs")
    torch.testing.assert_close(a["state"], b["state"], rtol=0, atol=0)


def analyze(run_name, *, preflight):
    directory = spec.ROOT / "results" / run_name
    results, audits, replays, probes, diagnostics, shared_warmup = [], [], [], [], [], []
    verification = {"primitive_steps": 0, "upper_inference_calls": 0, "lower_inference_calls": 0, "gate_inference_calls": 0}

    def charge(row):
        verification["primitive_steps"] += row["episode_length"]
        for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
            verification[key] += row[key]

    warm = spec.options(preflight=preflight)["warmup_iterations"]
    for root in spec.roots(preflight=preflight):
        source_cell = json.loads(spec.source_result(root, preflight=preflight).read_text())["cells"][0]
        payload = torch.load(source_cell["controller_checkpoint"], map_location="cpu", weights_only=False)
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**payload["state_dict"]["config"]))
        controller.load_state_dict(payload["state_dict"])
        args = spec.source.arguments(root, preflight=preflight)
        warmups = {}
        for method in spec.METHODS:
            path = directory / "cells" / method / f"replicate_{root}" / "result.json"
            result = json.loads(path.read_text())
            if (result["root"], result["method"], result["preflight"]) != (root, method, preflight):
                raise ValueError("execution cell identity changed")
            raw = path.parent.with_name(path.parent.name + "_raw")
            audits.append(audit_result(result, raw_path=raw))
            initial_model = joint.make_model(controller, "learned_history", root=root)
            initial, anchor = joint.inference_weights(initial_model), initial_model.lower_actor
            credit = spec.LOWER_CREDIT[method]
            roles = spec.seed_roles(root, preflight=preflight)
            probe_batch = None
            for phase, field in (("probe", "probe_credit"), ("training", "initial_training_credit")):
                seed = roles["probe" if phase == "probe" else "training"][0]
                kwargs = spec.rollout_arguments(root, method, seed, phase="probe" if phase == "probe" else "train", mode="warmup")
                torch.manual_seed(seed + root)
                batch, row, trace = joint.rollout(initial_model, args, "learned_history", seed=seed,
                                                capture=True, lower_credit=credit, **kwargs)
                audit_credit_batch(batch, row, trace, args=args, mode=credit)
                if {"seed": seed, **row["lower_training_credit"]} != result[field]:
                    raise ValueError("native initial execution credit differs from worker batch")
                if phase == "probe":
                    probe_batch = batch.lower
                    compare_trace(trace, raw / "probe" / f"episode_{seed}.npz")
                charge(row)
                probes.append({"root": root, "method": method, "phase": phase, "status": "passed"})
            for iteration in spec.snapshots(preflight=preflight):
                stage = result["snapshots"][str(iteration)]
                checkpoint = torch.load(stage["checkpoint"], map_location="cpu", weights_only=False)
                if (checkpoint["protocol"], checkpoint["root"], checkpoint["method"], checkpoint["iteration"]) != (
                        spec.EXPERIMENT_PROTOCOL, root, method, iteration):
                    raise ValueError("execution checkpoint identity changed")
                evaluated = joint.make_model(controller, "learned_history", root=root)
                if checkpoint["state_dict"]["config"] != evaluated.state_dict()["config"]:
                    raise ValueError("execution PPO configuration changed")
                evaluated.load_state_dict(checkpoint["state_dict"])
                measured = change_norms(joint.inference_weights(evaluated), initial)
                for key, value in measured.items():
                    np.testing.assert_allclose(value, stage["parameter_change_norms"][key], atol=1e-12, rtol=0)
                for key, value in probe_diagnostics(evaluated, probe_batch, anchor).items():
                    np.testing.assert_allclose(value, stage["diagnostics"][key], atol=1e-10, rtol=0)
                if iteration == warm:
                    warmups[method] = checkpoint["state_dict"]
                    seed = roles["training"][warm * spec.options(preflight=preflight)["rollouts_per_iteration"]]
                    kwargs = spec.rollout_arguments(root, method, seed, phase="train", mode="learning")
                    torch.manual_seed(seed + root)
                    batch, row, trace = joint.rollout(evaluated, args, "learned_history", seed=seed,
                                                    capture=True, lower_credit=credit, **kwargs)
                    audit_credit_batch(batch, row, trace, args=args, mode=credit)
                    if {"seed": seed, **row["lower_training_credit"]} != result["first_learning_credit"]:
                        raise ValueError("native aligned learning credit differs from worker batch")
                    charge(row)
                    probes.append({"root": root, "method": method, "phase": "learning", "status": "passed"})
                for mode in spec.MODES:
                    expected = stage["evaluation_rows"][mode][0]
                    kwargs = spec.rollout_arguments(root, method, expected["seed"], phase="eval", mode=mode)
                    torch.manual_seed(spec.policy_seed(root, expected["seed"]))
                    _, observed, trace = joint.rollout(evaluated, args, "learned_history", seed=expected["seed"],
                                                       capture=True, lower_credit=credit, **kwargs)
                    for key in spec.METRICS:
                        np.testing.assert_allclose(observed[key], expected[key], atol=1e-6, rtol=0)
                    compare_trace(trace, raw / f"iteration_{iteration}" / mode / f"episode_{expected['seed']}.npz")
                    charge(observed)
                    replays.append({"root": root, "method": method, "iteration": iteration, "mode": mode, "status": "passed"})
            diagnostics.append({"root": root, "method": method,
                                "snapshots": {k: v["diagnostics"] for k, v in result["snapshots"].items()},
                                "first_update": result["training"][warm], "last_update": result["training"][-1]})
            results.append(result)
            print(f"audited {root}/{method}: snapshots, paired lower noise and native learning execution", flush=True)
        for reward in ("intrinsic", "task"):
            audit_warmup_pair(warmups[reward + "_sampled"], warmups[reward + "_aligned"])
            shared_warmup.append({"root": root, "reward": reward, "status": "passed"})
    if verification["primitive_steps"] != spec.verification_budget(preflight=preflight)["total_primitive_steps"]:
        raise ValueError("execution verification budget changed")
    summary = {"status": "preflight_passed" if preflight else "complete", "run_name": run_name,
               "protocol": spec.EXPERIMENT_PROTOCOL, "audits": audits, "checkpoint_replays": replays,
               "probe_replays": probes, "shared_warmup": shared_warmup, "diagnostics": diagnostics,
               "verification_cost": verification,
               "optimizer_steps_by_method": {m: {k: sum(r["optimizer_steps"][k] for r in results if r["method"] == m)
                                                   for k in ("actor_optimizer_steps", "value_optimizer_steps")} for m in spec.METHODS},
               "method_cost": {"primitive_steps": sum(r["budget"]["total_primitive_steps"] for r in results),
                               **{k: sum(count[k] for r in results for count in r["inference_counts"].values())
                                  for k in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}},
               "aggregate": aggregate(results, preflight=preflight, specification=spec)}
    if not preflight:
        x = np.asarray([[r["endpoints"][key] for key in spec.ENDPOINTS] for r in summary["aggregate"]["root_rows"]])
        rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
        indices = rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0)
        for i, key in enumerate(spec.ENDPOINTS):
            np.testing.assert_allclose(bounds[:, i], summary["aggregate"]["primary_endpoints"][key]["ci"], atol=1e-10, rtol=0)
        summary["independent_statistics"] = {"status": "passed", "endpoints": len(spec.ENDPOINTS)}
    write_json(directory / "qualification_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    analyze(args.run_name, preflight=args.preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
