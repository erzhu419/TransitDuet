#!/usr/bin/env python3
"""Verify every calibration snapshot and its fixed native probe."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments.pointmaze_critic_calibration import aggregate, audit_result, probe_diagnostics
from freq_hrl.experiments.pointmaze_lower_credit import audit_credit_batch
from freq_hrl.experiments.pointmaze_root_response import write_json
from freq_hrl.experiments.pointmaze_update_isolation import change_norms
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_critic_calibration_stage39_spec as spec


def compare_trace(observed, path):
    with np.load(path) as expected:
        for key, value in observed.items():
            if key in ("decision_steps", "gate_steps", "gate_actions"):
                np.testing.assert_array_equal(value, expected[key])
            else:
                np.testing.assert_allclose(value, expected[key], atol=1e-7, rtol=1e-6)


def analyze(run_name, *, preflight):
    directory = spec.ROOT / "results" / run_name
    results, audits, replays, probes, diagnostics = [], [], [], [], []
    verification = {"primitive_steps": 0, "upper_inference_calls": 0, "lower_inference_calls": 0, "gate_inference_calls": 0}

    def charge(row):
        verification["primitive_steps"] += row["episode_length"]
        for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
            verification[key] += row[key]

    for root in spec.roots(preflight=preflight):
        source_cell = json.loads(spec.source_result(root, preflight=preflight).read_text())["cells"][0]
        payload = torch.load(source_cell["controller_checkpoint"], map_location="cpu", weights_only=False)
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**payload["state_dict"]["config"]))
        controller.load_state_dict(payload["state_dict"])
        args = spec.source.arguments(root, preflight=preflight)
        for method in spec.METHODS:
            path = directory / "cells" / method / f"replicate_{root}" / "result.json"
            result = json.loads(path.read_text())
            if (result["root"], result["method"], result["preflight"]) != (root, method, preflight):
                raise ValueError("calibration cell identity changed")
            raw = path.parent.with_name(path.parent.name + "_raw")
            audits.append(audit_result(result, raw_path=raw))
            model = joint.make_model(controller, "learned_history", root=root)
            initial, anchor = joint.inference_weights(model), model.lower_actor
            credit = spec.LOWER_CREDIT[method]
            probe_batch = None
            for phase, field in (("probe", "probe_credit"), ("training", "initial_training_credit")):
                seed = spec.seed_roles(root, preflight=preflight)[phase][0]
                torch.manual_seed(seed + root)
                batch, row, trace = joint.rollout(model, args, "learned_history", seed=seed,
                                                sample=True, capture=True, lower_credit=credit)
                audit_credit_batch(batch, row, trace, args=args, mode=credit)
                if {"seed": seed, **row["lower_training_credit"]} != result[field]:
                    raise ValueError("native probe or initial training credit differs from worker batch")
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
                    raise ValueError("calibration checkpoint identity changed")
                evaluated = joint.make_model(controller, "learned_history", root=root)
                if checkpoint["state_dict"]["config"] != evaluated.state_dict()["config"]:
                    raise ValueError("calibration PPO configuration changed")
                evaluated.load_state_dict(checkpoint["state_dict"])
                measured = change_norms(joint.inference_weights(evaluated), initial)
                for key, value in measured.items():
                    np.testing.assert_allclose(value, stage["parameter_change_norms"][key], atol=1e-12, rtol=0)
                measured_probe = probe_diagnostics(evaluated, probe_batch, anchor)
                for key, value in measured_probe.items():
                    np.testing.assert_allclose(value, stage["diagnostics"][key], atol=1e-10, rtol=0)
                for mode in spec.MODES:
                    expected = stage["evaluation_rows"][mode][0]
                    torch.manual_seed(spec.policy_seed(root, expected["seed"]))
                    _, observed, trace = joint.rollout(evaluated, args, "learned_history", seed=expected["seed"],
                                                       sample=False, capture=True, lower_sample=mode == "lower_sampled")
                    for key in spec.METRICS:
                        np.testing.assert_allclose(observed[key], expected[key], atol=1e-6, rtol=0)
                    compare_trace(trace, raw / f"iteration_{iteration}" / mode / f"episode_{expected['seed']}.npz")
                    charge(observed)
                    replays.append({"root": root, "method": method, "iteration": iteration, "mode": mode, "status": "passed"})
            diagnostics.append({"root": root, "method": method,
                                "snapshots": {key: value["diagnostics"] for key, value in result["snapshots"].items()},
                                "first_update": result["training"][spec.options(preflight=preflight)["warmup_iterations"]],
                                "last_update": result["training"][-1]})
            results.append(result)
            print(f"audited {root}/{method}: all snapshots, lower sampling and native probes", flush=True)
    if verification["primitive_steps"] != spec.verification_budget(preflight=preflight)["total_primitive_steps"]:
        raise ValueError("calibration verification budget changed")
    summary = {"status": "preflight_passed" if preflight else "complete", "run_name": run_name,
               "protocol": spec.EXPERIMENT_PROTOCOL, "audits": audits, "checkpoint_replays": replays,
               "probe_replays": probes, "diagnostics": diagnostics, "verification_cost": verification,
               "optimizer_steps_by_method": {method: {key: sum(r["optimizer_steps"][key] for r in results if r["method"] == method)
                                                       for key in ("actor_optimizer_steps", "value_optimizer_steps")}
                                             for method in spec.METHODS},
               "method_cost": {"primitive_steps": sum(r["budget"]["total_primitive_steps"] for r in results),
                               **{key: sum(count[key] for r in results for count in r["inference_counts"].values())
                                  for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}},
               "aggregate": aggregate(results, preflight=preflight)}
    if not preflight:
        x = np.asarray([[row["endpoints"][key] for key in spec.ENDPOINTS] for row in summary["aggregate"]["root_rows"]])
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
