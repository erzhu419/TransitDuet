#!/usr/bin/env python3
"""Audit component freezes and replay both final and selected native policies."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments.pointmaze_update_isolation import aggregate, audit_result, change_norms
from freq_hrl.experiments.pointmaze_root_response import write_json
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_update_isolation_stage36_spec as spec


def collect_results(run_name, *, preflight, specification=spec):
    spec = specification
    directory = spec.ROOT / "results" / run_name
    results, audits, replays = [], [], []
    verification = {"primitive_steps": 0, "upper_inference_calls": 0, "lower_inference_calls": 0, "gate_inference_calls": 0}
    for root in spec.roots(preflight=preflight):
        source_cell = json.loads(spec.source_result(root, preflight=preflight).read_text())["cells"][0]
        checkpoint = torch.load(source_cell["controller_checkpoint"], map_location="cpu", weights_only=False)
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**checkpoint["state_dict"]["config"]))
        controller.load_state_dict(checkpoint["state_dict"])
        for method in spec.METHODS:
            path = directory / "cells" / method / f"replicate_{root}" / "result.json"
            result = json.loads(path.read_text())
            if (result["root"], result["method"], result["preflight"]) != (root, method, preflight):
                raise ValueError("result cell identity changed")
            raw = path.parent.with_name(path.parent.name + "_raw")
            audits.append(audit_result(result, raw_path=raw, specification=spec))
            native = spec.native_method(method)
            initial = joint.inference_weights(joint.make_model(controller, native, root=root))
            for cohort in spec.COHORTS:
                field = "final_checkpoint" if cohort == "final" else "checkpoint"
                checkpoint = torch.load(result[field], map_location="cpu", weights_only=False)
                iteration = result["options"]["iterations"] if cohort == "final" else result["selected_iteration"]
                if (checkpoint["protocol"], checkpoint["root"], checkpoint["method"], checkpoint["iteration"]) != (
                        spec.EXPERIMENT_PROTOCOL, root, method, iteration):
                    raise ValueError("evaluation checkpoint identity changed")
                model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**checkpoint["state_dict"]["config"]))
                model.load_state_dict(checkpoint["state_dict"])
                reported = result["trained_parameter_change_norms" if cohort == "final" else "selected_parameter_change_norms"]
                measured = change_norms(joint.inference_weights(model), initial)
                for name in measured:
                    np.testing.assert_allclose(measured[name], reported[name], atol=1e-12, rtol=0)
                expected = result["evaluation_rows"][cohort][0]
                _, observed, _ = joint.rollout(model, spec.source.arguments(root, preflight=preflight), native,
                                              seed=expected["seed"], sample=False)
                for key in ("episode_return", "tracking_squared_error_integral", "charged_utility"):
                    np.testing.assert_allclose(observed[key], expected[key], atol=1e-6, rtol=0)
                for key in ("decision_steps", "gate_steps", "gate_actions"):
                    np.testing.assert_array_equal(observed[key], expected[key])
                verification["primitive_steps"] += observed["episode_length"]
                for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
                    verification[key] += observed[key]
                replays.append({"root": root, "method": method, "cohort": cohort, "iteration": iteration, "status": "passed"})
            results.append(result)
            print(f"audited {root}/{method}: final and selected", flush=True)
    summary = {"status": "preflight_passed" if preflight else "complete", "run_name": run_name,
               "protocol": spec.EXPERIMENT_PROTOCOL, "audits": audits, "checkpoint_replays": replays,
               "verification_cost": verification,
               "method_cost": {"primitive_steps": sum(r["budget"]["total_primitive_steps"] for r in results),
                               **{k: sum(count[k] for r in results for count in r["inference_counts"].values())
                                  for k in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}},
               "selection": [{"root": r["root"], "method": r["method"], "iteration": r["selected_iteration"],
                              "optimizer_steps": r["optimizer_steps"]} for r in results],
               "selection_curves": {m: {k: np.mean([[row[k] for row in r["selection_history"]]
                                       for r in results if r["method"] == m], axis=0).tolist()
                                       for k in ("iteration", "utility", "return", "ise", "calls")} for m in spec.METHODS}}
    if not preflight:
        summary["aggregate"] = aggregate(results, specification=spec)
    return summary


def analyze(run_name, *, preflight, specification=spec):
    summary = collect_results(run_name, preflight=preflight, specification=specification)
    directory = specification.ROOT / "results" / run_name
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
