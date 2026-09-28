#!/usr/bin/env python3
"""Audit server-only trajectories and independently replay selected policies."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from freq_hrl.experiments.pointmaze_joint_renewal import aggregate, audit_result, rollout
from freq_hrl.experiments.pointmaze_root_response import write_json
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_joint_renewal_stage35_spec as spec


def analyze(run_name, *, preflight):
    directory = spec.ROOT / "results" / run_name
    results, audits = [], []
    replay_counts = {"primitive_steps": 0, "upper_inference_calls": 0, "lower_inference_calls": 0, "gate_inference_calls": 0}
    for root in spec.roots(preflight=preflight):
        for method in spec.METHODS:
            path = directory / "cells" / method / f"replicate_{root}" / "result.json"
            result = json.loads(path.read_text())
            if (result["root"], result["method"], result["preflight"]) != (root, method, preflight):
                raise ValueError("result cell identity changed")
            raw = path.parent.with_name(path.parent.name + "_raw")
            audits.append(audit_result(result, raw_path=raw))
            checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
            if (checkpoint["protocol"], checkpoint["root"], checkpoint["method"], checkpoint["iteration"]) != (spec.EXPERIMENT_PROTOCOL, root, method, result["selected_iteration"]):
                raise ValueError("selected checkpoint identity changed")
            model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**checkpoint["state_dict"]["config"]))
            model.load_state_dict(checkpoint["state_dict"])
            expected = result["evaluation_rows"][0]
            _, observed, _ = rollout(model, spec.source.arguments(root, preflight=preflight), method,
                                     seed=expected["seed"], sample=False)
            for key in ("episode_return", "tracking_squared_error_integral", "charged_utility"):
                np.testing.assert_allclose(observed[key], expected[key], atol=1e-6, rtol=0)
            for key in ("decision_steps", "gate_steps", "gate_actions"):
                np.testing.assert_array_equal(observed[key], expected[key])
            replay_counts["primitive_steps"] += observed["episode_length"]
            for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
                replay_counts[key] += observed[key]
            results.append(result)
            print(f"audited {root}/{method}", flush=True)
    summary = {"status": "preflight_passed" if preflight else "complete", "run_name": run_name,
               "protocol": spec.EXPERIMENT_PROTOCOL, "audits": audits, "verification_cost": replay_counts,
               "method_cost": {"primitive_steps": sum(result["budget"]["total_primitive_steps"] for result in results),
                               **{key: sum(count[key] for result in results for count in result["inference_counts"].values())
                                  for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}},
               "selection": [{"root": r["root"], "method": r["method"], "iteration": r["selected_iteration"],
                              "trained_parameter_change_norms": r["trained_parameter_change_norms"],
                              "optimizer_steps": r["optimizer_steps"]} for r in results]}
    if not preflight:
        summary["aggregate"] = aggregate(results)
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
