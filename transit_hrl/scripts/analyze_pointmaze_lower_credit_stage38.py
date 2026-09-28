#!/usr/bin/env python3
"""Audit final/selected policies and replay the initial native training credit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments.pointmaze_lower_credit import audit_credit_batch, audit_training_credit
from freq_hrl.experiments.pointmaze_root_response import write_json
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts.analyze_pointmaze_update_isolation_stage36 import collect_results
from scripts import pointmaze_lower_credit_stage38_spec as spec


def analyze(run_name, *, preflight):
    directory = spec.ROOT / "results" / run_name
    summary = collect_results(run_name, preflight=preflight, specification=spec)
    audits, diagnostics = [], []
    for root in spec.roots(preflight=preflight):
        cell = json.loads(spec.source_result(root, preflight=preflight).read_text())["cells"][0]
        checkpoint = torch.load(cell["controller_checkpoint"], map_location="cpu", weights_only=False)
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**checkpoint["state_dict"]["config"]))
        controller.load_state_dict(checkpoint["state_dict"])
        args = spec.source.arguments(root, preflight=preflight)
        for method in spec.METHODS:
            result = json.loads((directory / "cells" / method / f"replicate_{root}" / "result.json").read_text())
            audit_training_credit(result, specification=spec)
            mode = spec.LOWER_CREDIT[method]
            model = joint.make_model(controller, spec.native_method(method), root=root)
            seed = spec.seed_roles(root, preflight=preflight)["training"][0]
            torch.manual_seed(seed + root)
            batch, row, raw = joint.rollout(model, args, spec.native_method(method), seed=seed,
                                           sample=True, capture=True, lower_credit=mode)
            audit_credit_batch(batch, row, raw, args=args, mode=mode)
            observed = {"seed": seed, **row["lower_training_credit"]}
            if observed != result["initial_credit_probe"]:
                raise ValueError("native initial training credit differs from the actual worker batch")
            for key in summary["verification_cost"]:
                summary["verification_cost"][key] += row["episode_length"] if key == "primitive_steps" else row[key]
            audits.append({"root": root, "method": method, "mode": mode, "status": "passed"})
            diagnostics.append({"root": root, "method": method, "first": result["training_credit"][0],
                                "last": result["training_credit"][-1]})
            print(f"audited initial native credit {root}/{method}", flush=True)
    summary["credit_audits"], summary["credit_diagnostics"] = audits, diagnostics
    if summary["verification_cost"]["primitive_steps"] != spec.verification_budget(preflight=preflight)["total_primitive_steps"]:
        raise ValueError("Stage-38 verification budget changed")
    if not preflight:
        combined = summary["aggregate"]["primary_endpoints"]
        x = np.asarray([[r["endpoints"][k] for k in spec.ENDPOINTS] for r in summary["aggregate"]["root_rows"]])
        rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
        indices = rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0)
        for i, key in enumerate(spec.ENDPOINTS):
            np.testing.assert_allclose(bounds[:, i], combined[key]["ci"], atol=1e-10, rtol=0)
        summary["independent_statistics"] = {"status": "passed", "endpoints": len(combined),
                                             "method": "paired_root_count_weight_bootstrap"}
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
