#!/usr/bin/env python3
"""Close the nine-endpoint level-update and gate-deployment diagnosis."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scripts import pointmaze_level_deployment_stage37_spec as spec
from scripts.analyze_pointmaze_update_isolation_stage36 import collect_results as analyze_training
from freq_hrl.experiments import pointmaze_gate_deployment as gate
from freq_hrl.experiments.pointmaze_root_response import write_json


def analyze(run_name, *, preflight):
    directory = spec.ROOT / "results" / run_name
    summary = analyze_training(run_name, preflight=preflight, specification=spec)
    results, audits = [], []
    for root in spec.roots(preflight=preflight):
        path = directory / "cells" / spec.GATE_TASK / f"replicate_{root}" / "result.json"
        result = json.loads(path.read_text())
        if (result["root"], result["preflight"]) != (root, preflight):
            raise ValueError("gate diagnosis cell identity changed")
        audits.append(gate.audit_result(result, raw_path=path.parent.with_name(path.parent.name + "_raw")))
        results.append(result)
        print(f"audited cached gate modes root{root}", flush=True)
    summary["gate_audits"] = audits
    for key in summary["method_cost"]:
        if key == "primitive_steps":
            summary["method_cost"][key] += sum(r["budget"]["total_primitive_steps"] for r in results)
        else:
            summary["method_cost"][key] += sum(r["inference_counts"][key] for r in results)
    for key in summary["verification_cost"]:
        summary["verification_cost"][key] += sum(a["verification_cost"][key] for a in audits)
    if not preflight:
        summary["gate_aggregate"] = gate.aggregate(results)
        combined = {**summary["aggregate"]["primary_endpoints"], **summary["gate_aggregate"]["primary_endpoints"]}
        if set(combined) != set(spec.ALL_ENDPOINTS):
            raise ValueError("nine-endpoint diagnosis incomplete")
        summary["primary_endpoints"] = combined
        # A root-count implementation independently checks the two aggregators.
        x = np.array([[r["endpoints"][k] for k in spec.ENDPOINTS] +
                      [g["endpoints"][k] for k in spec.GATE_ENDPOINTS]
                      for r, g in zip(summary["aggregate"]["root_rows"], summary["gate_aggregate"]["root_rows"])])
        rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
        indices = rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0)
        for i, key in enumerate(spec.ALL_ENDPOINTS):
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
