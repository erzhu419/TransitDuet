#!/usr/bin/env python3
"""Summarize frozen prefix-matched interventions without performance claims."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_plan_credit_stage158 as spec
from scripts.analyze_native_transit_residual_plan_stage157 import validate_credit
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    expected = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != expected:
        raise ValueError("Incomplete frozen plan-credit diagnostic cells")
    panels = []
    for (root, scenario), cell in cells.items():
        scene = spec.source.source.source_spec.scene_seeds(spec.source.LOWER_ROOT[root], scenario, preflight=False)[0]
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["scenario"] == scenario and cell["scene_seed"] == scene
                and cell["software_qualified"] and cell["training_updates"] == 0
                and cell["native_steps"] == 9 * 61380 and cell["source_reproduced"] == list(spec.CONTROLS)
                and set(cell["controls"]) == set(spec.CONTROLS)):
            raise ValueError("Changed frozen plan-credit diagnostic protocol")
        learned = cell["controls"]["learned"]
        for control in cell["controls"].values():
            check_episode(control, learned, baseline=False)
            validate_credit(control["credit"], control, 61380)
            if control["execution"]["endpoint_error_max_s"] != 0:
                raise ValueError("Frozen diagnostic changed the service-window endpoints")
        paired = spec.paired_credit(learned["trace"], cell["controls"]["forecast"]["trace"])
        if not (np.allclose(cell["credit_diagnostics"]["paired_rewards"], paired["paired_rewards"], rtol=1e-10, atol=1e-8)
                and np.isclose(cell["credit_diagnostics"]["paired_reward_sum"], paired["paired_reward_sum"], rtol=1e-10, atol=1e-8)):
            raise ValueError("Reported paired credit differs from its actual traces")
        probes = cell["probes"]
        if len(probes) != 6 or {(p["decision_index"], p["alternative"]) for p in probes} != {
                (i, a) for i in spec.PROBES for a in spec.ALTERNATIVES}:
            raise ValueError("Incomplete or duplicate single-plan interventions")
        for p in probes:
            d = learned["trace"][p["decision_index"]]
            action = ([0, 0] if p["alternative"] == "zero" else -np.asarray(d["action"]))
            check_episode(p, learned, baseline=False)
            if not (p["prefix_matched"] and p["time_s"] == d["time_s"] and p["state"] == d["state"]
                    and p["reference_action"] == d["action"] and np.array_equal(p["intervention_action"], action)
                    and p["execution"]["endpoint_error_max_s"] == 0
                    and p["credit"]["decisions"] == learned["credit"]["decisions"]
                    and np.isclose(p["physical_cost_delta"], p["credit"]["final_cost"] - learned["credit"]["final_cost"], atol=1e-12, rtol=0)
                    and np.isclose(p["physical_return_delta"], -100 * p["physical_cost_delta"], atol=1e-10, rtol=0)
                    and np.isclose(p["outcome_delta"]["service_cost_restricted"], p["physical_cost_delta"], atol=1e-6, rtol=0)
                    and np.all(np.isfinite(p["critic_twin_values"])) and np.isfinite(p["critic_margin"])):
                raise ValueError("Unmatched intervention, physical outcome, or critic diagnostic")
            validate_credit(p["credit"], {"service_cost_restricted": p["credit"]["final_cost"]}, 61380)
        active = [p for p in probes if abs(p["physical_cost_delta"]) > 1e-6]
        panels.append({"root": root, "scenario": scenario,
            "raw_reward_std": paired["raw_learned"]["std"], "paired_reward_std": paired["paired"]["std"],
            "terminal_advantage": paired["terminal_advantage"],
            "interventions": len(probes), "effects_above_1e_6_cost": len(active),
            "lower_cost_interventions": sum(p["physical_cost_delta"] < -1e-6 for p in probes),
            "critic_deployment_rank_matches": sum(p["critic_margin"] * p["physical_return_delta"] > 0 for p in active),
            "mean_abs_intervention_cost_delta": float(np.mean([abs(p["physical_cost_delta"]) for p in probes])),
            "action_abs_gt_0_95_fraction": cell["action_abs_gt_0_95_fraction"],
            "state_abs_max": float(np.max(np.abs([cell["state_min"], cell["state_max"]])))})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": sum(c["native_steps"] for c in cells.values()), "panels": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {(r, s): json.loads((directory / "cells" / s / f"seed_{r}/result.json").read_text())
             for r in spec.ROOTS for s in spec.SCENARIOS}
    summary = summarize(cells)
    write_json(directory / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "contract"}, indent=2))


if __name__ == "__main__":
    main()
