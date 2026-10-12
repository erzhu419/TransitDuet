#!/usr/bin/env python3
"""Separate cached cost-ranking errors from wait-feasibility errors."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_coverage_value_stage188 as source
from scripts.submit_native_transit_coverage_value_stage188_scheduleurm import result_path
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_value_error_stage189_v1"
SOURCE_RUN = "native_transit_coverage_value_stage188_fit_20261012_r1"
ARMS = ("predicted", "oracle_wait", "oracle_cost", "oracle_both")


def diagnose(prediction, row):
    labels = row["labels"]
    cost = np.asarray([l["actual_cost_delta"] for l in labels])
    wait = np.asarray([l["wait_delta_min"] for l in labels])
    estimated_cost = np.asarray(prediction["predicted_heads"]).sum(axis=1)
    estimated_wait = np.asarray(prediction["predicted_wait_delta_min"])
    original = row["original_choice"]
    arms = {}
    for arm in ARMS:
        c = cost if arm in ("oracle_cost", "oracle_both") else estimated_cost
        w = wait if arm in ("oracle_wait", "oracle_both") else estimated_wait
        chosen = source.paired.choice_from_cost(np.where(w <= 0, c, np.inf), original)
        arms[arm] = {"selected": chosen, "cost_delta": float(cost[chosen]), "wait_delta_min": float(wait[chosen])}
    selected = arms["predicted"]["selected"]
    if selected != prediction["selected"]:
        raise ValueError("Cached predicted arm differs from its fixed wait-feasible selector")
    return {"root": row["root"], "scene_seed": row["scene_seed"], "probe_index": row["probe_index"], "arms": arms,
        "all_candidate_cost_bias": float(np.mean(estimated_cost-cost)),
        "selected_cost_bias": float(estimated_cost[selected]-cost[selected]),
        "all_candidate_wait_bias_min": float(np.mean(estimated_wait-wait)),
        "selected_wait_bias_min": float(estimated_wait[selected]-wait[selected]),
        "all_candidate_cost_mse": float(np.mean((estimated_cost-cost)**2)),
        "all_candidate_wait_mse": float(np.mean((estimated_wait-wait)**2))}


def summarize(rows):
    return {"states": len(rows), "arms": {arm: {
        "mean_cost_delta": float(np.mean([r["arms"][arm]["cost_delta"] for r in rows])),
        "mean_wait_delta_min": float(np.mean([r["arms"][arm]["wait_delta_min"] for r in rows])),
        "positive_wait_choices": sum(r["arms"][arm]["wait_delta_min"] > 0 for r in rows)} for arm in ARMS},
        **{k: float(np.mean([r[k] for r in rows])) for k in ("all_candidate_cost_bias", "selected_cost_bias",
            "all_candidate_wait_bias_min", "selected_wait_bias_min", "all_candidate_cost_mse", "all_candidate_wait_mse")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    validated = json.loads((directory/"report/result.json").read_text())
    if not (validated["protocol"] == source.EXPERIMENT_PROTOCOL and validated["contract"] == source.contract()
            and validated["software_qualified"] and validated["fits"] == 216):
        raise ValueError("Error decomposition requires completed matched coverage validation")
    bank = json.loads(source.bank_path(SOURCE_RUN).read_text())
    rows = {(r["root"], r["scene_seed"], r["probe_index"]): r for r in bank["training_data"]}
    result = {}
    for mode in source.MODES:
        result[mode] = {}
        for split in source.SPLITS:
            diagnosed = []
            for fold in bank["manifest"]["folds"]:
                if fold["split"] != split: continue
                cell = json.loads(result_path(directory, mode, fold).read_text())
                for p in cell["predictions"]["wait_feasible"]:
                    diagnosed.append(diagnose(p, rows[p["root"], p["scene_seed"], p["probe_index"]]))
            if len(diagnosed) != 600 or len({(r["root"], r["scene_seed"], r["probe_index"]) for r in diagnosed}) != 600:
                raise ValueError("Error decomposition lost a held-out state")
            result[mode][split] = {"overall": summarize(diagnosed),
                "by_root": {str(root): summarize([r for r in diagnosed if r["root"] == root]) for root in source.collection.ROOTS}}
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "source_run": SOURCE_RUN,
        "source_protocol": source.EXPERIMENT_PROTOCOL, "software_qualified": True, "diagnostics": result,
        "critic_updates": 0, "actor_updates": 0, "new_native_steps": 0,
        "role": "hindsight_mask_decomposition_not_deployable_oracle_or_new_performance_validation"})
    print("NATIVE_VALUE_ERROR_DECOMPOSITION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
