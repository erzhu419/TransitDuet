#!/usr/bin/env python3
"""Matched-root linear expressivity with decision-only optimal-action targets."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_linear_floor_stage217 as source
from scripts import run_native_transit_value_expansion_stage187 as prefixes
from freq_hrl.domains.transit.native_action_value import choice_from_cost
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_best_action_stage218_v1"
SOURCE_RUN = "native_transit_linear_floor_stage217_diagnostic_20261013_r2"
PARTITIONS, state_key = source.PARTITIONS, source.state_key
PROBLEMS = ("all_orders", "same_targets_all_orders", "best_action", "wait_sign")
VIEWS = ("root_397", "root_401", "mixed_a", "mixed_b")


def contract():
    parent = source.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "original240_mean8_development_states_only_replicas0to3_six_callbacks_per_prefix",
        "matching": "each_view120_states20_prefixes_four_prefixes_per_regime_all_states_equal_weight",
        "mixed_a": "root397_replicas0and1_root401_replicas2and3_per_regime",
        "mixed_b": "root397_replicas2and3_root401_replicas0and1_per_regime",
        "features": "same98_causal_cost_features_same198_causal_wait_features_plus_bias_no_root_or_regime_id",
        "normalization": "selected120_states_candidate_mean_std_floor1e-8_before_original_centering",
        "best_target": "same_identity_first_argmin_in_both_future_halves_on_common_true_safe_menu",
        "best_constraints": "competitor_minus_best_ge1_if_competitor_precedes_best_in_tie_order_else_ge0",
        "near_ties": "exclude_state_if_a_preceding_competitor_cost_gap_le1e-12_in_either_half",
        "all_orders": "Stage217_both_half_stable_strict_cost_pairs_on_common_true_safe_menu",
        "same_targets_all_orders": "best_action_constraints_plus_all_orders_on_exact_same_eligible_states",
        "wait_sign": "unchanged_Stage217_nonidentity_both_half_safe_le0_unsafe_ge1",
        "partitions": parent["partitions"], "problems": list(PROBLEMS),
        "solver": parent["witness"],
        "budget": {"views": 4, "max_linear_programs": 32, "optimizer_updates": 0,
            "actor_updates": 0, "new_native_episodes": 0},
        "storage": "normalizers_target_rows_witnesses_remote_compact_report_only",
        "boundary": "finite_training_class_diagnosis_oracle_safe_menu_not_learned_wait_gate_or_heldout_policy_validation"}


def matched_views(rows, registered):
    keys = {state_key(r) for r in registered["mean_states"]}
    lookup = {state_key(r): r for r in rows if state_key(r) in keys}
    expected = {(root, prefixes.scene_seed(root, regime, replica), probe)
        for root in prefixes.ROOTS for regime in prefixes.SCENARIOS
        for replica in range(4) for probe in prefixes.PROBES}
    if keys != expected or set(lookup) != keys or len(lookup) != 240:
        raise ValueError("Require exactly the original240 registered development states")
    views = {name: [] for name in VIEWS}
    for root in prefixes.ROOTS:
        for regime in prefixes.SCENARIOS:
            for replica in range(4):
                mixed = "mixed_a" if (root == 397) == (replica < 2) else "mixed_b"
                for probe in prefixes.PROBES:
                    row = lookup[(root, prefixes.scene_seed(root, regime, replica), probe)]
                    if row["scenario"] != regime:
                        raise ValueError("Prefix regime differs from its registered seed")
                    views[f"root_{root}"].append(row)
                    views[mixed].append(row)
    for view in views.values():
        groups = {(r["root"], r["scene_seed"]) for r in view}
        if len(view) != 120 or len(groups) != 20:
            raise ValueError("Changed the matched sample or complete-prefix count")
    return views


def feature_design(rows):
    raw = source.wait_source.neural.batches(rows)[0]
    center, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    originals = [r["original_choice"] for r in rows]
    wait = source.wait_source.source.feasibility.design(raw, originals, center, scale)
    return wait[..., :98], wait, {"center": center.tolist(), "scale": scale.tolist()}


def best_constraints(rows, panels, design, partition):
    directions, margins, targets = [], [], []
    eligible = np.zeros(len(rows), dtype=bool)
    disagreed = near_ties = 0
    for i, row in enumerate(rows):
        a, b = [panels[state_key(row)]["values"][list(indices)].mean(axis=0)
            for indices in PARTITIONS[partition]]
        safe = (a[:, 3] <= 0) & (b[:, 3] <= 0)
        original = row["original_choice"]
        if not safe[original] or a[original, 0] != 0 or b[original, 0] != 0:
            raise ValueError("Original action is not the exact safe identity")
        best_a, best_b = [choice_from_cost(np.where(safe, values[:, 0], np.inf), original)
            for values in (a, b)]
        if best_a != best_b:
            disagreed += 1
            continue
        order = [original, *[j for j in range(6) if j != original]]
        preceding = [j for j in order[:order.index(best_a)] if safe[j]]
        if any(a[j, 0]-a[best_a, 0] <= 1e-12 or b[j, 0]-b[best_a, 0] <= 1e-12 for j in preceding):
            near_ties += 1
            continue
        eligible[i] = True
        competitors = [j for j in order if safe[j] and j != best_a]
        for j in competitors:
            directions.append(design[i, j]-design[i, best_a])
            margins.append(1. if j in preceding else 0.)
        targets.append({"root": row["root"], "scenario": row["scenario"],
            "scene_seed": row["scene_seed"], "probe_index": row["probe_index"],
            "original_choice": original, "best_choice": best_a, "safe_menu": np.flatnonzero(safe).tolist()})
    diagnostic = {"states": len(rows), "eligible_states": len(targets), "half_best_disagreement_states": disagreed,
        "preceding_near_tie_states": near_ties, "nonidentity_best_states": sum(t["best_choice"] != t["original_choice"] for t in targets),
        "eligible_prefixes": len({(t["root"], t["scene_seed"]) for t in targets}),
        "by_root": {str(root): sum(t["root"] == root for t in targets) for root in prefixes.ROOTS},
        "by_regime": {s: sum(t["scenario"] == s for t in targets) for s in prefixes.SCENARIOS},
        "strict_constraints": int(np.sum(margins)), "weak_constraints": len(margins)-int(np.sum(margins))}
    return (np.asarray(directions).reshape(-1, design.shape[-1]), np.asarray(margins)), eligible, targets, diagnostic


def diagnose_view(rows, panels, save_references):
    cost, wait, normalizer = feature_design(rows)
    partitions = {}
    for partition in PARTITIONS:
        best, eligible, targets, diagnostic = best_constraints(rows, panels, cost, partition)
        all_orders = source.stable_constraints(rows, panels, cost, wait, np.ones(len(rows), dtype=bool), partition)
        same_orders = source.stable_constraints(rows, panels, cost, wait, eligible, partition)["cost_order"]
        matched = tuple(np.concatenate((b, a), axis=0) for b, a in zip(best, same_orders))
        problems = {"all_orders": all_orders["cost_order"], "same_targets_all_orders": matched,
            "best_action": best, "wait_sign": all_orders["wait_sign"]}
        witnesses = {name: source.linear_witness(*problem) for name, problem in problems.items()}
        save_references(partition, {"normalizer": normalizer, "targets": targets, "witnesses": witnesses})
        partitions[partition] = {"target_coverage": diagnostic, "bounds": {name:
            {k: v for k, v in witness.items() if k != "coefficients"} for name, witness in witnesses.items()}}
    return {"states": len(rows), "prefixes": len({(r["root"], r["scene_seed"]) for r in rows}),
        "states_by_root": {str(root): sum(r["root"] == root for r in rows) for root in prefixes.ROOTS},
        "states_by_regime": {s: sum(r["scenario"] == s for r in rows) for s in prefixes.SCENARIOS},
        "prefix_groups": [{"root": r, "scene_seed": seed} for r, seed in sorted({(r["root"], r["scene_seed"]) for r in rows})],
        "partitions": partitions}


def analyze(bank, panels, registered, save_references):
    views = matched_views(bank["training_data"], registered)
    if set(panels) != {state_key(r) for r in registered["mean_states"]}:
        raise ValueError("Conditional panels changed the registered development states")
    results = {}
    for name, rows in views.items():
        results[name] = diagnose_view(rows, panels, lambda p, r: save_references(name, p, r))
        print(f"BEST_ACTION_VIEW {name} " + json.dumps({p: {k: b["feasible"]
            for k, b in v["bounds"].items()} for p, v in results[name]["partitions"].items()}), flush=True)
    bounds = [b for v in results.values() for p in v["partitions"].values() for b in p["bounds"].values()]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "views": results, "linear_programs": sum(b["linear_programs"] for b in bounds),
        "unresolved_bounds": sum(b["feasible"] is None for b in bounds),
        "linear_bounds_qualified": all(b["feasible"] is not None for b in bounds),
        "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != source.EXPERIMENT_PROTOCOL or prior["contract"] != source.contract() or not prior["linear_bounds_qualified"]:
        raise ValueError("Complete qualified Stage217 before matched action diagnosis")
    bank, _, registered, _ = source.load_data()
    def save_references(view, partition, references):
        write_json(raw_directory(args.output)/view/f"{partition}.json", references)
    result = analyze(bank, source.wait_source.source.diagnostic.load_panels(), registered, save_references)
    write_json(args.output, result)
    print("NATIVE_BEST_ACTION_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
