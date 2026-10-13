#!/usr/bin/env python3
"""Separate bounded-mask reachability, wait errors and frozen cost ranking."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mean_wait_stage213 as source
from scripts import analyze_native_transit_oracle_factorial_stage205 as factorial
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_bound_ranking_stage214_v1"
SOURCE_RUN = "native_transit_mean_wait_stage213_development_20261013_r1"
SELECTORS = ("LL", "NN", "LO", "OL", "OO", "bound_best", "free_best")
PARTITIONS = factorial.PARTITIONS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "states": 240, "prefix_groups": 40, "folds": 47, "bound": source.bounded.BOUND,
        "partitions": {p: [list(a), list(b)] for p, (a, b) in PARTITIONS.items()},
        "reachability": "enumerate_all_flexible_nonidentity_masks_preserve_forced_safe_unsafe_original_ties_frozen_cost_order",
        "oracle": "minimize_selection_half_wait_harm_then_actual_cost_then_source_tie_order_within_reachable_actions",
        "selectors": {"LL": "mean8_GLM", "NN": "mean8_shrunk", "LO": "learned_cost_oracle_wait",
            "OL": "oracle_cost_learned_wait", "OO": "oracle_cost_oracle_wait",
            "bound_best": "best_reachable_choice_fixed_bound_and_frozen_cost_order",
            "free_best": "best_reachable_choice_unbounded_mask_and_frozen_cost_order"},
        "evaluation": "A_to_B_and_B_to_A_other_half_only_hindsight_mean8_separate_no_test_half_selection",
        "errors": "stable_wrong_wait_signs_locked_by_bound_stable_feasible_cost_inversions_and_unreachable_joint_opportunities",
        "budget": {"new_fits": 0, "critic_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "existing_bank_models_draw_panels_remote_small_report_only",
        "boundary": "optimistic_independent_action_mask_envelope_not_shared_network_realizability_independent_CI_or_deployable_oracle"}


def reachable(cost, logits, original, bound):
    cost, logits = np.asarray(cost), np.asarray(logits)
    if bound is None:
        forced_safe, forced_unsafe = np.zeros(6, dtype=bool), np.zeros(6, dtype=bool)
    else:
        forced_safe, forced_unsafe = logits+bound <= 0, logits-bound > 0
    forced_safe[original], forced_unsafe[original] = True, False
    flexible = np.flatnonzero(~(forced_safe | forced_unsafe))
    actions = set()
    for bits in product((False, True), repeat=len(flexible)):
        mask = forced_safe.copy()
        mask[flexible] = bits
        actions.add(source.source.source.paired.choice_from_cost(np.where(mask, cost, np.inf), original))
    return sorted(actions), forced_safe, forced_unsafe


def oracle_choice(actions, truth, original):
    order = [original, *[i for i in range(6) if i != original]]
    return min((i for i in order if i in actions), key=lambda i: (max(truth[i, 1], 0.), truth[i, 0]))


def outcome(selected, hindsight, halves, mean, original):
    own = [halves[i][selected[i]] for i in range(2)]
    cross = [halves[1-i][selected[i]] for i in range(2)]
    return {"selected": selected, "selection_cost": float(np.mean([v[0] for v in own])),
        "selection_wait_min": float(np.mean([v[1] for v in own])),
        "cross_cost": float(np.mean([v[0] for v in cross])), "cross_wait_min": float(np.mean([v[1] for v in cross])),
        "direction_cost": [float(v[0]) for v in cross], "direction_wait_min": [float(v[1]) for v in cross],
        "half_agreement": int(selected[0] == selected[1]), "directional_changes": sum(i != original for i in selected),
        "hindsight_cost": float(mean[hindsight, 0]), "hindsight_wait_min": float(mean[hindsight, 1])}


def panel(values, cost, scores, logits, original, neural_choice, partition):
    result = factorial.panel(values, cost, scores, original, partition)
    result["selectors"] = {k: result["selectors"][k] for k in ("LL", "LO", "OL", "OO")}
    truth = values[..., [0, 3]]
    halves = [truth[list(indices)].mean(axis=0) for indices in PARTITIONS[partition]]
    mean = truth.mean(axis=0)
    actions, forced_safe, forced_unsafe = reachable(cost, logits, original, source.bounded.BOUND)
    free, _, _ = reachable(cost, logits, original, None)
    for name, candidates in (("bound_best", actions), ("free_best", free)):
        choices = [oracle_choice(candidates, h, original) for h in halves]
        result["selectors"][name] = outcome(choices, oracle_choice(candidates, mean, original), halves, mean, original)
    result["selectors"]["NN"] = outcome([neural_choice]*2, neural_choice, halves, mean, original)
    a, b = halves
    nonidentity = np.arange(6) != original
    stable_safe, stable_unsafe = (a[:, 1] <= 0) & (b[:, 1] <= 0), (a[:, 1] > 0) & (b[:, 1] > 0)
    false_safe, false_unsafe = nonidentity & (scores <= 0) & stable_unsafe, nonidentity & (scores > 0) & stable_safe
    joint = (a[:, 0] < 0) & (b[:, 0] < 0) & stable_safe
    result["errors"].update(locked_false_safe_candidates=int(np.sum(false_safe & forced_safe)),
        locked_false_unsafe_candidates=int(np.sum(false_unsafe & forced_unsafe)),
        repairable_false_safe_candidates=int(np.sum(false_safe & ~forced_safe)),
        repairable_false_unsafe_candidates=int(np.sum(false_unsafe & ~forced_unsafe)),
        neural_selected_both_halves_wait_harm=int(stable_unsafe[neural_choice]),
        neural_selected_locked_wait_harm=int(stable_unsafe[neural_choice] and forced_safe[neural_choice]))
    result["opportunities"].update(mean_reachable_actions=len(actions), mean_free_reachable_actions=len(free),
        no_safe_reachable_A=int(not np.any(a[actions, 1] <= 0)), no_safe_reachable_B=int(not np.any(b[actions, 1] <= 0)),
        stable_joint_reachable_states=int(np.any(joint[actions])),
        stable_joint_blocked_by_bound_states=int(np.any(joint[free]) and not np.any(joint[actions])),
        stable_joint_blocked_by_cost_states=int(np.any(joint) and not np.any(joint[free])))
    return result


def summarize(rows):
    result = {"states": len(rows), "prefix_groups": len({(r["root"], r["scene_seed"]) for r in rows}),
        "directional_evaluations": 2*len(rows), "selectors": {}}
    metrics = ("selection_cost", "selection_wait_min", "cross_cost", "cross_wait_min", "hindsight_cost", "hindsight_wait_min")
    for selector in SELECTORS:
        values = [r["selectors"][selector] for r in rows]
        result["selectors"][selector] = {**{"mean_"+m: float(np.mean([v[m] for v in values])) for m in metrics},
            **{m: sum(v[m] for v in values) for m in ("half_agreement", "directional_changes")},
            "directions": {name: {"mean_cost": float(np.mean([v["direction_cost"][i] for v in values])),
                "mean_wait_min": float(np.mean([v["direction_wait_min"][i] for v in values]))}
                for i, name in enumerate(("A_to_B", "B_to_A"))}}
    for category in ("errors", "opportunities"):
        result[category] = {k: sum(r[category][k] for r in rows) for k in rows[0][category]}
    for k in ("mean_reachable_actions", "mean_free_reachable_actions"):
        result["opportunities"][k] /= len(rows)
    result["contrasts"] = {name: {m: result["selectors"][a]["mean_"+m]-result["selectors"][b]["mean_"+m]
        for m in ("cross_cost", "cross_wait_min")} for name, (a, b) in
        {"wait_repair": ("LO", "LL"), "cost_repair": ("OL", "LL"), "joint_repair": ("OO", "LL"),
         "bound_envelope_gain": ("bound_best", "LL"), "free_minus_bound": ("free_best", "bound_best"),
         "cost_order_gap": ("OO", "free_best")}.items()}
    return result


def analyze(bank, panels, registered, load_fold):
    keys = {source.state_key(r) for r in registered["mean_states"]}
    rows = [r for r in bank["training_data"] if source.state_key(r) in keys]
    if len(rows) != 240 or set(panels) != keys or len(registered["folds"]) != 47:
        raise ValueError("Require the original240 states and47 outer folds")
    batch = source.neural.batches(rows)
    indexed = {source.state_key(r): i for i, r in enumerate(rows)}
    pooled = {p: {s: [] for s in source.SPLITS} for p in PARTITIONS}
    for fold in registered["folds"]:
        cell, model = load_fold(fold)
        source.qualify(cell, fold, registered)
        if source.stopping.metadata(model) != cell["fits"][source.NEW_GLM]:
            raise ValueError("Changed the fitted mean8 GLM reference")
        logits, _, scores = source.transfer.model_scores(batch[0], [r["original_choice"] for r in rows], model)
        neural = {source.state_key(r): r for r in cell["predictions"][source.PRIMARY]}
        for prediction in cell["predictions"][source.NEW_GLM]:
            key, original = source.state_key(prediction), prediction["original_choice"]
            p, i = panels[key], indexed[key]
            if p["original_choice"] != original or np.any(p["values"][:, original] != 0):
                raise ValueError("Changed paired original identity outcomes")
            archived = np.asarray(prediction["predicted_feasibility_score"])
            if not np.allclose(scores[i], archived, atol=1e-12, rtol=0):
                raise ValueError("GLM inference differs from the archived score")
            cost = np.asarray(prediction["predicted_heads"]).sum(axis=1)
            selected = source.source.choose(prediction, original)
            if selected != prediction["selected"] or neural[key]["predicted_heads"] != prediction["predicted_heads"]:
                raise ValueError("Changed frozen cost order or learned choice")
            for partition in PARTITIONS:
                record = panel(p["values"], cost, archived, logits[i], original, neural[key]["selected"], partition)
                pooled[partition][fold["split"]].append({**{k: prediction[k] for k in ("root", "scenario", "scene_seed", "probe_index")}, **record})
    partitions = {}
    for p, splits in pooled.items():
        validation = {}
        for split, records in splits.items():
            if len(records) != 240 or {source.state_key(r) for r in records} != keys:
                raise ValueError("Lost or repeated an evaluation state")
            validation[split] = {"overall": summarize(records),
                "by_root": {str(r): summarize([x for x in records if x["root"] == r]) for r in source.collection.ROOTS},
                "by_regime": {s: summarize([x for x in records if x["scenario"] == s]) for s in source.collection.SCENARIOS}}
        partitions[p] = {"validation": validation, "gates": {selector: {
            "cost": all(m["selectors"][selector]["mean_cross_cost"] < 0 for s in validation.values() for m in s["by_root"].values()),
            "wait": all(m["selectors"][selector]["mean_cross_wait_min"] <= 0 for s in validation.values() for m in s["by_root"].values()),
            "each_direction_cost": all(d["mean_cost"] < 0 for s in validation.values() for m in s["by_root"].values()
                for d in m["selectors"][selector]["directions"].values()),
            "each_direction_wait": all(d["mean_wait_min"] <= 0 for s in validation.values() for m in s["by_root"].values()
                for d in m["selectors"][selector]["directions"].values())} for selector in SELECTORS}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        **contract()["budget"], "reused_glms": 47, "partitions": partitions,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    prior = json.loads((directory/"report/result.json").read_text())
    if not (prior["protocol"] == source.EXPERIMENT_PROTOCOL and prior["contract"] == source.contract()
            and prior["software_qualified"] and prior["controls_reproduced"]):
        raise ValueError("Complete Stage213 before reachability diagnosis")
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    bank = json.loads(source.collection.mean_source.source.bank_path(source.collection.mean_source.SOURCE_RUN).read_text())
    def load_fold(fold):
        d = raw_directory(source.result_path(SOURCE_RUN, fold))
        return json.loads((d/"result.json").read_text()), json.loads((d/f"{source.NEW_GLM}.json").read_text())
    result = analyze(bank, source.source.diagnostic.load_panels(), registered, load_fold)
    for partition in PARTITIONS:
        for split in source.SPLITS:
            full = result["partitions"][partition]["validation"][split]["overall"]["selectors"]
            for selector, variant in (("LL", source.NEW_GLM), ("NN", source.PRIMARY)):
                old = prior["variants"][variant]["validation"][split]["overall"]
                if not np.allclose([full[selector]["mean_hindsight_cost"], full[selector]["mean_hindsight_wait_min"]],
                        [old["mean_actual_cost_delta"], old["mean_wait_delta_min"]], atol=1e-12, rtol=0):
                    raise ValueError("Diagnostic changed the original learned control outcomes")
    write_json(args.output, result)
    print(json.dumps({p: r["gates"] for p, r in result["partitions"].items()}), flush=True)
    print("NATIVE_BOUND_RANKING_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
