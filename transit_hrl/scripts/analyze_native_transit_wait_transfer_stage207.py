#!/usr/bin/env python3
"""Compare frozen feasibility models' in-sample and grouped held-out errors."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_wait_feasibility_stage206 as source
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_wait_transfer_stage207_v1"
SOURCE_RUN = "native_transit_wait_feasibility_stage206_development_20261013_r1"
VARIANTS = tuple(source.feasibility.FALSE_SAFE_COSTS)
GAP_METRICS = ("false_safe_rate_informative", "false_unsafe_rate_informative",
    "own_loss_error_rate_informative", "mass_weighted_excess_nats", "mean_expected_boundary_regret_min")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "unchanged600_states_240_mean8_targets_360_single_targets_47_folds_94_frozen_GLMS",
        "variants": list(VARIANTS), "inference": "batched_original199_feature_design_outer_train_scales_exact_archived_heldout_scores",
        "scopes": "training_all_training_mean8_training_single_held_out_mean8_and_scene_same_root_regime_training_mean8",
        "errors": "expected_wait_boundary_separate_from_lambda_mass_boundary_zero_loss_signal_separate",
        "loss": "draw_count_weighted_cross_entropy_excess_over_unrestricted_pointwise_mass_optimum",
        "halves": source.diagnostic.contract()["partitions"],
        "selection": "reuse_exact_Stage206_cost_selected_actions_heldout_harm_and_benefit_contributions",
        "counts": "state_model_evaluations_and_unique_states_prefixes_separate_no_new_independent_samples",
        "budget": {"new_fits": 0, "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "models_bank_targets_and_draws_server_only_small_summary_sync",
        "boundary": "descriptive_fitting_transfer_and_objective_diagnostic_not_CI_or_feature_ablation_proof"}


def model_scores(raw, originals, model):
    x = source.feasibility.design(raw, originals, model["center"], model["scale"])
    logits = x@np.asarray(model["coefficients"])
    shares = expit(logits)
    shares[np.arange(len(raw)), originals] = 0.
    scores = shares-.5
    scores[np.arange(len(raw)), originals] = 0.
    return logits, shares, scores


def records(rows, masses, logits, scores, false_safe_cost, panels):
    positive = np.asarray([masses[source.state_key(r)]["positive_wait_min"] for r in rows])
    negative = np.asarray([masses[source.state_key(r)]["negative_wait_min"] for r in rows])
    weights = np.asarray([masses[source.state_key(r)]["draw_count"] for r in rows])[:, None]
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    expected = positive-negative
    harm, benefit = false_safe_cost*positive, negative
    total, safe = harm+benefit, scores <= 0
    signal, true_safe, loss_safe = total > 0, expected <= 0, harm-benefit <= 0
    loss = harm*np.logaddexp(0, -logits)+benefit*np.logaddexp(0, logits)
    oracle = xlogy(total, total)-xlogy(harm, harm)-xlogy(benefit, benefit)
    target = np.divide(harm, total, out=np.zeros_like(harm), where=signal)
    counts = {"candidate_evaluations": mask, "informative_candidates": signal & mask,
        "zero_signal_candidates": ~signal & mask, "predicted_safe_candidates": safe & mask,
        "predicted_safe_informative": safe & signal & mask, "true_safe_informative": true_safe & signal & mask,
        "expected_false_safe": safe & ~true_safe & mask,
        "expected_false_unsafe_informative": ~safe & true_safe & signal & mask,
        "expected_false_unsafe_zero_signal": ~safe & ~signal & mask,
        "own_loss_false_safe": safe & ~loss_safe & signal & mask,
        "own_loss_false_unsafe": ~safe & loss_safe & signal & mask,
        "intentional_conservative_rejections": ~safe & true_safe & ~loss_safe & signal & mask}
    sums = {"weighted_loss_sum_min": (weights*loss*mask).sum(axis=1),
        "weighted_oracle_loss_sum_min": (weights*oracle*mask).sum(axis=1),
        "weighted_mass_sum_min": (weights*total*mask).sum(axis=1),
        "weighted_share_absolute_error": (weights*total*abs(expit(logits)-target)*mask).sum(axis=1),
        "expected_boundary_regret_sum_min": (abs(expected)*(safe != true_safe)*mask).sum(axis=1)}
    count_values = {k: v.sum(axis=1) for k, v in counts.items()}
    result = []
    for i, row in enumerate(rows):
        record = {**{k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
            **{k: int(v[i]) for k, v in count_values.items()}, **{k: float(v[i]) for k, v in sums.items()}}
        for partition, halves in source.diagnostic.PARTITIONS.items():
            half_counts = {"candidate_evaluations": 0, "stable_false_safe": 0, "stable_false_unsafe": 0, "sign_disagreement": 0}
            if source.state_key(row) in panels:
                values = panels[source.state_key(row)]["values"][..., 3]
                a, b = [values[list(indices)].mean(axis=0) for indices in halves]
                half_counts = {"candidate_evaluations": int(mask[i].sum()),
                    "stable_false_safe": int(np.sum(mask[i] & safe[i] & (a > 0) & (b > 0))),
                    "stable_false_unsafe": int(np.sum(mask[i] & ~safe[i] & (a <= 0) & (b <= 0))),
                    "sign_disagreement": int(np.sum(mask[i] & ((a > 0) != (b > 0))))}
            record[partition] = half_counts
        result.append(record)
    return result


def ratio(a, b):
    return a/b if b else None


def summary(rows):
    result = {"state_model_evaluations": len(rows),
        "unique_states": len({source.state_key(r) for r in rows}),
        "unique_prefix_groups": len({(r["root"], r["scene_seed"]) for r in rows})}
    if not rows: return result
    for key in rows[0]:
        if key in ("root", "scenario", "scene_seed", "probe_index", *source.diagnostic.PARTITIONS): continue
        result[key] = sum(r[key] for r in rows)
    result.update(false_safe_rate_informative=ratio(result["expected_false_safe"], result["predicted_safe_informative"]),
        false_unsafe_rate_informative=ratio(result["expected_false_unsafe_informative"], result["true_safe_informative"]),
        own_loss_error_rate_informative=ratio(result["own_loss_false_safe"]+result["own_loss_false_unsafe"], result["informative_candidates"]),
        mass_weighted_excess_nats=ratio(result["weighted_loss_sum_min"]-result["weighted_oracle_loss_sum_min"], result["weighted_mass_sum_min"]),
        mass_weighted_share_MAE=ratio(result["weighted_share_absolute_error"], result["weighted_mass_sum_min"]),
        mean_expected_boundary_regret_min=ratio(result["expected_boundary_regret_sum_min"], result["candidate_evaluations"]))
    result["halves"] = {p: {k: sum(r[p][k] for r in rows) for k in rows[0][p]} for p in source.diagnostic.PARTITIONS}
    return result


def scopes(rows):
    return {"overall": summary(rows),
        "by_root": {str(root): summary([r for r in rows if r["root"] == root]) for root in sorted({r["root"] for r in rows})},
        "by_regime": {s: summary([r for r in rows if r["scenario"] == s]) for s in sorted({r["scenario"] for r in rows})},
        "by_root_regime": {f"{root}:{s}": summary([r for r in rows if r["root"] == root and r["scenario"] == s])
            for root, s in sorted({(r["root"], r["scenario"]) for r in rows})}}


def gaps(train, held):
    return {k: held[k]-train[k] if train.get(k) is not None and held.get(k) is not None else None for k in GAP_METRICS}


def selected_summary(rows):
    return {"states": len(rows), "overrides": sum(r["selected"] != r["original_choice"] for r in rows),
        "false_safe_overrides": sum(r["selected"] != r["original_choice"] and r["wait_delta_min"] > 0 for r in rows),
        "mean_harm_min": float(np.mean([max(r["wait_delta_min"], 0) for r in rows])),
        "mean_benefit_min": float(np.mean([max(-r["wait_delta_min"], 0) for r in rows])),
        "mean_wait_min": float(np.mean([r["wait_delta_min"] for r in rows])),
        "mean_cost": float(np.mean([r["actual_cost_delta"] for r in rows]))}


def selected_scopes(rows):
    return {"overall": selected_summary(rows),
        "by_root": {str(root): selected_summary([r for r in rows if r["root"] == root]) for root in source.source.mean.ROOTS},
        "by_root_regime": {f"{root}:{s}": selected_summary([r for r in rows if r["root"] == root and r["scenario"] == s])
            for root in source.source.mean.ROOTS for s in source.source.mean.SCENARIOS}}


def analyze(bank, masses, panels, registered, load_fold):
    mean_keys = {source.state_key(r) for r in registered["mean_states"]}
    rows = bank["training_data"]
    if len(rows) != 600 or set(masses) != {source.state_key(r) for r in rows} or set(panels) != mean_keys or len(registered["folds"]) != 47:
        raise ValueError("Transfer diagnosis requires the exact600/240 state cohort and47 folds")
    raw = np.stack([source.source.wait.features(r, "conditional_wait") for r in rows])
    originals = np.asarray([r["original_choice"] for r in rows])
    pooled = {v: {s: {k: [] for k in ("training_all", "training_mean8", "training_single", "held_out_mean8", "scene_matched_training_mean8")}
        for s in source.SPLITS} for v in VARIANTS}
    selections = {v: {s: [] for s in source.SPLITS} for v in VARIANTS}
    fold_gaps = {v: [] for v in VARIANTS}
    for fold in registered["folds"]:
        cell, models = load_fold(fold)
        source.qualify(cell, fold, registered)
        held = np.asarray([source.source.credit.group_key(r, fold["split"]) == fold["held_out"] for r in rows])
        mean = np.asarray([source.state_key(r) in mean_keys for r in rows])
        for variant in VARIANTS:
            model = models[variant]
            if {k: model[k] for k in cell["fits"][variant]} != cell["fits"][variant]:
                raise ValueError("Frozen model does not match its archived fitting metadata")
            logits, shares, scores = model_scores(raw, originals, model)
            predictions = {source.state_key(r): r for r in cell["predictions"][variant]}
            for i in np.flatnonzero(held & mean):
                prediction = predictions[source.state_key(rows[i])]
                if not (np.allclose(shares[i], prediction["predicted_harm_mass_share"], atol=1e-12, rtol=0)
                        and np.allclose(scores[i], prediction["predicted_feasibility_score"], atol=1e-12, rtol=0)):
                    raise ValueError("Batched diagnostic changed a frozen held-out prediction")
                scores[i] = prediction["predicted_feasibility_score"]
            records_all = records(rows, masses, logits, scores, model["false_safe_cost"], panels)
            masks = {"training_all": ~held, "training_mean8": ~held & mean, "training_single": ~held & ~mean, "held_out_mean8": held & mean}
            if fold["split"] == "scene":
                exemplar = rows[int(np.flatnonzero(held & mean)[0])]
                masks["scene_matched_training_mean8"] = ~held & mean & np.asarray([
                    (r["root"], r["scenario"]) == (exemplar["root"], exemplar["scenario"]) for r in rows])
            groups = {k: [records_all[i] for i in np.flatnonzero(mask)] for k, mask in masks.items()}
            for group, records_group in groups.items(): pooled[variant][fold["split"]][group].extend(records_group)
            reference = "scene_matched_training_mean8" if fold["split"] == "scene" else "training_mean8"
            train_summary, held_summary = summary(groups[reference]), summary(groups["held_out_mean8"])
            fold_gaps[variant].append({"fold": {k: fold[k] for k in ("split", "index", "held_out")}, "training_reference": reference,
                "training": {k: train_summary[k] for k in GAP_METRICS}, "held_out": {k: held_summary[k] for k in GAP_METRICS},
                "gap": gaps(train_summary, held_summary)})
            selections[variant][fold["split"]].extend(predictions.values())
    variants = {}
    for variant in VARIANTS:
        validation = {}
        for split in source.SPLITS:
            groups = pooled[variant][split]
            held = groups["held_out_mean8"]
            if len(held) != 240 or {source.state_key(r) for r in held} != mean_keys:
                raise ValueError("Held-out diagnostic duplicated or lost a mean8 state")
            validation[split] = {k: scopes(rs) if k in ("training_mean8", "held_out_mean8", "scene_matched_training_mean8") else {"overall": summary(rs)}
                for k, rs in groups.items() if rs}
            reference = "scene_matched_training_mean8" if split == "scene" else "training_mean8"
            validation[split]["held_minus_training_gap"] = gaps(summary(groups[reference]), summary(held))
            validation[split]["held_out_selection"] = selected_scopes(selections[variant][split])
        variants[variant] = {"validation": validation, "fold_gaps": fold_gaps[variant]}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "reused_models": 94, "evaluation_states": 240, "evaluation_prefix_groups": 40, **contract()["budget"],
        "variants": variants, "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    prior = json.loads((directory/"report/result.json").read_text())
    if not (prior["protocol"] == source.EXPERIMENT_PROTOCOL and prior["contract"] == source.contract()
            and prior["software_qualified"] and prior["new_wait_fits"] == 94):
        raise ValueError("Complete qualified Stage206 before diagnosing frozen-model transfer")
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    bank = json.loads(source.source.coverage.bank_path(source.source.mean.SOURCE_RUN).read_text())
    prepared = json.loads(source.mass_path(SOURCE_RUN).read_text())
    if prepared["contract"] != source.contract(): raise ValueError("Changed frozen draw-mass targets")
    def load_fold(fold):
        directory = raw_directory(source.result_path(SOURCE_RUN, fold))
        return json.loads((directory/"result.json").read_text()), {v: json.loads((directory/f"{v}.json").read_text()) for v in VARIANTS}
    result = analyze(bank, {source.state_key(r): r for r in prepared["targets"]}, source.diagnostic.load_panels(), registered, load_fold)
    write_json(args.output, result)
    print(json.dumps({v: {s: d["held_minus_training_gap"] for s, d in x["validation"].items()} for v, x in result["variants"].items()}, indent=2), flush=True)
    print("NATIVE_WAIT_TRANSFER_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
