#!/usr/bin/env python3
"""Separate frozen training fit, ordinary generalization and domain transfer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mean_cost_stage215 as source
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_fit_transfer_stage216_v1"
SOURCE_RUN = "native_transit_mean_cost_stage215_development_20261013_r1"
VARIANTS, SPLITS, state_key = source.VARIANTS, source.SPLITS, source.state_key
wait_source, transfer = source.previous, source.previous.transfer
PARTITIONS = source.diagnosis.PARTITIONS
GAP_METRICS = ("cost_pair_mse", "cost_head_pair_normalized_mse", "mass_weighted_excess_nats",
    "mean_expected_boundary_regret_min", "first_last_cost_inversion_rate", "even_odd_cost_inversion_rate",
    "first_last_stable_false_safe_rate", "even_odd_stable_false_safe_rate")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "same600_mean8_states_original240_future_panels_40_prefixes_47_outer_folds",
        "variants": list(VARIANTS), "primary": source.PRIMARY,
        "inference": "same_cost_coefficients_GLM_and_selected_shrunk_wait_models_training_scales_masks_identity_ties",
        "scopes": "training_all600_training_original240_scene_root_regime_matched_reference_held_original240",
        "weights": wait_source.collection.contract()["weights"],
        "comparison": "same240_held_states_scene_vs_regime_vs_root_plus_within_fold_train_held_gaps",
        "errors": "physical_and_normalized_cost_pair_MSE_wait_mass_excess_stable_cost_order_and_false_safe_signs",
        "partitions": source.diagnosis.factorial.contract()["partitions"],
        "feature_support": "causal_cost_pair_wait_pair_and_absolute_wait_context_outside_training_coordinate_ranges",
        "counts": "model_state_evaluations_and_unique_states_prefixes_separate",
        "budget": {"new_fits": 0, "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0},
        "storage": "models_targets_bank_and_future_panels_server_only_small_summary_sync",
        "boundary": "descriptive_current_regularized_model_fit_and_transfer_not_feature_impossibility_or_independent_CI"}


def masks(rows, fold, mean_keys):
    held = np.asarray([source.base.credit.group_key(r, fold["split"]) == fold["held_out"] for r in rows])
    mean = np.asarray([state_key(r) in mean_keys for r in rows])
    reference = ~held & mean
    if fold["split"] == "scene":
        exemplar = rows[int(np.flatnonzero(held & mean)[0])]
        reference &= np.asarray([(r["root"], r["scenario"]) == (exemplar["root"], exemplar["scenario"]) for r in rows])
    return {"training_all": ~held, "training_mean8": ~held & mean,
        "training_reference": reference, "held": held & mean}


def inference(rows, batch, cost_model, wait_model, wait_name):
    heads = source.base.cost_predictions(batch[0], rows, cost_model)
    if wait_name == "glm":
        logits, shares, scores = transfer.model_scores(batch[0], [r["original_choice"] for r in rows], wait_model)
    else:
        logits = wait_source.bounded.logits_many(rows, wait_model, batch=batch)
        shares = expit(logits)
        shares[np.arange(len(rows)), [r["original_choice"] for r in rows]] = 0.
        scores = shares-.5
        scores[np.arange(len(rows)), [r["original_choice"] for r in rows]] = 0.
    return heads, logits, shares, scores


def records(rows, masses, panels, heads, logits, scores, cost_scale):
    result = transfer.records(rows, masses, logits, scores, 1., panels)
    weights = wait_source.source.feasibility.sample_weights(rows, masses)
    left, right = np.triu_indices(6, 1)
    for i, (row, record) in enumerate(zip(rows, result, strict=True)):
        labels = {l["candidate"]: l for l in row["labels"]}
        actual = np.asarray([[labels[j][h] for h in ("fleet_cost_delta", "nonfleet_cost_delta")] for j in range(6)])
        residual = (heads[i, left]-heads[i, right])-(actual[left]-actual[right])
        cost = heads[i].sum(axis=1)
        selected = source.base.paired.choice_from_cost(np.where(scores[i] <= 0, cost, np.inf), row["original_choice"])
        record.update(sample_weight_sum=float(weights[i]),
            weighted_cost_pair_sse=float(weights[i]*np.mean(residual.sum(axis=1)**2)),
            weighted_cost_head_pair_sse=float(weights[i]*np.mean((residual/np.asarray(cost_scale[:2]))**2)),
            selected_cost_sum=labels[selected]["actual_cost_delta"], selected_wait_sum_min=labels[selected]["wait_delta_min"])
        for partition, halves in PARTITIONS.items():
            counts = record[partition]
            counts.update(stable_safe_candidates=0, stable_unsafe_candidates=0,
                stable_feasible_cost_pairs=0, stable_cost_inversions=0, stable_cost_prediction_ties=0)
            if state_key(row) not in panels: continue
            values = panels[state_key(row)]["values"]
            a, b = [values[list(indices), :, 3].mean(axis=0) for indices in halves]
            nonidentity = np.arange(6) != row["original_choice"]
            counts["stable_safe_candidates"] = int(np.sum(nonidentity & (a <= 0) & (b <= 0)))
            counts["stable_unsafe_candidates"] = int(np.sum(nonidentity & (a > 0) & (b > 0)))
            e = source.diagnosis.factorial.panel(values, cost, scores[i], row["original_choice"], partition)["errors"]
            for key in ("stable_feasible_cost_pairs", "stable_cost_inversions", "stable_cost_prediction_ties"):
                counts[key] = e[key]
    return result


def summary(rows):
    result = transfer.summary(rows)
    if not rows: return result
    result.update(cost_pair_mse=transfer.ratio(result["weighted_cost_pair_sse"], result["sample_weight_sum"]),
        cost_head_pair_normalized_mse=transfer.ratio(result["weighted_cost_head_pair_sse"], result["sample_weight_sum"]),
        mean_selected_cost=result["selected_cost_sum"]/len(rows), mean_selected_wait_min=result["selected_wait_sum_min"]/len(rows))
    for partition, counts in result["halves"].items():
        result[partition+"_cost_inversion_rate"] = transfer.ratio(counts["stable_cost_inversions"], counts["stable_feasible_cost_pairs"])
        result[partition+"_stable_false_safe_rate"] = transfer.ratio(counts["stable_false_safe"], counts["stable_unsafe_candidates"])
    return result


def scopes(rows):
    return {"overall": summary(rows), "by_root": {str(r): summary([x for x in rows if x["root"] == r])
        for r in wait_source.collection.ROOTS}}


def gaps(training, held):
    return {k: held[k]-training[k] if training.get(k) is not None and held.get(k) is not None else None for k in GAP_METRICS}


def feature_support(rows, batch, mask):
    originals = np.asarray([r["original_choice"] for r in rows])
    pair = batch[0]-batch[0][np.arange(len(rows)), originals][:, None]
    nonidentity = np.arange(6)[None] != originals[:, None]
    support = [{} for _ in rows]
    for name, values in (("cost_pair", pair[..., :98]), ("wait_pair", pair), ("wait_context", batch[1])):
        training = values[mask] if name == "wait_context" else values[mask][nonidentity[mask]]
        outside = (values < training.min(axis=0)-1e-12) | (values > training.max(axis=0)+1e-12)
        for i in range(len(rows)):
            evaluated = outside[i] if name == "wait_context" else outside[i][nonidentity[i]]
            support[i][name] = float(evaluated.mean())
    return [{**{k: r[k] for k in ("root", "scenario", "scene_seed", "probe_index")}, **s} for r, s in zip(rows, support)]


def support_summary(rows):
    return {"states": len(rows), "fraction_outside": {name: float(np.mean([r[name] for r in rows]))
        for name in ("cost_pair", "wait_pair", "wait_context")}} if rows else {"states": 0}


def analyze(bank, masses, panels, registered, load_fold):
    rows = bank["training_data"]
    keys = {state_key(r) for r in registered["mean_states"]}
    if len(rows) != 600 or set(panels) != keys or len(keys) != 240 or set(masses) != {state_key(r) for r in rows} or len(registered["folds"]) != 47:
        raise ValueError("Require all600 mean8 states, original240 panels and47 frozen folds")
    batch = wait_source.neural.batches(rows)
    pooled = {v: {s: {scope: [] for scope in ("training_all", "training_mean8", "training_reference", "held")}
        for s in SPLITS} for v in VARIANTS}
    fold_gaps, directions = {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
    support = {s: [] for s in SPLITS}
    for fold in registered["folds"]:
        cell, cost_models, wait_models = load_fold(fold)
        split, groups = fold["split"], masks(rows, fold, keys)
        coverage = feature_support(rows, batch, groups["training_all"])
        support[split].extend(coverage[i] for i in np.flatnonzero(groups["held"]))
        for price, cost_model in cost_models.items():
            for name, wait_model in wait_models.items():
                variant = f"{price}_{name}"
                heads, logits, shares, scores = inference(rows, batch, cost_model, wait_model, name)
                archived = {state_key(r): r for r in cell["predictions"][variant]}
                for i in np.flatnonzero(groups["held"]):
                    old = archived[state_key(rows[i])]
                    if not (np.allclose(heads[i], old["predicted_heads"], atol=1e-12, rtol=0)
                            and np.allclose(scores[i], old["predicted_feasibility_score"], atol=1e-12, rtol=0)
                            and np.allclose(shares[i], old["predicted_harm_mass_share"], atol=1e-12, rtol=0)):
                        raise ValueError("Frozen model did not reproduce archived held-out predictions")
                    heads[i], scores[i] = old["predicted_heads"], old["predicted_feasibility_score"]
                all_records = records(rows, masses, panels, heads, logits, scores, cost_model["target_scale"])
                populations = {scope: [all_records[i] for i in np.flatnonzero(mask)] for scope, mask in groups.items()}
                for scope, rs in populations.items(): pooled[variant][split][scope].extend(rs)
                train, held = summary(populations["training_reference"]), summary(populations["held"])
                fold_gaps[variant].append({"fold": fold, "training_states": int(groups["training_all"].sum()),
                    "training_reference_states": train["state_model_evaluations"], "held_states": held["state_model_evaluations"],
                    "training": {k: train.get(k) for k in GAP_METRICS}, "held": {k: held.get(k) for k in GAP_METRICS}, "gap": gaps(train, held)})
                if split == "lower_root":
                    directions[variant].append({"held_root": fold["held_out"],
                        **{k: summary(rs) for k, rs in populations.items()},
                        "cost_ridge_loss": cost_model["ridge_loss"],
                        "wait_regularization": .5*wait_model["ridge"]*float(np.sum(np.asarray(wait_model["coefficients"])**2)) if name == "glm" else wait_model["final_functional_penalty"]})
    variants = {}
    for variant in VARIANTS:
        validation = {}
        for split in SPLITS:
            populations = pooled[variant][split]
            held = populations["held"]
            if len(held) != 240 or {state_key(r) for r in held} != keys:
                raise ValueError("Lost or repeated a held-out state in a complete split")
            validation[split] = {scope: scopes(rs) for scope, rs in populations.items()}
            validation[split]["held_minus_training_reference"] = gaps(summary(populations["training_reference"]), summary(held))
        scene = {state_key(r): r for r in pooled[variant]["scene"]["held"]}
        contrasts = {}
        for split in ("regime", "lower_root"):
            held = pooled[variant][split]["held"]
            baseline = [scene[state_key(r)] for r in held]
            contrasts[split] = {"states": len(held), "overall": gaps(summary(baseline), summary(held)),
                "by_root": {str(root): gaps(summary([r for r in baseline if r["root"] == root]), summary([r for r in held if r["root"] == root]))
                    for root in wait_source.collection.ROOTS}}
        variants[variant] = {"validation": validation, "same_state_held_minus_scene": contrasts,
            "fold_gaps": fold_gaps[variant], "lower_root_directions": directions[variant]}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "cost_models": 94, "wait_models": 94, **contract()["budget"], "variants": variants,
        "held_feature_support": {s: {"overall": support_summary(rs),
            "by_root": {str(root): support_summary([r for r in rs if r["root"] == root]) for root in wait_source.collection.ROOTS}}
            for s, rs in support.items()}, "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/SOURCE_RUN
    prior = json.loads((directory/"report/result.json").read_text())
    if prior["protocol"] != source.EXPERIMENT_PROTOCOL or prior["contract"] != source.contract() or not prior["software_qualified"]:
        raise ValueError("Complete Stage215 before frozen fit/transfer diagnosis")
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    wait_registered = json.loads((ROOT/"results"/source.SOURCE_RUN/"inputs/manifest.json").read_text())
    bank = json.loads(wait_source.collection.mean_source.source.bank_path(wait_source.collection.mean_source.SOURCE_RUN).read_text())
    prepared = json.loads((raw_directory(ROOT/"results"/wait_source.SOURCE_RUN/"prepare/receipt.json")/"targets.json").read_text())
    original = json.loads(wait_source.source.mass_path(wait_source.collection.MASS_RUN).read_text())
    masses = wait_source.checked_masses(bank, prepared, original, wait_registered)
    old_targets = json.loads(wait_source.collection.mean_source.expected.targets_path(wait_source.collection.MODEL_RUN).read_text())["targets"]
    bank["training_data"] = source.base.mean.expected.replace_targets(bank["training_data"], [*old_targets, *prepared["new_mean_labels"]])
    def load_fold(fold):
        d = raw_directory(source.result_path(SOURCE_RUN, fold))
        cell = json.loads((d/"result.json").read_text())
        source.qualify(cell, fold, registered)
        wd = raw_directory(wait_source.result_path(source.SOURCE_RUN, fold))
        wait_cell = json.loads((wd/"result.json").read_text())
        wait_source.qualify(wait_cell, fold, wait_registered)
        cost_models = {v: json.loads((d/f"{v}.json").read_text()) for v in ("old_cost", "mean_cost")}
        wait_models = {name: json.loads((wd/f"{v}.json").read_text()) for name, v in source.WAIT_HEADS.items()}
        if any({k: x for k, x in m.items() if k != "coefficients"} != cell["fits"][v] for v, m in cost_models.items()):
            raise ValueError("Cost model differs from its frozen fitting receipt")
        if any(wait_source.stopping.metadata(m) != wait_cell["fits"][source.WAIT_HEADS[name]] for name, m in wait_models.items()):
            raise ValueError("Wait model differs from its frozen fitting receipt")
        return cell, cost_models, wait_models
    result = analyze(bank, masses, wait_source.source.diagnostic.load_panels(), registered, load_fold)
    for variant in VARIANTS:
        for split in SPLITS:
            roots = result["variants"][variant]["validation"][split]["held"]["by_root"]
            for root, observed in roots.items():
                expected = prior["variants"][variant]["validation"][split]["by_root"][root]
                if not np.allclose([observed["mean_selected_cost"], observed["mean_selected_wait_min"]],
                        [expected["mean_actual_cost_delta"], expected["mean_wait_delta_min"]], atol=1e-12, rtol=0):
                    raise ValueError("Fit/transfer diagnosis changed the archived selected outcomes")
    result["controls_reproduced"] = True
    write_json(args.output, result)
    print(json.dumps(result["variants"][source.PRIMARY]["same_state_held_minus_scene"]), flush=True)
    print("NATIVE_FIT_TRANSFER_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
