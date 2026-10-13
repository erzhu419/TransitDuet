#!/usr/bin/env python3
"""Fixed-feature value projection and stable ordinal/sign expressivity bounds."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_fit_transfer_stage216 as previous
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_linear_floor_stage217_v1"
SOURCE_RUN = "native_transit_fit_transfer_stage216_diagnostic_20261013_r2"
source, wait_source = previous.source, previous.wait_source
SPLITS, PARTITIONS, state_key = previous.SPLITS, previous.PARTITIONS, previous.state_key
VARIANTS = ("ridge_glm", "projection_glm")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "data": "same600_mean8_targets_original240_panels_47_folds_original240_weight8_360_weight1",
        "value_bound": "weighted_numpy_lstsq_rcond_None_same98_causal_features_three_RHS_original_training_scales",
        "decomposition": "physical_total_cost_MSE_projection_floor_plus_ridge_minus_projection_energy",
        "ordinal_bound": "train_original240_only_both_half_stable_true_safe_cost_pairs_strict_order_margin1",
        "wait_bound": "same199_GLM_design_train_original240_stable_safe_logit_le0_unsafe_logit_ge1_identity_excluded",
        "witness": "minimum_L1_coefficients_HiGHS_threads1_no_coefficient_cap_numerical_feasibility_tolerance1e-7",
        "partitions": source.diagnosis.factorial.contract()["partitions"], "variants": list(VARIANTS),
        "evaluation": "projection_with_frozen_GLM_threshold_and_identity_ties_training_and_held_physical_units",
        "budget": {"cost_projection_fits": 47, "max_linear_programs": 188, "optimizer_updates": 0,
            "actor_updates": 0, "new_native_episodes": 0},
        "storage": "projection_models_witnesses_bank_targets_panels_remote_small_report_only",
        "boundary": "finite_training_numerical_linear_class_bounds_not_all_feature_impossibility_or_population_CI"}


def project(raw, targets, weights, reference):
    left, right = np.triu_indices(6, 1)
    x = (raw-reference["center"])/reference["scale"]
    z = (x[:, left]-x[:, right]).reshape(-1, raw.shape[-1])
    y = ((targets[:, left]-targets[:, right])/reference["target_scale"]).reshape(-1, 3)
    pair_weights = np.repeat(weights, 15)
    root_weight = np.sqrt(pair_weights/pair_weights.sum())[:, None]
    a, b = root_weight*z, root_weight*y
    coefficients, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    ridge = np.asarray(reference["coefficients"])
    cost_scale = np.asarray(reference["target_scale"][:2])
    physical = lambda v: v[:, :2]@cost_scale
    floor = float(np.sum(physical(a@coefficients-b)**2))
    ridge_loss = float(np.sum(physical(a@ridge-b)**2))
    shrinkage = float(np.sum(physical(a@(ridge-coefficients))**2))
    gap = ridge_loss-floor-shrinkage
    if not np.isclose(gap, 0., atol=1e-10, rtol=0):
        raise ValueError("Projection residual is not orthogonal enough for cost-MSE decomposition")
    if not np.isclose(float(np.sum((a@ridge-b)**2)/3), reference["final_pair_loss"], atol=1e-10, rtol=0):
        raise ValueError("Projection data/weights do not reproduce the mean8 ridge training loss")
    model = {**reference, "method": "weighted_linear_projection", "coefficients": coefficients.tolist(),
        "ridge": 0., "ridge_loss": 0., "solver": "numpy_lstsq_rcond_None",
        "final_pair_loss": float(np.sum((a@coefficients-b)**2)/3)}
    return model, {"training_states": len(raw), "sample_weight_sum": float(np.sum(weights)), "feature_dimensions": raw.shape[-1],
        "numerical_rank": int(rank), "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[rank-1]) if rank else None,
        "ridge_cost_pair_mse": ridge_loss, "projection_cost_pair_mse": floor,
        "ridge_excess_cost_pair_mse": shrinkage, "decomposition_residual": gap,
        "projection_fraction_of_ridge_error": previous.transfer.ratio(floor, ridge_loss),
        "coefficient_l2": float(np.linalg.norm(coefficients)), "ridge_coefficient_l2": float(np.linalg.norm(ridge))}


def linear_witness(directions, margins):
    n, dimensions = directions.shape
    if not n:
        return {"constraints": 0, "dimensions": dimensions, "linear_programs": 0, "feasible": True,
            "status": 0, "coefficient_l1": 0., "constraint_violation": 0., "coefficients": [0.]*dimensions}
    # Positive/negative coefficient parts minimize L1 without constraining the class.
    result = linprog(np.ones(2*dimensions), A_ub=-np.c_[directions, -directions], b_ub=-margins,
        bounds=(0., None), method="highs", options={"threads": 1})
    if result.status not in (0, 2):
        raise RuntimeError(f"Linear expressivity probe was not solved: {result.message}")
    metadata = {"constraints": n, "dimensions": dimensions, "linear_programs": 1, "feasible": result.status == 0,
        "status": int(result.status), "solver_iterations": int(result.nit), "coefficients": None}
    if result.status == 0:
        coefficients = result.x[:dimensions]-result.x[dimensions:]
        violation = float(np.maximum(margins-directions@coefficients, 0).max())
        if violation > 1e-7: raise ValueError("LP witness violates its registered training inequalities")
        metadata.update(coefficients=coefficients.tolist(), coefficient_l1=float(abs(coefficients).sum()), constraint_violation=violation)
    return metadata


def stable_constraints(rows, panels, cost_design, wait_design, training, partition):
    left, right = np.triu_indices(6, 1)
    cost_rows, wait_rows, wait_margins = [], [], []
    for i in np.flatnonzero(training):
        row = rows[i]
        if state_key(row) not in panels: continue
        values = panels[state_key(row)]["values"]
        a, b = [values[list(indices)].mean(axis=0) for indices in PARTITIONS[partition]]
        safe, unsafe = (a[:, 3] <= 0) & (b[:, 3] <= 0), (a[:, 3] > 0) & (b[:, 3] > 0)
        ga, gb = a[left, 0]-a[right, 0], b[left, 0]-b[right, 0]
        stable = safe[left] & safe[right] & (abs(ga) > 1e-12) & (abs(gb) > 1e-12) & (ga*gb > 0)
        cost_rows.extend(np.sign(ga[stable])[:, None]*(cost_design[i, left[stable]]-cost_design[i, right[stable]]))
        for j in range(6):
            if j == row["original_choice"] or not (safe[j] or unsafe[j]): continue
            wait_rows.append((-1. if safe[j] else 1.)*wait_design[i, j])
            wait_margins.append(0. if safe[j] else 1.)
    cost_rows = np.asarray(cost_rows).reshape(-1, cost_design.shape[-1])
    wait_rows = np.asarray(wait_rows).reshape(-1, wait_design.shape[-1])
    return {"cost_order": (cost_rows, np.ones(len(cost_rows))), "wait_sign": (wait_rows, np.asarray(wait_margins))}


def diagnose_fold(rows, masses, panels, registered, fold, cell, ridge, glm, batch):
    keys = {state_key(r) for r in registered["mean_states"]}
    masks = previous.masks(rows, fold, keys)
    weights = wait_source.source.feasibility.sample_weights(rows, masses)
    targets = np.asarray([[[l[h] for h in source.base.cost.value.HEADS] for l in sorted(r["labels"], key=lambda x: x["candidate"])] for r in rows])
    projection, decomposition = project(batch[0][masks["training_all"], :, :98], targets[masks["training_all"]],
        weights[masks["training_all"]], ridge)
    cost_design = source.base.centered(batch[0][..., :98], rows, ridge)
    wait_design = wait_source.source.feasibility.design(batch[0], [r["original_choice"] for r in rows], glm["center"], glm["scale"])
    witnesses = {p: {name: linear_witness(*problem) for name, problem in
        stable_constraints(rows, panels, cost_design, wait_design, masks["training_all"], p).items()} for p in PARTITIONS}
    archived = {state_key(r): r for r in cell["predictions"][source.PRIMARY]}
    groups = {}
    for variant, model in (("ridge_glm", ridge), ("projection_glm", projection)):
        heads, logits, shares, scores = previous.inference(rows, batch, model, glm, "glm")
        for i in np.flatnonzero(masks["held"]):
            old = archived[state_key(rows[i])]
            if not (np.allclose(shares[i], old["predicted_harm_mass_share"], atol=1e-12, rtol=0)
                    and np.allclose(scores[i], old["predicted_feasibility_score"], atol=1e-12, rtol=0)):
                raise ValueError("Projection changed its frozen wait control")
            if variant == "ridge_glm":
                if not np.allclose(heads[i], old["predicted_heads"], atol=1e-12, rtol=0):
                    raise ValueError("Mean8 ridge predictor did not reproduce its control")
                heads[i] = old["predicted_heads"]
            scores[i] = old["predicted_feasibility_score"]
        records = previous.records(rows, masses, panels, heads, logits, scores, ridge["target_scale"])
        groups[variant] = {name: [records[i] for i in np.flatnonzero(mask)] for name, mask in masks.items() if name != "training_mean8"}
    return groups, {"fold": fold, "value_bound": decomposition,
        "stable_training_constraints": {p: {name: {k: v for k, v in witness.items() if k != "coefficients"}
            for name, witness in problems.items()} for p, problems in witnesses.items()}}, projection, witnesses


def analyze(bank, masses, panels, registered, load_fold, save_references):
    rows = bank["training_data"]
    keys = {state_key(r) for r in registered["mean_states"]}
    if len(rows) != 600 or len(keys) != 240 or set(panels) != keys or len(registered["folds"]) != 47:
        raise ValueError("Require the unchanged600/240/47 mean8 cohort")
    batch = wait_source.neural.batches(rows)
    pooled = {v: {s: {scope: [] for scope in ("training_all", "training_reference", "held")} for s in SPLITS} for v in VARIANTS}
    folds = []
    for fold in registered["folds"]:
        cell, ridge, glm = load_fold(fold)
        groups, diagnostic, projection, witnesses = diagnose_fold(rows, masses, panels, registered, fold, cell, ridge, glm, batch)
        save_references(fold, projection, witnesses)
        folds.append(diagnostic)
        for variant, scopes in groups.items():
            for scope, rs in scopes.items(): pooled[variant][fold["split"]][scope].extend(rs)
        print(f"LINEAR_FLOOR_FOLD {fold['split']}/{fold['index']} rank={diagnostic['value_bound']['numerical_rank']}", flush=True)
    variants = {}
    for variant in VARIANTS:
        validation = {}
        for split, scopes in pooled[variant].items():
            held = scopes["held"]
            if len(held) != 240 or {state_key(r) for r in held} != keys: raise ValueError("Changed the held evaluation cohort")
            validation[split] = {scope: previous.scopes(rs) for scope, rs in scopes.items()}
        roots = [r for v in validation.values() for r in v["held"]["by_root"].values()]
        variants[variant] = {"validation": validation,
            "gates": {"cost": all(r["mean_selected_cost"] < 0 for r in roots), "wait": all(r["mean_selected_wait_min"] <= 0 for r in roots)}}
    counts = {p: {name: sum(f["stable_training_constraints"][p][name]["feasible"] for f in folds)
        for name in ("cost_order", "wait_sign")} for p in PARTITIONS}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "cost_projection_fits": len(folds), "linear_programs": sum(w["linear_programs"] for f in folds for p in f["stable_training_constraints"].values() for w in p.values()),
        "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0, "folds": folds,
        "feasible_fold_counts": counts, "variants": variants, "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["controls_reproduced"]:
        raise ValueError("Complete corrected Stage216 before diagnosing the linear class")
    registered = json.loads((ROOT/"results"/previous.SOURCE_RUN/"inputs/manifest.json").read_text())
    wait_registered = json.loads((ROOT/"results"/source.SOURCE_RUN/"inputs/manifest.json").read_text())
    bank = json.loads(wait_source.collection.mean_source.source.bank_path(wait_source.collection.mean_source.SOURCE_RUN).read_text())
    prepared = json.loads((raw_directory(ROOT/"results"/wait_source.SOURCE_RUN/"prepare/receipt.json")/"targets.json").read_text())
    original = json.loads(wait_source.source.mass_path(wait_source.collection.MASS_RUN).read_text())
    masses = wait_source.checked_masses(bank, prepared, original, wait_registered)
    old_targets = json.loads(wait_source.collection.mean_source.expected.targets_path(wait_source.collection.MODEL_RUN).read_text())["targets"]
    bank["training_data"] = source.base.mean.expected.replace_targets(bank["training_data"], [*old_targets, *prepared["new_mean_labels"]])
    def load_fold(fold):
        d = raw_directory(source.result_path(previous.SOURCE_RUN, fold))
        cell, ridge = [json.loads((d/name).read_text()) for name in ("result.json", "mean_cost.json")]
        source.qualify(cell, fold, registered)
        if {k: v for k, v in ridge.items() if k != "coefficients"} != cell["fits"]["mean_cost"]:
            raise ValueError("Changed the mean8 ridge model")
        wd = raw_directory(wait_source.result_path(source.SOURCE_RUN, fold))
        wait_cell, glm = [json.loads((wd/name).read_text()) for name in ("result.json", f"{wait_source.NEW_GLM}.json")]
        wait_source.qualify(wait_cell, fold, wait_registered)
        if wait_source.stopping.metadata(glm) != wait_cell["fits"][wait_source.NEW_GLM]:
            raise ValueError("Changed the frozen mean8 GLM")
        return cell, ridge, glm
    def save_references(fold, model, witnesses):
        d = raw_directory(args.output)/fold["split"]/f"fold_{fold['index']:03}"
        write_json(d/"projection.json", model)
        write_json(d/"witnesses.json", witnesses)
    result = analyze(bank, masses, wait_source.source.diagnostic.load_panels(), registered, load_fold, save_references)
    for split in SPLITS:
        for root, observed in result["variants"]["ridge_glm"]["validation"][split]["held"]["by_root"].items():
            expected = prior["variants"][source.PRIMARY]["validation"][split]["held"]["by_root"][root]
            if not np.allclose([observed["mean_selected_cost"], observed["mean_selected_wait_min"]],
                    [expected["mean_selected_cost"], expected["mean_selected_wait_min"]], atol=1e-12, rtol=0):
                raise ValueError("Linear-class diagnosis did not reproduce Stage216 physical controls")
    result["controls_reproduced"] = True
    write_json(args.output, result)
    print(json.dumps({"feasible": result["feasible_fold_counts"], "gates": {v: d["gates"] for v, d in result["variants"].items()}}), flush=True)
    print("NATIVE_LINEAR_FLOOR_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
