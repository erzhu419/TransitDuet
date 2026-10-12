"""Training-scene cross-fitted wait margins with a frozen deployed critic."""

from collections import defaultdict
from statistics import NormalDist

import numpy as np

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_pair_ridge as ridge

MEMBERS, PARTITION_SEED = 4, 200
POLICIES = ("uncalibrated", "mean_bias", "cluster_margin90")
Z90 = NormalDist().inv_cdf(.9)


def partition_groups(groups, mean_groups):
    strata = defaultdict(list)
    for group in groups:
        key = paired.scene_group(group)
        strata[group["root"], group["scenario"], key in mean_groups].append(key)
    rng = np.random.default_rng(PARTITION_SEED)
    held = [[] for _ in range(MEMBERS)]
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        for i, j in enumerate(rng.permutation(len(keys))): held[i % MEMBERS].append(keys[j])
    return [sorted(map(list, bucket)) for bucket in held]


def corrected_wait(prediction, original, margin):
    wait = np.asarray(prediction["predicted_wait_delta_min"])+margin
    wait[original] = 0.
    return wait


def choose(prediction, original, margin):
    cost = np.asarray(prediction["predicted_heads"]).sum(axis=1)
    return paired.choice_from_cost(np.where(corrected_wait(prediction, original, margin) <= 0, cost, np.inf), original)


def residual_summary(rows, predictions):
    totals = defaultdict(lambda: [0., 0])
    for row, prediction in zip(rows, predictions, strict=True):
        bucket = totals[paired.scene_group(row)]
        selected = choose(prediction, row["original_choice"], 0.)
        if selected != row["original_choice"]:
            labels = {l["candidate"]: l for l in row["labels"]}
            bucket[0] += labels[selected]["wait_delta_min"]-prediction["predicted_wait_delta_min"][selected]
            bucket[1] += 1
    sums, counts = np.asarray(list(totals.values())).T
    changed, groups = int(counts.sum()), len(totals)
    bias = float(sums.sum()/changed) if changed else 0.
    se = (float(np.sqrt(groups/(groups-1)*np.sum((sums-bias*counts)**2))/changed)
        if groups > 1 and changed else 0.)
    return {"mean_selected_residual_min": bias, "group_ratio_standard_error_min": se,
        "changed_choices": changed, "calibration_states": len(rows), "calibration_groups": groups,
        "groups": sorted(map(list, totals)),
        "group_totals": [{"group": list(g), "residual_sum_min": v[0], "changed_choices": v[1]} for g, v in sorted(totals.items())],
        "margins": {"uncalibrated": 0., "mean_bias": max(0., bias), "cluster_margin90": max(0., bias+Z90*se)}}


def crossfit(rows, mean_keys, partition):
    calibration_rows, predictions, ledgers = [], [], []
    for held_groups in partition:
        held = {tuple(g) for g in held_groups}
        if not held <= {paired.scene_group(r) for r in rows}:
            raise ValueError("Inner held groups must belong to the outer training population")
        train = [r for r in rows if paired.scene_group(r) not in held]
        calibration = [r for r in rows if paired.scene_group(r) in held and
            (r["root"], r["scene_seed"], r["probe_index"]) in mean_keys]
        model = ridge.fit_value(train, sample_weights=[8 if (r["root"], r["scene_seed"], r["probe_index"]) in mean_keys else 1 for r in train])
        calibration_rows.extend(calibration)
        predictions.extend(ridge.predict(r, model) for r in calibration)
        ledgers.append({"training_groups": model["training_groups"], "held_groups": held_groups,
            "training_states": len(train), "calibration_mean_states": len(calibration),
            "parameter_count": model["parameter_count"], "sample_weight_sum": model["sample_weight_sum"],
            "center": model["center"], "scale": model["scale"], "target_scale": model["target_scale"]})
    keys = [(r["root"], r["scene_seed"], r["probe_index"]) for r in calibration_rows]
    expected = {(r["root"], r["scene_seed"], r["probe_index"]) for r in rows} & mean_keys
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Cross-fitting lost or duplicated a training mean8 state")
    return {**residual_summary(calibration_rows, predictions), "partition": partition, "fits": ledgers}


def evaluate(rows, frozen_predictions, margin):
    result = paired.evaluate_action_value(rows, {},
        predictor=lambda r, _: np.asarray(frozen_predictions[r["root"], r["scene_seed"], r["probe_index"]]["predicted_heads"]),
        selector=lambda r, _: choose(frozen_predictions[r["root"], r["scene_seed"], r["probe_index"]], r["original_choice"], margin))
    for row, observed in zip(rows, result):
        prediction = frozen_predictions[row["root"], row["scene_seed"], row["probe_index"]]
        observed.update(predicted_wait_delta_min=prediction["predicted_wait_delta_min"],
            calibrated_wait_delta_min=corrected_wait(prediction, row["original_choice"], margin).tolist(), wait_margin_min=margin)
    return result
