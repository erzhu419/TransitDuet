"""Fixed-fold label attribution and support geometry, without new fits."""

import numpy as np

from .compute_budget import count_work
from .plan_validity_boundary_parent_scale import scaled_basis


BLOCKS = (("state", slice(0, 39)), ("clock", slice(39, 42)),
    ("state_x_remaining", slice(42, 81)), ("state_x_spent", slice(81, 120)))


def raw_weights(model):
    w = np.asarray(model["weights"]).copy()
    mean, scale = np.asarray(model["parent_mean"]), np.asarray(model["parent_scale"])
    for block in (slice(0, 39), slice(42, 81), slice(81, 120)):
        w[block] /= scale
    w[40] -= mean @ w[42:81]
    w[41] -= mean @ w[81:120]
    return w


def attribute(data, frozen_model, saved_views, original):
    keys, labels = np.asarray(data["window_keys"]), np.asarray(data["training_replicas"])
    groups = keys[:, 0]
    prefixes = np.unique(groups)
    difference = data["wait_one_check"] - data["now"]
    if labels.shape != (384, 2) or len(prefixes) != 32 or len(saved_views) != 2:
        raise ValueError("need the saved two-view 32-prefix Wide diagnostic")
    blocks = np.empty((2, 32, 384, 4))
    held = np.empty((2, 384))
    for r, view in enumerate(saved_views):
        if view["fitting_A_column"] != r or [f["excluded_prefix"] for f in view["folds"]] != prefixes.tolist():
            raise ValueError("saved fold view or prefix ordering differs")
        for j, fold in enumerate(view["folds"]):
            if fold["training_prefixes"] != prefixes[prefixes != prefixes[j]].tolist():
                raise ValueError("saved training-prefix roster differs")
            w = raw_weights(fold["model"])
            blocks[r, j] = np.column_stack([difference[:, b] @ w[b] for _, b in BLOCKS])
            count_work("saved_fold_window_predictions", len(groups))
            mask = groups == prefixes[j]
            held[r, mask] = blocks[r, j, mask].sum(axis=1)
    scores = blocks.sum(axis=3)
    views = []
    for r, view in enumerate(saved_views):
        held_blocks = np.empty((384, 4))
        for j, g in enumerate(prefixes):
            held_blocks[groups == g] = blocks[r, j, groups == g]
        mean = (scores[r].sum(axis=0) - held[r]) / 31
        square = ((scores[r] ** 2).sum(axis=0) - held[r] ** 2) / 31
        if any(not np.allclose(value, view[key], rtol=1e-10, atol=1e-10) for key, value in (
            ("prefix_held_predictions", held[r]), ("matched_prediction_mean", mean), ("matched_prediction_square_mean", square))):
            raise ValueError("raw-coordinate predictions do not reproduce saved fold moments")
        mean_blocks = (blocks[r].sum(axis=0) - held_blocks) / 31
        cross_blocks = ((scores[r, :, :, None] * blocks[r]).sum(axis=0) - held[r, :, None] * held_blocks) / 31
        square_parts = (held[r, :, None] * held_blocks - cross_blocks).mean(axis=0)
        alignment_parts = (-2 * labels[:, 1 - r, None] * (held_blocks - mean_blocks)).mean(axis=0)
        gap = float(np.sum(square_parts + alignment_parts))
        if not np.isclose(gap, original["views"][r]["overall"]["held_minus_matched"], rtol=1e-10, atol=1e-10):
            raise ValueError("feature-block loss attribution does not close")
        views.append({"fitting_A_column": r, "source_overall": original["views"][r]["overall"],
            "block_gap_allocations": [{"block": name, "squared_prediction_term": float(sq),
                "alignment_term": float(al), "gap": float(sq + al)} for (name, _), sq, al in zip(BLOCKS, square_parts, alignment_parts)]})
    donor = np.zeros((2, 32, 32))
    own = np.zeros((2, 32))
    geometry = []
    reference_scale = np.asarray(frozen_model["parent_scale"])
    reference_mean = np.asarray(frozen_model["parent_mean"])
    label_variance = (labels[:, 0] - labels[:, 1]) ** 2 / 2
    for j, g in enumerate(prefixes):
        models = [v["folds"][j]["model"] for v in saved_views]
        for key in ("parent_mean", "parent_scale", "alpha", "total_effective_df"):
            if models[0][key] != models[1][key]:
                raise ValueError("reciprocal views must share label-independent fold geometry")
        model = models[0]
        x = scaled_basis(data["wait_one_check"], model["parent_mean"], model["parent_scale"]) - scaled_basis(
            data["now"], model["parent_mean"], model["parent_scale"])
        train, probe = groups != g, groups == g
        u, s, vt = np.linalg.svd(x[train], full_matrices=False)
        count_work("geometry_svd_calls")
        eigenvalues, alpha = s ** 2, model["alpha"]
        shrink = eigenvalues / (eigenvalues + alpha)
        df = float(shrink.sum())
        if not np.isclose(df, model["total_effective_df"], rtol=1e-10, atol=1e-10):
            raise ValueError("saved ridge geometry does not reproduce its fixed DF")
        projected = x[probe] @ vt.T
        response = (projected * (s / (eigenvalues + alpha))) @ u.T
        # This is the label response of the stored fit, never an alternative fit.
        if not np.allclose(response @ labels[train], held[:, probe].T, rtol=1e-10, atol=1e-10):
            raise ValueError("fixed-model label contributions do not reconstruct saved predictions")
        train_groups = groups[train]
        for k, h in enumerate(prefixes):
            if h == g:
                continue
            selected = train_groups == h
            contribution = response[:, selected] @ labels[groups == h]
            for r in (0, 1):
                donor[r, j, k] = np.mean(contribution[:, r] * (held[r, probe] - 2 * labels[probe, 1 - r]))
            # Own-prefix contribution in every already fitted model including h.
            ug = u[selected]
            own_response = (ug * shrink) @ ug.T
            for r in (0, 1):
                c = own_response @ labels[groups == h, r]
                own[r, k] += np.mean(c * (scores[r, j, groups == h] - 2 * labels[groups == h, 1 - r])) / 31
        leverage = (projected ** 2 / (eigenvalues + alpha)).sum(axis=1)
        null = eigenvalues <= max(1., eigenvalues[0]) * 1e-12
        null_leverage = float((projected[:, null] ** 2 / (eigenvalues[null] + alpha)).sum())
        scales = np.asarray(model["parent_scale"]) / reference_scale
        geometry.append({"seed": int(g), "held_to_training_mean_ridge_leverage": float(leverage.mean() / (df / train.sum())),
            "held_ridge_leverage_max": float(leverage.max()), "contrast_rank": int((~null).sum()),
            "held_null_leverage_fraction": null_leverage / float(leverage.sum()) if leverage.sum() else None,
            "parent_scale_ratio_min": float(scales.min()), "parent_scale_ratio_max": float(scales.max()),
            "parent_mean_shift_RMS_in_full_scale": float(np.sqrt(np.mean(((np.asarray(model["parent_mean"]) - reference_mean) / reference_scale) ** 2))),
            "estimated_single_A_prediction_noise_variance": float(np.mean(response ** 2 @ label_variance[train])),
            "observed_reciprocal_prediction_change_MSE": float(np.mean((held[0, probe] - held[1, probe]) ** 2))})
    for r, view in enumerate(views):
        allocated = donor[r].sum(axis=1)
        expected = [p["held_minus_zero"] for p in original["views"][r]["prefix_reports"]]
        if not np.allclose(allocated, expected, rtol=1e-10, atol=1e-10):
            raise ValueError("training-prefix label loss allocations do not close")
        own_mean = float(own[r].mean())
        view.update(matched_self_label_allocation=own_mean,
            matched_other_label_allocation=view["source_overall"]["matched_minus_zero"] - own_mean,
            held_label_allocation_by_training_prefix=[{"seed": int(g), "allocation": float(donor[r, :, k].mean())} for k, g in enumerate(prefixes)])
    prefix_gaps = np.mean([[p["held_minus_matched"] for p in v["prefix_reports"]] for v in original["views"]], axis=0)
    for row, gap in zip(geometry, prefix_gaps):
        row["mean_A_view_prefix_gap"] = float(gap)
    return {"views": views, "prefix_geometry": geometry, "reproduced_saved_predictions_and_loss_allocations": True,
        "statistics": "fixed_model_additive_allocations_not_causal_deletion_or_geometry_counterfactual_effects",
        "noise_scope": "descriptive_two_A_label_variance_propagation_given_independent_suffix_noise_not_CI_or_gap_correction"}, {
            "prefixes": prefixes, "held_loss_allocation_by_training_prefix": donor,
            "matched_self_label_loss_allocation": own}
