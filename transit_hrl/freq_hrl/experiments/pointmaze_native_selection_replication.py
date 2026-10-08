"""Root-cluster inference including the cheaper fitted-return selector."""

import numpy as np

from .statistics import bootstrap_mean_ci
from scripts import pointmaze_native_selection_replication_stage135_spec as spec


def aggregate(cells):
    if len(cells) != len(spec.ROOTS) or {c["root"] for c in cells} != set(spec.ROOTS):
        raise ValueError("Native selection replication requires exactly six source roots")
    values, root_rows = {k: [] for k in spec.ENDPOINTS}, []
    for c in sorted(cells, key=lambda c: c["root"]):
        if (c["status"], c["protocol"], c["cost"], c["source_protocol"], c["inherited_source_cost"], c["kind"]) != (
                "complete", spec.PROTOCOL, spec.budget(), spec.source.PROTOCOL, spec.source.budget(), spec.EVIDENCE_ROLE):
            raise ValueError("Native selection protocol, source or budget changed")
        root_row = {"root": c["root"], "periods": {}}
        for period in spec.PERIODS:
            g = c["groups"][str(period)]
            if (g["training_roles"] != spec.training_roles(c["root"])
                    or g["validation_roles"] != spec.validation_roles(c["root"])
                    or g["evaluation_seeds"] != spec.evaluation_seeds(c["root"])
                    or g["lower_and_critics_frozen"] != "passed"):
                raise ValueError("Native selection paired roster or freeze changed")
            scores = {"two_step": 0.}
            for m in spec.METHODS:
                paired = np.asarray(g["selection"]["paired_differences"][m], dtype=np.float64)
                if paired.shape != (len(spec.PANELS)*spec.VALIDATION_SCENARIOS,) or not np.isfinite(paired).all():
                    raise ValueError("Native validation paired differences changed")
                scores[m] = float(paired.mean())
                np.testing.assert_allclose(g["selection"]["mean_incremental_returns"][m], scores[m], atol=1e-12, rtol=0)
            method = max(scores, key=scores.get)
            if g["selection"]["method"] != method or g["selection"]["mean_incremental_returns"]["two_step"] != 0.:
                raise ValueError("Native selection no longer follows validation")
            if g["mean_metrics"]["selected"] != g["mean_metrics"][method]:
                raise ValueError("Selected policy evaluation alias changed")
            if g["effects"]["refresh_minus_source_forecast"] != g["effects"]["refresh_minus_refresh_blinded"]:
                raise ValueError("Native selection blinded forecast identity changed")
            arrays = {}
            for a, b in spec.CONTRASTS:
                key = f"{a}_minus_{b}"
                e = g["effects"][key]
                paired = np.asarray(e["paired_differences"], dtype=np.float64)
                if paired.shape != (spec.EVALUATION_EPISODES,) or not np.isfinite(paired).all():
                    raise ValueError("Native evaluation paired differences changed")
                np.testing.assert_allclose(e["mean"], paired.mean(), atol=1e-12, rtol=0)
                np.testing.assert_allclose(e["mean"], g["mean_metrics"][a]["episode_return"]-g["mean_metrics"][b]["episode_return"], atol=1e-10, rtol=0)
                arrays[key] = paired
            current = arrays["selected_minus_two_step"]
            for m, paired in (("two_step", np.zeros_like(current)), ("refresh", arrays["refresh_minus_two_step"]), ("stale", arrays["stale_minus_two_step"])):
                np.testing.assert_allclose(arrays[f"selected_minus_{m}"], current-paired, atol=1e-10, rtol=0)
                if method == m:
                    np.testing.assert_allclose(current, paired, atol=1e-10, rtol=0)
            fitted = {"two_step": 0., **{m:g["training_native_return_fits"][m]["pooled"]["predicted_gain"] for m in spec.METHODS}}
            calibration_choice = max(fitted, key=fitted.get)
            # This comparator uses the declared fit-only rule, never evaluation winners.
            arrays["selected_minus_calibration_choice"] = arrays[f"selected_minus_{calibration_choice}"]
            means = {}
            for control in spec.PRIMARY_CONTROLS:
                key = f"{period}/selected_minus_{control}"
                mean = float(arrays[f"selected_minus_{control}"].mean())
                values[key].append(mean)
                means[control] = mean
            root_row["periods"][str(period)] = {"selected_method": method, "calibration_choice": calibration_choice, "effects": means}
        root_rows.append(root_row)
    endpoints = {}
    for i, (key, x) in enumerate(values.items()):
        ci = bootstrap_mean_ci(x, n_boot=spec.BOOTSTRAP_DRAWS, seed=spec.BOOTSTRAP_SEED+i, alpha=.05/len(spec.ENDPOINTS))
        threshold = spec.MINIMUM_GAIN if key.endswith("source_forecast") else 0.
        endpoints[key] = {"mean": float(np.mean(x)), "ci": list(ci), "root_means": x, "threshold": threshold,
            "positive_gain_supported": ci[0] > 0., "threshold_supported": ci[0] > threshold}
    def gate(control):
        return {str(p): endpoints[f"{p}/selected_minus_{control}"]["threshold_supported"] for p in spec.PERIODS}
    material, increment, refresh, radial, validation = [gate(c) for c in spec.PRIMARY_CONTROLS]
    return {"status": "complete", "protocol": spec.PROTOCOL, "root_rows": root_rows, "endpoints": endpoints,
        "period_material_gate": material, "period_increment_gate": increment,
        "period_fixed_refresh_gate": refresh, "period_fixed_radial_gate": radial, "period_validation_added_value_gate": validation,
        "material_continuation_gate": "supported_both_periods" if all(material.values()) and all(increment.values()) else "not_closed",
        "fixed_method_selection_gate": "supported_both_periods" if all(refresh.values()) and all(radial.values()) else "not_closed",
        "validation_added_value_gate": "supported_both_periods" if all(validation.values()) else "not_closed",
        "statistics": {"independent_unit": "source_policy_optimizer_root", "n_independent": len(cells),
            "paired_evaluation_scenarios_per_root_period": spec.EVALUATION_EPISODES,
            "validation_scenarios_per_root_period": spec.VALIDATION_SCENARIOS, "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
            "familywise_alpha": .05, "bonferroni_family_size": len(spec.ENDPOINTS), "family_scope": "within_stage_not_adaptive_research_sequence"},
        "cost": {k:sum(c["cost"][k] for c in cells) for k in spec.budget()},
        "inherited_source_cost": {k:sum(c["inherited_source_cost"][k] for c in cells) for k in spec.source.budget()},
        "fit_only_selector_validation_episodes_saved_per_root": spec.budget()["selection_episodes"],
        "evidence_role": spec.EVIDENCE_ROLE, "prior_stage133_composite_gate": "unchanged_not_closed",
        "prior_joint_HRL_and_frequency_specific_gates": "unchanged"}
