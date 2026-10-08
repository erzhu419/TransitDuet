"""Root-cluster inference for the frozen compact continuation method."""

import numpy as np

from .statistics import bootstrap_mean_ci
from scripts import pointmaze_continuation_replication_stage131_spec as spec


def aggregate(cells, *, protocol_spec=spec):
    spec = protocol_spec
    if len(cells) != len(spec.ROOTS) or {c["root"] for c in cells} != set(spec.ROOTS):
        raise ValueError("Stage131 requires exactly six additional source roots")
    cells = sorted(cells, key=lambda c: c["root"])
    values, root_rows = {key: [] for key in spec.ENDPOINTS}, []
    for c in cells:
        if (c["status"], c["protocol"], c["cost"], c["source_protocol"], c["inherited_source_cost"]) != (
                "complete", spec.PROTOCOL, spec.budget(), spec.source.PROTOCOL, spec.source.budget()):
            raise ValueError("Stage131 protocol, source or budget changed")
        means = {}
        for period in spec.PERIODS:
            g = c["groups"][str(period)]
            if (g["evaluation_seeds"] != spec.evaluation_seeds(c["root"])
                    or g["training_roles"] != spec.training_roles(c["root"])
                    or g["lower_and_critics_frozen"] != "passed"):
                raise ValueError("Stage131 paired roster or lower/critic freeze changed")
            if g["effects"]["refresh_minus_source_forecast"] != g["effects"]["refresh_minus_refresh_blinded"]:
                raise ValueError("Stage131 blinded forecast identity changed")
            for a, b in spec.PRIMARY_CONTRASTS:
                key = f"{period}/{a}_minus_{b}"
                e = g["effects"][f"{a}_minus_{b}"]
                paired = np.asarray(e["paired_differences"], dtype=np.float64)
                if paired.shape != (spec.EVALUATION_EPISODES,) or not np.isfinite(paired).all():
                    raise ValueError("Stage131 primary paired differences changed")
                np.testing.assert_allclose(e["mean"], paired.mean(), atol=1e-12, rtol=0)
                values[key].append(float(paired.mean()))
                means[key] = float(paired.mean())
        root_rows.append({"root": c["root"], "effects": means})
    endpoints = {}
    for index, (key, x) in enumerate(values.items()):
        ci = bootstrap_mean_ci(x, n_boot=spec.BOOTSTRAP_DRAWS, seed=spec.BOOTSTRAP_SEED+index,
            alpha=.05/len(spec.ENDPOINTS))
        threshold = spec.MINIMUM_GAIN if key.endswith("source_forecast") else 0.
        endpoints[key] = {"mean": float(np.mean(x)), "ci": list(ci), "root_means": x,
            "threshold": threshold, "positive_gain_supported": ci[0] > 0., "threshold_supported": ci[0] > threshold}
    material = {str(p): endpoints[f"{p}/refresh_minus_source_forecast"]["threshold_supported"] for p in spec.PERIODS}
    increment = {str(p): endpoints[f"{p}/refresh_minus_{spec.SOURCE_BASELINE}"]["threshold_supported"] for p in spec.PERIODS}
    relabel = {str(p): endpoints[f"{p}/refresh_minus_stale"]["threshold_supported"] for p in spec.PERIODS}
    return {"status": "complete", "protocol": spec.PROTOCOL, "root_rows": root_rows, "endpoints": endpoints,
        "period_material_gate": material, "period_increment_gate": increment, "period_relabeling_gate": relabel,
        "material_continuation_gate": "supported_both_periods" if all(material.values()) and all(increment.values()) else "not_closed",
        "universal_relabeling_gate": "supported_both_periods" if all(relabel.values()) else "not_closed",
        "statistics": {"independent_unit": "source_policy_optimizer_root", "n_independent": len(cells),
            "paired_scenarios_per_root_period": spec.EVALUATION_EPISODES, "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
            "familywise_alpha": .05, "bonferroni_family_size": len(spec.ENDPOINTS)},
        "cost": {k: sum(c["cost"][k] for c in cells) for k in spec.budget()},
        "inherited_source_cost": {k: sum(c["inherited_source_cost"][k] for c in cells) for k in spec.source.budget()},
        "evidence_role": spec.EVIDENCE_ROLE, "prior_joint_HRL_and_frequency_specific_gates": "unchanged"}
