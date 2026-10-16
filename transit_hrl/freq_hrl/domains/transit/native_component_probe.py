"""Cross-future component-information probes, not deployable controllers."""

import numpy as np

from freq_hrl.domains.transit import native_risk_policy as policy
from freq_hrl.domains.transit import native_shared_risk as weights

CONTROLS = ("raw_prior", "raw_linear_clip", "observed_context_df_clip")
PROBES = ("mc_service_raw", "mc_service_linear9", "mc_service_observed",
    "mc_risk_raw_service", "mc_both")
ARMS = (*CONTROLS, *(f"{h}__{p}" for h in policy.DRAW_GROUPS for p in PROBES))
SLICES = {"all16": slice(None), "first8": slice(0, 8), "last8": slice(8, 16)}


def propose(data, risks, selection_values):
    """Only the assigned bank enters here; evaluation draws are not an input."""
    control, error = policy.decisions(data, risks)
    choices = {a: control[a] for a in CONTROLS}
    total = policy.joint.relative(data["heads"], data["originals"])[..., 0]
    service = total-risks["raw_prior"]
    components = {"service_raw": service,
        **{f"risk_{a}": risks[a] for a in CONTROLS}}
    for h, ds in SLICES.items():
        mean = policy.joint.relative(selection_values[:, ds].mean(axis=1), data["originals"])
        fleet, served = mean[..., 1], mean[..., 2]
        components[f"risk_mc_{h}"] = fleet
        components[f"service_mc_{h}"] = served
        totals = {**{p: served+risks[a] for p, a in zip(PROBES[:3], CONTROLS)},
            "mc_risk_raw_service": fleet+service, "mc_both": fleet+served}
        for p, cost in totals.items():
            choices[f"{h}__{p}"] = policy.joint.binding.policy.selected(
                cost, data["raw_logits"], data["originals"])
    return choices, components, error


def component_errors(data, predictions, values):
    eligible = (data["raw_logits"] <= 0)&(np.arange(6)[None] != data["originals"][:, None])
    w = weights.state_weights(eligible)
    target = policy.joint.relative(values.mean(axis=1), data["originals"])
    return {"cases": int(np.count_nonzero(eligible.sum(axis=1))),
        "sums": {name: float(np.sum(w*(pred-target[..., 1 if name.startswith("risk_") else 2])**2))
            for name, pred in predictions.items()}}


def summarize_errors(rows):
    n = sum(r["cases"] for r in rows)
    return {"eligible_state_exposures": n,
        "MSE": {k: sum(r["sums"][k] for r in rows)/n if n else None for k in rows[0]["sums"]}}


def selection_stability(choices):
    return {"cases": len(choices["raw_prior"]), "counts": {p: {
        f"{a}_vs_{b}": int(np.count_nonzero(choices[f"{a}__{p}"] != choices[f"{b}__{p}"]))
        for a, b in (("first8", "last8"), ("all16", "first8"), ("all16", "last8"))} for p in PROBES}}


def summarize_stability(rows):
    n = sum(r["cases"] for r in rows)
    return {p: {k: sum(r["counts"][p][k] for r in rows)/n for k in rows[0]["counts"][p]} for p in PROBES}
