"""Fixed selected-pair margin accounting, not a new selector or loss."""

import numpy as np

from freq_hrl.domains.transit import native_risk_policy as policy
from freq_hrl.domains.transit.native_signed_risk import ZERO

PAIRS = {"observed_vs_linear9": ("observed_context_df_clip", "raw_linear_clip"),
    "observed_vs_action_df": ("observed_context_df_clip", "action_context_df_clip"),
    "observed_vs_raw": ("observed_context_df_clip", "raw_prior"),
    "observed_vs_bounded": ("observed_context_df_clip", "raw_bounded_clip"),
    "linear9_vs_raw": ("raw_linear_clip", "raw_prior")}
MEANS = ("predicted_total", "predicted_service", "predicted_risk", "control_predicted_risk",
    "actual_cost", "actual_fleet", "actual_service", "actual_wait_min", "fleet_error", "service_error")
ERRORS = ("risk_MSE_proposed", "risk_MSE_control", "risk_MC_variance_of_mean")


def predictions(data, risks, choices):
    raw_total = policy.joint.relative(data["heads"], data["originals"])[..., 0]
    service = raw_total-risks["raw_prior"]
    ix = np.arange(len(raw_total));out = {}
    for name, (a, b) in PAIRS.items():
        ca, cb = np.asarray(choices[a]), np.asarray(choices[b])
        margin = lambda values: values[ix, ca]-values[ix, cb]
        s, r, c = margin(service), margin(risks[a]), margin(risks[b])
        if np.max(s+r, initial=0.) > 1e-12 or np.min(s+c, initial=0.) < -1e-12:
            raise ValueError("Cached pair does not match its frozen controller minima")
        out[name] = {"proposed": ca, "control": cb, "different": ca != cb,
            "predicted_total": s+r, "predicted_service": s, "predicted_risk": r, "control_predicted_risk": c}
    return out


def audit(values, predicted):
    ix = np.arange(len(values));output = {h: {} for h in policy.DRAW_GROUPS}
    maximum = 0.
    for name, p in predicted.items():
        paired = values[ix, :, p["proposed"]]-values[ix, :, p["control"]]
        for h, ds in (("all16", slice(None)), ("first8", slice(0, 8)), ("last8", slice(8, 16))):
            v = paired[:, ds];mean = v.mean(axis=1)
            parts = {f: mean[:, j] for j, f in enumerate(("actual_cost", "actual_fleet", "actual_service", "actual_wait_min"))}
            parts.update({f: p[f] for f in MEANS[:4]})
            parts["fleet_error"] = parts["actual_fleet"]-p["predicted_risk"]
            parts["service_error"] = parts["actual_service"]-p["predicted_service"]
            error = max(float(np.max(np.abs(parts["actual_cost"]-parts["actual_fleet"]-parts["actual_service"]), initial=0.)),
                float(np.max(np.abs(parts["actual_cost"]-p["predicted_total"]-parts["fleet_error"]-parts["service_error"]), initial=0.)))
            maximum = max(maximum, error)
            output[h][name] = {"cases": len(ix), "disagreement_cases": int(p["different"].sum()),
                "disagreement_draws": int(p["different"].sum()*v.shape[1]),
                "fleet_add_draws": int(np.count_nonzero(v[..., 1] > ZERO)), "fleet_save_draws": int(np.count_nonzero(v[..., 1] < -ZERO)),
                "sums": {**{f: float(parts[f].sum()) for f in MEANS},
                    "risk_MSE_proposed": float(np.sum(parts["fleet_error"]**2)),
                    "risk_MSE_control": float(np.sum((parts["actual_fleet"]-p["control_predicted_risk"])**2)),
                    "risk_MC_variance_of_mean": float(np.sum(v[..., 1].var(axis=1, ddof=1)/v.shape[1]))}}
    if maximum > 1e-12: raise ValueError("Selected-pair component decomposition failed")
    return output, maximum


def summarize(rows):
    n = sum(r["cases"] for r in rows);different = sum(r["disagreement_cases"] for r in rows)
    means = {f: sum(r["sums"][f] for r in rows)/n for f in MEANS}
    result = {"cases": n, "disagreement_cases": different, "disagreement_rate": different/n, **means,
        **{f: sum(r["sums"][f] for r in rows)/different if different else None for f in ERRORS},
        **{f: sum(r[f] for r in rows) for f in ("disagreement_draws", "fleet_add_draws", "fleet_save_draws")}}
    if abs(means["actual_cost"]-means["predicted_total"]-means["fleet_error"]-means["service_error"]) > 1e-12:
        raise ValueError("Aggregate margin decomposition failed")
    return result
