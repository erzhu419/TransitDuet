"""Frozen risk-only plug-in decisions and paired cached physical outcomes."""

import numpy as np

from freq_hrl.domains.transit import native_joint_advantage as joint

ARMS = ("raw_prior", "raw_bounded_clip", "raw_linear_clip", "action_context_clip",
    "observed_context_clip", "action_context_df_clip", "observed_context_df_clip")
FIELDS = ("delta_cost", "delta_fleet", "delta_service", "delta_wait_min")
DRAW_GROUPS = ("all16", "first8", "last8")


def decisions(data, risks):
    originals = data["originals"]
    raw_risk = joint.relative(data["components"], originals)[..., 1]
    raw_total = joint.relative(data["heads"], originals)[..., 0]
    error = float(np.max(np.abs(risks["raw_prior"]-raw_risk)))
    if error > 1e-12: raise ValueError("Frozen raw risk changed")
    choices = {a: joint.binding.policy.selected(raw_total+(risks[a]-risks["raw_prior"]), data["raw_logits"], originals) for a in ARMS}
    if not np.array_equal(choices["raw_prior"], data["raw_choices"]): raise ValueError("Raw choice reproduction failed")
    for chosen in choices.values():
        if np.any((chosen != originals)&(data["raw_logits"][np.arange(len(chosen)), chosen] > 0)):
            raise ValueError("Risk plug-in changed feasibility")
    return choices, error


def outcomes(values, choices):
    ix = np.arange(len(values));raw = choices["raw_prior"]
    output = {s: {} for s in DRAW_GROUPS}
    for a, chosen in choices.items():
        paired = values[ix, :, chosen]-values[ix, :, raw]
        if np.max(np.abs(paired[..., 0]-paired[..., 1]-paired[..., 2]), initial=0.) > 1e-12:
            raise ValueError("Physical total does not equal fleet plus service")
        for label, ds in (("all16", slice(None)), ("first8", slice(0, 8)), ("last8", slice(8, 16))):
            mean = paired[:, ds].mean(axis=1)
            output[label][a] = {"cases": len(ix), "sums": dict(zip(FIELDS, map(float, mean.sum(axis=0)))),
                "changed_count": int(np.count_nonzero(chosen != raw)),
                "linear9_disagreement_count": int(np.count_nonzero(chosen != choices["raw_linear_clip"]))}
    return output


def summarize(rows):
    n = sum(r["cases"] for r in rows)
    result = {"cases": n, **{f: sum(r["sums"][f] for r in rows)/n for f in FIELDS},
        "changed_rate": sum(r["changed_count"] for r in rows)/n,
        "linear9_disagreement_rate": sum(r["linear9_disagreement_count"] for r in rows)/n}
    if abs(result["delta_cost"]-result["delta_fleet"]-result["delta_service"]) > 1e-12:
        raise ValueError("Aggregate physical-cost identity failed")
    return result
