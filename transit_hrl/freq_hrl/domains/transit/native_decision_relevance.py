"""Frozen wait-gate diagnostics on causal cost-ranked candidate subsets."""

import numpy as np
from scipy.special import expit

from freq_hrl.domains.transit import native_refit_binding as binding

STRATA = ("all", "contender", "front", "excluded")


def masks(data):
    all_candidates = np.arange(6)[None, :] != data["originals"][:, None]
    contenders = all_candidates & (data["costs"] < 0)
    front_choices = binding.policy.selected(data["costs"], np.zeros_like(data["costs"]), data["originals"])
    front = np.zeros_like(all_candidates)
    front[np.arange(len(front)), front_choices] = True
    return {"all": all_candidates, "contender": contenders, "front": front & contenders,
        "excluded": all_candidates & ~contenders}


def logits(data, calibration):
    output = {"glm": data["offset"], **{f"raw_{v}": data[v] for v in binding.WAIT_VARIANTS}}
    for mode in ("risk", "oof", "final", "joint"):
        for v in binding.WAIT_VARIANTS:
            output[f"{mode}_{v}"] = binding.policy.blended(data["offset"], data[v], calibration[mode][v]["alpha"])
    return output


def truth(rows, panels, indices, prediction):
    _, masses = binding.learned.half_targets(rows, panels, indices)
    return {**prediction,
        "values": np.stack([panels[binding.learned.state_key(r)]["values"][list(indices)].mean(axis=0) for r in rows]),
        **{name: np.asarray([masses[binding.learned.state_key(r)][field] for r in rows])
            for name, field in (("positive", "positive_wait_min"), ("negative", "negative_wait_min"))}}


def evaluate(data, other, calibration):
    n = len(data["originals"])
    subsets, heads = masks(data), logits(data, calibration)
    wait, other_wait = data["values"][..., 3], other["values"][..., 3]
    flipped = (wait > 0) != (other_wait > 0)
    mass = {g: data["positive"]*np.logaddexp(0., -z)+data["negative"]*np.logaddexp(0., z) for g, z in heads.items()}
    gradients = {}
    for v in binding.WAIT_VARIANTS:
        residual = data[v]-data["offset"]
        gradients[v] = {name: ((data["positive"]+data["negative"])*expit(binding.policy.blended(data["offset"], data[v], alpha))
            -data["positive"])*residual for name, alpha in (("zero", 0.), ("one", 1.), ("uniform_risk", calibration["risk"][v]["alpha"]))}
    strata = {}
    for name, mask in subsets.items():
        strata[name] = {"candidate_cases": int(mask.sum()), "state_cases_with_members": int(mask.any(axis=1).sum()),
            "flipped_wait_sign_count": int((mask & flipped).sum()),
            "heads": {}, "gradients": {v: {k: float(x[mask].sum()) for k, x in g.items()} for v, g in gradients.items()}}
        for g, z in heads.items():
            false_safe, false_unsafe = mask & (z <= 0) & (wait > 0), mask & (z > 0) & (wait <= 0)
            strata[name]["heads"][g] = {"loss_sum_min": float(mass[g][mask].sum()),
                "false_safe_count": int(false_safe.sum()), "false_unsafe_count": int(false_unsafe.sum()),
                "stable_false_safe_count": int((false_safe & ~flipped).sum()),
                "stable_false_unsafe_count": int((false_unsafe & ~flipped).sum()),
                "false_safe_harm_sum_min": float(wait[false_safe].sum()),
                "false_unsafe_missed_sum_min": float(-wait[false_unsafe].sum())}
    oracle_choice = binding.policy.selected(data["costs"], wait, data["originals"])
    oracle = data["values"][np.arange(n), oracle_choice]
    choices = {g: binding.policy.selected(data["costs"], z, data["originals"]) for g, z in heads.items()}
    decisions, switches = {}, {}
    for g, choice in choices.items():
        physical = data["values"][np.arange(n), choice]
        decisions[g] = {"cost_sum": float(physical[:, 0].sum()), "wait_sum_min": float(physical[:, 3].sum()),
            "changed_count": int((choice != data["originals"]).sum()), "unsafe_count": int((physical[:, 3] > 0).sum()),
            "oracle_choice_mismatch_count": int((choice != oracle_choice).sum())}
    for mode in ("risk", "oof", "final", "joint"):
        for v in binding.WAIT_VARIANTS:
            g, raw = f"{mode}_{v}", choices[f"raw_{v}"]
            chosen, changed = choices[g], choices[g] != raw
            delta = data["values"][np.arange(n), chosen]-data["values"][np.arange(n), raw]
            alternate = other["values"][np.arange(n), chosen]-other["values"][np.arange(n), raw]
            worse, other_worse = delta[:, 3] > 0, alternate[:, 3] > 0
            switches[g] = {"changed_count": int(changed.sum()), "delta_cost_sum": float(delta[:, 0].sum()),
                "delta_wait_sum_min": float(delta[:, 3].sum()), "other_delta_cost_sum": float(alternate[:, 0].sum()),
                "other_delta_wait_sum_min": float(alternate[:, 3].sum()),
                "wait_worse_both_count": int((changed & worse & other_worse).sum()),
                "wait_worse_current_only_count": int((changed & worse & ~other_worse).sum()),
                "wait_worse_other_only_count": int((changed & ~worse & other_worse).sum()),
                "wait_worse_neither_count": int((changed & ~worse & ~other_worse).sum())}
    return {"state_cases": n, "strata": strata, "decisions": decisions, "switches": switches,
        "oracle": {"cost_sum": float(oracle[:, 0].sum()), "wait_sum_min": float(oracle[:, 3].sum()),
            "changed_count": int((oracle_choice != data["originals"]).sum())}}


def combine(items):
    first = items[0]
    return {k: combine([x[k] for x in items]) if isinstance(v, dict) else sum(x[k] for x in items) for k, v in first.items()}


def summarized(totals):
    n, all_count = totals["state_cases"], totals["strata"]["all"]["candidate_cases"]
    mean = lambda x, count: x/count if count else None
    output = {"state_cases": n, "strata": {}, "decisions": {}, "switches": {},
        "oracle": {"mean_cost": totals["oracle"]["cost_sum"]/n, "mean_wait_min": totals["oracle"]["wait_sum_min"]/n,
            "changed_fraction": totals["oracle"]["changed_count"]/n}}
    for name, s in totals["strata"].items():
        count = s["candidate_cases"]
        output["strata"][name] = {"candidate_cases": count, "candidate_fraction": count/all_count,
            "state_cases_with_members": s["state_cases_with_members"], "flipped_wait_sign_fraction": mean(s["flipped_wait_sign_count"], count),
            "heads": {g: {"physical_mass_loss_min": mean(x["loss_sum_min"], count),
                **{k: v for k, v in x.items() if k != "loss_sum_min"}, "loss_contribution_all_min": x["loss_sum_min"]/all_count}
                for g, x in s["heads"].items()},
            "gradients": {v: {k: {"mean": mean(x, count), "contribution_all": x/all_count} for k, x in g.items()}
                for v, g in s["gradients"].items()}}
    for g, d in totals["decisions"].items():
        output["decisions"][g] = {"mean_cost": d["cost_sum"]/n, "mean_wait_min": d["wait_sum_min"]/n,
            "changed_fraction": d["changed_count"]/n, "unsafe_fraction": d["unsafe_count"]/n,
            "oracle_choice_mismatch_fraction": d["oracle_choice_mismatch_count"]/n}
    for g, s in totals["switches"].items():
        output["switches"][g] = {**s, "delta_cost_per_state": s["delta_cost_sum"]/n,
            "delta_wait_per_state_min": s["delta_wait_sum_min"]/n,
            "delta_cost_per_switch": mean(s["delta_cost_sum"], s["changed_count"]),
            "delta_wait_per_switch_min": mean(s["delta_wait_sum_min"], s["changed_count"])}
    return output
