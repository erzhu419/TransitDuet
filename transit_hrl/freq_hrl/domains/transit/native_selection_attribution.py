"""Account for admission and candidate-key changes without selecting a policy."""

from freq_hrl.domains.transit import native_objective_oof as rule

FIELDS = ("delta_cost", "delta_wait_min", "delta_risk", "delta_service", "delta_unsafe", "changed_rate")
CATEGORIES = ("unchanged", "removed", "added", "rekey")


def transition(before, after):
    before, after = before or "zero", after or "zero"
    if before == after: return "unchanged"
    if after == "zero": return "removed"
    if before == "zero": return "added"
    return "rekey"


def gate(summary, mode):
    """Explain the existing selector: strict worst-coordinate gain plus nonharm."""
    blocks = []
    if summary["pooled"]["score"] >= -rule.TOL: blocks.append("strict_gain")
    curves = [("pooled", summary["pooled"])]
    if mode == "half_safe": curves += [(f"half{h}", curve) for h, curve in enumerate(summary["halves"])]
    for scope, curve in curves:
        for root, stats in curve["by_root"].items():
            for field in ("delta_cost", "delta_wait_min", "delta_unsafe"):
                if stats[field] > (0. if field == "delta_unsafe" else rule.TOL):
                    blocks.append(f"{scope}:{root}:{field}")
    return {"eligible": not blocks, "blocks": blocks, "strict_gain_only": blocks == ["strict_gain"],
        "nonharm_some_gain": blocks == ["strict_gain"] and any(x[f] < -rule.TOL
            for x in summary["pooled"]["by_root"].values() for f in ("scaled_cost", "scaled_wait"))}


def gate_counts(summaries, selected, mode):
    statuses = [gate(value, mode) for candidate, value in summaries.items() if candidate != "zero"]
    count = sum(s["eligible"] for s in statuses)
    if (selected is None) != (count == 0) or (selected is not None and not gate(summaries[selected], mode)["eligible"]):
        raise ValueError("Gate explanation disagrees with the cached selection")
    return {"candidate_exposures": len(statuses), "eligible": count,
        "strict_gain_only": sum(s["strict_gain_only"] for s in statuses),
        "nonharm_some_gain": sum(s["nonharm_some_gain"] for s in statuses)}


def account(pieces):
    """Contributions use the entire bank-cell denominator, not each category's."""
    n = len(pieces)
    def mean(xs, side):
        return {f: sum(x[side][f] for x in xs)/len(xs) for f in FIELDS} if xs else None
    before, after = mean(pieces, "before"), mean(pieces, "after")
    total = {f: after[f]-before[f] for f in FIELDS}
    if any(abs(p["after"][f]-p["before"][f]) > 1e-12 for p in pieces if p["category"] == "unchanged" for f in FIELDS):
        raise ValueError("An unchanged key changed its frozen full-model outcomes")
    categories = {}
    for category in CATEGORIES:
        xs = [p for p in pieces if p["category"] == category]
        contribution = {f: sum(x["after"][f]-x["before"][f] for x in xs)/n for f in FIELDS}
        patterns = sorted({"|".join(x["old_key_under_new_gate"]["blocks"]) or "eligible" for x in xs if x["old_key_under_new_gate"] is not None})
        categories[category] = {"bank_cell_exposures": len(xs), "fraction": len(xs)/n,
            "before_conditional_mean": mean(xs, "before"), "after_conditional_mean": mean(xs, "after"),
            "contribution": contribution,
            "old_key_new_gate_patterns": {s: sum(("|".join(x["old_key_under_new_gate"]["blocks"]) or "eligible") == s
                for x in xs if x["old_key_under_new_gate"] is not None) for s in patterns},
            "strict_gain_only": sum(x["old_key_under_new_gate"]["strict_gain_only"] for x in xs if x["old_key_under_new_gate"] is not None),
            "nonharm_some_gain": sum(x["old_key_under_new_gate"]["nonharm_some_gain"] for x in xs if x["old_key_under_new_gate"] is not None),
            "rekey_rank": sum(x["rekey_kind"] == "rank" for x in xs),
            "rekey_strength": sum(x["rekey_kind"] == "strength" for x in xs)}
    maximum = max(abs(sum(c["contribution"][f] for c in categories.values())-total[f]) for f in FIELDS)
    if maximum > 1e-12: raise ValueError("Selection attribution does not sum to the paired change")
    for side in (before, after, total, *(c["contribution"] for c in categories.values())):
        if abs(side["delta_cost"]-side["delta_risk"]-side["delta_service"]) > 1e-12:
            raise ValueError("Selection attribution changed cost-component accounting")
    return {"bank_cell_exposures": n, "before": before, "after": after, "change": total, "categories": categories,
        "gate_candidates": {bundle: {f: sum(p["gate_candidates"][bundle][f] for p in pieces)
            for f in pieces[0]["gate_candidates"][bundle]} for bundle in ("prefix_refit", "fixed_bundle")},
        "decomposition_max_abs_difference": maximum}
