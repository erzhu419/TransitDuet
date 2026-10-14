"""Validation-block influence on a fixed rank/strength grid, without new policies."""

import numpy as np

from freq_hrl.domains.transit import native_cost_strength as strength

joint = strength.joint
FIELDS = dict(zip(strength.NEW_MODES, ("mse_level", "mean_level", "safe_level")))


def select(curves, ledger, kind):
    return strength.base.select(curves, ledger, kind, levels=tuple(strength.CANDIDATES))


def eligible(curve, mode, candidate):
    if mode == "cost_mse" or candidate in (None, "zero"): return None
    return curve["joint_score"] < 0 and (mode != "safe_selected" or
        all(x["mean_delta_unsafe"] <= 0 for x in curve["by_root"].values()))


def rescore(curves, ledger, kind, full, indices):
    reduced = select([curves[i] for i in indices], [ledger[i] for i in indices], kind)
    return {mode: {"candidate": reduced[field], "changed": reduced[field] != full[field],
        "fixed_eligible": eligible(reduced["curves"][full[field] or "zero"], mode, full[field]),
        "fixed_curve": reduced["curves"][full[field] or "zero"]} for mode, field in FIELDS.items()}


def concentration(prefixes, root, mode, field):
    pieces = [(p["group"], p["modes"][mode][field]) for p in prefixes if str(p["group"][0]) == root]
    benefits = sorted((max(-value, 0.) for _, value in pieces), reverse=True)
    total = sum(benefits)
    return {"prefixes": len(pieces), "improving_prefixes": sum(v < 0 for _, v in pieces),
        "worsening_prefixes": sum(v > 0 for _, v in pieces), "net_sum": sum(v for _, v in pieces),
        "benefit_sum": total, "top1_benefit_share": benefits[0]/total if total else None,
        "top3_benefit_share": sum(benefits[:3])/total if total else None}


def diagnose(curves, ledger, kind, expected):
    full = select(curves, ledger, kind)
    if full != expected: raise ValueError("Stage234 inner selection changed during prefix replay")
    groups = sorted({tuple(f["held_group"]) for f in ledger})
    prefixes, deletions = [], []
    for group in groups:
        held = [i for i, f in enumerate(ledger) if tuple(f["held_group"]) == group]
        modes = {}
        for mode, field in FIELDS.items():
            pieces = [curves[i][full[field] or "zero"] for i in held]
            xs = [p["by_root"][str(group[0])] for p in pieces]
            modes[mode] = {"cases": sum(x["cases"] for x in xs),
                **{k: sum(x[k] for x in xs) for k in ("cost_sum", "wait_sum_min", "unsafe_delta_count")},
                "changed_count": sum(p["changed_count"] for p in pieces)}
        prefixes.append({"group": list(group), "modes": modes})
        keep = [i for i, f in enumerate(ledger) if tuple(f["held_group"]) != group]
        deletions.append({"removed_group": list(group), "removed_blocks": len(held),
            "modes": rescore(curves, ledger, kind, full, keep)})
    halves = []
    for direction in (0, 1):
        indices = [i for i, f in enumerate(ledger) if f["direction"] == direction]
        f = ledger[indices[0]]
        halves.append({"direction": direction, "training_draws": f["training_draws"],
            "validation_draws": f["validation_draws"], "modes": rescore(curves, ledger, kind, full, indices)})
    roots = sorted({str(g[0]) for g in groups})
    return {"full": {mode: {"candidate": full[field], "curve": full["curves"][full[field] or "zero"]}
            for mode, field in FIELDS.items()}, "prefixes": prefixes, "deletions": deletions, "future_halves": halves,
        "concentration": {mode: {root: {name: concentration(prefixes, root, mode, field)
            for name, field in (("cost", "cost_sum"), ("wait", "wait_sum_min"))} for root in roots} for mode in FIELDS}}


def outer_stats(rows, name):
    raw = [r["decisions"]["raw"] for r in rows]
    chosen = [r["decisions"][name] for r in rows]
    return {"cases": len(rows),
        "mean_delta_cost": float(np.mean([c["cost"]-b["cost"] for c, b in zip(chosen, raw)])),
        "mean_delta_wait_min": float(np.mean([c["wait"]-b["wait"] for c, b in zip(chosen, raw)])),
        "mean_delta_unsafe": float(np.mean([int(c["unsafe"])-int(b["unsafe"]) for c, b in zip(chosen, raw)])),
        "switch_count": sum(c["selected"] != b["selected"] for c, b in zip(chosen, raw))}


def weighted(pieces):
    n = sum(p["cases"] for p in pieces)
    return {"cases": n, **{k: sum(p[k]*p["cases"] for p in pieces)/n
        for k in ("mean_delta_cost", "mean_delta_wait_min", "mean_delta_unsafe")}}


def summarize_anchor(cells, kind):
    output = {}
    for mode in FIELDS:
        ds = [c["anchors"][kind] for c in cells]
        roots = sorted({r for d in ds for r in d["full"][mode]["curve"]["by_root"]})
        deletes = [x["modes"][mode] for d in ds for x in d["deletions"]]
        halves = [x["modes"][mode] for d in ds for x in d["future_halves"]]
        full = [d["full"][mode] for d in ds]
        output[mode] = {"candidates": [f["candidate"] for f in full],
            "validation_switch_count": sum(f["curve"]["changed_count"] for f in full),
            "deletion_checks": len(deletes), "deletion_choice_changes": sum(x["changed"] for x in deletes),
            "deletion_eligibility_checks": sum(x["fixed_eligible"] is not None for x in deletes),
            "deletion_eligibility_losses": sum(x["fixed_eligible"] is False for x in deletes),
            "future_half_checks": len(halves), "future_half_choice_changes": sum(x["changed"] for x in halves),
            "future_half_eligibility_checks": sum(x["fixed_eligible"] is not None for x in halves),
            "future_half_eligibility_losses": sum(x["fixed_eligible"] is False for x in halves),
            "inner_by_root": {r: weighted([f["curve"]["by_root"][r] for f in full]) for r in roots},
            "outer_scopes": {scope: {r: {**weighted([c["outer"][kind][mode][scope][r] for c in cells]),
                "switch_count": sum(c["outer"][kind][mode][scope][r]["switch_count"] for c in cells)}
                for r in roots} for scope in cells[0]["outer"][kind][mode]},
            "inner_by_cell": [f["curve"]["by_root"] for f in full],
            "outer_scopes_by_cell": [c["outer"][kind][mode] for c in cells],
            "future_halves_by_cell": [[{"direction": h["direction"],
                "candidate": h["modes"][mode]["candidate"], "fixed_eligible": h["modes"][mode]["fixed_eligible"],
                "fixed_by_root": h["modes"][mode]["fixed_curve"]["by_root"]} for h in d["future_halves"]] for d in ds],
            "concentration_by_cell": [{r: d["concentration"][mode][r] for r in roots} for d in ds]}
    return output
