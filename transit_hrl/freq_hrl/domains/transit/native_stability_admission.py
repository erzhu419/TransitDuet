"""Training-only admission of fixed candidates using cached stability curves."""

import numpy as np

from freq_hrl.domains.transit import native_prefix_stability as diagnostic

joint = diagnostic.joint
MODES = ("mean_selected", "safe_selected")
FILTERS = ("future_only", "prefix_only", "joint")
NONHARM_TOL = 1e-12
FROZEN = diagnostic.strength.VARIANTS


def control_name(anchor, mode): return f"cost_strength_{anchor}_{mode}"
def variant_name(anchor, mode, strategy): return f"admission_{anchor}_{mode}_{strategy}"


VARIANTS = (*FROZEN, *(variant_name(a, m, f) for a in joint.ANCHORS for m in MODES for f in FILTERS))


def nonharm(curve, mode):
    return all(x["mean_delta_cost"] <= NONHARM_TOL and x["mean_delta_wait_min"] <= NONHARM_TOL
        and (mode != "safe_selected" or x["mean_delta_unsafe"] <= 0.) for x in curve["by_root"].values())


def decide(diagnostic_anchor, mode):
    full = diagnostic_anchor["full"][mode]
    candidate, curve = full["candidate"], full["curve"]
    pooled = candidate not in (None, "zero") and curve["joint_score"] < 0 and (
        mode != "safe_selected" or all(x["mean_delta_unsafe"] <= 0 for x in curve["by_root"].values()))
    future = all(nonharm(h["modes"][mode]["fixed_curve"], mode) for h in diagnostic_anchor["future_halves"])
    prefix = all(nonharm(d["modes"][mode]["fixed_curve"], mode) for d in diagnostic_anchor["deletions"])
    return {"candidate": candidate, "pooled_improvement": pooled, "future_nonharm": future, "prefix_nonharm": prefix,
        "accepted": {"future_only": pooled and future, "prefix_only": pooled and prefix, "joint": pooled and future and prefix}}


def admit_cell(source, stability):
    admission = {}
    for anchor in joint.ANCHORS:
        admission[anchor] = {}
        for mode in MODES:
            gate = decide(stability["anchors"][anchor], mode)
            if gate["candidate"] != source["selection"][anchor][diagnostic.FIELDS[mode]]:
                raise ValueError("Stage234/235 fixed candidates differ")
            admission[anchor][mode] = gate
    scopes, choices, scores, changed, maximum, records = {}, 0, 0, 0, 0., 0
    for scope, rows in source["scopes"].items():
        output = []
        for row in rows:
            decisions = dict(row["decisions"])
            for anchor in joint.ANCHORS:
                for mode in MODES:
                    for strategy in FILTERS:
                        name = control_name(anchor, mode) if admission[anchor][mode]["accepted"][strategy] else "raw"
                        decisions[variant_name(anchor, mode, strategy)] = dict(row["decisions"][name])
            for name in FROZEN:
                before, after = row["decisions"][name], decisions[name]
                changed += int(before["selected"] != after["selected"])
                maximum = max(maximum, abs(before["cost"]-after["cost"]), abs(before["wait"]-after["wait"]))
                choices += 1
                scores += 2
            output.append({**row, "decisions": decisions})
        scopes[scope] = output
        for anchor in joint.ANCHORS:
            for mode in MODES:
                for root, expected in stability["outer"][anchor][mode][scope].items():
                    actual = diagnostic.outer_stats([r for r in output if str(r["root"]) == root], control_name(anchor, mode))
                    if actual != expected: raise ValueError("Stage235 outer physical scores changed")
                    records += 1
    if changed or maximum: raise ValueError("Frozen Stage234 controls changed")
    return {"admission": admission, "scopes": scopes, "source_reproduction": {"candidate_comparisons": 4,
        "choice_comparisons": choices, "score_comparisons": scores, "changed_choices": changed,
        "max_abs_difference": maximum, "outer_metric_records": records}}


def metrics(rows):
    raw = [r["decisions"]["raw"] for r in rows]
    output = {}
    for name in VARIANTS:
        ds = [r["decisions"][name] for r in rows]
        output[name] = {"mean_cost": float(np.mean([d["cost"] for d in ds])),
            "mean_wait_min": float(np.mean([d["wait"] for d in ds])),
            "unsafe_fraction": float(np.mean([d["unsafe"] for d in ds])),
            "changed_fraction": float(np.mean([d["changed"] for d in ds])),
            "switch_fraction_to_raw": float(np.mean([d["selected"] != b["selected"] for d, b in zip(ds, raw)])),
            "mean_delta_cost": float(np.mean([d["cost"]-b["cost"] for d, b in zip(ds, raw)])),
            "mean_delta_wait_min": float(np.mean([d["wait"]-b["wait"] for d, b in zip(ds, raw)])),
            "mean_delta_unsafe": float(np.mean([int(d["unsafe"])-int(b["unsafe"]) for d, b in zip(ds, raw)]))}
    return {"prediction_cases": len(rows), "unique_states": len({joint.binding.learned.state_key(r) for r in rows}),
        "variants": output}
