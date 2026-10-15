"""Evaluate cached OOF actions under new futures without fitting or selection."""

import numpy as np

from freq_hrl.domains.transit import native_objective_oof as previous

joint, VARIANTS = previous.joint, previous.VARIANTS
SCOPES = ("old_oof2", "old_unused4", "fresh_half0", "fresh_half1", "fresh8")


def state_key(row):
    return tuple(row[k] for k in ("root", "scene_seed", "probe_index"))


def outcomes(block, values):
    """Values are draw x state x action x (cost, fleet, nonfleet, wait)."""
    values = np.asarray(values)
    n = len(block["states"])
    if values.ndim != 4 or values.shape[1:] != (n, 6, 4):
        raise ValueError("Changed paired future panel dimensions")
    ix = np.arange(n)
    raw = np.asarray(block["choices"]["raw"])
    mean = values.mean(axis=0)
    base, base_draws = mean[ix, raw], values[:, ix, raw]
    oracle = np.where(block["feasible"], mean[..., 0], np.inf).min(axis=1)
    result = {}
    for name, choices in block["choices"].items():
        chosen = np.asarray(choices)
        actual, draws = mean[ix, chosen], values[:, ix, chosen]
        result[name] = {
            "delta_cost": float((actual[:, 0]-base[:, 0]).mean()),
            "delta_wait_min": float((actual[:, 3]-base[:, 3]).mean()),
            "delta_unsafe": float(((actual[:, 3] > 0).astype(int)-(base[:, 3] > 0).astype(int)).mean()),
            "delta_draw_wait_harm_rate": float(((draws[..., 3] > 0).astype(int)-(base_draws[..., 3] > 0).astype(int)).mean()),
            "gate_cost_regret": float((actual[:, 0]-oracle).mean()),
            "switch_raw": float((chosen != raw).mean())}
    return {"cases": n, "variants": result}


def seal_cell(train, panels, refit_source, references, source, models):
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i)
        for i, t in enumerate(refit_source["trials"]) if t["removed_group"] and t["half"] is not None}
    blocks, maximum, comparisons = [], 0., 0
    for old in source["blocks"]:
        group, half = tuple(old["held_group"]), old["half"]
        rows = [r for r in train if (r["root"], r["scene_seed"]) == group]
        index = lookup[group, half]
        data = joint.causal_data(rows, references[index])
        predictions = previous.predict(data, models[index], source["selection"])
        block = {"held_group": list(group), "half": half,
            "states": [{k: r[k] for k in ("root", "scene_seed", "probe_index")} for r in rows],
            "choices": {v: p["chosen"].tolist() for v, p in predictions.items()},
            "feasible": (data["raw_logits"] <= 0).tolist()}
        values = np.stack([panels[state_key(r)]["values"] for r in rows], axis=1)
        block["old_oof2"] = outcomes(block, values[old["validation_draws"]])
        block["old_unused4"] = outcomes(block, values[source["evaluation_draws"]])
        for anchor in joint.ANCHORS:
            for family in previous.FAMILIES:
                for mode in previous.FILTERS:
                    choice = source["selection"][anchor][family][mode]
                    old_stats = old["curves"][anchor][choice["objective"] or "all_mse"][choice["candidate"] or "zero"]
                    expected = old_stats["by_root"][str(group[0])]
                    actual = block["old_oof2"]["variants"][f"{family}_{anchor}_{mode}"]
                    maximum = max(maximum, *(abs(actual[f]*len(rows)-expected[g]) for f, g in (
                        ("delta_cost", "cost_sum"), ("delta_wait_min", "wait_sum_min"), ("delta_unsafe", "unsafe_delta_count"))),
                        abs(actual["switch_raw"]*len(rows)-old_stats["changed_count"]))
                    comparisons += 1
        blocks.append(block)
    if maximum > previous.TOL:
        raise ValueError("Sealed actions did not reproduce Stage239 selected OOF outcomes")
    return {"selection": source["selection"], "blocks": blocks,
        "reproduction": {"records": comparisons, "max_abs_difference": maximum}}


def evaluate_cell(sealed, panels):
    blocks = []
    for frozen in sealed["blocks"]:
        values = np.stack([panels[state_key(r)] for r in frozen["states"]], axis=1)
        blocks.append({"held_group": frozen["held_group"], "half": frozen["half"],
            "scopes": {"old_oof2": frozen["old_oof2"], "old_unused4": frozen["old_unused4"],
                **{name: outcomes(frozen, values[ds]) for name, ds in (
                    ("fresh_half0", slice(0, 4)), ("fresh_half1", slice(4, 8)), ("fresh8", slice(0, 8)))}}})
    return {"selection": sealed["selection"], "blocks": blocks}


def aggregate(cells, scopes=SCOPES):
    blocks = [b for c in cells for b in c["blocks"]]
    def average(xs):
        return {"matched_blocks": len(xs), "mean_cases": float(np.mean([x["cases"] for x in xs])),
            "variants": {v: {f: float(np.mean([x["variants"][v][f] for x in xs]))
                for f in xs[0]["variants"][v]} for v in VARIANTS}}
    return {s: {"all": average([b["scopes"][s] for b in blocks]),
        "by_root": {str(r): average([b["scopes"][s] for b in blocks if b["held_group"][0] == r])
            for r in sorted({b["held_group"][0] for b in blocks})}} for s in scopes}


def prefix_effects(cell):
    output = {}
    for anchor in joint.ANCHORS:
        for family in previous.FAMILIES:
            for mode in previous.FILTERS:
                if cell["selection"][anchor][family][mode]["candidate"] is None:
                    continue
                name = f"{family}_{anchor}_{mode}"
                output[name] = {}
                for scope in SCOPES:
                    groups = sorted({tuple(b["held_group"]) for b in cell["blocks"]})
                    xs = [{"group": list(g), **{f: float(np.mean([b["scopes"][scope]["variants"][name][f]
                        for b in cell["blocks"] if tuple(b["held_group"]) == g]))
                        for f in ("delta_cost", "delta_wait_min", "delta_unsafe")}} for g in groups]
                    concentration = {}
                    for root in sorted({g[0] for g in groups}):
                        concentration[str(root)] = {}
                        for f in ("delta_cost", "delta_wait_min"):
                            values = [x[f] for x in xs if x["group"][0] == root]
                            benefits = [max(-x, 0.) for x in values]
                            concentration[str(root)][f] = {"improving_prefixes": sum(x < 0 for x in values),
                                "worsening_prefixes": sum(x > 0 for x in values),
                                "top1_benefit_share": max(benefits)/sum(benefits) if sum(benefits) else None}
                    output[name][scope] = {"prefixes": xs, "concentration": concentration}
    return output


def conditional_mc(cells, panels, *, variants=VARIANTS):
    """Fixed-prefix sampling SE; shared state/future outcomes count only once."""
    keys = sorted({state_key(r) for c in cells for b in c["blocks"] for r in b["states"]})
    index = {k: i for i, k in enumerate(keys)}
    draws = panels[keys[0]]["values"].shape[0]
    output = {}
    for name in variants:
        effects = np.zeros((len(keys), draws, 4))
        changed, fleet_events = set(), set()
        for cell in cells:
            for block in cell["blocks"]:
                weight = 1./(len(cells)*len(cell["blocks"])*len(block["states"]))
                for row, chosen, raw in zip(block["states"], block["choices"][name], block["choices"]["raw"]):
                    key = state_key(row)
                    values = panels[key]["values"]
                    effects[index[key]] += weight*(values[:, chosen]-values[:, raw])
                    if chosen != raw: changed.add(key)
                    fleet_events.update((key, int(d)) for d in np.flatnonzero(values[:, chosen, 1] != values[:, raw, 1]))
        means = effects.mean(axis=1).sum(axis=0)
        se = np.sqrt(effects.var(axis=1, ddof=1).sum(axis=0)/draws)
        output[name] = {"changed_states": len(changed), "sampled_fleet_event_state_draws": len(fleet_events),
            "fields": {f: {"mean": float(means[i]), "conditional_mc_se": float(se[i]),
                "mean_over_mc_se": float(means[i]/se[i]) if se[i] else None}
                for i, f in enumerate(("cost", "fleet", "nonfleet", "wait_min"))}}
    return {"fixed_states": len(keys), "draws_per_state": draws,
        "state_exposures": sum(len(b["states"]) for c in cells for b in c["blocks"]), "variants": output,
        "boundary": "conditional_innovation_sampling_SE_not_training_seed_CI_or_unobserved_tail_guarantee"}
