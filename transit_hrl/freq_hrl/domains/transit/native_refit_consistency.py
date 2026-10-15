"""Frozen OOF candidates under paired residual and reference refits."""

import numpy as np

from freq_hrl.domains.transit import native_objective_oof as previous

joint, strength = previous.joint, previous.strength
ENDPOINTS = ("54_2_cached", "60_2_fixed54ref", "60_2_native", "60_4_fixed60ref", "60_4_native")
SCOPES = ("prefix_probe", *previous.SCOPES)
VARIANTS = previous.VARIANTS


def prefix_diagnostics(source):
    groups = sorted({tuple(b["held_group"]) for b in source["blocks"]})
    output = {}
    for anchor in joint.ANCHORS:
        roots = sorted({str(g[0]) for g in groups})
        scales = {r: np.mean([b["scales"][anchor] for b in source["blocks"] if str(b["held_group"][0]) == r], axis=0) for r in roots}
        for family in previous.FAMILIES:
            for mode in previous.FILTERS:
                s = source["selection"][anchor][family][mode]
                name = f"{family}_{anchor}_{mode}"
                if s["candidate"] is None:
                    output[name] = {"admitted": False, "deletions": [], "concentration": {}}
                    continue
                def summary(blocks):
                    pieces = [b["curves"][anchor][s["objective"]][s["candidate"]] for b in blocks]
                    return previous.curve_summary(pieces, scales)
                deletions = []
                for group in groups:
                    keep = [b for b in source["blocks"] if tuple(b["held_group"]) != group]
                    pooled = summary(keep)
                    halves = [summary([b for b in keep if b["half"] == h]) for h in (0, 1)]
                    eligible = pooled["score"] < -previous.TOL and previous.nonharm(pooled) and (
                        mode != "half_safe" or all(previous.nonharm(h) for h in halves))
                    deletions.append({"removed_group": list(group), "eligible": eligible, "pooled": pooled, "halves": halves})
                concentration = {}
                for root in roots:
                    concentration[root] = {}
                    for field in ("cost_sum", "wait_sum_min"):
                        xs = [sum(b["curves"][anchor][s["objective"]][s["candidate"]]["by_root"][root][field]
                            for b in source["blocks"] if tuple(b["held_group"]) == g) for g in groups if str(g[0]) == root]
                        benefits = sorted((max(-x, 0.) for x in xs), reverse=True)
                        total = sum(benefits)
                        concentration[root][field] = {"prefixes": len(xs), "improving_prefixes": sum(x < 0 for x in xs),
                            "worsening_prefixes": sum(x > 0 for x in xs), "net_sum": sum(xs),
                            "top1_benefit_share": benefits[0]/total if total else None, "benefit_sum": total}
                output[name] = {"admitted": True, "deletions": deletions, "concentration": concentration}
    return output


def fit_bundle(rows, data, values, paths=None):
    models, fits, maximum = {}, {}, 0.
    for anchor in joint.ANCHORS:
        cached = paths[anchor] if paths else strength.lowrank.fit_path(rows, data, values, anchor)[0]
        models[anchor], fits[anchor], difference = previous.fit_paths(data, values, cached)
        maximum = max(maximum, difference)
    return models, fits, maximum


def metrics(data, common, values, predictions, initial, preceding):
    out = previous.base.previous.metrics(data, common, values, predictions)
    groups = {"all": np.arange(len(values)), **{str(r): np.flatnonzero(data["roots"] == r) for r in set(data["roots"])}}
    for root, indices in groups.items():
        piece = out["all"] if root == "all" else out["by_root"][root]
        for name, p in predictions.items():
            stats = piece["variants"][name]
            stats["score_mse_against_cost"] = stats.pop("cost_mse")
            stats["switch_initial"] = float(np.mean(p["chosen"][indices] != initial[name]["chosen"][indices]))
            stats["switch_preceding"] = float(np.mean(p["chosen"][indices] != preceding[name]["chosen"][indices]))
    return out


def fit_cell(train, held, panels, reference, refits, refit_source, source, cached):
    a, b = tuple(source["training_draws"]), tuple(source["evaluation_draws"])
    lookup = {(tuple(t["removed_group"]) if t["removed_group"] else None, t["half"]): str(i)
        for i, t in enumerate(refit_source["trials"])}
    full_data = {n: joint.causal_data(rs, reference) for n, rs in (("train", train), ("held", held))}
    values = {(n, ds): joint.physical_values(rs, panels, ds) for n, rs in (("train", train), ("held", held)) for ds in (a, b, a[:2], a[2:])}
    half_paths, half_data, half_fits = {}, {}, {}
    models, maximum = {}, 0.
    for half in (0, 1):
        index = lookup[None, half]
        ref = refits["references"][index]
        data = {n: joint.causal_data(rs, ref) for n, rs in (("train", train), ("held", held))}
        native, fits, diff = fit_bundle(train, data["train"], values["train", a[2*half:2*half+2]], refits["paths"][index]["changing_native_x"])
        fixed, fixed_fits, fixed_diff = fit_bundle(train, data["train"], values["train", a])
        half_paths[half] = {ENDPOINTS[2]: native, ENDPOINTS[3]: fixed}
        half_data[half] = data
        half_fits[str(half)] = {ENDPOINTS[2]: fits, ENDPOINTS[3]: fixed_fits}
        models[f"half_{half}"] = half_paths[half]
        maximum = max(maximum, diff, fixed_diff)
    blocks, probe_max, full_max, probe_count, full_count = [], 0., 0., 0, 0
    for group, _, validation in joint.binding.policy.prefix_folds(train):
        for half in (0, 1):
            index = lookup[tuple(group), half]
            ref = refits["references"][index]
            data = {n: joint.causal_data(rs, ref) for n, rs in (("train", train), ("held", held))}
            expanded, fits, difference = fit_bundle(train, data["train"], values["train", a[2*half:2*half+2]])
            maximum = max(maximum, difference)
            models[index] = expanded
            paths = {ENDPOINTS[0]: cached[index], ENDPOINTS[1]: expanded, **half_paths[half], ENDPOINTS[4]: cached["full"]}
            datasets = {ENDPOINTS[0]: data, ENDPOINTS[1]: data, ENDPOINTS[2]: half_data[half], ENDPOINTS[3]: half_data[half], ENDPOINTS[4]: full_data}
            refs = {ENDPOINTS[0]: ref, ENDPOINTS[1]: ref, ENDPOINTS[2]: refits["references"][lookup[None, half]],
                ENDPOINTS[3]: refits["references"][lookup[None, half]], ENDPOINTS[4]: reference}
            probe_rows = [train[i] for i in validation]
            probe_values = joint.physical_values(probe_rows, panels, a[2*(1-half):2*(1-half)+2])
            probe = {e: joint.causal_data(probe_rows, refs[e]) for e in ENDPOINTS}
            record = {"held_group": list(group), "half": half, "expanded_fits": fits, "endpoints": {e: {} for e in ENDPOINTS}}
            old_block = next(x for x in source["blocks"] if x["held_group"] == list(group) and x["half"] == half)
            for scope in SCOPES:
                n = "train" if scope.startswith("train") else "held"
                ds = a if scope.endswith("same") else b
                eval_data = probe if scope == "prefix_probe" else {e: datasets[e][n] for e in ENDPOINTS}
                vv = probe_values if scope == "prefix_probe" else values[n, ds]
                predictions = {e: previous.predict(eval_data[e], paths[e], source["selection"]) for e in ENDPOINTS}
                for i, e in enumerate(ENDPOINTS):
                    out = metrics(eval_data[e], eval_data[ENDPOINTS[-1]], vv, predictions[e], predictions[ENDPOINTS[0]], predictions[ENDPOINTS[max(i-1, 0)]])
                    record["endpoints"][e][scope] = out
                    if scope == "prefix_probe" and i == 0:
                        for anchor in joint.ANCHORS:
                            for family in previous.FAMILIES:
                                for mode in previous.FILTERS:
                                    s = source["selection"][anchor][family][mode]
                                    obj, key = s["objective"] or "all_mse", s["candidate"] or "zero"
                                    old = old_block["curves"][anchor][obj][key]
                                    stats = out["all"]["variants"][f"{family}_{anchor}_{mode}"]
                                    expected = old["by_root"][str(group[0])]
                                    probe_max = max(probe_max, abs(stats["delta_cost_own"]*len(vv)-expected["cost_sum"]),
                                        abs(stats["delta_wait_min_own"]*len(vv)-expected["wait_sum_min"]),
                                        abs(stats["delta_unsafe_own"]*len(vv)-expected["unsafe_delta_count"]),
                                        abs(stats["switch_own"]*len(vv)-old["changed_count"]))
                                    probe_count += 1
                    if len(blocks) == 0 and scope != "prefix_probe" and i == 4:
                        old = source["scopes"][scope]
                        for xs, ys in [(out["all"], old["all"]), *((p, old["by_root"][r]) for r, p in out["by_root"].items())]:
                            for name, stats in ys["variants"].items():
                                full_max = max(full_max, *(abs(xs["variants"][name][f]-v) for f, v in stats.items() if f != "feasible_score_mse_against_cost"))
                                full_count += 1
            blocks.append(record)
            print(f"REFIT_CONSISTENCY_BLOCK_COMPLETE {group}/{half}", flush=True)
    if max(probe_max, full_max, maximum) > previous.TOL: raise ValueError("Cached OOF or full endpoint changed")
    return {"selection": source["selection"], "training_draws": list(a), "evaluation_draws": list(b), "blocks": blocks,
        "half_fits": half_fits, "prefix_diagnostics": prefix_diagnostics(source), "reproduction": {
            "probe_records": probe_count, "full_metric_records": full_count, "probe_max_abs_difference": probe_max,
            "full_max_abs_difference": full_max, "subspace_max_abs_difference": maximum}}, models


def averaged(pieces):
    return {"matched_blocks": len(pieces), "mean_cases": float(np.mean([p["cases"] for p in pieces])),
        "variants": {v: {f: float(np.mean([p["variants"][v][f] for p in pieces])) for f in pieces[0]["variants"][v]} for v in VARIANTS}}
