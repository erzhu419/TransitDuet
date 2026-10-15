"""Cached refit endpoints across seen/held prefixes and independent futures."""

import numpy as np

from freq_hrl.domains.transit import native_refit_consistency as previous

joint, ENDPOINTS, VARIANTS = previous.joint, previous.ENDPOINTS, previous.VARIANTS
SCOPES = ("probe6", "seen54", "train60", "held60")
LABELS = ("old_inner2", "old_outer4", "old_matched", "fresh_half0", "fresh_half1", "fresh8")


def write_panel_cache(path, old, fresh):
    keys = sorted(old)
    if set(keys) != set(fresh): raise ValueError("Old/fresh prefix coverage differs")
    arrays = {name: np.stack([panels[k]["values"] for k in keys]) for name, panels in (("old", old), ("fresh", fresh))}
    if any(x.shape != (len(keys), 8, 6, 4) for x in arrays.values()): raise ValueError("Changed cached panel axes")
    np.savez_compressed(path, keys=np.asarray(keys), **arrays)


def read_panel_cache(path):
    with np.load(path) as pack:
        keys = [tuple(k) for k in pack["keys"].tolist()]
        return tuple({k: {"values": values} for k, values in zip(keys, pack[name])} for name in ("old", "fresh"))


def freeze(train, held, reference, refits, refit_source, source, cached, consistency):
    lookup = {(tuple(t["removed_group"]) if t["removed_group"] else None, t["half"]): str(i)
        for i, t in enumerate(refit_source["trials"])}
    output = []
    for group, seen, probe in joint.binding.policy.prefix_folds(train):
        rows = {"probe6": [train[i] for i in probe], "seen54": [train[i] for i in seen], "train60": train, "held60": held}
        for half in (0, 1):
            index = lookup[tuple(group), half]
            references = {ENDPOINTS[0]: refits["references"][index], ENDPOINTS[1]: refits["references"][index],
                ENDPOINTS[2]: refits["references"][lookup[None, half]], ENDPOINTS[3]: refits["references"][lookup[None, half]],
                ENDPOINTS[4]: reference}
            models = {ENDPOINTS[0]: cached[index], ENDPOINTS[1]: consistency[index],
                **consistency[f"half_{half}"], ENDPOINTS[4]: cached["full"]}
            data, predictions = {}, {}
            for endpoint in ENDPOINTS:
                data[endpoint] = {s: joint.causal_data(rows[s], references[endpoint]) for s in ("probe6", "train60", "held60")}
                data[endpoint]["seen54"] = {k: v[seen] for k, v in data[endpoint]["train60"].items()}
                predictions[endpoint] = {s: previous.previous.predict(data[endpoint][s], models[endpoint], source["selection"]) for s in SCOPES}
            output.append({"held_group": list(group), "half": half, "rows": rows, "data": data, "predictions": predictions})
    return output


def action_cache(frozen):
    return [{"held_group": b["held_group"], "half": b["half"],
        "states": {s: [[r[k] for k in ("root", "scene_seed", "probe_index")] for r in b["rows"][s]] for s in SCOPES},
        "choices": {e: {s: {v: p["chosen"].tolist() for v, p in b["predictions"][e][s].items()} for s in SCOPES} for e in ENDPOINTS}}
        for b in frozen]


def match_sealed_actions(frozen, sealed):
    lookup = {(tuple(b["held_group"]), b["half"]): b for b in sealed["blocks"]}
    comparisons = 0
    for block in frozen:
        old = lookup[tuple(block["held_group"]), block["half"]]
        rows = block["rows"]["probe6"]
        if [[r[k] for k in ("root", "scene_seed", "probe_index")] for r in rows] != [
                [r[k] for k in ("root", "scene_seed", "probe_index")] for r in old["states"]]:
            raise ValueError("Changed sealed OOF state order")
        for v, p in block["predictions"][ENDPOINTS[0]]["probe6"].items():
            if p["chosen"].tolist() != old["choices"][v]: raise ValueError("Changed pre-future sealed OOF action")
            comparisons += len(rows)
    return comparisons


def metrics(block, endpoint, scope, draws):
    index = ENDPOINTS.index(endpoint)
    data, common = block["data"][endpoint][scope], block["data"][ENDPOINTS[-1]][scope]
    pred = block["predictions"][endpoint][scope]
    values = draws.mean(axis=0)
    out = previous.metrics(data, common, values[:, :, (0, 3)], pred,
        block["predictions"][ENDPOINTS[0]][scope], block["predictions"][ENDPOINTS[max(index-1, 0)]][scope])
    ix = np.arange(len(values))
    groups = {"all": ix, **{str(r): np.flatnonzero(data["roots"] == r) for r in set(data["roots"])}}
    for root, indices in groups.items():
        piece = out["all"] if root == "all" else out["by_root"][root]
        for name, p in pred.items():
            actual = values[ix, p["chosen"]]
            for suffix, choices in (("own", data["raw_choices"]), ("common", common["raw_choices"])):
                base = values[ix, choices]
                for component, column in (("fleet", 1), ("nonfleet", 2)):
                    piece["variants"][name][f"delta_{component}_cost_{suffix}"] = float((actual-base)[indices, column].mean())
                harm = (draws[:, ix, p["chosen"], 3] > 0).astype(int)-(draws[:, ix, choices, 3] > 0).astype(int)
                piece["variants"][name][f"delta_draw_wait_harm_rate_{suffix}"] = float(harm[:, indices].mean())
    return out


def evaluate(frozen, old, fresh, source, consistency):
    a, b = source["training_draws"], source["evaluation_draws"]
    blocks, maximum, records = [], 0., 0
    lookup = {(tuple(x["held_group"]), x["half"]): x for x in consistency["blocks"]}
    for frozen_block in frozen:
        half = frozen_block["half"]
        record = {"held_group": frozen_block["held_group"], "half": half, "endpoints": {e: {} for e in ENDPOINTS}}
        for scope, rows in frozen_block["rows"].items():
            keys = [joint.binding.learned.state_key(r) for r in rows]
            past = np.stack([old[k]["values"] for k in keys], axis=1)
            future = np.stack([fresh[k]["values"] for k in keys], axis=1)
            panels = {"old_inner2": past[a[2*half:2*half+2]], "old_outer4": past[a],
                "old_matched": past[a[2*(1-half):2*(1-half)+2] if scope == "probe6" else b],
                "fresh_half0": future[:4], "fresh_half1": future[4:], "fresh8": future}
            for endpoint in ENDPOINTS:
                stats = {label: metrics(frozen_block, endpoint, scope, panel) for label, panel in panels.items()}
                record["endpoints"][endpoint][scope] = stats
                if scope == "seen54": continue
                old_scope = {"probe6": "prefix_probe", "train60": "train_other", "held60": "held_other"}[scope]
                expected = lookup[tuple(record["held_group"]), half]["endpoints"][endpoint][old_scope]
                actual = stats["old_matched"]
                for xs, ys in [(actual["all"], expected["all"]), *((actual["by_root"][r], p) for r, p in expected["by_root"].items())]:
                    if xs["cases"] != ys["cases"]: raise ValueError("Cached endpoint dropped states")
                    for name, fields in ys["variants"].items():
                        maximum = max(maximum, *(abs(xs["variants"][name][f]-v) for f, v in fields.items()))
                        records += 1
        blocks.append(record)
    if maximum > 1e-12: raise ValueError("Cached Stage240 endpoint changed")
    return {"selection": source["selection"], "blocks": blocks,
        "reproduction": {"records": records, "max_abs_difference": maximum}}


def aggregate(cells):
    blocks = [b for c in cells for b in c["blocks"]]
    return {e: {s: {label: {"all": previous.averaged([b["endpoints"][e][s][label]["all"] for b in blocks]),
        "by_root": {r: previous.averaged([b["endpoints"][e][s][label]["by_root"][r] for b in blocks
            if r in b["endpoints"][e][s][label]["by_root"]]) for r in sorted({r for b in blocks for r in b["endpoints"][e][s][label]["by_root"]})}}
        for label in LABELS} for s in SCOPES} for e in ENDPOINTS}


def merge_matrices(matrices):
    # Cell caches are block means; source-only probe means have fewer blocks.
    def merge(parts):
        weights = [p["matched_blocks"] for p in parts]
        return {"matched_blocks": sum(weights), "mean_cases": float(np.average([p["mean_cases"] for p in parts], weights=weights)),
            "variants": {v: {f: float(np.average([p["variants"][v][f] for p in parts], weights=weights))
                for f in parts[0]["variants"][v]} for v in VARIANTS}}
    return {e: {s: {label: {"all": merge([m[e][s][label]["all"] for m in matrices]),
        "by_root": {r: merge([m[e][s][label]["by_root"][r] for m in matrices if r in m[e][s][label]["by_root"]])
            for r in sorted({r for m in matrices for r in m[e][s][label]["by_root"]})}} for label in LABELS} for s in SCOPES} for e in ENDPOINTS}
