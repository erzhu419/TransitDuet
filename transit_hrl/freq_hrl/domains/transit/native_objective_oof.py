"""Clean prefix/future objective selection from cached native references."""

import copy

import numpy as np

from freq_hrl.domains.transit import native_cost_objectives as base

joint, strength = base.joint, base.strength
OBJECTIVES = (*base.OBJECTIVES, "wait_rank")
FILTERS = ("pooled_safe", "half_safe")
FAMILIES = (*OBJECTIVES, "joint")
SCOPES = ("train_same", "train_other", "held_same", "held_other")
TOL = 1e-12
VARIANTS = ("raw", *(f"{o}_{a}_{f}" for o in FAMILIES for a in joint.ANCHORS for f in FILTERS))


def fit_paths(data, values, cached):
    levels = set(strength.lowrank.LEVELS[1:])
    models, fits, maximum = base.fit_objectives(data, values, cached, levels)
    models["wait_rank"] = copy.deepcopy(cached)
    anchor = joint.anchors(data, cached["anchor"])
    mask = np.arange(6)[None, :] != anchor[:, None]
    x = joint.relative(data["x"][..., :198], anchor)
    _, s, vt = np.linalg.svd(x[mask], full_matrices=False)
    numerical = int((s > s[0]*max(x[mask].shape)*np.finfo(float).eps).sum())
    scales = np.asarray(cached["scales"])
    utility = (joint.relative(values, anchor)/scales).sum(axis=-1)
    offset = joint.relative(data["heads"], anchor)[..., 0]/scales[0]
    for level in sorted(levels, key=int):
        k = min(int(level), numerical)
        basis = vt[:k].T*(np.sqrt(mask.sum())/s[:k])
        old = np.asarray(cached["weights"][level][0])
        initial = (vt[:k]@old)*s[:k]/np.sqrt(mask.sum())
        loss, hessian = base.losses(x@basis, offset, utility, data["raw_logits"] <= 0, anchor)["rank_regret"]
        theta, fit = base.constrained_fit(loss, hessian, initial)
        models["wait_rank"]["weights"][level][0] = (basis@theta).tolist()
        fits.append({"objective": "wait_rank", "level": level, "requested_rank": int(level), "retained_rank": k,
            "parameter_count": k, "residual_rms_limit_physical": float(np.linalg.norm(initial)*scales[0]),
            "residual_rms_physical": float(np.linalg.norm(theta)*scales[0]), **fit})
    return models, fits, maximum


def curve_summary(pieces, scales):
    roots = sorted({r for p in pieces for r in p["by_root"]})
    output = {}
    for root in roots:
        xs = [p["by_root"][root] for p in pieces if root in p["by_root"]]
        n = sum(x["cases"] for x in xs)
        cost, wait = sum(x["cost_sum"] for x in xs)/n, sum(x["wait_sum_min"] for x in xs)/n
        output[root] = {"cases": n, "delta_cost": cost, "delta_wait_min": wait,
            "delta_unsafe": sum(x["unsafe_delta_count"] for x in xs)/n,
            "scaled_cost": cost/scales[root][0], "scaled_wait": wait/scales[root][1]}
    return {"by_root": output, "score": max(x[f] for x in output.values() for f in ("scaled_cost", "scaled_wait")),
        "changed_count": sum(p["changed_count"] for p in pieces)}


def nonharm(curve):
    return all(x["delta_cost"] <= TOL and x["delta_wait_min"] <= TOL and x["delta_unsafe"] <= 0 for x in curve["by_root"].values())


def choose(curves):
    output = {}
    for family in FAMILIES:
        objectives = OBJECTIVES if family == "joint" else (family,)
        output[family] = {}
        for mode in FILTERS:
            eligible = [(o, c, curves[o][c]) for o in objectives for c in strength.CANDIDATES if c != "zero"
                and curves[o][c]["pooled"]["score"] < -TOL and nonharm(curves[o][c]["pooled"])
                and (mode != "half_safe" or all(nonharm(h) for h in curves[o][c]["halves"]))]
            best = min(eligible, key=lambda x: x[2]["pooled"]["score"]) if eligible else None
            output[family][mode] = {"objective": best[0] if best else None, "candidate": best[1] if best else None,
                "eligible_count": len(eligible), "curve": best[2] if best else curves[objectives[0]]["zero"]}
    return output


def collect(train, panels, source, cached, reused_oof):
    a = tuple(source["trials"][0]["training_draws"])
    lookup = {(tuple(t["removed_group"]), t["half"]): i for i, t in enumerate(source["trials"]) if t["removed_group"] and t["half"] is not None}
    batch = joint.binding.learned.neural.batches(train)
    blocks, cache, oof, maximum = [], {}, [], 0.
    for group, indices, validation in joint.binding.policy.prefix_folds(train):
        rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
        for half in (0, 1):
            index = str(lookup[tuple(group), half])
            reference = cached["references"][index]
            data = joint.causal_data(rows, reference, batch=tuple(x[indices] for x in batch))
            val = joint.causal_data(val_rows, reference, batch=tuple(x[validation] for x in batch))
            oof.append(val)
            draws, other = a[2*half:2*half+2], a[2*(1-half):2*(1-half)+2]
            values = joint.physical_values(rows, panels, draws)
            vv = joint.physical_values(val_rows, panels, other)
            block = {"held_group": list(group), "half": half, "training_states": len(rows), "held_states": len(val_rows),
                "training_draws": list(draws), "validation_draws": list(other), "fits": {}, "curves": {}, "scales": {}}
            models = {}
            for kind in joint.ANCHORS:
                path = cached["paths"][index]["changing_native_x"][kind]
                models[kind], block["fits"][kind], difference = fit_paths(data, values, path)
                maximum = max(maximum, difference)
                block["scales"][kind] = path["scales"]
                block["curves"][kind] = {o: {c: strength.base.stats(val, vv, models[kind][o], **p)
                    for c, p in strength.CANDIDATES.items()} for o in OBJECTIVES}
            blocks.append(block)
            cache[index] = models
            print(f"OBJECTIVE_OOF_BLOCK_COMPLETE {group}/{half}", flush=True)
    difference = max(float(np.max(np.abs(np.concatenate([d["heads"][..., 0] for d in oof])-reused_oof["costs"]))),
        float(np.max(np.abs(np.concatenate([d["raw_logits"] for d in oof])-reused_oof["state_conditioned_pair"]))))
    identity = all(np.array_equal(np.concatenate([d[k] for d in oof]), reused_oof[k]) for k in ("originals", "roots"))
    choices = np.concatenate([d["raw_choices"] for d in oof])
    expected = joint.binding.policy.selected(reused_oof["costs"], reused_oof["state_conditioned_pair"], reused_oof["originals"])
    if not identity or difference > TOL or np.any(choices != expected): raise ValueError("Cached clean OOF raw changed")
    roots = sorted({str(r["root"]) for r in train})
    summaries, selections, legacy = {}, {}, {}
    for kind in joint.ANCHORS:
        scales = {r: np.mean([b["scales"][kind] for b in blocks if str(b["held_group"][0]) == r], axis=0) for r in roots}
        summaries[kind] = {o: {c: {"pooled": curve_summary([b["curves"][kind][o][c] for b in blocks], scales),
            "halves": [curve_summary([b["curves"][kind][o][c] for b in blocks if b["half"] == h], scales) for h in (0, 1)]}
            for c in strength.CANDIDATES} for o in OBJECTIVES}
        for o in OBJECTIVES:
            zero = summaries[kind][o]["zero"]
            if zero["pooled"]["changed_count"] or zero["pooled"]["score"] != 0: raise ValueError("OOF zero not raw")
        selections[kind] = choose(summaries[kind])
        ledger = [{**b, "fits": {k: {"scales": b["scales"][k]} for k in joint.ANCHORS}} for b in blocks]
        legacy[kind] = strength.base.select([b["curves"][kind]["all_mse"] for b in blocks], ledger, kind, levels=tuple(strength.CANDIDATES))
    return {"blocks": blocks, "curves": summaries, "selection": selections, "legacy_selection": legacy, "reproduction": {
        "oof_choice_comparisons": len(choices), "oof_changed_choices": 0, "oof_max_abs_difference": difference,
        "subspace_max_abs_difference": maximum}}, cache


def predict(data, models, selection):
    output = {"raw": {"chosen": data["raw_choices"], "cost": data["heads"][..., 0]}}
    for anchor in joint.ANCHORS:
        for family in FAMILIES:
            for mode in FILTERS:
                s = selection[anchor][family][mode]
                if s["candidate"] is None:
                    cost, chosen = data["heads"][..., 0], data["raw_choices"]
                else:
                    p = strength.CANDIDATES[s["candidate"]]
                    cost = strength.base.costs(data, models[anchor][s["objective"]], **p)
                    chosen = joint.binding.policy.selected(cost, data["raw_logits"], joint.anchors(data, anchor))
                output[f"{family}_{anchor}_{mode}"] = {"chosen": chosen, "cost": cost}
    return output


def fit_cell(train, held, panels, reference, paths, source, cached, reused_oof, frozen_selection):
    inner, models = collect(train, panels, source, cached, reused_oof)
    maximum, changed = 0., 0
    for kind in joint.ANCHORS:
        actual, old = inner["legacy_selection"][kind], frozen_selection[kind]
        changed += sum(actual[f] != old[f] for f in base.previous.diagnostic.FIELDS.values())
        for c in strength.CANDIDATES:
            xs, ys = actual["curves"][c], old["curves"][c]
            maximum = max(maximum, *(abs(xs[f]-ys[f]) for f in ("mse", "joint_score")))
            for r, x in xs["by_root"].items():
                maximum = max(maximum, *(abs(x[f]-ys["by_root"][r][f]) for f in x))
    if changed or maximum > TOL: raise ValueError("Cached all-MSE clean inner selection changed")
    inner["reproduction"].update(legacy_choice_comparisons=6, legacy_changed_choices=changed, legacy_max_abs_difference=maximum)
    a = tuple(source["trials"][0]["training_draws"])
    b = tuple(i for i in range(8) if i not in a)
    data = {n: joint.causal_data(rs, reference) for n, rs in (("train", train), ("held", held))}
    values = {(n, ds): joint.physical_values(rs, panels, ds) for n, rs in (("train", train), ("held", held)) for ds in (a, b)}
    full, fits = {}, {}
    for kind in joint.ANCHORS:
        full[kind], fits[kind], difference = fit_paths(data["train"], values["train", a], paths[kind])
        inner["reproduction"]["subspace_max_abs_difference"] = max(inner["reproduction"]["subspace_max_abs_difference"], difference)
    scopes = {}
    for scope in SCOPES:
        n, ds = ("train" if scope.startswith("train") else "held"), (a if scope.endswith("same") else b)
        stats = base.metrics(data[n], values[n, ds], predict(data[n], full, inner["selection"]))
        # Wait-aware outputs are utility scores, not calibrated cost predictions.
        for piece in (stats["all"], *stats["by_root"].values()):
            for v in piece["variants"].values():
                v["score_mse_against_cost"] = v.pop("cost_mse")
                v["feasible_score_mse_against_cost"] = v.pop("feasible_cost_mse")
        scopes[scope] = stats
    models["full"] = full
    return {**inner, "training_draws": list(a), "evaluation_draws": list(b), "full_fits": fits, "scopes": scopes}, models


def averaged(pieces):
    return {"cells": len(pieces), "cases": sum(p["cases"] for p in pieces),
        "variants": {v: {f: float(np.mean([p["variants"][v][f] for p in pieces])) for f in pieces[0]["variants"][v]} for v in VARIANTS}}
