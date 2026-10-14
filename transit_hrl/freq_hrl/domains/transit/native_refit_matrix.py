"""Matched residual refits with fixed reference and coordinate interventions."""

import numpy as np

from freq_hrl.domains.transit import native_prefix_stability as diagnostic

strength, joint = diagnostic.strength, diagnostic.joint
ARMS = ("fixed", "changing_fixed_x", "changing_native_x")
VARIANTS = ("raw", *(f"cost_strength_{a}_{m}" for a in joint.ANCHORS for m in diagnostic.FIELDS))
SCOPES = ("fit_training", "train_same", "train_other", "held_same", "held_other")


def sliced(data, indices): return {k: v[indices] for k, v in data.items()}


def predict(data, paths, selection):
    output = {"raw": {"chosen": data["raw_choices"], "cost": data["heads"][..., 0]}}
    for anchor in joint.ANCHORS:
        for mode, field in diagnostic.FIELDS.items():
            key = selection[anchor][field]
            if key in (None, "zero"):
                cost = joint.relative(data["heads"], joint.anchors(data, anchor))[..., 0]
                chosen = data["raw_choices"]
            else:
                c = strength.CANDIDATES[key]
                cost = strength.base.costs(data, paths[anchor], c["level"], c["alpha"])
                chosen = joint.binding.policy.selected(cost, data["raw_logits"], joint.anchors(data, anchor))
            output[f"cost_strength_{anchor}_{mode}"] = {"chosen": chosen, "cost": cost}
    return output


def metrics(data, common, values, prediction):
    ix = np.arange(len(values))
    own, base = values[ix, data["raw_choices"]], values[ix, common["raw_choices"]]
    mask = np.arange(6)[None, :] != data["originals"][:, None]
    truth = joint.relative(values, data["originals"])[..., 0]
    oracle = np.where(data["raw_logits"] <= 0, values[..., 0], np.inf).min(axis=1)
    fields = {}
    for name, p in prediction.items():
        actual = values[ix, p["chosen"]]
        error = (p["cost"]-p["cost"][ix, data["originals"]][:, None]-truth)**2
        regret = actual[:, 0]-oracle
        fields[name] = {"cost_mse": error[mask].reshape(len(values), 5).mean(axis=1), "gate_cost_regret": regret,
            "delta_cost_common": actual[:, 0]-base[:, 0], "delta_wait_min_common": actual[:, 1]-base[:, 1],
            "delta_unsafe_common": (actual[:, 1] > 0).astype(int)-(base[:, 1] > 0).astype(int),
            "delta_cost_own": actual[:, 0]-own[:, 0], "delta_wait_min_own": actual[:, 1]-own[:, 1],
            "delta_unsafe_own": (actual[:, 1] > 0).astype(int)-(own[:, 1] > 0).astype(int),
            "switch_common": p["chosen"] != common["raw_choices"], "switch_own": p["chosen"] != data["raw_choices"]}
    def summarize(indices):
        return {"cases": len(indices), "variants": {name: {k: float(v[indices].mean()) for k, v in xs.items()}
            for name, xs in fields.items()}}
    return {"all": summarize(ix), "by_root": {str(root): summarize(np.flatnonzero(data["roots"] == root))
        for root in sorted(set(data["roots"]))}}


def fit_matrix(train, held, panels, reference, paths, source, stability, reused_oof):
    a, b = tuple(source["training_draws"]), tuple(source["evaluation_draws"])
    folds = list(joint.binding.policy.prefix_folds(train))
    batches = {"train": joint.binding.learned.neural.batches(train), "held": joint.binding.learned.neural.batches(held)}
    fixed = {n: joint.causal_data(rs, reference, batch=batches[n]) for n, rs in (("train", train), ("held", held))}
    values = {(n, draws): joint.physical_values(rs, panels, draws) for n, rs in (("train", train), ("held", held))
        for draws in (a, b, a[:2], a[2:])}
    designs = [(None, None, np.arange(len(train)), None)]
    designs.extend((None, h, np.arange(len(train)), None) for h in (0, 1))
    for group, training, validation in folds:
        designs.append((tuple(group), None, training, validation))
        designs.extend((tuple(group), h, training, validation) for h in (0, 1))
    trials, lookup, cache, native_oof, prefix_stats = [], {}, {"references": {}, "paths": {}}, {}, {}
    choices, scores, changed, maximum = 0, 0, 0, 0.
    for group, half, indices, validation in designs:
        index = len(trials)
        lookup[group, half] = index
        draws = a if half is None else a[2*half:2*half+2]
        rows = [train[i] for i in indices]
        full = group is None and half is None
        if full:
            changing, ref_fit, natural, fit_natural = reference, None, fixed, fixed["train"]
        else:
            batch = tuple(x[indices] for x in batches["train"])
            changing, ref_fit = joint.fit_reference(rows, panels, draws, batch=batch)
            natural = {n: joint.causal_data(rs, changing, batch=batches[n]) for n, rs in (("train", train), ("held", held))}
            fit_natural = joint.causal_data(rows, changing, batch=batch)
            cache["references"][str(index)] = changing
        trial = {"removed_group": list(group) if group else None, "half": half, "training_states": len(rows),
            "training_draws": list(draws), "reference_fit": ref_fit,
            "reference_shift": {"raw_switch_train": float(np.mean(natural["train"]["raw_choices"] != fixed["train"]["raw_choices"])),
                "raw_logit_rms_train": float(np.sqrt(np.mean((natural["train"]["raw_logits"]-fixed["train"]["raw_logits"])**2))),
                "cost_head_rms_train": float(np.sqrt(np.mean((natural["train"]["heads"][..., 0]-fixed["train"]["heads"][..., 0])**2))),
                "residual_coordinate_rms_train": float(np.sqrt(np.mean((natural["train"]["x"][..., :198]-fixed["train"]["x"][..., :198])**2)))},
            "arms": {}}
        cached_paths = {}
        for arm in ARMS:
            data = fixed if arm == "fixed" else natural
            fit_data = sliced(fixed["train"], indices) if arm == "fixed" else fit_natural
            if arm == "changing_fixed_x":
                data = {n: {**d, "x": fixed[n]["x"]} for n, d in data.items()}
                fit_data = {**fit_data, "x": fixed["train"]["x"][indices]}
            if full: fitted, fits = paths, {}
            else:
                fitted, fits = {}, {}
                for kind in joint.ANCHORS:
                    fitted[kind], fits[kind], _ = strength.lowrank.fit_path(rows, fit_data, values["train", draws][indices], kind)
                cached_paths[arm] = fitted
            datasets = {"fit_training": (fit_data, sliced(fixed["train"], indices), values["train", draws][indices])}
            datasets.update({scope: (data[n], fixed[n], values[n, a if scope.endswith("same") else b])
                for scope, n in (("train_same", "train"), ("train_other", "train"), ("held_same", "held"), ("held_other", "held"))})
            outcome = {}
            for scope, (d, common, v) in datasets.items():
                pred = predict(d, fitted, source["selection"])
                outcome[scope] = metrics(d, common, v, pred)
                if full and arm == "fixed" and scope != "fit_training":
                    old = {joint.binding.learned.state_key(r): r for r in source["scopes"][scope]}
                    eval_rows = train if scope.startswith("train") else held
                    for i, r in enumerate(eval_rows):
                        for name, p in pred.items():
                            previous = old[joint.binding.learned.state_key(r)]["decisions"][name]
                            j = int(p["chosen"][i])
                            changed += int(j != previous["selected"])
                            maximum = max(maximum, abs(v[i, j, 0]-previous["cost"]), abs(v[i, j, 1]-previous["wait"]))
                            choices += 1
                            scores += 2
            trial["arms"][arm] = {"fits": fits, "scopes": outcome}
            if group is not None and half is not None and arm == "changing_native_x":
                val_rows = [train[i] for i in validation]
                val = joint.causal_data(val_rows, changing, batch=tuple(x[validation] for x in batches["train"]))
                native_oof[group, half] = val
                vv = values["train", a[2*(1-half):2*(1-half)+2]][validation]
                for kind in joint.ANCHORS:
                    for mode, field in diagnostic.FIELDS.items():
                        key = source["selection"][kind][field]
                        c = strength.CANDIDATES[key or "zero"]
                        stat = strength.base.stats(val, vv, fitted[kind], c["level"], alpha=c["alpha"])
                        xs = stat["by_root"][str(group[0])]
                        target = prefix_stats.setdefault((group, kind, mode), {"cases": 0, "cost_sum": 0., "wait_sum_min": 0.,
                            "unsafe_delta_count": 0, "changed_count": 0})
                        for k in ("cases", "cost_sum", "wait_sum_min", "unsafe_delta_count"): target[k] += xs[k]
                        target["changed_count"] += stat["changed_count"]
        if not full: cache["paths"][str(index)] = cached_paths
        trials.append(trial)
        print(f"REFIT_MATRIX_DESIGN_COMPLETE {index} states={len(rows)} draws={list(draws)}", flush=True)
    oof = [native_oof[tuple(group), h] for group, _, _ in folds for h in (0, 1)]
    difference = max(float(np.abs(np.concatenate([d["heads"][..., 0] for d in oof])-reused_oof["costs"]).max()),
        float(np.abs(np.concatenate([d["raw_logits"] for d in oof])-reused_oof["state_conditioned_pair"]).max()))
    oof_changed = int(np.sum(np.concatenate([d["raw_choices"] for d in oof]) != joint.binding.policy.selected(
        reused_oof["costs"], reused_oof["state_conditioned_pair"], reused_oof["originals"])))
    prefix_max, prefix_changes = 0., 0
    for kind in joint.ANCHORS:
        for p in stability["anchors"][kind]["prefixes"]:
            for mode in diagnostic.FIELDS:
                actual, expected = prefix_stats[tuple(p["group"]), kind, mode], p["modes"][mode]
                prefix_max = max(prefix_max, *(abs(actual[k]-expected[k]) for k in ("cost_sum", "wait_sum_min")))
                prefix_changes += sum(actual[k] != expected[k] for k in ("cases", "unsafe_delta_count", "changed_count"))
    identity = all(np.array_equal(np.concatenate([d[k] for d in oof]), reused_oof[k]) for k in ("originals", "roots"))
    if not identity or changed or maximum > 1e-12 or oof_changed or difference > 1e-12 or prefix_changes or prefix_max > 1e-12:
        raise ValueError("Cached outer or clean inner refit endpoints changed")
    blocks = [{"removed_group": list(group), "half": h, "trials": {
        f"{n}/{k}": lookup[g, q] for n, g in ((len(train)-6, tuple(group)), (len(train), None))
        for k, q in ((2, h), (4, None))}} for group, _, _ in folds for h in (0, 1)]
    return {"trials": trials, "blocks": blocks, "reproduction": {"choice_comparisons": choices, "score_comparisons": scores,
        "changed_choices": changed, "max_abs_difference": maximum, "oof_max_abs_difference": difference,
        "oof_changed_choices": oof_changed, "prefix_max_abs_difference": prefix_max, "prefix_count_differences": prefix_changes}}, cache


def averaged(pieces):
    first = pieces[0]
    return {"matched_blocks": len(pieces), "mean_cases": float(np.mean([p["cases"] for p in pieces])),
        "variants": {name: {field: float(np.mean([p["variants"][name][field] for p in pieces]))
            for field in first["variants"][name]} for name in VARIANTS}}
