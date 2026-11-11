"""Audit and assemble A-only Dense/Wide banks without artificial repeat labels."""

from collections import Counter
import numpy as np

from .plan_validity_boundary_information import draw_seed
from .plan_validity_boundary_tail import basis, predict
from .plan_validity_boundary_parent_scale import predict_parent


def audit_public(design, release, records):
    if len(records) != len(release["shards"]) or len(records) != 32:
        raise ValueError("need every frozen acquisition shard")
    costs = {}
    for shard, record in zip(release["shards"], records):
        cell = design["cells"][shard["cell_index"]]
        if (record["run_name"], record["task"], record["optimizer_seed"], record["method"], record["shard_index"],
            record["status"], record["stage"], record["sampled_prefix_seeds"], record["completed_states"],
            record["completed_pairs"], record["new_environment_steps"], record["fits"], record["evaluation_prefixes_accessed"]) != (
            design["run_name"], cell["task"], cell["optimizer_seed"], shard["method"], shard["shard_index"],
            "complete", "boundary_information_sample_complete", shard["prefix_seeds"], 12,
            shard["new_pairs"], shard["environment_steps"], 0, False):
            raise ValueError("completed public record differs from its declared shard")
        counts = record["measured_cost"]["counts"]
        if counts.get("environment_steps") != shard["environment_steps"] or any(counts.get(k, 0)
            for k in ("ridge_fit_calls", "constant_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates")):
            raise ValueError("acquisition ledger includes missing native work or fitting")
        key = f"{cell['task']}/{cell['optimizer_seed']}/{shard['method']}"
        row = costs.setdefault(key, {"environment_steps": 0, "new_pairs": 0, "cpu_seconds": 0., "shards": 0})
        row["environment_steps"] += counts["environment_steps"]
        row["new_pairs"] += record["completed_pairs"]
        row["cpu_seconds"] += record["measured_cost"]["cpu_seconds"]
        row["shards"] += 1
    if sum(c["environment_steps"] for c in costs.values()) != release["released_environment_steps"]:
        raise ValueError("acquisition budget did not close")
    for key, row in costs.items():
        method = key.rsplit("/", 1)[1]
        if row["environment_steps"] != design[method]["new_steps_per_root"] or row["new_pairs"] != design[method]["new_pairs"]:
            raise ValueError("Dense/Wide root budgets are not the frozen matched budgets")
    return costs


def row_map(rows, prefixes):
    keyed = {(r["seed"], r["opportunity_step"]): r for r in rows}
    if (len(keyed) != len(rows) or set(r["seed"] for r in rows) != set(prefixes)
        or any(sum(r["seed"] == s for r in rows) != 3 for s in prefixes)):
        raise ValueError("private prefix/state roster differs")
    return keyed


def checked_window(window, *, root, seed, step, j, indices, cached=False):
    if window["window_seed"] != draw_seed(root, seed, step, j, 11401):
        raise ValueError("fixed window seed changed")
    if cached:
        replicas = [{"suffix_index": k, **window["suffix_draws"][k]} for k in indices]
    else:
        if window["window_index"] != j:
            raise ValueError("new window order changed")
        replicas = window["A_replicas"]
    if [r["suffix_index"] for r in replicas] != indices:
        raise ValueError("training bank includes wrong A indices or B/E/precheck labels")
    w = window["window_costs"][1] - window["window_costs"][0]
    for k, r in zip(indices, replicas):
        if (r["suffix_seed"] != draw_seed(root, seed, step, j, k, 11402)
            or not np.all(np.isfinite([r["W"], r["tail"], r["G"]]))
            or not np.isclose(r["W"], w) or not np.isclose(r["W"] + r["tail"], r["G"], rtol=1e-10, atol=1e-10)):
            raise ValueError("A label provenance or credit reconstruction changed")
    return [r["tail"] for r in replicas]


def assemble(cell, design, cached_rows, new_rows, method):
    base = row_map(cached_rows, cell["development_prefix_seeds"])
    new_prefixes = cell["development_prefix_seeds"] if method == "dense" else cell["wide_training_prefix_seeds"]
    added = row_map(new_rows, new_prefixes)
    if method == "dense" and set(added) != set(base):
        raise ValueError("Dense extensions do not match the saved state roster")
    states = base if method == "dense" else {**base, **added}
    extras = {(k["seed"], k["state_index"], k["window_index"]) for k in cell["dense_extra_windows"]}
    origins, now, wait, targets, variances, sizes, keys = [], [], [], [], [], [], []
    state_indices = {}
    for seed in sorted({key[0] for key in states}):
        for i, key in enumerate(sorted(k for k in states if k[0] == seed)):
            state_indices[key] = i
    for (seed, step), row in sorted(states.items()):
        if len(row["windows"]) != 4:
            raise ValueError("each training state needs its four frozen windows")
        for j, window in enumerate(row["windows"]):
            original = (seed, step) in base
            labels = checked_window(window, root=cell["optimizer_seed"], seed=seed, step=step, j=j,
                indices=[0, 1] if original else design["wide"]["new_A_indices"], cached=original)
            if method == "dense":
                extra = added[seed, step]
                if (extra["feature_names"] != row["feature_names"] or extra["causal_features"] != row["causal_features"]
                    or len(extra["windows"]) != 4):
                    raise ValueError("Dense extension changed its intervention features")
                extended = extra["windows"][j]
                if extended["boundary_context"] != window["boundary_context"] or extended["window_costs"] != window["window_costs"]:
                    raise ValueError("Dense extension changed its cached boundary state")
                indices = [*design["dense"]["new_A_indices"], *([design["dense"]["extra_A_index"]]
                    if (seed, state_indices[seed, step], j) in extras else [])]
                labels += checked_window(extended, root=cell["optimizer_seed"], seed=seed, step=step, j=j, indices=indices)
            context = window["boundary_context"]
            now.append(context["now"])
            wait.append(context["wait_one_check"])
            origins.append({**row, "offset_fraction": (step % design["period"]) / 25,
                "remaining_fraction": (design["horizon"] - step) / design["horizon"], "call_spent": False})
            targets.append(float(np.mean(labels)))
            variances.append(float(np.var(labels, ddof=1) / len(labels)))
            sizes.append(len(labels))
            keys.append([seed, step, j])
    if sum(sizes) != design[method]["training_pairs"]:
        raise ValueError("assembled A information differs from the declared pair budget")
    return {"now": basis(now), "wait_one_check": basis(wait), "causal_t": basis(origins),
        "training_mean": np.asarray(targets), "A_mean_noise_estimate": np.asarray(variances),
        "label_counts": sizes, "window_keys": keys, "prefixes": sorted({s for s, _, _ in keys})}


def frozen_predictions(data, models):
    predictions = {"zero": np.zeros(len(data["now"])),
        "constant": np.full(len(data["now"]), models["constant"]["value"]),
        "causal_t": predict(models["causal_t"], data["causal_t"])}
    for name in ("base", "dense", "wide"):
        predictions[name] = predict_parent(models[name], data["now"], data["wait_one_check"])
    return predictions


def training_readout(data, models):
    predictions = frozen_predictions(data, models)
    return {"prefixes": len(data["prefixes"]), "windows": len(data["training_mean"]),
        "A_pairs": sum(data["label_counts"]), "A_count_histogram": dict(Counter(data["label_counts"])),
        "empirical_A_mean_noise_variance": float(data["A_mean_noise_estimate"].mean()),
        "MSE_to_own_training_A_mean": {name: float(np.mean((p - data["training_mean"]) ** 2)) for name, p in predictions.items()},
        "scope": "training_descriptive_only_not_independent_quality_or_dense_wide_ranking"}
