"""Measure fixed-state MC control variates without moving policies or references."""

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
from itertools import combinations
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_actor_credit as scores
from . import pointmaze_continuing_credit as continuing
from . import pointmaze_credit_diagnostics as credit
from . import pointmaze_credit_reliability as reliability
from . import pointmaze_independent_credit as independent
from . import pointmaze_horizon_value as values
from . import pointmaze_normalized_update as normalized
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as native
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import write_json
from scripts import pointmaze_mc_control_variate_stage70_spec as spec


def variance_decomposition(common, state):
    common, state = np.asarray(common, dtype=np.float64), np.asarray(state, dtype=np.float64)
    baseline = common - state
    a, b = common - common.mean(0), baseline - baseline.mean(0)
    v0 = independent.gradient_noise(common)["covariance_trace"]
    vs = independent.gradient_noise(state)["covariance_trace"]
    vh = independent.gradient_noise(baseline)["covariance_trace"]
    covariance = float(np.sum(a * b) / (len(common) - 1))
    residual = abs(vs - (v0 + vh - 2 * covariance))
    relative = residual / max(v0 + vh + 2 * abs(covariance), 1.)
    if relative > 1e-10:
        raise ValueError("Stage70 control-variate covariance identity failed")
    return {"common_variance": v0, "state_variance": vs, "baseline_variance": vh,
        "trace_common_baseline_covariance": covariance, "identity_relative_error": relative,
        "state_over_common_variance": None if v0 == 0 else vs / v0,
        "variance_reduction_fraction": None if v0 == 0 else 1. - vs / v0,
        "baseline_empirical_noise": independent.gradient_noise(baseline)}


def compare_batches(batches, mask):
    masks = {"all": np.ones(len(mask), dtype=bool), "mean": ~mask, "log_std": mask}
    raw = {k: np.concatenate([b["gradients"][k] for b in batches]) for k in spec.ESTIMATORS}
    result = {}
    for part, select in masks.items():
        cos = lambda a, b: scores.cosine(a[select], b[select])
        stat = reliability.cosine_statistics
        means = {k: [b["gradients"][k].mean(0) for b in batches] for k in spec.ESTIMATORS}
        result[part] = {"noise": {k: independent.gradient_noise(g[:, select]) for k, g in raw.items()},
            "within_raw": {k: stat([cos(g[i], g[j]) for i, j in combinations(range(len(batches)), 2)])
                for k, g in means.items()},
            "within_normalized": {k: stat([cos(batches[i]["directions"][k], batches[j]["directions"][k])
                for i, j in combinations(range(len(batches)), 2)]) for k in spec.ESTIMATORS},
            "cross_raw_common_reference": {k: stat([cos(means[k][i], means["mc_common"][j])
                for i in range(len(batches)) for j in range(len(batches)) if i != j]) for k in spec.ESTIMATORS},
            "control_variates": {k: variance_decomposition(raw["mc_common"][:, select], raw[k][:, select])
                for k in ("mc_control", "mc_factored")},
            "GAE_over_MC_variance": {t: None if independent.gradient_noise(raw["mc_" + t][:, select])["covariance_trace"] == 0
                else independent.gradient_noise(raw["gae_" + t][:, select])["covariance_trace"] /
                independent.gradient_noise(raw["mc_" + t][:, select])["covariance_trace"] for t in ("control", "factored")}}
    return result


def reproduce_stage69(observed, source):
    for part in ("all", "mean", "log_std"):
        old, row = source[part], observed[part]
        for new_key, old_key in (("mc_common", "mc_common"), ("gae_control", "mc_normalized"), ("gae_factored", "mc_factored")):
            for metric, value in row["noise"][new_key].items():
                expected = old["raw_episode_noise"][old_key][metric]
                if value is None or expected is None:
                    if value != expected:
                        raise ValueError("Stage70 zero-variance definition changed")
                else:
                    np.testing.assert_allclose(value, expected, atol=1e-5, rtol=1e-4)
        for new_key, expected in (("mc_common", old["within_common_MC"]),
                ("gae_control", old["critics"]["mc_normalized"]["within_GAE"]),
                ("gae_factored", old["critics"]["mc_factored"]["within_GAE"])):
            actual = row["within_raw"][new_key] if new_key == "mc_common" else row["within_normalized"][new_key]
            for metric in expected:
                if actual[metric] is None or expected[metric] is None:
                    if actual[metric] != expected[metric]:
                        raise ValueError("Stage70 zero-gradient cosine definition changed")
                else:
                    np.testing.assert_allclose(actual[metric], expected[metric], atol=1e-5, rtol=1e-4)


def replay(root, *, preflight, output):
    file = spec.source_result(root, preflight=preflight)
    source = json.loads(file.read_text())
    if ((source["status"], source["root"], source["preflight"], source["protocol"]) !=
            ("complete", root, preflight, spec.source.EXPERIMENT_PROTOCOL) or source["contract"] != spec.source.contract()
            or source["seed_roles"]["fresh_batches"] != spec.seed_roles(root, preflight=preflight)["archive_batches"]):
        raise ValueError("Stage70 requires the completed frozen Stage69 archives")
    controls = json.loads(spec.values_source.source_result(root, preflight=preflight).read_text())
    factored = json.loads(spec.source.source_result(root, preflight=preflight).read_text())
    clones, _, initialization = native.load_source(root, preflight=preflight)
    opt, args, roles = spec.options(preflight=preflight), spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    archive = file.parent.with_name(file.parent.name + "_raw")
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=diagnostics.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            clone_snapshot = copy.deepcopy(clone.state_dict())
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                saved = torch.load(controls["groups"][p][arm]["treatments"]["mc_normalized"]["checkpoint"], map_location="cpu", weights_only=False)
                fits = {"control": normalized.restore_fit(clone, saved, root=root, period=period, arm=arm, treatment="mc_normalized")}
                saved = torch.load(factored["groups"][p][arm]["candidate_checkpoint"], map_location="cpu", weights_only=False)
                if (saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["horizon"]) != (
                        spec.values_source.EXPERIMENT_PROTOCOL, root, period, arm, args.horizon):
                    raise ValueError("Stage70 factored baseline identity changed")
                fits["factored"] = values.FactoredValueFit.restore(copy.deepcopy(clone), saved)
                snapshots = {t: copy.deepcopy(f.model.state_dict()) for t, f in fits.items()}
                cost["critic_checkpoint_loads"] += len(fits)
                batches, rows = [], []
                for i, seeds in enumerate(roles["archive_batches"]):
                    path = archive / p / arm / f"batch_{i + 1}"
                    old = source["groups"][p][arm]["batches"][i]
                    if seeds != old["seeds"]:
                        raise ValueError("Stage70 archived batch order changed")
                    pairs = list(pool.map(diagnostics.worker_reconstruct, [(joint.inference_weights(clone),
                        str(path / f"episode_{seed}.npz"), seed, period) for seed in seeds]))
                    for (_, row), reward in zip(pairs, old["frozen_episode_returns"]):
                        if row["episode_return"] != reward or row["action_check"] != "passed":
                            raise ValueError("Stage70 archived actions or rewards changed")
                        cost["archive_episodes"] += 1
                        cost["reconstructed_lower_calls"] += row["lower_calls"]
                        cost["reconstructed_upper_calls"] += row["upper_calls"]
                        cost["archive_network_checks"] += 1
                    batch = concat_hierarchical_batches([b for b, _ in pairs])
                    lower = continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower
                    mc = independent.exact_returns(lower, clone.config.gamma)
                    b0 = independent.common_baseline(lower, horizon=args.horizon, gamma=clone.config.gamma,
                        rate_location=fits["factored"].location)
                    signals, value_rows = {"mc_common": mc - b0}, {}
                    cost["mc_calls"] += 1
                    for t, fit in fits.items():
                        pred = values.control_predictions(fit, lower, clone) if t == "control" else fit.predictions(lower)
                        state = replace(lower, old_value=pred)
                        gae, _ = fit.model._gae(state.reward, state.done, state.duration, state.old_value, state.next_value, state.terminal)
                        value_rows[t] = credit.value_metrics(pred, mc)
                        original_t = "mc_normalized" if t == "control" else "mc_factored"
                        if value_rows[t] != old["TD"][original_t]["value_mc"]:
                            raise ValueError("Stage70 fixed value predictions differ from Stage69")
                        signals["mc_" + t], signals["gae_" + t] = mc - pred, gae
                        cost["probe_value_rows"] += lower.size
                        cost["gae_calls"] += 1
                        cost["source_value_checks"] += 1
                    g, arrays, mask, score_cost = reliability.episode_scores(clone.lower_actor, lower, signals,
                        horizon=args.horizon, clip_ratio=clone.config.clip_ratio)
                    for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
                        cost[key] += score_cost[key]
                    batches.append({"gradients": g, "directions": reliability.fold_gradients(g, arrays, range(len(seeds)))})
                    rows.append({"seeds": seeds, "value_MC": value_rows, "score_cost": score_cost})
                observed = compare_batches(batches, mask)
                reproduce_stage69(observed, source["groups"][p][arm]["comparisons"])
                cost["source_gradient_checks"] += 1
                cost["control_variate_identity_checks"] += 6
                for t, fit in fits.items():
                    independent.assert_frozen(fit.model, snapshots[t])
                    cost["frozen_model_checks"] += 1
                groups[p][arm] = {"comparisons": observed, "batches": rows,
                    "source_reproduction": "passed", "model_and_Adam_unchanged": "passed"}
                print(f"MC control variate {root}/period{period}/{arm}: Stage69 reproduced, fixed baselines compared", flush=True)
            independent.assert_frozen(clone, clone_snapshot)
            cost["frozen_model_checks"] += 1
    cell = {"status": "complete", "root": root, "preflight": preflight, "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "cost": cost, "groups": groups, "seed_roles": roles, "source_initialization": initialization,
        "native_steps": 0, "optimizer_steps": 0, "critic_fits": 0, "checkpoint_writes": 0,
        "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight)
            or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or any(cell[k] for k in ("native_steps", "optimizer_steps", "critic_fits", "checkpoint_writes"))
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage70 fixed archive diagnosis or cost changed")
    for arms in cell["groups"].values():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage70 execution roster changed")
        for g in arms.values():
            if (g["source_reproduction"] != "passed" or g["model_and_Adam_unchanged"] != "passed"
                    or [b["seeds"] for b in g["batches"]] != cell["seed_roles"]["archive_batches"]):
                raise ValueError("Stage70 source or model changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage70 requires every frozen root")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    means = {}
    average = lambda v: None if any(x is None for x in v) else float(np.mean(v))
    for p in map(str, spec.PERIODS):
        means[p] = {}
        for arm in spec.TRAIN_POLICIES:
            parts = {}
            for part in ("all", "mean", "log_std"):
                cs = [r["groups"][p][arm]["comparisons"][part] for r in rows]
                parts[part] = {"estimators": {k: {"covariance_trace": average([c["noise"][k]["covariance_trace"] for c in cs]),
                    "debiased_mean_snr": average([c["noise"][k]["debiased_mean_snr"] for c in cs]),
                    **{name: average([c[name][k]["mean"] for c in cs]) for name in
                        ("within_raw", "within_normalized", "cross_raw_common_reference")}} for k in spec.ESTIMATORS},
                    "state_over_common_variance": {k: average([c["control_variates"][k]["state_over_common_variance"] for c in cs])
                        for k in ("mc_control", "mc_factored")},
                    "GAE_over_MC_variance": {k: average([c["GAE_over_MC_variance"][k] for c in cs]) for k in ("control", "factored")}}
            means[p][arm] = parts
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows, "equal_root_group_means": means,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_frozen_control_variate_diagnosis"}
