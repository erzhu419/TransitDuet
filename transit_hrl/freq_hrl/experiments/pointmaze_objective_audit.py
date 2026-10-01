"""Audit objective-aligned score estimates without changing the controller."""

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
from . import pointmaze_credit_reliability as reliability
from . import pointmaze_independent_credit as independent
from . import pointmaze_horizon_value as values
from . import pointmaze_normalized_update as normalized
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as native
from . import pointmaze_joint_renewal as joint
from . import pointmaze_mc_control_variate as legacy
from . import pointmaze_calibrated_cv as previous
from .pointmaze_root_response import write_json
from scripts import pointmaze_objective_audit_stage72_spec as spec


def objective_signals(lower, *, horizon, gamma, discounted_location, native_location):
    time_index = np.arange(lower.size) % horizon
    discounted = independent.exact_returns(lower, gamma)
    native_returns = independent.exact_returns(lower, 1.)
    np.testing.assert_allclose(native_returns.reshape(-1, horizon)[:, 0],
        lower.reward.astype(np.float64).reshape(-1, horizon).sum(1), atol=1e-9, rtol=0)
    common = discounted - independent.common_baseline(lower, horizon=horizon, gamma=gamma, rate_location=discounted_location)
    return {"mc_common": common, "mc_discounted_objective": np.power(gamma, time_index) * common,
        "mc_native": native_returns - (horizon - time_index) * native_location,
        "mc_native_zero": native_returns}, discounted


def compare_batches(batches, mask):
    result = {}
    for part, select in {"all": np.ones(len(mask), dtype=bool), "mean": ~mask, "log_std": mask}.items():
        raw = {k: np.concatenate([b["gradients"][k][:, select] for b in batches]) for k in spec.ESTIMATORS}
        means = {k: [b["gradients"][k].mean(0)[select] for b in batches] for k in spec.ESTIMATORS}
        row = {"noise": {k: independent.gradient_noise(g) for k, g in raw.items()},
            "within_raw": {k: reliability.cosine_statistics([scores.cosine(g[i], g[j])
                for i, j in combinations(range(len(batches)), 2)]) for k, g in means.items()},
            "within_normalized": {k: reliability.cosine_statistics([
                scores.cosine(batches[i]["directions"][k][select], batches[j]["directions"][k][select])
                for i, j in combinations(range(len(batches)), 2)]) for k in spec.ESTIMATORS}}
        for name, reference in (("cross_raw_native_reference", "mc_native"),
                ("cross_raw_discounted_reference", "mc_discounted_objective")):
            row[name] = {k: reliability.cosine_statistics([scores.cosine(g[i], means[reference][j])
                for i in range(len(batches)) for j in range(len(batches)) if i != j]) for k, g in means.items()}
        denominator = row["noise"]["mc_native_zero"]["covariance_trace"]
        row["native_baseline_over_zero_variance"] = None if denominator == 0 else row["noise"]["mc_native"]["covariance_trace"] / denominator
        result[part] = row
    return result


def replay(root, *, preflight, output):
    prerequisite = json.loads(spec.prerequisite_result(root, preflight=preflight).read_text())
    previous.qualify(prerequisite, preflight=preflight)
    if prerequisite["root"] != root:
        raise ValueError("Stage72 prerequisite root changed")
    controls = json.loads(spec.legacy.values_source.source_result(root, preflight=preflight).read_text())
    factored = json.loads(spec.legacy.source.source_result(root, preflight=preflight).read_text())
    historical_file = spec.legacy.values_source.training_result(root, preflight=preflight)
    historical = json.loads(historical_file.read_text())
    historical_archive = historical_file.parent.with_name(historical_file.parent.name + "_raw")
    probe_file = spec.legacy.source_result(root, preflight=preflight)
    probe = json.loads(probe_file.read_text())
    probe_archive = probe_file.parent.with_name(probe_file.parent.name + "_raw")
    reference = json.loads(spec.source.source_result(root, preflight=preflight).read_text())
    clones, _, initialization = native.load_source(root, preflight=preflight)
    opt, args, roles = spec.options(preflight=preflight), spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=diagnostics.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        def reconstruct(clone, period, path, seeds, rewards, role):
            pairs = list(pool.map(diagnostics.worker_reconstruct, [(joint.inference_weights(clone),
                str(path / f"episode_{seed}.npz"), seed, period) for seed in seeds]))
            for (_, row), reward in zip(pairs, rewards):
                if row["episode_return"] != reward or row["action_check"] != "passed":
                    raise ValueError("Stage72 archived actions or rewards changed")
                cost[role + "_archive_episodes"] += 1
                cost["reconstructed_lower_calls"] += row["lower_calls"]
                cost["reconstructed_upper_calls"] += row["upper_calls"]
                cost["archive_network_checks"] += 1
            batch = concat_hierarchical_batches([b for b, _ in pairs])
            return continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower

        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            clone_snapshot = copy.deepcopy(clone.state_dict())
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                first = historical["calibration"][p][arm]["history"][0]
                if first["iteration"] != 1 or [r["seed"] for r in first["rows"]] != roles["first_calibration"]:
                    raise ValueError("Stage72 first historical frame roster changed")
                lower = reconstruct(clone, period, historical_archive / p / arm / "warmup" / "1" / "training",
                    roles["first_calibration"], [r["episode_return"] for r in first["rows"]], "calibration")
                location = float(lower.reward.astype(np.float64).mean())
                frame = {"location": location, "data_role": "first_historical_calibration_only",
                    "episodes": len(roles["first_calibration"]), "sample_count": lower.size}
                cost["historical_reward_rate_fits"] += 1
                saved = torch.load(controls["groups"][p][arm]["treatments"]["mc_normalized"]["checkpoint"], map_location="cpu", weights_only=False)
                fits = {"control": normalized.restore_fit(clone, saved, root=root, period=period, arm=arm, treatment="mc_normalized")}
                saved = torch.load(factored["groups"][p][arm]["candidate_checkpoint"], map_location="cpu", weights_only=False)
                if (saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["horizon"]) != (
                        spec.legacy.values_source.EXPERIMENT_PROTOCOL, root, period, arm, args.horizon):
                    raise ValueError("Stage72 factored critic identity changed")
                fits["factored"] = values.FactoredValueFit.restore(copy.deepcopy(clone), saved)
                snapshots = {t: copy.deepcopy(f.model.state_dict()) for t, f in fits.items()}
                cost["critic_checkpoint_loads"] += len(fits)
                batches, rows = [], []
                for i, seeds in enumerate(roles["archive_batches"]):
                    old = probe["groups"][p][arm]["batches"][i]
                    if seeds != old["seeds"]:
                        raise ValueError("Stage72 probe order changed")
                    lower = reconstruct(clone, period, probe_archive / p / arm / f"batch_{i + 1}",
                        seeds, old["frozen_episode_returns"], "probe")
                    signals, discounted = objective_signals(lower, horizon=args.horizon, gamma=clone.config.gamma,
                        discounted_location=fits["factored"].location, native_location=location)
                    cost["mc_calls"] += 2
                    cost["native_return_identity_checks"] += 1
                    for t, fit in fits.items():
                        pred = values.control_predictions(fit, lower, clone) if t == "control" else fit.predictions(lower)
                        b = replace(lower, old_value=pred)
                        signals["mc_" + t] = discounted - pred
                        signals["gae_" + t], _ = fit.model._gae(b.reward, b.done, b.duration, b.old_value, b.next_value, b.terminal)
                        cost["probe_value_rows"] += lower.size
                        cost["gae_calls"] += 1
                    g, arrays, mask, score_cost = reliability.episode_scores(clone.lower_actor, lower, signals,
                        horizon=args.horizon, clip_ratio=clone.config.clip_ratio)
                    for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
                        cost[key] += score_cost[key]
                    batches.append({"gradients": g, "directions": reliability.fold_gradients(g, arrays, range(len(seeds)))})
                    rounding = signals["mc_native_zero"].reshape(-1, args.horizon)[:, 0] - old["frozen_episode_returns"]
                    rows.append({"seeds": seeds, "score_cost": score_cost, "max_native_return_rounding_error": float(np.max(np.abs(rounding)))})
                previous.reproduce_stage70(legacy.compare_batches(batches, mask), reference["groups"][p][arm]["comparisons"])
                cost["source_gradient_checks"] += 1
                cost["legacy_variance_identity_checks"] += 6
                observed = compare_batches(batches, mask)
                for t, fit in fits.items():
                    independent.assert_frozen(fit.model, snapshots[t])
                    cost["frozen_model_checks"] += 1
                groups[p][arm] = {"native_baseline_frame": frame, "comparisons": observed, "batches": rows,
                    "source_reproduction": "passed", "model_and_Adam_unchanged": "passed"}
                print(f"objective audit {root}/{p}/{arm}: native/discounted score estimates; Stage70 reproduced", flush=True)
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
        raise ValueError("Stage72 frozen objective audit or budget changed")
    for arms in cell["groups"].values():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage72 execution roster changed")
        for g in arms.values():
            frame = g["native_baseline_frame"]
            if (g["source_reproduction"] != "passed" or g["model_and_Adam_unchanged"] != "passed"
                    or [b["seeds"] for b in g["batches"]] != cell["seed_roles"]["archive_batches"]
                    or frame["data_role"] != "first_historical_calibration_only"
                    or frame["episodes"] != len(cell["seed_roles"]["first_calibration"])):
                raise ValueError("Stage72 baseline frame or frozen source changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage72 requires every frozen root")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    means = {}
    avg = lambda v: None if any(x is None for x in v) else float(np.mean(v))
    for p in map(str, spec.PERIODS):
        means[p] = {}
        for arm in spec.TRAIN_POLICIES:
            parts = {}
            for part in ("all", "mean", "log_std"):
                cs = [r["groups"][p][arm]["comparisons"][part] for r in rows]
                parts[part] = {"estimators": {k: {"covariance_trace": avg([c["noise"][k]["covariance_trace"] for c in cs]),
                    "debiased_mean_snr": avg([c["noise"][k]["debiased_mean_snr"] for c in cs]),
                    **{name: avg([c[name][k]["mean"] for c in cs]) for name in
                        ("within_raw", "within_normalized", "cross_raw_native_reference", "cross_raw_discounted_reference")}}
                    for k in spec.ESTIMATORS}, "native_baseline_over_zero_variance": avg([c["native_baseline_over_zero_variance"] for c in cs])}
            means[p][arm] = parts
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows, "equal_root_group_means": means,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_frozen_objective_audit"}
