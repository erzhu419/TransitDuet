"""Independent gradient noise and exact TD/value-error attribution with fixed policies."""

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
from . import pointmaze_horizon_value as values
from . import pointmaze_normalized_update as normalized
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as native
from . import pointmaze_joint_renewal as joint
from . import pointmaze_learned_plan as learned
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_independent_credit_stage69_spec as spec


def common_baseline(lower, *, horizon, gamma, rate_location):
    remaining = np.tile(np.arange(horizon, 0, -1), lower.size // horizon)
    np.testing.assert_array_equal(np.rint(lower.value_state[:, -1] * horizon), remaining)
    return values.discounted_mass(remaining, gamma).astype(np.float64) * rate_location


def exact_returns(lower, gamma):
    target, successor = np.empty(lower.size, dtype=np.float64), 0.
    for i in range(lower.size - 1, -1, -1):
        # Native float32 done must not downcast the Python/double recurrence.
        successor = float(lower.reward[i]) + gamma ** int(lower.duration[i]) * (1. - float(lower.done[i])) * successor
        target[i] = successor
    return target


def gradient_noise(episodes):
    g = np.asarray(episodes, dtype=np.float64)
    n, mean = len(g), g.mean(0)
    variance = float(np.square(g - mean).sum() / (n - 1))
    power = float(np.dot(mean, mean))
    return {"episodes": n, "mean_squared_norm": power, "covariance_trace": variance,
        "unbiased_signal_power": power - variance / n,
        "debiased_mean_snr": None if variance == 0 else n * power / variance - 1.}


def td_attribution(lower, pred, mc, advantage, *, gamma, lam, horizon, period):
    if lower.next_value is not None or lower.terminal is not None or np.any(lower.duration != 1):
        raise ValueError("Stage69 requires native primitive true-episode bootstrap semantics")
    done = np.tile(np.r_[np.zeros(horizon - 1), 1.], lower.size // horizon)
    np.testing.assert_array_equal(lower.done, done)
    c, error = 1. - done, np.asarray(pred, dtype=np.float64) - mc
    successor = np.r_[pred[1:], 0.].astype(np.float64)
    next_error = np.r_[error[1:], 0.]
    td = lower.reward.astype(np.float64) + gamma * c * successor - pred
    mc_td = lower.reward.astype(np.float64) + gamma * c * np.r_[mc[1:], 0.] - mc
    np.testing.assert_allclose(mc_td, 0., atol=1e-10, rtol=0)
    np.testing.assert_allclose(td, mc_td + gamma * c * next_error - error, atol=1e-10, rtol=0)
    correction, last = np.zeros_like(error), 0.
    for i in range(len(error) - 1, -1, -1):
        last = gamma * (1. - lam) * c[i] * next_error[i] + gamma * lam * c[i] * last
        correction[i] = last
    difference = advantage.astype(np.float64) - (mc - pred)
    np.testing.assert_allclose(difference, correction, atol=2e-4, rtol=1e-6)
    steps = np.arange(len(error)) % horizon
    renew = (steps % period == period - 1) & (done == 0)
    summary = lambda x: {"count": int(len(x)), "mse": float(np.mean(np.square(x))),
        "bias": float(np.mean(x)), "std": float(np.std(x))}
    return {"value_mc": credit.value_metrics(pred, mc), "TD": summary(td),
        "renewal_TD": summary(td[renew]), "other_TD": summary(td[~renew & (done == 0)]),
        "tail_value_error": summary(error[steps >= int(.9 * horizon)]),
        "filtered_error": summary(correction), "identity_max_abs_error": float(np.max(np.abs(difference - correction)))}


def compare_batches(old, batches, mask):
    masks = {"all": np.ones(len(mask), dtype=bool), "mean": ~mask, "log_std": mask}
    directions = [b["directions"] for b in batches]
    raw = {k: np.concatenate([b["gradients"][k] for b in batches]) for k in ("mc_common", *spec.TREATMENTS)}
    result = {}
    for part, select in masks.items():
        cos = lambda a, b: scores.cosine(a[select], b[select])
        stats = reliability.cosine_statistics
        mc = [b["gradients"]["mc_common"].mean(0) for b in batches]
        result[part] = {"raw_episode_noise": {k: gradient_noise(v[:, select]) for k, v in raw.items()},
            "within_common_MC": stats([cos(mc[i], mc[j]) for i, j in combinations(range(len(batches)), 2)]),
            "critics": {}}
        for t in spec.TREATMENTS:
            result[part]["critics"][t] = {
                "within_GAE": stats([cos(directions[i][t], directions[j][t]) for i, j in combinations(range(len(batches)), 2)]),
                "same_batch_GAE_common_MC": stats([cos(d[t], m) for d, m in zip(directions, mc)]),
                "cross_batch_GAE_common_MC": stats([cos(directions[i][t], mc[j])
                    for i in range(len(batches)) for j in range(len(batches)) if i != j]),
                "anchor_GAE_fresh_common_MC": stats([cos(old["directions"][t], m) for m in mc]),
                "cross_batch_entropy_adjusted_common_MC": stats([cos(directions[i][t + "_entropy"], mc[j])
                    for i in range(len(batches)) for j in range(len(batches)) if i != j])}
    return result


def init_worker(config, args):
    diagnostics.init_worker(config, args)
    native.init_worker(config, args)


def replay(root, *, preflight, output):
    source = json.loads(spec.source_result(root, preflight=preflight).read_text())
    if ((source["status"], source["root"], source["preflight"], source["protocol"]) !=
            ("complete", root, preflight, spec.previous.source.EXPERIMENT_PROTOCOL)
            or source["contract"] != spec.previous.source.contract()):
        raise ValueError("Stage69 requires the frozen complete Stage67 source")
    original = json.loads(spec.training_result(root, preflight=preflight).read_text())
    controls = json.loads(spec.previous.source.source_result(root, preflight=preflight).read_text())
    clones, predictor, initialization = native.load_source(root, preflight=preflight)
    opt, args, roles = spec.options(preflight=preflight), spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    historical = spec.native.seed_roles(root, preflight=preflight)
    used = {s for seeds in historical.values() for s in seeds}
    fresh = [s for seeds in roles["fresh_batches"] for s in seeds]
    if len(set(fresh)) != len(fresh) or used.intersection(fresh):
        raise ValueError("Stage69 fresh sampling overlaps frozen historical roles")
    archive = raw_directory(spec.training_result(root, preflight=preflight))
    raw = raw_directory(output)
    cost, counts = dict.fromkeys(spec.budget(preflight=preflight), 0), dict.fromkeys(learned.COUNT_KEYS, 0)
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    groups, started = {}, time.monotonic()
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            frozen_clone = copy.deepcopy(clone.state_dict())
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                saved = torch.load(controls["groups"][p][arm]["treatments"]["mc_normalized"]["checkpoint"], map_location="cpu", weights_only=False)
                fits = {"mc_normalized": normalized.restore_fit(clone, saved, root=root, period=period, arm=arm, treatment="mc_normalized")}
                saved = torch.load(source["groups"][p][arm]["candidate_checkpoint"], map_location="cpu", weights_only=False)
                if (saved["protocol"], saved["root"], saved["period"], saved["arm"]) != (spec.previous.source.EXPERIMENT_PROTOCOL, root, period, arm):
                    raise ValueError("Stage69 factored checkpoint identity changed")
                fits["mc_factored"] = values.FactoredValueFit.restore(copy.deepcopy(clone), saved)
                if fits["mc_factored"].horizon != args.horizon:
                    raise ValueError("Stage69 factored horizon differs from native rollout")
                snapshots = {t: copy.deepcopy(f.model.state_dict()) for t, f in fits.items()}
                cost["critic_checkpoint_loads"] += len(fits)
                first = original["training"][p][arm]["history"][0]
                if [r["seed"] for r in first["rows"]] != roles["anchor"]:
                    raise ValueError("Stage69 anchor roster changed")
                path = archive / p / arm / "train" / "1" / "training"
                weights = joint.inference_weights(clone)
                pairs = list(pool.map(diagnostics.worker_reconstruct, [(weights,
                    str(path / f"episode_{r['seed']}.npz"), r["seed"], period) for r in first["rows"]]))
                for (_, row), old in zip(pairs, first["rows"]):
                    if row["episode_return"] != old["episode_return"] or row["action_check"] != "passed":
                        raise ValueError("Stage69 anchor trajectory changed")
                    cost["archive_episodes"] += 1
                    cost["reconstructed_lower_calls"] += row["lower_calls"]
                    cost["reconstructed_upper_calls"] += row["upper_calls"]
                    cost["archive_network_checks"] += 1

                def probe(outputs, anchor=False):
                    batch = concat_hierarchical_batches([b for b, _ in outputs])
                    lower = continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower
                    mc = exact_returns(lower, clone.config.gamma)
                    historical_mc = values.values.monte_carlo_returns(lower, clone.config.gamma) if anchor else None
                    if anchor:
                        cost["source_mc_reproduction_calls"] += 1
                    baseline = common_baseline(lower, horizon=args.horizon, gamma=clone.config.gamma,
                        rate_location=fits["mc_factored"].location)
                    signals, rows = {"mc_common": mc - baseline}, {}
                    cost["mc_calls"] += 1
                    for t, fit in fits.items():
                        pred = values.control_predictions(fit, lower, clone) if t == "mc_normalized" else fit.predictions(lower)
                        b = replace(lower, old_value=pred)
                        advantage, _ = fit.model._gae(b.reward, b.done, b.duration, b.old_value, b.next_value, b.terminal)
                        rows[t] = td_attribution(b, pred, mc, advantage, gamma=clone.config.gamma,
                            lam=clone.config.gae_lambda, horizon=args.horizon, period=period)
                        if anchor:
                            expected = source["groups"][p][arm]["treatments"][t]
                            if credit.value_metrics(pred, historical_mc) != expected["value_mc"] or float(np.std(advantage)) != expected["gae_advantage_std"]:
                                raise ValueError("Stage69 anchor value/GAE differs from Stage67")
                            cost["source_probe_checks"] += 1
                        signals[t] = advantage
                        cost["probe_value_rows"] += lower.size
                        cost["gae_calls"] += 1
                        cost["td_identity_checks"] += 1
                    g, arrays, mask, score_cost = reliability.episode_scores(clone.lower_actor, lower, signals,
                        horizon=args.horizon, clip_ratio=clone.config.clip_ratio)
                    for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
                        cost[key] += score_cost[key]
                    direction = reliability.fold_gradients(g, arrays, range(lower.size // args.horizon))
                    for t in spec.TREATMENTS:
                        direction[t + "_entropy"] = direction[t] + clone.config.entropy_coef * direction["entropy"]
                    return {"gradients": g, "directions": direction, "TD": rows, "score_cost": score_cost,
                        "historical_MC_rounding_max_abs": None if not anchor else float(np.max(np.abs(mc - historical_mc)))}, mask

                anchor, mask = probe(pairs, anchor=True)
                batches, batch_rows = [], []
                for i, seeds in enumerate(roles["fresh_batches"]):
                    path = raw / p / arm / f"batch_{i + 1}"
                    path.mkdir(parents=True, exist_ok=True)
                    outputs = list(pool.map(native.worker_rollout, [(weights, s, arm, period, "train", "training", predictor,
                        str(path / f"episode_{s}.npz")) for s in seeds]))
                    rows = [r for _, r in outputs]
                    joint.audit_trajectories(rows, args=args, method=f"fixed{period}", raw_path=path)
                    if [r["seed"] for r in rows] != seeds or any(r["rollout_network_check"] != "passed" for r in rows):
                        raise ValueError("Stage69 frozen native roster or weights changed")
                    for row in rows:
                        counts["primitive_steps"] += row["episode_length"]
                        for key in learned.COUNT_KEYS[1:]:
                            counts[key] += row[key]
                    for key in ("native_episodes", "native_trace_audits", "native_network_checks"):
                        cost[key] += len(rows)
                    measured, new_mask = probe(outputs)
                    np.testing.assert_array_equal(new_mask, mask)
                    batches.append(measured)
                    batch_rows.append({"seeds": seeds, "TD": measured["TD"], "score_cost": measured["score_cost"],
                        "frozen_episode_returns": [r["episode_return"] for r in rows]})
                    print(f"independent credit {root}/period{period}/{arm}: batch{i + 1}/{opt['batches']} complete", flush=True)
                for t, fit in fits.items():
                    torch.testing.assert_close(fit.model.state_dict(), snapshots[t], atol=0, rtol=0)
                    cost["frozen_model_checks"] += 1
                groups[p][arm] = {"common_rate_location": fits["mc_factored"].location,
                    "anchor": {"TD": anchor["TD"], "score_cost": anchor["score_cost"],
                        "historical_MC_rounding_max_abs": anchor["historical_MC_rounding_max_abs"]}, "batches": batch_rows,
                    "comparisons": compare_batches(anchor, batches, mask), "model_and_Adam_unchanged": "passed"}
            torch.testing.assert_close(clone.state_dict(), frozen_clone, atol=0, rtol=0)
            cost["frozen_model_checks"] += 1
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "cost": cost, "native_counts": counts, "groups": groups, "seed_roles": roles,
        "source_initialization": initialization, "optimizer_steps": 0, "critic_fits": 0, "checkpoint_writes": 0,
        "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return result


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["native_counts"] != spec.native_budget(preflight=preflight)
            or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "checkpoint_writes"))):
        raise ValueError("Stage69 frozen sampling, identity or cost changed")
    if set(cell["groups"]) != {str(p) for p in spec.PERIODS}:
        raise ValueError("Stage69 period roster changed")
    for arms in cell["groups"].values():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage69 execution roster changed")
        for group in arms.values():
            if (group["model_and_Adam_unchanged"] != "passed" or
                    [b["seeds"] for b in group["batches"]] != cell["seed_roles"]["fresh_batches"]):
                raise ValueError("Stage69 policy or independent batch roster changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage69 requires every frozen root")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    group_means = {}
    for period in spec.PERIODS:
        p = str(period)
        group_means[p] = {}
        for arm in spec.TRAIN_POLICIES:
            groups = [c["groups"][p][arm] for c in rows]
            mean = lambda v: None if any(x is None for x in v) else float(np.mean(v))
            parts = {}
            for part in ("all", "mean", "log_std"):
                comparisons = [g["comparisons"][part] for g in groups]
                parts[part] = {"within_common_MC": mean([c["within_common_MC"]["mean"] for c in comparisons]),
                    "raw_common_MC_mean_snr": mean([c["raw_episode_noise"]["mc_common"]["debiased_mean_snr"] for c in comparisons]),
                    "critics": {t: {**{k: mean([c["critics"][t][k]["mean"] for c in comparisons])
                        for k in comparisons[0]["critics"][t]},
                        "raw_GAE_mean_snr": mean([c["raw_episode_noise"][t]["debiased_mean_snr"] for c in comparisons])}
                        for t in spec.TREATMENTS}}
            td = {t: {k: float(np.mean([b["TD"][t][section][metric] for g in groups for b in g["batches"]]))
                for k, section, metric in (("value_MSE", "value_mc", "mse"), ("TD_MSE", "TD", "mse"),
                    ("renewal_TD_MSE", "renewal_TD", "mse"), ("filtered_error_MSE", "filtered_error", "mse"),
                    ("tail_value_error_bias", "tail_value_error", "bias"))} for t in spec.TREATMENTS}
            group_means[p][arm] = {"gradient": parts, "fresh_TD": td}
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_counts": {k: sum(c["native_counts"][k] for c in rows) for k in spec.native_budget(preflight=preflight)},
        "equal_root_group_means": group_means, "statistics": "descriptive_root_means_no_batch_pair_CI",
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_frozen_policy_diagnosis"}
