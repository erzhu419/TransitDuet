"""Separate same-batch reference movement from episode-level gradient stability."""

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
from itertools import combinations
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_actor_credit as scores
from . import pointmaze_continuing_credit as continuing
from . import pointmaze_credit_diagnostics as credit
from . import pointmaze_horizon_value as horizon_values
from . import pointmaze_normalized_update as normalized
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as previous
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import write_json
from scripts import pointmaze_credit_reliability_stage68_spec as spec


def balanced_partitions(count):
    half, all_indices = count // 2, set(range(count))
    for others in combinations(range(1, count), half - 1):
        first = (0, *others)
        yield first, tuple(sorted(all_indices - set(first)))


def episode_scores(actor, lower, signals, *, horizon, clip_ratio):
    signals = {k: np.asarray(v, dtype=np.float32).reshape(-1, horizon) for k, v in signals.items()}
    rows, mask, cost = [], None, {"actor_score_forward_batches": 0, "actor_score_backward_batches": 0,
        "max_abs_old_logp_difference": 0.}
    for i in range(lower.size // horizon):
        start, stop = i * horizon, (i + 1) * horizon
        part = replace(lower, **{k: v[start:stop] for k, v in vars(lower).items() if v is not None})
        g, mask, row = scores.actor_gradients(actor, part,
            {**{k: v[i] for k, v in signals.items()}, "one": np.ones(horizon, dtype=np.float32)},
            clip_ratio=clip_ratio, chunk_size=spec.CHUNK_SIZE)
        if row["max_abs_old_logp_difference"] >= min(-math.log(1. - clip_ratio), math.log(1. + clip_ratio)):
            raise ValueError("clipped preupdate ratios invalidate linear fold reconstruction")
        rows.append(g)
        for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
            cost[key] += row[key]
        cost["max_abs_old_logp_difference"] = max(cost["max_abs_old_logp_difference"], row["max_abs_old_logp_difference"])
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}, signals, mask, cost


def fold_gradients(gradients, signals, indices):
    idx = list(indices)
    one = gradients["one"][idx].mean(0)
    result = {"entropy": gradients["entropy"][idx].mean(0)}
    for key, values in signals.items():
        subset = values[idx].reshape(-1)
        # A separately centered fold reproduces its own PPO normalization.
        result[key] = (gradients[key][idx].mean(0) - float(subset.mean()) * one) / (float(subset.std()) + 1e-8)
    return result


def cosine_statistics(values):
    valid = [v for v in values if v is not None]
    return {"count": len(values), "defined": len(valid), "undefined": len(values) - len(valid),
        "mean": None if not valid else float(np.mean(valid)), "minimum": None if not valid else min(valid),
        "maximum": None if not valid else max(valid),
        "positive_fraction": None if not valid else float(np.mean(np.asarray(valid) > 0))}


def compare_gradients(episodes, signals, sigma_mask):
    count = len(next(iter(episodes.values()))["one"])
    masks = {"all": np.ones(len(sigma_mask), dtype=bool), "mean": ~sigma_mask, "log_std": sigma_mask}
    full = {t: fold_gradients(g, signals[t], range(count)) for t, g in episodes.items()}
    metrics = {part: {"within": {t: {k: [] for k in ("gae", "mc")} for t in episodes},
        "cross_reference": {f"{a}_GAE/{b}_MC": [] for a in episodes for b in episodes}} for part in masks}
    partitions = list(balanced_partitions(count))
    for first, second in partitions:
        folds = {t: (fold_gradients(g, signals[t], first), fold_gradients(g, signals[t], second)) for t, g in episodes.items()}
        for part, mask in masks.items():
            for t, (left, right) in folds.items():
                for key in ("gae", "mc"):
                    metrics[part]["within"][t][key].append(scores.cosine(left[key][mask], right[key][mask]))
            for a, (left, right) in folds.items():
                for b, (other_left, other_right) in folds.items():
                    values = metrics[part]["cross_reference"][f"{a}_GAE/{b}_MC"]
                    values.extend((scores.cosine(left["gae"][mask], other_right["mc"][mask]),
                        scores.cosine(right["gae"][mask], other_left["mc"][mask])))
    result = {}
    for part, mask in masks.items():
        result[part] = {"within": {t: {k: cosine_statistics(v) for k, v in row.items()}
                for t, row in metrics[part]["within"].items()},
            "cross_reference": {k: cosine_statistics(v) for k, v in metrics[part]["cross_reference"].items()},
            "same_batch_reference": {f"{a}_GAE/{b}_MC": scores.cosine(full[a]["gae"][mask], full[b]["mc"][mask])
                for a in episodes for b in episodes},
            "between_critics": {key: scores.cosine(full[spec.TREATMENTS[0]][key][mask], full[spec.TREATMENTS[1]][key][mask])
                for key in ("gae", "mc")}}
    return result, full, len(partitions)


def replay(root, *, preflight, output):
    source = json.loads(spec.source_result(root, preflight=preflight).read_text())
    if ((source["status"], source["root"], source["preflight"], source["protocol"]) !=
            ("complete", root, preflight, spec.source.EXPERIMENT_PROTOCOL) or source["contract"] != spec.source.contract()):
        raise ValueError("Stage68 requires the completed frozen Stage67 source")
    original = json.loads(spec.training_result(root, preflight=preflight).read_text())
    critics = json.loads(spec.source.source_result(root, preflight=preflight).read_text())
    clones, _, initialization = previous.load_source(root, preflight=preflight)
    roles, args = spec.seed_roles(root, preflight=preflight), spec.arguments(root, preflight=preflight)
    directory = spec.training_result(root, preflight=preflight).parent
    directory = directory.with_name(directory.name + "_raw")
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"), initializer=diagnostics.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                saved = torch.load(critics["groups"][p][arm]["treatments"]["mc_normalized"]["checkpoint"], map_location="cpu", weights_only=False)
                fits = {"mc_normalized": normalized.restore_fit(clone, saved, root=root, period=period, arm=arm, treatment="mc_normalized")}
                saved = torch.load(source["groups"][p][arm]["candidate_checkpoint"], map_location="cpu", weights_only=False)
                if (saved["protocol"], saved["root"], saved["period"], saved["arm"]) != (spec.source.EXPERIMENT_PROTOCOL, root, period, arm):
                    raise ValueError("Stage68 factored checkpoint identity changed")
                fits[spec.source.CANDIDATE] = horizon_values.FactoredValueFit.restore(copy.deepcopy(clone), saved)
                cost["critic_checkpoint_loads"] += len(fits)
                first = original["training"][p][arm]["history"][0]
                if [r["seed"] for r in first["rows"]] != roles["first_training_probe"]:
                    raise ValueError("Stage68 first training episode roster changed")
                path = directory / p / arm / "train" / "1" / "training"
                pairs = list(pool.map(diagnostics.worker_reconstruct, [(joint.inference_weights(clone),
                    str(path / f"episode_{r['seed']}.npz"), r["seed"], period) for r in first["rows"]]))
                for (_, row), old in zip(pairs, first["rows"]):
                    if row["episode_return"] != old["episode_return"] or row["action_check"] != "passed":
                        raise ValueError("Stage68 archived action or reward changed")
                    cost["archive_episodes"] += 1
                    cost["reconstructed_lower_calls"] += row["lower_calls"]
                    cost["reconstructed_upper_calls"] += row["upper_calls"]
                    cost["archive_network_checks"] += 1
                batch = concat_hierarchical_batches([b for b, _ in pairs])
                lower = continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower
                mc = horizon_values.values.monte_carlo_returns(lower, clone.config.gamma)
                cost["mc_calls"] += 1
                episode_gradients, signal_arrays, sigma_mask, snapshots, treatment_rows = {}, {}, None, {}, {}
                for t, fit in fits.items():
                    snapshots[t] = copy.deepcopy(fit.model.state_dict())
                    pred = horizon_values.control_predictions(fit, lower, clone) if t == "mc_normalized" else fit.predictions(lower)
                    b = replace(lower, old_value=pred)
                    advantage, _ = fit.model._gae(b.reward, b.done, b.duration, b.old_value, b.next_value, b.terminal)
                    expected = source["groups"][p][arm]["treatments"][t]
                    if credit.value_metrics(pred, mc) != expected["value_mc"] or float(np.std(advantage)) != expected["gae_advantage_std"]:
                        raise ValueError("Stage68 does not reproduce the Stage67 value/GAE probe")
                    cost["probe_value_rows"] += lower.size
                    cost["gae_calls"] += 1
                    cost["source_probe_checks"] += 1
                    g, arrays, sigma_mask, score_cost = episode_scores(fit.model.lower_actor, b,
                        {"gae": advantage, "mc": mc - pred}, horizon=args.horizon, clip_ratio=fit.model.config.clip_ratio)
                    episode_gradients[t], signal_arrays[t] = g, arrays
                    for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
                        cost[key] += score_cost[key]
                    treatment_rows[t] = {"score_cost": score_cost}
                comparisons, full, count = compare_gradients(episode_gradients, signal_arrays, sigma_mask)
                cost["balanced_partitions"] += count
                for t, fit in fits.items():
                    observed = scores.gradient_summary(full[t], sigma_mask, fit.model.config.entropy_coef)
                    expected = source["groups"][p][arm]["treatments"][t]["gradient"]
                    for part in ("all", "mean", "log_std"):
                        for key in observed[part]:
                            a, b = observed[part][key], expected[part][key]
                            if a is None or b is None:
                                if a != b:
                                    raise ValueError("Stage68 zero-gradient definition differs from Stage67")
                            else:
                                np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
                    before, after = snapshots[t], fit.model.state_dict()
                    if before.pop("config") != after.pop("config"):
                        raise ValueError("Stage68 configuration changed")
                    torch.testing.assert_close(before, after, atol=0, rtol=0)
                    cost["source_gradient_checks"] += 1
                    cost["frozen_model_checks"] += 1
                    treatment_rows[t].update(full_gradient=observed, model_and_Adam_unchanged="passed", source_reproduction="passed")
                groups[p][arm] = {"partition_count": count, "comparisons": comparisons, "treatments": treatment_rows}
                print(f"credit reliability {root}/period{period}/{arm}: {count} balanced partitions; Stage67 probe/gradient reproduced", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "cost": cost, "groups": groups, "source_initialization": initialization,
        "new_native_steps": 0, "optimizer_steps": 0, "critic_fits": 0, "checkpoint_writes": 0,
        "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return result


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("new_native_steps", "optimizer_steps", "critic_fits", "checkpoint_writes"))):
        raise ValueError("Stage68 frozen roster, diagnosis or cost changed")
    episodes = spec.source.options(preflight=preflight)["rollouts_per_iteration"]
    count = math.comb(episodes - 1, episodes // 2 - 1)
    for arms in cell["groups"].values():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage68 arm roster changed")
        for group in arms.values():
            if (group["partition_count"] != count or set(group["treatments"]) != set(spec.TREATMENTS)
                    or any(r["model_and_Adam_unchanged"] != "passed" or r["source_reproduction"] != "passed" for r in group["treatments"].values())):
                raise ValueError("Stage68 source gradient or model changed")
            for part in group["comparisons"].values():
                for row in part["within"].values():
                    if any(v["count"] != count or v["defined"] + v["undefined"] != count for v in row.values()):
                        raise ValueError("Stage68 within-credit split accounting changed")
                if any(v["count"] != 2 * count or v["defined"] + v["undefined"] != 2 * count for v in part["cross_reference"].values()):
                    raise ValueError("Stage68 cross-credit split accounting changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage68 requires all frozen roots")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_readonly_reliability"}
