"""Fit score-covariance scalars on historical paths, then freeze them on probes."""

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
from . import pointmaze_mc_control_variate as previous
from .pointmaze_root_response import write_json
from scripts import pointmaze_calibrated_cv_stage71_spec as spec


class CovarianceFit:
    """Streaming episode covariance; never retain calibration gradient matrices."""

    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, 0.
        self.baselines = {}

    def update(self, gradients):
        x = gradients["mc_common"]
        n, mean = len(x), x.mean(0)
        total = self.count + n
        delta = mean if self.mean is None else mean - self.mean
        weight = self.count * n / total
        a = x - mean
        self.m2 += float(np.sum(a * a) + weight * np.dot(delta, delta))
        for t in ("control", "factored"):
            h = x - gradients["mc_" + t]
            hm = h.mean(0)
            old = self.baselines.get(t, {"mean": hm, "m2": 0., "cov": 0.})
            hd, b = hm - old["mean"], h - hm
            old["m2"] += float(np.sum(b * b) + weight * np.dot(hd, hd))
            old["cov"] += float(np.sum(a * b) + weight * np.dot(delta, hd))
            old["mean"] = old["mean"] + n / total * hd
            self.baselines[t] = old
        self.mean = mean if self.mean is None else self.mean + n / total * delta
        self.count = total

    def finish(self):
        if self.count < 2:
            raise ValueError("Stage71 covariance fit requires multiple historical episodes")
        result = {}
        for t, row in self.baselines.items():
            alpha = 0. if row["m2"] == 0. else row["cov"] / row["m2"]
            result[t] = {"alpha": alpha, "episodes": self.count,
                "common_variance": self.m2 / (self.count - 1),
                "baseline_variance": row["m2"] / (self.count - 1),
                "trace_common_baseline_covariance": row["cov"] / (self.count - 1),
                "calibration_optimal_variance": (self.m2 + alpha * alpha * row["m2"] - 2 * alpha * row["cov"]) / (self.count - 1),
                "coefficient_parameters": "all", "data_role": "historical_calibration_only"}
        return result


def candidate_signals(signals, coefficients):
    return {"mc_calibrated_" + t: signals["mc_common"] - row["alpha"] *
        (signals["mc_common"] - signals["mc_" + t]) for t, row in coefficients.items()}


def compare_batches(batches, mask):
    result = previous.compare_batches(batches, mask)
    for part, select in {"all": np.ones(len(mask), dtype=bool), "mean": ~mask, "log_std": mask}.items():
        row = result[part]
        common = np.concatenate([b["gradients"]["mc_common"][:, select] for b in batches])
        reference = [b["gradients"]["mc_common"].mean(0)[select] for b in batches]
        for key in spec.ESTIMATORS[len(spec.source.ESTIMATORS):]:
            raw = np.concatenate([b["gradients"][key][:, select] for b in batches])
            means = [b["gradients"][key].mean(0)[select] for b in batches]
            row["noise"][key] = independent.gradient_noise(raw)
            row["within_raw"][key] = reliability.cosine_statistics([scores.cosine(means[i], means[j])
                for i, j in combinations(range(len(batches)), 2)])
            row["within_normalized"][key] = reliability.cosine_statistics([
                scores.cosine(batches[i]["directions"][key][select], batches[j]["directions"][key][select])
                for i, j in combinations(range(len(batches)), 2)])
            row["cross_raw_common_reference"][key] = reliability.cosine_statistics([
                scores.cosine(means[i], reference[j]) for i in range(len(batches)) for j in range(len(batches)) if i != j])
            row["control_variates"][key] = previous.variance_decomposition(common, raw)
    return result


def reproduce_stage70(observed, source):
    def compare(actual, expected):
        if isinstance(expected, dict):
            for key, value in expected.items():
                compare(actual[key], value)
        elif actual is None or expected is None:
            if actual != expected:
                raise ValueError("Stage71 zero-gradient definition differs from Stage70")
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)

    for part in ("all", "mean", "log_std"):
        compare(observed[part], source[part])


def replay(root, *, preflight, output):
    source = json.loads(spec.source_result(root, preflight=preflight).read_text())
    previous.qualify(source, preflight=preflight)
    if source["root"] != root:
        raise ValueError("Stage71 source root changed")
    historical_file = spec.source.values_source.training_result(root, preflight=preflight)
    historical = json.loads(historical_file.read_text())
    historical_archive = historical_file.parent.with_name(historical_file.parent.name + "_raw")
    probe_file = spec.source.source_result(root, preflight=preflight)
    probe = json.loads(probe_file.read_text())
    probe_archive = probe_file.parent.with_name(probe_file.parent.name + "_raw")
    controls = json.loads(spec.source.values_source.source_result(root, preflight=preflight).read_text())
    factored = json.loads(spec.source.source.source_result(root, preflight=preflight).read_text())
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
                    raise ValueError("Stage71 archived actions or rewards changed")
                cost[role + "_archive_episodes"] += 1
                cost["reconstructed_lower_calls"] += row["lower_calls"]
                cost["reconstructed_upper_calls"] += row["upper_calls"]
                cost["archive_network_checks"] += 1
            batch = concat_hierarchical_batches([b for b, _ in pairs])
            return continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower

        def get_signals(clone, fits, lower, *, gae):
            mc = independent.exact_returns(lower, clone.config.gamma)
            b0 = independent.common_baseline(lower, horizon=args.horizon, gamma=clone.config.gamma,
                rate_location=fits["factored"].location)
            result = {"mc_common": mc - b0}
            cost["mc_calls"] += 1
            for t, fit in fits.items():
                pred = values.control_predictions(fit, lower, clone) if t == "control" else fit.predictions(lower)
                result["mc_" + t] = mc - pred
                cost["value_prediction_rows"] += lower.size
                if gae:
                    b = replace(lower, old_value=pred)
                    result["gae_" + t], _ = fit.model._gae(b.reward, b.done, b.duration, b.old_value, b.next_value, b.terminal)
                    cost["gae_calls"] += 1
            return result

        def get_scores(clone, lower, signals):
            g, arrays, mask, score_cost = reliability.episode_scores(clone.lower_actor, lower, signals,
                horizon=args.horizon, clip_ratio=clone.config.clip_ratio)
            for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
                cost[key] += score_cost[key]
            return g, arrays, mask, score_cost

        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            clone_snapshot = copy.deepcopy(clone.state_dict())
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                saved = torch.load(controls["groups"][p][arm]["treatments"]["mc_normalized"]["checkpoint"], map_location="cpu", weights_only=False)
                fits = {"control": normalized.restore_fit(clone, saved, root=root, period=period, arm=arm, treatment="mc_normalized")}
                saved = torch.load(factored["groups"][p][arm]["candidate_checkpoint"], map_location="cpu", weights_only=False)
                if (saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["horizon"]) != (
                        spec.source.values_source.EXPERIMENT_PROTOCOL, root, period, arm, args.horizon):
                    raise ValueError("Stage71 factored critic identity changed")
                fits["factored"] = values.FactoredValueFit.restore(copy.deepcopy(clone), saved)
                snapshots = {t: copy.deepcopy(f.model.state_dict()) for t, f in fits.items()}
                cost["critic_checkpoint_loads"] += len(fits)
                items = historical["calibration"][p][arm]["history"]
                if (len(items) != opt["calibration_batches"] or
                        [r["seed"] for item in items for r in item["rows"]] != roles["calibration"]):
                    raise ValueError("Stage71 calibration roster changed")
                accumulator = CovarianceFit()
                for item in items:
                    path = historical_archive / p / arm / "warmup" / str(item["iteration"]) / "training"
                    lower = reconstruct(clone, period, path, [r["seed"] for r in item["rows"]],
                        [r["episode_return"] for r in item["rows"]], "calibration")
                    g, _, _, _ = get_scores(clone, lower, get_signals(clone, fits, lower, gae=False))
                    accumulator.update(g)
                coefficients = accumulator.finish()
                cost["control_variate_coefficient_fits"] += len(coefficients)
                print(f"calibrated CV {root}/{p}/{arm}: historical coefficients frozen { {t: r['alpha'] for t, r in coefficients.items()} }", flush=True)
                batches, rows = [], []
                for i, seeds in enumerate(roles["archive_batches"]):
                    old = probe["groups"][p][arm]["batches"][i]
                    if seeds != old["seeds"]:
                        raise ValueError("Stage71 probe order changed")
                    lower = reconstruct(clone, period, probe_archive / p / arm / f"batch_{i + 1}",
                        seeds, old["frozen_episode_returns"], "probe")
                    signals = get_signals(clone, fits, lower, gae=True)
                    signals.update(candidate_signals(signals, coefficients))
                    g, arrays, mask, score_cost = get_scores(clone, lower, signals)
                    batches.append({"gradients": g, "directions": reliability.fold_gradients(g, arrays, range(len(seeds)))})
                    rows.append({"seeds": seeds, "score_cost": score_cost})
                observed = compare_batches(batches, mask)
                reproduce_stage70(observed, source["groups"][p][arm]["comparisons"])
                cost["source_gradient_checks"] += 1
                cost["control_variate_identity_checks"] += 12
                for t, fit in fits.items():
                    independent.assert_frozen(fit.model, snapshots[t])
                    cost["frozen_model_checks"] += 1
                groups[p][arm] = {"coefficients": coefficients, "comparisons": observed, "batches": rows,
                    "source_reproduction": "passed", "model_and_Adam_unchanged": "passed"}
                print(f"calibrated CV {root}/{p}/{arm}: probes complete; Stage70 reproduced", flush=True)
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
        raise ValueError("Stage71 frozen calibration, probe or cost changed")
    for arms in cell["groups"].values():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage71 execution roster changed")
        for group in arms.values():
            if (group["source_reproduction"] != "passed" or group["model_and_Adam_unchanged"] != "passed"
                    or [b["seeds"] for b in group["batches"]] != cell["seed_roles"]["archive_batches"]
                    or set(group["coefficients"]) != {"control", "factored"}):
                raise ValueError("Stage71 source, model or coefficients changed")
            for row in group["coefficients"].values():
                if (row["data_role"] != "historical_calibration_only" or row["coefficient_parameters"] != "all"
                        or row["episodes"] != len(cell["seed_roles"]["calibration"]) or not np.isfinite(row["alpha"])):
                    raise ValueError("Stage71 coefficient fit not frozen on historical episodes")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage71 requires every frozen root")
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
                        for k in cs[0]["control_variates"]}}
            means[p][arm] = {"parts": parts, "coefficients": {t: {"mean": average([r["groups"][p][arm]["coefficients"][t]["alpha"] for r in rows]),
                "minimum": min(r["groups"][p][arm]["coefficients"][t]["alpha"] for r in rows),
                "maximum": max(r["groups"][p][arm]["coefficients"][t]["alpha"] for r in rows)} for t in ("control", "factored")}}
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows, "equal_root_group_means": means,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_frozen_calibrated_CV_diagnosis"}
