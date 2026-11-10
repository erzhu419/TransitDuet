"""Matched L/W/G supervision on a shared frozen-policy state and future bank."""

from __future__ import annotations

import numpy as np

from .plan_validity_reliability import decompose_probe
from .plan_validity_transfer import run_episode, time_contract
from .pointmaze_compact_plan_validity import _base_matrix, _causal_validity_interactions
from .pointmaze_timing_baselines import TimingRule, _ridge_at_df


TARGETS = {"L": "local_advantage", "W": "adaptive_window_advantage", "G": "full_advantage"}


def matrix(rows):
    return _causal_validity_interactions(*_base_matrix(rows))


def labels(rows, target, block):
    values = []
    for row in rows:
        futures = row["conditional_replicas"]
        half = len(futures) // 2
        if half < 1 or len(futures) % 2:
            raise ValueError("labels require equal nonempty future blocks")
        selected = futures[:half] if block == "A" else futures[half:] if block == "B" else futures
        values.append([r[TARGETS[target]] for r in selected])
    return np.asarray(values, dtype=float)


def collect_bank(controller, rule, *, spec, args, seeds, probes_per_path, replicas):
    scale = time_contract(args)
    grid = tuple(range(50, args.horizon - 199, 50))
    rows = []
    for seed in seeds:
        def run(**kwargs):
            return run_episode(controller, task_spec=spec, seed=seed, args=args,
                               time_scale=scale, rule=rule, **kwargs)[1]

        factual = run(feature_steps=grid)
        features = {r["opportunity_step"]: r for r in factual["feature_rows"]}
        queries = {r["step"]: r for r in factual["queries"]}
        rng = np.random.default_rng(np.random.SeedSequence([args.optimizer_seed, seed, 617]))
        checks = sorted(rng.choice(grid, probes_per_path, replace=False).tolist())
        for step in checks:
            factual_pair = [run(intervention_step=step, arm=arm) for arm in ("now", "wait_one_check")]
            replay = factual_pair[0] if queries[step]["stop"] else factual_pair[1]
            if (replay["decision_steps"] != factual["decision_steps"] or not np.isclose(
                replay["tracking_squared_error_integral"], factual["tracking_squared_error_integral"],
                rtol=1e-10, atol=1e-10)):
                raise RuntimeError("training-bank intervention failed factual replay")
            future_rows = []
            for replica in range(replicas):
                future = int(np.random.SeedSequence([args.optimizer_seed, seed, step, replica, 618]).generate_state(1)[0])
                local = [run(branch_step=step, renew=value, continuation_seed=future) for value in (False, True)]
                full = [run(intervention_step=step, arm=arm, continuation_seed=future)
                        for arm in ("now", "wait_one_check")]
                if any(not np.array_equal(r["prefix"], factual_pair[0]["prefix"]) for r in (*local, *full)):
                    raise RuntimeError("target-bank future changed the fixed causal prefix")
                if not np.array_equal(local[0]["features"], features[step]["causal_features"]):
                    raise RuntimeError("target-bank features changed from the factual state")
                future_rows.append({"future_seed": future, **decompose_probe(*local, *full)})
            rows.append({**features[step], "factual_stop": queries[step]["stop"],
                         "conditional_replicas": future_rows})
    return rows


def fit_targets(rows, *, effective_df, threshold_quantile):
    x, names = matrix(rows)
    groups = np.asarray([r["seed"] for r in rows])
    if len(np.unique(groups)) < 3:
        raise ValueError("target comparison needs at least three fitting paths")
    rules, oof = {}, {}
    for target in TARGETS:
        y = labels(rows, target, "A").mean(axis=1)
        prediction = np.empty(len(rows))
        for group in np.unique(groups):
            train = groups != group
            prediction[~train], _ = _ridge_at_df(x[train], y[train], x[~train], effective_df)
        _, model = _ridge_at_df(x, y, x, effective_df)
        fitted = {"feature_names": list(names), "model": model,
            "threshold_source": "training_block_A_leave_one_path_out_fixed_quantile",
            "threshold_quantile": threshold_quantile}
        rules[target] = TimingRule(f"credit_{target}", "legacy", view="causal_validity_interactions",
            threshold=float(np.quantile(prediction, threshold_quantile)), model=fitted)
        oof[target] = prediction
    return rules, oof


def predict(rule, rows):
    x = matrix(rows)[0]
    model = rule.model["model"]
    return model["weights"][0] + ((x - model["feature_mean"]) / model["feature_scale"]) @ np.asarray(model["weights"][1:])


def prediction_error(scores, rows, target, block, *, independent):
    values = labels(rows, target, block)
    means = values.mean(axis=1)
    noise = values.var(axis=1, ddof=1) / values.shape[1]
    mse = float(np.mean((scores - means) ** 2))
    return {"paths": len(set(r["seed"] for r in rows)), "states": len(rows),
        "mse_to_repeat_mean": mse, "label_mean_variance": float(noise.mean()),
        "noise_corrected_mse": float(mse - noise.mean()) if independent else None,
        "mse_minus_zero": float(np.mean(scores ** 2 - 2 * scores * means)),
        "bias": float(np.mean(scores - means))}


def fitting_readout(rows, rules, oof):
    return {target: {"own_training_A": prediction_error(predict(rule, rows), rows, target, "A", independent=False),
        "own_independent_future_B": prediction_error(predict(rule, rows), rows, target, "B", independent=True),
        "own_prefix_oof_B": prediction_error(oof[target], rows, target, "B", independent=True),
        "episode_independent_future_B": prediction_error(predict(rule, rows), rows, "G", "B", independent=True),
        "effective_df": rule.model["model"]["effective_df_slopes"], "threshold": rule.threshold}
        for target, rule in rules.items()}


def held_readout(rows, rules):
    return {target: {block: {"own": prediction_error(predict(rule, rows), rows, target, block, independent=True),
        "episode": prediction_error(predict(rule, rows), rows, "G", block, independent=True)}
        for block in ("A", "B", "all")} for target, rule in rules.items()}


def label_uncertainty(rows):
    output = {}
    for target in TARGETS:
        values = labels(rows, target, "all")
        half = values.shape[1] // 2
        a, b = values[:, :half].mean(axis=1), values[:, half:].mean(axis=1)
        output[target] = {"mean": float(values.mean()),
            "future_variance": float(values.var(axis=1, ddof=1).mean()),
            "mean_label_variance": float((values.var(axis=1, ddof=1) / values.shape[1]).mean()),
            "cross_block_conditional_second_moment": float(np.mean(a * b)),
            "half_sign_agreement": float(np.mean((a > 0) == (b > 0)))}
    return output
