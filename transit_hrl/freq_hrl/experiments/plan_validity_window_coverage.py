"""Window-credit coverage versus effective complexity on equal fitting budgets."""

from __future__ import annotations

import numpy as np

from .plan_validity_target_credit import matrix, predict
from .plan_validity_transfer import run_episode, time_contract
from .pointmaze_timing_baselines import TimingRule, _ridge_at_df


ARMS = ("dense_df2", "dense_df8", "wide_df2", "wide_df8")


def cached_rows(rows, fitting_futures):
    return [{**{k: v for k, v in row.items() if k != "conditional_replicas"},
        "A_replicas": row["conditional_replicas"][:fitting_futures],
        "B_replicas": row["conditional_replicas"][4:8]} for row in rows]


def values(rows, target, block):
    key = "adaptive_window_advantage" if target == "W" else "full_advantage"
    return np.asarray([[r[key] for r in row[block + "_replicas"]] for row in rows], dtype=float)


def collect_window_bank(controller, rule, *, spec, args, seeds, probes_per_path, fitting_futures, evaluation_futures):
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
        for step in sorted(rng.choice(grid, probes_per_path, replace=False).tolist()):
            factual_pair = [run(intervention_step=step, arm=arm) for arm in ("now", "wait_one_check")]
            replay = factual_pair[0] if queries[step]["stop"] else factual_pair[1]
            if (replay["decision_steps"] != factual["decision_steps"] or not np.isclose(
                replay["tracking_squared_error_integral"], factual["tracking_squared_error_integral"],
                rtol=1e-10, atol=1e-10)):
                raise RuntimeError("window bank failed factual replay")
            replicas = []
            for replica in range(fitting_futures + evaluation_futures):
                future = int(np.random.SeedSequence([args.optimizer_seed, seed, step, replica, 728]).generate_state(1)[0])
                now, wait = [run(intervention_step=step, arm=arm, continuation_seed=future)
                             for arm in ("now", "wait_one_check")]
                if any(not np.array_equal(r["prefix"], factual_pair[0]["prefix"]) for r in (now, wait)):
                    raise RuntimeError("window bank changed its causal prefix")
                if not np.array_equal(now["features"], features[step]["causal_features"]):
                    raise RuntimeError("window bank changed its causal features")
                window = wait["window_ise"] - now["window_ise"]
                tail = wait["tail_ise"] - now["tail_ise"]
                full = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
                if not np.isclose(full, window + tail, rtol=1e-10, atol=1e-10):
                    raise RuntimeError("window and tail do not reconstruct episode credit")
                replicas.append({"future_seed": future, "adaptive_window_advantage": window,
                                 "tail_advantage": tail, "full_advantage": full})
            rows.append({**features[step], "factual_stop": queries[step]["stop"],
                "A_replicas": replicas[:fitting_futures], "B_replicas": replicas[fitting_futures:]})
    return rows


def fit_window(rows, *, effective_df, threshold_quantile):
    x, names = matrix(rows)
    y = values(rows, "W", "A").mean(axis=1)
    groups = np.asarray([r["seed"] for r in rows])
    oof = np.empty(len(rows))
    for group in np.unique(groups):
        train = groups != group
        oof[~train], _ = _ridge_at_df(x[train], y[train], x[~train], effective_df)
    _, model = _ridge_at_df(x, y, x, effective_df)
    return TimingRule(f"window_df{int(effective_df)}", "legacy", view="causal_validity_interactions",
        threshold=float(np.quantile(oof, threshold_quantile)), model={"feature_names": list(names), "model": model,
            "threshold_source": "training_A_path_oof_fixed_quantile", "threshold_quantile": threshold_quantile}), oof


def error(scores, rows, *, target, block, independent):
    samples = values(rows, target, block)
    mean = samples.mean(axis=1)
    mse = float(np.mean((scores - mean) ** 2))
    zero = float(np.mean(mean ** 2))
    noise = float(np.mean(samples.var(axis=1, ddof=1) / samples.shape[1])) if samples.shape[1] > 1 else None
    return {"paths": len(set(r["seed"] for r in rows)), "states": len(rows), "mse": mse,
        "mse_minus_zero": float(np.mean(scores ** 2 - 2 * scores * mean)),
        "mse_over_zero": mse / zero if zero else None, "label_mean_variance": noise,
        "noise_corrected_mse": mse - noise if independent and noise is not None else None,
        "bias": float(np.mean(scores - mean))}


def training_readout(rows, rule, oof):
    scores = predict(rule, rows)
    return {"fitting_future_count": sum(len(r["A_replicas"]) for r in rows),
        "effective_df": rule.model["model"]["effective_df_slopes"], "threshold": rule.threshold,
        "own_train_A": error(scores, rows, target="W", block="A", independent=False),
        "own_independent_B": error(scores, rows, target="W", block="B", independent=True),
        "own_prefix_oof_B": error(oof, rows, target="W", block="B", independent=True)}


def held_readout(rows, rule):
    return {block: {target: error(predict(rule, rows), rows, target=target, block=block, independent=True)
            for target in ("W", "G")} for block in ("A", "B")}
