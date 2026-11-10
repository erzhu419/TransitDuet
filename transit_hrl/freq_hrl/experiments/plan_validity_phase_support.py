"""Paired bin-start/reachable-check sampling and clock-conditioned W heads."""

from __future__ import annotations

from collections import Counter
import numpy as np

from .plan_validity_target_credit import matrix as legacy_matrix
from .plan_validity_transfer import run_episode, time_contract
from .plan_validity_window_coverage import error, values
from .pointmaze_timing_baselines import TimingRule, _ridge_at_df


ARMS = ("start_plain", "start_clock", "checks_plain", "checks_clock")


def clock_row(row, *, horizon):
    step = row["opportunity_step"]
    return {**row, "offset_fraction": (step % 50) / 25, "remaining_fraction": (horizon - step) / horizon}


def feature_matrix(rows, *, clock):
    x, names = legacy_matrix(rows)
    if clock:
        x = np.column_stack((x, [r["offset_fraction"] for r in rows], [r["remaining_fraction"] for r in rows]))
        names = (*names, "offset_fraction", "remaining_fraction")
    return x, names


def predict(rule, rows):
    x = feature_matrix(rows, clock=rule.kind == "phase_ridge")[0]
    model = rule.model["model"]
    return model["weights"][0] + ((x - model["feature_mean"]) / model["feature_scale"]) @ np.asarray(model["weights"][1:])


def fit(rows, *, clock, effective_df, threshold_quantile):
    x, names = feature_matrix(rows, clock=clock)
    y = values(rows, "W", "A").mean(axis=1)
    groups = np.asarray([r["seed"] for r in rows])
    oof = np.empty(len(rows))
    for group in np.unique(groups):
        train = groups != group
        oof[~train], _ = _ridge_at_df(x[train], y[train], x[~train], effective_df)
    _, model = _ridge_at_df(x, y, x, effective_df)
    return TimingRule("phase_clock" if clock else "phase_plain", "phase_ridge" if clock else "legacy",
        view="causal_validity_interactions", threshold=float(np.quantile(oof, threshold_quantile)),
        model={"feature_names": list(names), "model": model, "threshold_source": "training_A_path_oof_fixed_quantile",
               "threshold_quantile": threshold_quantile}), oof


def observe(controller, rule, *, spec, args, seed, probes_per_path):
    grid = tuple(s + offset for s in range(50, args.horizon - 199, 50) for offset in (0, 5, 10, 15, 20)
                 if s + offset + 200 <= args.horizon)
    factual = run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=time_contract(args),
                          rule=rule, feature_steps=grid)[1]
    starts = np.asarray(list(range(50, args.horizon - 199, 50)))
    rng = np.random.default_rng(np.random.SeedSequence([args.optimizer_seed, seed, 617]))
    starts = sorted(rng.choice(starts, probes_per_path, replace=False).tolist())
    available = {q["step"]: q for q in factual["queries"] if q["step"] in grid}
    checks = []
    for start in starts:
        options = sorted(s for s in available if s // 50 == start // 50)
        rng = np.random.default_rng(np.random.SeedSequence([args.optimizer_seed, seed, start, 839]))
        checks.append(int(rng.choice(options)))
    return {"seed": seed, "factual": factual, "start_steps": starts, "check_steps": checks,
            "available_steps": sorted(available)}


def sample(controller, rule, observation, steps, *, spec, args, fitting_futures, evaluation_futures):
    factual = observation["factual"]
    features = {r["opportunity_step"]: r for r in factual["feature_rows"]}
    queries = {r["step"]: r for r in factual["queries"]}
    rows = []
    for step in steps:
        def run(**kwargs):
            return run_episode(controller, task_spec=spec, seed=observation["seed"], args=args,
                time_scale=time_contract(args), rule=rule, intervention_step=step, **kwargs)[1]

        factual_pair = [run(arm=arm) for arm in ("now", "wait_one_check")]
        replay = factual_pair[0] if queries[step]["stop"] else factual_pair[1]
        if (replay["decision_steps"] != factual["decision_steps"] or not np.isclose(
            replay["tracking_squared_error_integral"], factual["tracking_squared_error_integral"],
            rtol=1e-10, atol=1e-10)):
            raise RuntimeError("phase intervention failed factual replay")
        replicas = []
        for replica in range(fitting_futures + evaluation_futures):
            future = int(np.random.SeedSequence([args.optimizer_seed, observation["seed"], step, replica, 840]).generate_state(1)[0])
            now, wait = [run(arm=arm, continuation_seed=future) for arm in ("now", "wait_one_check")]
            if (any(not np.array_equal(r["prefix"], factual_pair[0]["prefix"]) for r in (now, wait))
                or not np.array_equal(now["features"], features[step]["causal_features"])):
                raise RuntimeError("phase future changed its causal prefix or features")
            window = wait["window_ise"] - now["window_ise"]
            tail = wait["tail_ise"] - now["tail_ise"]
            full = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
            if not np.isclose(full, window + tail, rtol=1e-10, atol=1e-10):
                raise RuntimeError("phase window and tail do not reconstruct episode credit")
            replicas.append({"future_seed": future, "adaptive_window_advantage": window,
                             "tail_advantage": tail, "full_advantage": full})
        rows.append(clock_row({**features[step], "factual_stop": queries[step]["stop"],
            "A_replicas": replicas[:fitting_futures], "B_replicas": replicas[fitting_futures:]}, horizon=args.horizon))
    return rows


def phase_counts(rows):
    return {str(offset): sum(r["opportunity_step"] % 50 == offset for r in rows) for offset in (0, 5, 10, 15, 20)}


def training_readout(rows, rule, oof):
    scores = predict(rule, rows)
    return {"fitting_future_count": sum(len(r["A_replicas"]) for r in rows),
        "effective_df": rule.model["model"]["effective_df_slopes"], "threshold": rule.threshold,
        "phase_counts": phase_counts(rows),
        "own_train_A": error(scores, rows, target="W", block="A", independent=False),
        "own_independent_B": error(scores, rows, target="W", block="B", independent=True),
        "own_prefix_oof_B": error(oof, rows, target="W", block="B", independent=True)}


def held_readout(rows, rule):
    if not rows:
        return None
    scores = predict(rule, rows)
    return {block: {target: error(scores, rows, target=target, block=block, independent=True)
            for target in ("W", "G")} for block in ("A", "B")}


def deployment_counts(row):
    return {"queries": dict(Counter(str(q["offset"]) for q in row["queries"])),
            "renewals": dict(Counter(str(s % 50) for s in row["decision_steps"][1:]))}
