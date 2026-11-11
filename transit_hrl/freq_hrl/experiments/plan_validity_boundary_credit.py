"""Separate realized-window and conditional-suffix variation in paired credit."""

import numpy as np

from .plan_validity_transfer import run_episode, time_contract


def variance_components(samples):
    """Equal-size nested samples: rows are windows, columns are suffix redraws."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("use at least two finite windows and two suffixes per window")
    window_means = x.mean(axis=1)
    within = float(x.var(axis=1, ddof=1).mean())
    mean_variance = float(window_means.var(ddof=1))
    between = mean_variance - within / x.shape[1]
    return {"mean": float(x.mean()), "within_window": within,
        "between_window_corrected": between, "total_corrected": within + between,
        "nested_mean_mc_se": float(np.sqrt(mean_variance / x.shape[0]))}


def boundary_row(rollout, *, step, horizon, period):
    row, = rollout["feature_rows"]
    if row["opportunity_step"] != step:
        raise RuntimeError("suffix context was not captured at the window boundary")
    start = step - step % period
    return {"step": step, "feature_names": row["feature_names"],
        "causal_features": row["causal_features"], "offset_fraction": (step % period) / 25,
        "remaining_fraction": (horizon - step) / horizon,
        "call_spent": any(start <= s < step for s in rollout["decision_steps"])}


def collect_boundary_pair(controller, rule, observation, step, *, spec, args, windows, suffixes):
    """Acquire independent window draws, each with paired suffix redraws under C0."""
    if min(windows, suffixes) < 2:
        raise ValueError("the nested design needs at least two draws at each level")
    scale = time_contract(args)
    boundary = step + scale.upper_period_steps
    seed = observation["seed"]
    arms = ("now", "wait_one_check")

    def pair(**kwargs):
        return [run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale,
            rule=rule, intervention_step=step, arm=arm, **kwargs)[1] for arm in arms]

    original = pair()
    query = next(q for q in observation["factual"]["queries"] if q["step"] == step)
    factual = observation["factual"]
    replay = original[0] if query["stop"] else original[1]
    if replay["decision_steps"] != factual["decision_steps"] or not np.isclose(
        replay["tracking_squared_error_integral"], factual["tracking_squared_error_integral"], rtol=1e-10, atol=1e-10):
        raise RuntimeError("boundary credit failed factual C0 replay")

    def draw(*key):
        return int(np.random.SeedSequence([args.optimizer_seed, seed, step, *key]).generate_state(1)[0])

    seeds = [(draw(j, 11401), [draw(j, k, 11402) for k in range(suffixes)]) for j in range(windows)]
    if len({s for w, tail in seeds for s in (w, *tail)}) != windows * (1 + suffixes):
        raise ValueError("window and suffix seeds overlap")
    output = []
    for window_seed, suffix_seeds in seeds:
        window = None
        for suffix_seed in suffix_seeds:
            now, wait = pair(continuation_seed=window_seed, boundary_future_seed=suffix_seed,
                feature_steps=(boundary,))
            if any(not np.array_equal(r["prefix"], original[0]["prefix"]) or not np.array_equal(
                r["features"], original[0]["features"]) for r in (*original, now, wait)):
                raise RuntimeError("nested draw changed the intervention's causal state")
            context = {arm: boundary_row(r, step=boundary, horizon=args.horizon, period=scale.upper_period_steps)
                for arm, r in zip(arms, (now, wait))}
            window_costs = [now["window_ise"], wait["window_ise"]]
            if window is None:
                window = {"window_seed": window_seed, "boundary_context": context,
                    "window_costs": window_costs, "suffix_draws": []}
            elif (context != window["boundary_context"] or window_costs != window["window_costs"]):
                raise RuntimeError("a suffix redraw changed its fixed window or boundary context")
            w = window_costs[1] - window_costs[0]
            tail = wait["tail_ise"] - now["tail_ise"]
            g = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
            if not np.isclose(w + tail, g, rtol=1e-10, atol=1e-10):
                raise RuntimeError("window and suffix credit do not reconstruct G")
            window["suffix_draws"].append({"suffix_seed": suffix_seed, "W": w, "tail": tail, "G": g})
        output.append(window)
    return {"seed": seed, "opportunity_step": step, "boundary_step": boundary,
        "feature_names": list(original[0]["feature_names"]), "causal_features": original[0]["features"].tolist(),
        "windows": output, "variance_components": {target: variance_components(
            [[r[target] for r in w["suffix_draws"]] for w in output]) for target in ("W", "tail", "G")}}
