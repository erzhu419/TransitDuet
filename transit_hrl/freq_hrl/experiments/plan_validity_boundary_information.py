"""Reuse fixed windows or sample new prefixes without reusing exposed B labels."""

from copy import deepcopy
import numpy as np

from .plan_validity_boundary_credit import boundary_row
from .plan_validity_phase_support import observe
from .plan_validity_transfer import run_episode, time_contract


def draw_seed(root, prefix, step, *key):
    return int(np.random.SeedSequence([root, prefix, step, *key]).generate_state(1)[0])


def dense_extra_windows(prefixes):
    keys = []
    for i, seed in enumerate(prefixes):
        keys.append({"seed": seed, "state_index": i % 3, "window_index": i % 4})
        if i < 4:
            keys.append({"seed": seed, "state_index": (i + 1) % 3, "window_index": (i + 2) % 4})
    return keys


def fresh_reference(observation, step, *, args, windows):
    feature, = [r for r in observation["factual"]["feature_rows"] if r["opportunity_step"] == step]
    return {**feature, "boundary_step": step + time_contract(args).upper_period_steps,
        "windows": [{"window_seed": draw_seed(args.optimizer_seed, observation["seed"], step, j, 11401)} for j in range(windows)]}


def acquire_windows(controller, rule, reference, suffix_indices_by_window, *, spec, args, prefix=None):
    scale = time_contract(args)
    if any(k < 4 for indices in suffix_indices_by_window.values() for k in indices):
        raise ValueError("new acquisitions cannot replay old A0/1 or use exposed B2/3")
    step, seed, boundary = reference["opportunity_step"], reference["seed"], reference["boundary_step"]
    output = {k: deepcopy(reference[k]) for k in ("seed", "opportunity_step", "boundary_step", "feature_names", "causal_features")}
    output["windows"] = []
    for j, indices in suffix_indices_by_window.items():
        old = reference["windows"][j]
        window = {"window_index": j, "window_seed": old["window_seed"], "A_replicas": []}
        for k in indices:
            future = draw_seed(args.optimizer_seed, seed, step, j, k, 11402)
            pair = [run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale,
                rule=rule, intervention_step=step, arm=arm, continuation_seed=old["window_seed"],
                boundary_future_seed=future, feature_steps=(boundary,))[1] for arm in ("now", "wait_one_check")]
            if prefix is None:
                prefix = pair[0]["prefix"].copy()
            if any(not np.array_equal(r["prefix"], prefix)
                or tuple(r["feature_names"]) != tuple(reference["feature_names"])
                or not np.array_equal(r["features"], reference["causal_features"]) for r in pair):
                raise RuntimeError("new future changed its fixed causal intervention state")
            context = {arm: boundary_row(r, step=boundary, horizon=args.horizon, period=scale.upper_period_steps)
                for arm, r in zip(("now", "wait_one_check"), pair)}
            costs = [r["window_ise"] for r in pair]
            for expected in (old, window):
                if ("boundary_context" in expected and (context != expected["boundary_context"] or costs != expected["window_costs"])):
                    raise RuntimeError("suffix redraw changed cached window costs or boundary context")
            window.update(boundary_context=context, window_costs=costs)
            w = costs[1] - costs[0]
            tail = pair[1]["tail_ise"] - pair[0]["tail_ise"]
            g = pair[1]["tracking_squared_error_integral"] - pair[0]["tracking_squared_error_integral"]
            if not np.isclose(w + tail, g, rtol=1e-10, atol=1e-10):
                raise RuntimeError("new credit does not reconstruct G")
            window["A_replicas"].append({"suffix_index": k, "suffix_seed": future, "W": w, "tail": tail, "G": g})
        output["windows"].append(window)
    return output


def native_precheck(controller, rule, cached, *, spec, args, seed):
    indices = {j: (1000, 1001) for j in (0, 1)}
    extension = acquire_windows(controller, rule, cached, indices, spec=spec, args=args)
    observation = observe(controller, rule, spec=spec, args=args, seed=seed, probes_per_path=1)
    step = next(s for s in observation["available_steps"] if s % 50 != 0)
    scale = time_contract(args)
    pair = [run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale,
        rule=rule, intervention_step=step, arm=arm)[1] for arm in ("now", "wait_one_check")]
    query, = [q for q in observation["factual"]["queries"] if q["step"] == step]
    replay = pair[0] if query["stop"] else pair[1]
    if (replay["decision_steps"] != observation["factual"]["decision_steps"] or not np.isclose(
        replay["tracking_squared_error_integral"], observation["factual"]["tracking_squared_error_integral"], rtol=1e-10, atol=1e-10)):
        raise RuntimeError("fresh-prefix factual replay changed C0")
    fresh = acquire_windows(controller, rule, fresh_reference(observation, step, args=args, windows=2),
        indices, spec=spec, args=args, prefix=pair[0]["prefix"])
    return {"cached_extension": extension, "fresh_prefix": fresh}
