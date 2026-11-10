"""Change label continuation on identical candidate-visited causal states."""

from __future__ import annotations

import numpy as np

from .plan_validity_phase_support import clock_row, fit, predict, training_readout
from .plan_validity_transfer import run_episode, time_contract
from .plan_validity_window_coverage import error, values


CONTINUATIONS = ("reference", "candidate")
HEADS = ("label_reference", "label_candidate")


def sample_labels(controller, rules, observation, step, *, spec, args, a, b, tag):
    factual = observation["factual"]
    feature = next(r for r in factual["feature_rows"] if r["opportunity_step"] == step)
    query = next(q for q in factual["queries"] if q["step"] == step)
    futures = [int(np.random.SeedSequence([args.optimizer_seed, observation["seed"], step, r, tag]).generate_state(1)[0])
        for r in range(a + b)]
    output, captured = {}, None
    for name in CONTINUATIONS:
        def pair(future=None):
            return [run_episode(controller, task_spec=spec, seed=observation["seed"], args=args,
                time_scale=time_contract(args), rule=rules["candidate"], intervention_step=step, arm=arm,
                continuation_rule=rules[name], continuation_seed=future)[1] for arm in ("now", "wait_one_check")]

        original = pair()
        if captured is None:
            captured = original[0]
        if name == "candidate":
            replay = original[0] if query["stop"] else original[1]
            if (replay["decision_steps"] != factual["decision_steps"] or not np.isclose(
                replay["tracking_squared_error_integral"], factual["tracking_squared_error_integral"], rtol=1e-10, atol=1e-10)):
                raise RuntimeError("candidate-continuation intervention failed factual replay")
        replicas = []
        for future in futures:
            now, wait = pair(future)
            if any(not np.array_equal(r["prefix"], captured["prefix"]) or not np.array_equal(
                r["features"], feature["causal_features"]) for r in (*original, now, wait)):
                raise RuntimeError("label continuation changed the candidate causal state")
            w = wait["window_ise"] - now["window_ise"]
            tail = wait["tail_ise"] - now["tail_ise"]
            g = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
            if not np.isclose(w + tail, g, rtol=1e-10, atol=1e-10):
                raise RuntimeError("label window and tail do not reconstruct full credit")
            replicas.append({"future_seed": future, "adaptive_window_advantage": w,
                "tail_advantage": tail, "full_advantage": g})
        output[name] = clock_row({**feature, "factual_stop": query["stop"],
            "A_replicas": replicas[:a], "B_replicas": replicas[a:]}, horizon=args.horizon)
    return output


def validate_banks(banks, *, seeds, probes, a, b):
    first, second = (banks[c] for c in CONTINUATIONS)
    roster = lambda rows: [(r["seed"], r["opportunity_step"]) for r in rows]
    if (len(first) != len(seeds) * probes or roster(first) != roster(second)
        or len(set(roster(first))) != len(first) or set(r["seed"] for r in first) != set(seeds)
        or any(sum(r["seed"] == seed for r in first) != probes for seed in seeds)):
        raise ValueError("paired candidate-state bank roster differs")
    for left, right in zip(first, second):
        if (left["feature_names"] != right["feature_names"] or not np.array_equal(left["causal_features"], right["causal_features"])
            or any(left[k] != right[k] for k in ("offset_fraction", "remaining_fraction", "factual_stop"))):
            raise ValueError("label arms do not share identical observable states")
        ids = []
        for block, count in (("A_replicas", a), ("B_replicas", b)):
            l, r = left[block], right[block]
            if len(l) != count or [v["future_seed"] for v in l] != [v["future_seed"] for v in r]:
                raise ValueError("label arms have different future counts or pairing")
            ids.extend(v["future_seed"] for v in l)
        if len(set(ids)) != a + b:
            raise ValueError("fitting and independent futures overlap")


def fit_labels(banks, *, effective_df, threshold_quantile):
    rules, report = {}, {}
    for continuation, name in zip(CONTINUATIONS, HEADS):
        rules[name], oof = fit(banks[continuation], clock=True, effective_df=effective_df,
            threshold_quantile=threshold_quantile)
        rules[name].name = name
        report[name] = {"own": training_readout(banks[continuation], rules[name], oof),
            "cross_target_B": {c: {"full_fit": error(predict(rules[name], rows), rows, target="W", block="B", independent=True),
                "prefix_oof": error(oof, rows, target="W", block="B", independent=True)} for c, rows in banks.items()}}
    return rules, report


def _effect(samples):
    return {"mean": float(samples.mean()), "conditional_MC_standard_error": float(np.sqrt(
        np.sum(samples.var(axis=1, ddof=1) / samples.shape[1])) / samples.shape[0])}


def single_step_readout(banks, rules):
    rows = banks["candidate"]
    scores = {name: np.asarray([rule.score(tuple(r["feature_names"]), r["causal_features"],
        r["offset_fraction"], r["remaining_fraction"]) for r in rows]) for name, rule in rules.items()}
    actions = {name: score >= rules[name].threshold for name, score in scores.items()}
    result = {}
    for block in ("A", "B", "all"):
        targets = {}
        for continuation in CONTINUATIONS:
            sample_rows = banks[continuation]
            samples = {key: (np.column_stack((values(sample_rows, key, "A"), values(sample_rows, key, "B")))
                if block == "all" else values(sample_rows, key, block)) for key in ("W", "G")}
            samples["tail"] = samples["G"] - samples["W"]
            y, noise = samples["W"].mean(axis=1), samples["W"].var(axis=1, ddof=1) / samples["W"].shape[1]
            heads = {name: {"W_mse": float(np.mean((score - y) ** 2)),
                "noise_corrected_W_mse": float(np.mean((score - y) ** 2 - noise)),
                "gains_against_source_C": {k: _effect((actions[name].astype(int) - actions["source_C"].astype(int))[:, None] * v)
                    for k, v in samples.items()}} for name, score in scores.items()}
            difference = actions["label_candidate"].astype(int) - actions["label_reference"].astype(int)
            targets[continuation] = {"heads": heads, "candidate_minus_reference_label": {
                "W_mse_difference": heads["label_candidate"]["W_mse"] - heads["label_reference"]["W_mse"],
                "gains": {k: _effect(difference[:, None] * v) for k, v in samples.items()}}}
        result[block] = targets
    return result
