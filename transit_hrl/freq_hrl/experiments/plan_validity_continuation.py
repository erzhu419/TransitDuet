"""Frozen prefix-by-continuation diagnostics with paired conditional futures."""

from __future__ import annotations

from collections import Counter
import numpy as np

from .plan_validity_reliability import state_distribution
from .plan_validity_transfer import run_episode, time_contract


POLICIES = ("reference", "candidate")
CONTEXTS = ("reference/reference", "reference/candidate", "candidate/reference", "candidate/candidate")
DECISION_CHOICES = ("candidate", "score_zero", "W_sign", "G_sign")
CONTRASTS = {
    "continuation_at_reference": [(-1, "reference/reference"), (1, "reference/candidate")],
    "continuation_at_candidate": [(-1, "candidate/reference"), (1, "candidate/candidate")],
    "prefix_under_reference": [(-1, "reference/reference"), (1, "candidate/reference")],
    "prefix_under_candidate": [(-1, "reference/candidate"), (1, "candidate/candidate")],
    "interaction": [(1, "reference/reference"), (-1, "reference/candidate"),
                    (-1, "candidate/reference"), (1, "candidate/candidate")],
}


def factual_pair(controller, rules, *, spec, args, seed):
    grid = tuple(s + k for s in range(50, args.horizon - 199, 50) for k in (0, 5, 10, 15, 20)
                 if s + k + 200 <= args.horizon)
    return {name: run_episode(controller, task_spec=spec, seed=seed, args=args,
        time_scale=time_contract(args), rule=rule, feature_steps=grid)[1] for name, rule in rules.items()}


def query_support(factual, *, horizon):
    sets = {name: {q["step"] for q in row["queries"] if 50 <= q["step"] <= horizon - 200 and q["offset"] < 25}
            for name, row in factual.items()}
    common = sets["reference"] & sets["candidate"]
    return {"sets": sets, "common": common,
        "counts": {"common": len(common), "reference_only": len(sets["reference"] - common),
                   "candidate_only": len(sets["candidate"] - common)},
        "phase_counts": {name: dict(Counter(str(s % 50) for s in steps)) for name, steps in sets.items()}}


def selected_checks(factual, *, args, seed, probes_per_path):
    support = query_support(factual, horizon=args.horizon)
    bins = np.arange(50, args.horizon - 199, 50)
    rng = np.random.default_rng(np.random.SeedSequence([args.optimizer_seed, seed, 960]))
    checks = []
    for start in sorted(rng.choice(bins, probes_per_path, replace=False).tolist()):
        available = sorted(s for s in support["common"] if s // 50 == start // 50)
        rng = np.random.default_rng(np.random.SeedSequence([args.optimizer_seed, seed, start, 961]))
        checks.append(int(rng.choice(available)))
    return checks, support


def paired_probe(controller, rules, factual, *, spec, args, seed, step, replicas):
    future_seeds = [int(np.random.SeedSequence([args.optimizer_seed, seed, step, r, 962]).generate_state(1)[0])
                    for r in range(replicas)]
    output = {"seed": seed, "step": step, "offset": step % 50, "features": {}, "actions": {}, "effects": {}}
    for prefix in POLICIES:
        captured = None
        query = next(q for q in factual[prefix]["queries"] if q["step"] == step)
        for continuation in POLICIES:
            def pair(future=None):
                return [run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=time_contract(args),
                    rule=rules[prefix], intervention_step=step, arm=arm, continuation_rule=rules[continuation],
                    continuation_seed=future)[1] for arm in ("now", "wait_one_check")]

            original_pair = pair()
            if captured is None:
                captured = original_pair[0]
                output["features"][prefix] = {"feature_names": captured["feature_names"],
                    "causal_features": captured["features"], "offset_fraction": (step % 50) / 25,
                    "remaining_fraction": (args.horizon - step) / args.horizon}
                output["actions"][prefix] = {name: rules[name].score(tuple(captured["feature_names"]), captured["features"],
                    (step % 50) / 25, (args.horizon - step) / args.horizon) >= rules[name].threshold for name in POLICIES}
                output.setdefault("candidate_scores", {})[prefix] = rules["candidate"].score(
                    tuple(captured["feature_names"]), captured["features"], (step % 50) / 25, (args.horizon - step) / args.horizon)
            if prefix == continuation:
                replay = original_pair[0] if query["stop"] else original_pair[1]
                if (replay["decision_steps"] != factual[prefix]["decision_steps"] or not np.isclose(
                    replay["tracking_squared_error_integral"], factual[prefix]["tracking_squared_error_integral"],
                    rtol=1e-10, atol=1e-10)):
                    raise RuntimeError("same-policy continuation failed factual replay")
            effects = []
            for future in future_seeds:
                now, wait = pair(future)
                if any(not np.array_equal(r["prefix"], captured["prefix"]) or not np.array_equal(
                    r["features"], captured["features"]) for r in (*original_pair, now, wait)):
                    raise RuntimeError("continuation or conditional future changed the causal prefix")
                window = wait["window_ise"] - now["window_ise"]
                tail = wait["tail_ise"] - now["tail_ise"]
                full = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
                if not np.isclose(full, window + tail, rtol=1e-10, atol=1e-10):
                    raise RuntimeError("continuation W and tail do not reconstruct G")
                effects.append({"future_seed": future, "W": window, "tail": tail, "G": full})
            output["effects"][prefix + "/" + continuation] = effects
    return output


def probe_summary(probes):
    output = {}
    for block in ("A", "B", "all"):
        arrays = {}
        for context in CONTEXTS:
            values = []
            for probe in probes:
                rows = probe["effects"][context]
                half = len(rows) // 2
                chosen = rows[:half] if block == "A" else rows[half:] if block == "B" else rows
                values.append([[r[k] for k in ("W", "tail", "G")] for r in chosen])
            arrays[context] = np.asarray(values, dtype=float)
        means = {k: v.mean(axis=1) for k, v in arrays.items()}
        contexts = {}
        for context in CONTEXTS:
            prefix = context.split("/")[0]
            score = np.asarray([p["candidate_scores"][prefix] for p in probes])
            delta = np.asarray([int(p["actions"][prefix]["candidate"]) - int(p["actions"][prefix]["reference"]) for p in probes])
            y = means[context]
            noise = arrays[context][:, :, 0].var(axis=1, ddof=1) / arrays[context].shape[1]
            contexts[context] = {"mean_W": float(y[:, 0].mean()), "mean_tail": float(y[:, 1].mean()),
                "mean_G": float(y[:, 2].mean()), "W_G_sign_disagreement": float(np.mean(y[:, 0] * y[:, 2] < 0)),
                "candidate_W_mse": float(np.mean((score - y[:, 0]) ** 2)),
                "candidate_W_noise_corrected_mse": float(np.mean((score - y[:, 0]) ** 2 - noise)),
                "candidate_action_gain_W": float(np.mean(delta * y[:, 0])),
                "candidate_action_gain_G": float(np.mean(delta * y[:, 2]))}
        contrasts = {}
        for name, terms in CONTRASTS.items():
            paired = sum(sign * arrays[context] for sign, context in terms)
            contrasts[name] = {key: {"mean": float(paired[:, :, i].mean()),
                "conditional_MC_standard_error": float(np.sqrt(np.sum(paired[:, :, i].var(axis=1, ddof=1)
                    / paired.shape[1])) / len(probes))} for i, key in enumerate(("W", "tail", "G"))}
        output[block] = {"contexts": contexts, "contrasts": contrasts}
    return output


def occupancy(fit_rows, factual_by_seed, *, horizon):
    return state_distribution(fit_rows,
        [r for f in factual_by_seed for r in f["reference"]["feature_rows"]],
        [r for f in factual_by_seed for r in f["candidate"]["feature_rows"]], horizon=horizon)


def cross_future_decisions(probes):
    """Choose with one cached future half, evaluate with the other; no policy fit."""
    output = {}
    for direction in ("A_to_B", "B_to_A"):
        contexts = {}
        for context in CONTEXTS:
            values = {name: [] for name in DECISION_CHOICES}
            disagreement = []
            prefix = context.split("/")[0]
            for probe in probes:
                rows = probe["effects"][context]
                if len(rows) < 4 or len(rows) % 2:
                    raise ValueError("cross-future choices need two halves of at least two futures")
                matrix = np.asarray([[r[k] for k in ("W", "tail", "G")] for r in rows])
                a, b = np.array_split(matrix, 2)
                train, evaluation = (a, b) if direction == "A_to_B" else (b, a)
                estimated, target = train.mean(axis=0), evaluation.mean(axis=0)
                reference = int(probe["actions"][prefix]["reference"])
                choices = {"candidate": int(probe["actions"][prefix]["candidate"]),
                    "score_zero": int(probe["candidate_scores"][prefix] >= 0),
                    "W_sign": int(estimated[0] >= 0), "G_sign": int(estimated[2] >= 0)}
                disagreement.append(choices["W_sign"] != choices["G_sign"])
                for name, action in choices.items():
                    values[name].append([*((action - reference) * target), action != reference, action])
            contexts[context] = {"W_G_choice_disagreement": float(np.mean(disagreement)),
                "choices": {name: dict(zip(("gain_W", "gain_tail", "gain_G", "changed_from_reference", "choose_now"),
                    np.asarray(rows).mean(axis=0).tolist())) for name, rows in values.items()}}
        output[direction] = contexts
    output["all"] = {c: {"W_G_choice_disagreement": float(np.mean([
        output[d][c]["W_G_choice_disagreement"] for d in ("A_to_B", "B_to_A")])),
        "choices": {name: {key: float(np.mean([output[d][c]["choices"][name][key]
            for d in ("A_to_B", "B_to_A")])) for key in output["A_to_B"][c]["choices"][name]}
            for name in output["A_to_B"][c]["choices"]}} for c in CONTEXTS}
    return output
