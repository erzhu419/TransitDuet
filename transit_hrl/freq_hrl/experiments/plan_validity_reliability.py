"""Frozen-policy diagnostics linking local renewal targets to complete episodes."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta, t

from .pointmaze_compact_plan_validity import _base_matrix


PROFILES = ("in_distribution", "fast_regime", "force_tail")
PROTOCOL_VERSION = "plan_validity_episode_reliability_v1_prepared"
FAMILY_ALPHA = 0.025


def profile_options(options, profile):
    result = dict(options)
    if profile == "fast_regime":
        result["regime_dwell_seconds"] = tuple(v * 0.5 for v in options["regime_dwell_seconds"])
    elif profile == "force_tail":
        result["force_pulse_amplitude"] *= 3
        result["force_pulse_duration_seconds"] = tuple(v * 2 for v in options["force_pulse_duration_seconds"])
    elif profile != "in_distribution":
        raise ValueError(f"unknown reliability profile: {profile}")
    return result


def episode_risk(costs, distances, clipped, *, dt, period, distance, duration_steps):
    costs, distances, clipped = map(np.asarray, (costs, distances, clipped))
    if (not np.isfinite(distance) or distance <= 0 or duration_steps < 1
        or not len(costs) or len(costs) != len(distances) or len(costs) != len(clipped)
        or not np.all(np.isfinite(distances)) or not np.all(np.isfinite(costs))):
        raise ValueError("risk measurement needs aligned finite costs and a positive operational threshold")
    above = distances >= distance
    runs, current, longest = 0, 0, 0
    for value in above:
        current = current + 1 if value else 0
        longest = max(longest, current)
        runs += current == duration_steps
    split = len(costs) // 2
    return {"bin_ise": [float(costs[i:i + period].sum()) for i in range(0, len(costs), period)],
            "first_half_ise": float(costs[:split].sum()), "second_half_ise": float(costs[split:].sum()),
            "sustained_failure": bool(runs), "sustained_failure_events": int(runs),
            "longest_excursion_steps": int(longest), "exceedance_seconds": float(above.sum() * dt),
            "excess_distance_ise": float(np.square(np.maximum(distances - distance, 0)).sum() * dt),
            "maximum_tracking_distance": float(distances.max()), "clipped_control_steps": int(clipped.sum()),
            "worst_50_step_ise": float(np.convolve(costs, np.ones(min(50, len(costs))), mode="valid").max())}


def paired_mean_interval(values, *, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("interval needs nonempty finite paired effects")
    mean = float(values.mean())
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "interval": None}
    radius = float(t.ppf(1 - alpha / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "interval": [mean - radius, mean + radius]}


def event_upper_bound(events, episodes, *, alpha=0.05):
    """One-sided exact binomial bound, conditional on a frozen trained root."""
    if not 0 <= events <= episodes or episodes < 1:
        raise ValueError("event counts require completed independent episodes")
    return 1.0 if events == episodes else float(beta.ppf(1 - alpha, events + 1, episodes - events))


def expected_shortfall(values, *, quantile=0.9):
    values = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil((1 - quantile) * len(values))))
    return {"quantile": quantile, "tail_episodes": count, "mean": float(values[-count:].mean())}


def state_distribution(fit_rows, reference_rows, deployed_rows, *, horizon):
    """Paired fixed-clock occupancy, scaled ONLY by the original fitting bank."""
    schema, fit = _base_matrix(fit_rows)
    keys = lambda rows: [(row["seed"], row["opportunity_step"]) for row in rows]
    if keys(reference_rows) != keys(deployed_rows) or len(set(keys(reference_rows))) != len(reference_rows):
        raise ValueError("reference/deployed occupancy needs the same unique path-clock grid")
    for rows in (reference_rows, deployed_rows):
        if _base_matrix(rows)[0] != schema:
            raise ValueError("occupancy and fitting feature schemas differ")
    ref, dep = _base_matrix(reference_rows)[1], _base_matrix(deployed_rows)[1]
    scale = np.where(fit.std(axis=0) > 1e-8, fit.std(axis=0), 1.0)
    low, high = fit.min(axis=0), fit.max(axis=0)
    output = {"scaling_source": "frozen_fit_bank_only", "feature_names": list(schema), "panels": {}}
    for name, mask in (("all", np.ones(len(ref), dtype=bool)),
                       ("first_half", np.array([r["opportunity_step"] < horizon / 2 for r in reference_rows])),
                       ("second_half", np.array([r["opportunity_step"] >= horizon / 2 for r in reference_rows]))):
        if not np.any(mask):
            output["panels"][name] = {"states": 0}
            continue
        r, d = ref[mask], dep[mask]
        output["panels"][name] = {"states": len(d),
            "deployed_minus_reference_mean_in_fit_sd": ((d.mean(axis=0) - r.mean(axis=0)) / scale).tolist(),
            "paired_standardized_state_rms": float(np.sqrt(np.mean(((d - r) / scale) ** 2))),
            "reference_outside_fit_range_fraction": float(np.mean((r < low) | (r > high))),
            "deployed_outside_fit_range_fraction": float(np.mean((d < low) | (d > high)))}
    return output


def decompose_probe(local_keep, local_renew, now, wait):
    arms = (local_keep, local_renew, now, wait)
    if any(not np.array_equal(row[key], arms[0][key]) for row in arms[1:] for key in ("prefix", "features")):
        raise RuntimeError("local and full continuations do not share the same causal state")
    local = local_keep["branch_ise"] - local_renew["branch_ise"]
    window = wait["window_ise"] - now["window_ise"]
    tail = wait["tail_ise"] - now["tail_ise"]
    full = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
    if not np.isclose(full, window + tail, rtol=1e-10, atol=1e-10):
        raise RuntimeError("window and tail do not reconstruct the full paired episode effect")
    return {"local_advantage": float(local), "adaptive_window_advantage": float(window),
            "opportunity_gap": float(window - local), "tail_advantage": float(tail),
            "full_advantage": float(full),
            "first_200_step_advantage": wait["first_200_step_ise"] - now["first_200_step_ise"],
            "local_to_full_sign_flip": bool(local * full < 0)}


def evaluate_profile(controller, comparator_controller, rule, comparator_rule, *, task_spec,
                     args, seeds, probe_seeds, fit_rows, probes_per_episode, risk_spec, conditional_replicates):
    from .plan_validity_transfer import jitter_schedule, run_episode, time_contract

    if conditional_replicates < 2 or conditional_replicates % 2:
        raise ValueError("conditional futures need two equal independent half-groups")
    scale = time_contract(args)
    episodes, probes, reference_states, deployed_states = [], [], [], []
    grid = tuple(range(50, args.horizon, 50))
    for seed in seeds:
        def run(model, **kwargs):
            return run_episode(model, task_spec=task_spec, seed=seed, args=args,
                               time_scale=scale, risk_spec=risk_spec, **kwargs)[1]

        deployed = run(controller, rule=rule, feature_steps=grid if seed in probe_seeds else ())
        comparator = run(comparator_controller, rule=comparator_rule)
        episodes.append({"seed": seed, "candidate": compact_episode(deployed),
                         "comparator": compact_episode(comparator)})
        if seed not in probe_seeds:
            continue
        schedule = jitter_schedule(args, seed)
        reference = run(controller, schedule=schedule, feature_steps=grid)
        reference_states.extend(reference["feature_rows"])
        deployed_states.extend(deployed["feature_rows"])
        # Uniform predeclared clock locations, not selected by score, loss or event severity.
        eligible = [step for step in grid if step + 200 <= args.horizon and step not in schedule]
        if len(eligible) < probes_per_episode:
            raise ValueError("not enough noncolliding complete-200-step probe locations")
        rng = np.random.default_rng(np.random.SeedSequence([args.optimizer_seed, seed, 402]))
        checks = sorted(rng.choice(eligible, probes_per_episode, replace=False).tolist())
        by_step = {row["opportunity_step"]: row for row in deployed["feature_rows"]}
        factual_checks = {row["step"]: row for row in deployed["queries"]}
        for step in checks:
            local_pair = [run(controller, rule=rule, branch_step=step, renew=value) for value in (False, True)]
            full_pair = [run(controller, rule=rule, intervention_step=step, arm=arm)
                         for arm in ("now", "wait_one_check")]
            factual_effect = decompose_probe(*local_pair, *full_pair)
            factual = full_pair[0] if factual_checks[step]["stop"] else full_pair[1]
            if (factual["decision_steps"] != deployed["decision_steps"] or
                not np.isclose(factual["tracking_squared_error_integral"], deployed["tracking_squared_error_integral"],
                               rtol=1e-10, atol=1e-10) or
                not np.array_equal(local_pair[0]["features"], by_step[step]["causal_features"])):
                raise RuntimeError("intervention does not reproduce the factual deployed trajectory")
            reference_pair = [run(controller, schedule=schedule, branch_step=step, renew=value)
                              for value in (False, True)]
            ref_score = rule.score(reference_pair[0]["feature_names"], reference_pair[0]["features"],
                                   0, (args.horizon - step) / args.horizon)
            replicas = []
            for replica in range(conditional_replicates):
                future = int(np.random.SeedSequence([args.optimizer_seed, seed, step, replica, 403]).generate_state(1)[0])
                ref_pair = [run(controller, schedule=schedule, branch_step=step, renew=value, continuation_seed=future)
                            for value in (False, True)]
                local = [run(controller, rule=rule, branch_step=step, renew=value, continuation_seed=future)
                         for value in (False, True)]
                full = [run(controller, rule=rule, intervention_step=step, arm=arm, continuation_seed=future)
                        for arm in ("now", "wait_one_check")]
                if any(not np.array_equal(row["prefix"], local_pair[0]["prefix"]) for row in (*local, *full)):
                    raise RuntimeError("conditional replica changed the frozen deployed prefix")
                if any(not np.array_equal(row["prefix"], reference_pair[0]["prefix"]) for row in ref_pair):
                    raise RuntimeError("conditional replica changed the frozen reference prefix")
                replicas.append({"future_seed": future, **decompose_probe(*local, *full),
                                 "reference_local_advantage": ref_pair[0]["branch_ise"] - ref_pair[1]["branch_ise"]})
            probes.append({"seed": seed, "step": step, "factual_effect": factual_effect,
                "conditional_replicas": replicas,
                "local_advantage": float(np.mean([r["local_advantage"] for r in replicas])),
                "reference_local_advantage": float(np.mean([r["reference_local_advantage"] for r in replicas])),
                "deployed_score": factual_checks[step]["score"], "reference_score": ref_score,
                "factual_stop": factual_checks[step]["stop"], "prefix_max_abs_difference": 0.0})
    return {"status": "complete", "episodes": episodes, "probes": probes,
            "occupancy": state_distribution(fit_rows, reference_states, deployed_states, horizon=args.horizon),
            "local_errors": {name: local_error_summary(probes, name) for name in ("reference", "deployed")},
            "probe_sampling": "paired_uniform_bin_starts_excluding_jitter_collisions_and_incomplete_200_step_tails"}


def compact_episode(row):
    return {key: row[key] for key in ("tracking_squared_error_integral", "episode_return",
                                     "upper_decision_count", "reliability")}


def local_error_summary(rows, population):
    score = np.array([row[f"{population}_score"] for row in rows])
    target = np.array([row["reference_local_advantage" if population == "reference" else "local_advantage"] for row in rows])
    key = "reference_local_advantage" if population == "reference" else "local_advantage"
    correction = np.array([np.var([r[key] for r in row["conditional_replicas"]], ddof=1) /
                           len(row["conditional_replicas"]) for row in rows])
    return {"probes": len(rows), "mse_to_conditional_repeat_mean": float(np.mean((score - target) ** 2)),
            "noise_corrected_conditional_mean_mse": float(np.mean((score - target) ** 2 - correction)),
            "finite_repeat_variance_correction": float(correction.mean()),
            "mse_minus_zero_predictor": float(np.mean(score ** 2 - 2 * score * target)),
            "bias": float(np.mean(score - target)), "target_second_moment": float(np.mean(target ** 2)),
            "zero_predictor_mse": float(np.mean(target ** 2)),
            "sign_accuracy": float(np.mean((score > 0) == (target > 0)))}


def aggregate_reliability(reports, *, expected_roots, tasks, profiles=PROFILES,
                          alpha=FAMILY_ALPHA, risk_rate_margin, tail_ise_margin):
    """Primary inference uses optimizer roots; probes never inflate the episode N."""
    if not 0 < alpha < 1 or not 0 <= risk_rate_margin < 1 or not np.isfinite(tail_ise_margin):
        raise ValueError("declare finite risk/ISE margins and a valid family alpha")
    keys = [(r["protocol"]["task"], r["protocol"]["optimizer_seed"]) for r in reports]
    if len(set(keys)) != len(keys) or len(set(expected_roots)) != len(expected_roots) or not expected_roots:
        raise ValueError("declare unique task/root reports and a nonempty root roster")
    expected = {(task, root) for task in tasks for root in expected_roots}
    if set(keys) - expected:
        raise ValueError("report outside the declared task/root roster")
    path_banks = {task: set() for task in tasks}
    for report in reports:
        task = report["protocol"]["task"]
        bank = set(report["protocol"]["evaluation_seeds"])
        if bank.intersection(path_banks[task]):
            raise ValueError("different optimizer roots need independent held path banks")
        path_banks[task].update(bank)
    by_key, output = dict(zip(keys, reports)), []
    endpoint_alpha = alpha / (len(tasks) * len(profiles) * 6)
    risk_alpha = alpha / (len(tasks) * len(profiles) * len(expected_roots) * 2)
    for task in tasks:
        for profile in profiles:
            cells, missing = [], []
            contracts = []
            for root in expected_roots:
                report = by_key.get((task, root))
                cell = None if report is None else report.get("profiles", {}).get(profile)
                if cell is None or cell["status"] != "complete":
                    missing.append(root)
                else:
                    if (report["protocol"]["family_alpha"] != alpha or
                        report["protocol"]["risk_rate_margin"] != risk_rate_margin or
                        report["protocol"]["tail_ise_margin"] != tail_ise_margin):
                        raise ValueError("analysis margins differ from the frozen case plan")
                    if [row["seed"] for row in cell["episodes"]] != report["protocol"]["evaluation_seeds"]:
                        raise ValueError("completed episode rows differ from the predeclared path roster")
                    cells.append(cell)
                    contracts.append((report["protocol"]["risk_spec"], report["protocol"]["comparison"]))
            if missing:
                output.append({"task": task, "profile": profile, "status": "incomplete", "missing_or_failed_roots": missing})
                continue
            if any(contract != contracts[0] for contract in contracts):
                raise ValueError("risk definition or candidate/comparator method differs across roots")
            effects, risks, tails, half_effects, risk_bounds = [], [], [], [[], []], []
            root_risk_counts = []
            for root, cell in zip(expected_roots, cells):
                rows = cell["episodes"]
                path_ids = [row["seed"] for row in rows]
                if len(rows) < 2 or len(rows) % 2 or len(set(path_ids)) != len(rows):
                    raise ValueError("episode inference needs distinct, complete paired paths and two equal half-panels")
                candidate = [row["candidate"] for row in rows]
                comparator = [row["comparator"] for row in rows]
                delta = np.array([a["tracking_squared_error_integral"] - b["tracking_squared_error_integral"]
                                  for a, b in zip(candidate, comparator)])
                a_event = np.array([r["reliability"]["sustained_failure"] for r in candidate], dtype=bool)
                b_event = np.array([r["reliability"]["sustained_failure"] for r in comparator], dtype=bool)
                effects.append(delta.mean())
                risks.append(float(a_event.mean() - b_event.mean()))
                # A zero-discordance sample still has a positive upper bound on harmful events.
                risk_bounds.append(event_upper_bound(int(np.sum(a_event & ~b_event)), len(rows), alpha=risk_alpha))
                root_risk_counts.append({"optimizer_seed": root, "episodes": len(rows),
                    "candidate_failure_episodes": int(a_event.sum()), "comparator_failure_episodes": int(b_event.sum()),
                    "harmful_discordances": int(np.sum(a_event & ~b_event)),
                    "beneficial_discordances": int(np.sum(~a_event & b_event)),
                    "harmful_probability_upper": risk_bounds[-1],
                    "descriptive_conditional_path_ise_interval": paired_mean_interval(delta)})
                tails.append(expected_shortfall([r["tracking_squared_error_integral"] for r in candidate])["mean"] -
                             expected_shortfall([r["tracking_squared_error_integral"] for r in comparator])["mean"])
                for i, half in enumerate(np.split(delta, 2)):
                    half_effects[i].append(half.mean())
            ise = paired_mean_interval(effects, alpha=endpoint_alpha)
            risk = paired_mean_interval(risks, alpha=endpoint_alpha)
            tail = paired_mean_interval(tails, alpha=endpoint_alpha)
            halves = [paired_mean_interval(values, alpha=endpoint_alpha) for values in half_effects]
            supported = (ise["interval"] is not None and ise["interval"][1] < 0
                         and tail["interval"][1] <= tail_ise_margin
                         and risk["interval"][1] <= risk_rate_margin
                         and np.mean(risk_bounds) <= risk_rate_margin
                         and all(row["mean"] < 0 for row in halves))
            output.append({"task": task, "profile": profile, "status": "analyzed", "optimizer_roots": len(cells),
                "candidate_minus_comparator_ise": ise, "sustained_failure_rate_difference": risk,
                "episode_ise_expected_shortfall_difference": tail, "evaluation_halves": halves,
                "root_risk_counts": root_risk_counts,
                "conditional_mean_harmful_event_probability_upper": float(np.mean(risk_bounds)),
                "local_transfer_diagnostics": transfer_summary(cells),
                "joint_reliability_supported": bool(supported)})
    return {"panels": output, "statistical_unit": "optimizer_root",
            "scope": "frozen_complete_episode_endpoints_not_a_per_state_value_guarantee",
            "root_interval_alpha": endpoint_alpha, "conditional_risk_bound_alpha": risk_alpha,
            "family_alpha": alpha,
            "risk_rate_margin": risk_rate_margin, "tail_ise_margin": tail_ise_margin,
            "joint_reliability_supported": all(row.get("joint_reliability_supported", False) for row in output)}


def transfer_summary(cells):
    metrics = {key: [] for key in ("deployed_minus_reference_corrected_mse", "deployed_mse_minus_zero",
               "local_advantage", "opportunity_gap", "tail_advantage", "full_advantage",
               "first_200_step_advantage", "conditional_mean_sign_flip_fraction",
               "future_first_half_full_advantage", "future_second_half_full_advantage")}
    for cell in cells:
        errors = cell["local_errors"]
        metrics["deployed_minus_reference_corrected_mse"].append(
            errors["deployed"]["noise_corrected_conditional_mean_mse"] -
            errors["reference"]["noise_corrected_conditional_mean_mse"])
        metrics["deployed_mse_minus_zero"].append(errors["deployed"]["mse_minus_zero_predictor"])
        probes = cell["probes"]
        for key in ("local_advantage", "opportunity_gap", "tail_advantage", "full_advantage", "first_200_step_advantage"):
            metrics[key].append(np.mean([np.mean([r[key] for r in row["conditional_replicas"]]) for row in probes]))
        metrics["conditional_mean_sign_flip_fraction"].append(np.mean([
            np.mean([r["local_advantage"] for r in row["conditional_replicas"]]) *
            np.mean([r["full_advantage"] for r in row["conditional_replicas"]]) < 0 for row in probes]))
        for i, key in enumerate(("future_first_half_full_advantage", "future_second_half_full_advantage")):
            metrics[key].append(np.mean([np.split(np.asarray([r["full_advantage"] for r in row["conditional_replicas"]]), 2)[i].mean()
                                         for row in probes]))
    return {"interval_role": "descriptive_optimizer_root_intervals_not_additional_confirmatory_tests",
            **{key: paired_mean_interval(values) for key, values in metrics.items()}}
