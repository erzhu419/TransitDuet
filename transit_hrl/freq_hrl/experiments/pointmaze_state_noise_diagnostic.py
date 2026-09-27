"""Separate observable history compression from conditional continuation noise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointmaze_adaptive_pair_diagnostic import rollout_intervention
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_continuation_credit import (
    PROTOCOL_VERSION as ENDPOINT_PROTOCOL, VALUE_EPOCHS, fit_continuation,
)
from .pointmaze_deployed_pair_diagnostic import replay_source_controller
from .pointmaze_goal_validation import _json_ready
from .pointmaze_onecheck_advantage import require_factual_match
from .pointmaze_paired_value_qualification import endpoints


PROTOCOL_VERSION = "pointmaze_state_noise_stage18_v1_diagnostic"
REPRESENTATIONS = ("compact", "full_history")


def history_endpoints(pairs, *, representation):
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown state representation")
    rows = endpoints(pairs)
    for row in rows:
        history = row["controller_history"]
        row["feature_names"] = [*row["feature_names"], *[f"history_{i}" for i in range(len(history))]]
        row["features"] = [*row["features"], *(history if representation == "full_history" else [0.0] * len(history))]
    return rows


def qualify_states(pairs, *, root, fit_seeds, eval_seeds, epochs=VALUE_EPOCHS):
    paths = set(fit_seeds)
    if len(paths) < 2 or paths.intersection(eval_seeds) or {r["seed"] for r in pairs} != paths:
        raise ValueError("state qualification has wrong seed roles")
    rows, folds = [], []
    for held_out in sorted(paths):
        held = [r for r in pairs if r["seed"] == held_out]
        train_pairs = [r for r in pairs if r["seed"] != held_out]
        seed = int(np.random.SeedSequence([root, held_out, 16_016]).generate_state(1)[0])
        predictions = {}
        for representation in REPRESENTATIONS:
            prediction, diagnostics = fit_continuation(
                history_endpoints(train_pairs, representation=representation),
                history_endpoints(held, representation=representation),
                seed=seed, epochs=epochs, objective="paired_contrast",
            )
            predictions[representation] = prediction[1::2] - prediction[::2]
            folds.append({"held_out_path": held_out, "representation": representation, **diagnostics})
        for i, row in enumerate(held):
            rows.append({"seed": held_out, "check_step": row["check_step"],
                         "actual_tail_advantage": row["actual_tail_advantage"],
                         **{f"{key}_prediction": float(v[i]) for key, v in predictions.items()}})
    truth = np.asarray([r["actual_tail_advantage"] for r in rows])
    mse = {"zero": float(np.mean(truth ** 2)), **{
        key: float(np.mean((np.asarray([r[f"{key}_prediction"] for r in rows]) - truth) ** 2))
        for key in REPRESENTATIONS
    }}
    return {"rows": rows, "folds": folds, "contrast_mse": mse,
            "history_qualification_passed": mse["full_history"] < min(mse["compact"], mse["zero"])}


def select_noise_checks(pairs, *, root, seed, count):
    checks = sorted(r["check_step"] for r in pairs if r["seed"] == seed)
    if not 1 <= count <= len(checks):
        raise ValueError("noise opportunity count exceeds available pairs")
    rng = np.random.default_rng(np.random.SeedSequence([root, seed, 18_018]))
    return sorted(map(int, rng.choice(checks, size=count, replace=False)))


def noise_summary(rows):
    samples = np.asarray([r["replicate_tail_advantages"] for r in rows], dtype=np.float64)
    if samples.shape[0] < 2 or samples.shape[1] < 2:
        raise ValueError("noise diagnostic needs multiple states and futures")
    means, variances = samples.mean(axis=1), samples.var(axis=1, ddof=1)
    correction = float(np.mean(variances) / samples.shape[1])
    return {
        "opportunities": len(rows), "replicates_per_opportunity": samples.shape[1],
        "mean_conditional_variance": float(np.mean(variances)),
        "variance_of_conditional_means_estimate": float(np.var(means, ddof=1) - correction),
        "conditional_mean_mse_estimate": {
            key: float(np.mean((np.asarray([r[f"{key}_prediction"] for r in rows]) - means) ** 2) - correction)
            for key in ("zero", *REPRESENTATIONS)
        },
    }


def require_same_boundary(actual, reference, *, step):
    a, b = actual["value_trace"][0], reference["value_trace"][0]
    if (a["step"] != b["step"] or a["step"] != step
            or any(not np.array_equal(a[k], b[k]) for k in ("features", "controller_history"))
            or [s for s in actual["decision_steps"] if s < step]
            != [s for s in reference["decision_steps"] if s < step]
            or abs(actual["window_ise"] - reference["window_ise"]) > 1e-10):
        raise RuntimeError("future resampling changed the pre-boundary trajectory")


def run_cell(args):
    cached = json.loads(args.endpoint_result.read_text(encoding="utf-8"))
    if (cached["status"] != "complete" or cached["protocol"]["protocol_version"] != ENDPOINT_PROTOCOL
            or cached["protocol"]["optimizer_seed"] != args.optimizer_seed):
        raise ValueError("endpoint input is not the completed Stage-16 root")
    source = cached["cells"][0]
    if (source["branch_fit_seeds"] != args.branch_fit_seeds
            or source["trigger_eval_seeds"] != args.trigger_eval_seeds
            or source["pairs_per_seed"] != args.pairs_per_seed or args.future_replicates < 2):
        raise ValueError("state/noise source roles or sampling differ")
    reference_source, controller, time_scale = replay_source_controller(args)
    predictor = reference_source["trigger_predictor"]
    options = dict(predictor=predictor, args=args, time_scale=time_scale)
    pairs, noise_rows = [], []
    for seed in args.branch_fit_seeds:
        reference = rollout_intervention(
            controller, seed=seed, intervention_step=None, arm=None, **options,
        )
        selected = select_noise_checks(source["branch_fit_rows"], root=args.optimizer_seed,
                                       seed=seed, count=args.noise_pairs_per_seed)
        for old in [r for r in source["branch_fit_rows"] if r["seed"] == seed]:
            check = old["check_step"]
            branch_options = dict(seed=seed, intervention_step=check, collect_value_trace=True,
                                  capture_endpoint_history=True, **options)
            arms = [rollout_intervention(controller, arm=arm, **branch_options)
                    for arm in ("now", "wait_one_check")]
            now, wait = arms
            if not np.array_equal(now["prefix"], wait["prefix"]):
                raise RuntimeError("state diagnostic pair prefixes differ")
            require_factual_match(now if now["score"] >= predictor["threshold"] else wait, reference)
            for name, result in zip(("now", "wait"), arms):
                actual, expected = result["value_trace"][0], old[f"{name}_endpoint"]
                if (actual["step"] != expected["step"]
                        or not np.array_equal(actual["features"], expected["features"])
                        or abs(actual["cost_to_go"] - expected["cost_to_go"]) > 1e-8):
                    raise RuntimeError("controller reconstruction differs from cached endpoints")
            pairs.append({**old, "now_endpoint": now["value_trace"][0], "wait_endpoint": wait["value_trace"][0]})
            if check in selected:
                replicates, seeds = [], []
                for replica in range(args.future_replicates):
                    future_seed = int(np.random.SeedSequence(
                        [args.optimizer_seed, seed, check, replica, 18_019],
                    ).generate_state(1)[0])
                    futures = [rollout_intervention(controller, arm=arm, continuation_seed=future_seed,
                                                    **branch_options) for arm in ("now", "wait_one_check")]
                    for actual, original in zip(futures, arms):
                        require_same_boundary(actual, original, step=check + time_scale.upper_period_steps)
                    replicates.append(futures[1]["value_trace"][0]["cost_to_go"]
                                      - futures[0]["value_trace"][0]["cost_to_go"])
                    seeds.append(future_seed)
                noise_rows.append({"seed": seed, "check_step": check,
                                   "future_seeds": seeds, "replicate_tail_advantages": replicates,
                                   "original_tail_advantage": old["actual_tail_advantage"]})
        print(f"state/noise fit path {seed} complete: {len(pairs)} pairs, {len(noise_rows)} noise states", flush=True)
    qualification = qualify_states(pairs, root=args.optimizer_seed, fit_seeds=args.branch_fit_seeds,
                                   eval_seeds=args.trigger_eval_seeds)
    predictions = {(r["seed"], r["check_step"]): r for r in qualification["rows"]}
    for row in noise_rows:
        fitted = predictions[(row["seed"], row["check_step"])]
        row.update({f"{key}_prediction": fitted[f"{key}_prediction"] for key in REPRESENTATIONS})
        row["zero_prediction"] = 0.0
    return {
        "optimizer_seed": args.optimizer_seed,
        "branch_fit_seeds": args.branch_fit_seeds, "trigger_eval_seeds": args.trigger_eval_seeds,
        "controller_selected_iteration": reference_source["controller_selected_iteration"],
        "replayed_controller_training_iterations": args.iterations,
        "inherited_branch_fit_primitive_steps": source["inherited_branch_fit_primitive_steps"],
        "horizon": args.horizon, "pairs": len(pairs),
        "state_qualification": qualification, "noise_rows": noise_rows,
        "noise_summary": noise_summary(noise_rows),
        "critic_optimizer_steps": len(qualification["folds"]) * VALUE_EPOCHS,
        "additional_primitive_steps": {
            "reference_fit_replay": len(args.branch_fit_seeds) * args.horizon,
            "original_pair_replay": len(pairs) * 2 * args.horizon,
            "conditional_future_pair_replay": len(noise_rows) * args.future_replicates * 2 * args.horizon,
        },
    }


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--endpoint-result", type=Path, required=True)
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    parser.add_argument("--noise-pairs-per-seed", type=int, default=2)
    parser.add_argument("--future-replicates", type=int, default=8)
    args = parser.parse_args(argv)
    output = {
        "status": "dry_run" if args.dry_run else "complete",
        "protocol": {
            "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
            "source_result": str(args.source_result), "endpoint_result": str(args.endpoint_result),
            "pairs_per_seed": args.pairs_per_seed, "noise_pairs_per_seed": args.noise_pairs_per_seed,
            "future_replicates": args.future_replicates, "value_epochs": VALUE_EPOCHS,
            "evidence_role": "state_compression_and_conditional_noise_diagnostic_only",
            "latent_state_used_only_for_future_resampling": True, "policy_deployment": False,
        },
        "cells": [] if args.dry_run else [run_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
