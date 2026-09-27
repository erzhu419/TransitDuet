"""Matched single-draw versus averaged continuation labels without new replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_continuation_credit import (
    PROTOCOL_VERSION as ENDPOINT_PROTOCOL, VALUE_EPOCHS, fit_continuation,
)
from .pointmaze_goal_validation import _json_ready
from .pointmaze_paired_value_qualification import endpoints
from .pointmaze_state_noise_diagnostic import PROTOCOL_VERSION as SOURCE_PROTOCOL


PROTOCOL_VERSION = "pointmaze_averaged_label_stage19_v1_diagnostic"
LABELS = ("single_draw", "averaged")
HISTORY_SLOTS = 384


def join_cached_pairs(endpoint_pairs, noise_rows):
    lookup = {(r["seed"], r["check_step"]): r for r in endpoint_pairs}
    keys = [(r["seed"], r["check_step"]) for r in noise_rows]
    if len(set(keys)) != len(keys) or any(k not in lookup for k in keys):
        raise ValueError("noise opportunities must uniquely match cached endpoints")
    pairs = []
    for noise in sorted(noise_rows, key=lambda r: (r["seed"], r["check_step"])):
        old = lookup[(noise["seed"], noise["check_step"])]
        if not np.isclose(old["actual_tail_advantage"], noise["original_tail_advantage"],
                          atol=1e-10, rtol=1e-10):
            raise ValueError("noise opportunity original label differs from endpoint cache")
        pairs.append({**old, "replicate_tail_advantages": noise["replicate_tail_advantages"]})
    return pairs


def compact_endpoints(pairs):
    rows = endpoints(pairs)
    for row in rows:
        row["feature_names"] = [*row["feature_names"], *[f"history_{i}" for i in range(HISTORY_SLOTS)]]
        row["features"] = [*row["features"], *[0.0] * HISTORY_SLOTS]
    return rows


def label_targets(pairs, label):
    samples = np.asarray([r["replicate_tail_advantages"] for r in pairs], dtype=np.float64)
    if label == "single_draw":
        return samples[:, 0]
    if label == "averaged":
        return samples.mean(axis=1)
    raise ValueError("unknown label treatment")


def metrics(rows):
    truth = np.asarray([r["repeat_mean"] for r in rows])
    correction = float(np.mean([r["repeat_variance"] / r["replicates"] for r in rows]))
    raw = {"zero": float(np.mean(truth ** 2)), **{
        key: float(np.mean((np.asarray([r[f"{key}_prediction"] for r in rows]) - truth) ** 2))
        for key in LABELS
    }}
    corrected = {key: value - correction for key, value in raw.items()}
    return {"mse_to_repeat_mean": raw, "finite_repeat_correction": correction,
            "conditional_mean_mse_estimate": corrected,
            "averaging_qualification_passed": corrected["averaged"] < min(
                corrected["single_draw"], corrected["zero"])}


def qualify(pairs, *, root, fit_seeds, eval_seeds, epochs=VALUE_EPOCHS):
    paths = set(fit_seeds)
    if len(paths) < 2 or paths.intersection(eval_seeds) or {r["seed"] for r in pairs} != paths:
        raise ValueError("averaged-label diagnostic has wrong seed roles")
    samples = np.asarray([r["replicate_tail_advantages"] for r in pairs], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("averaged-label diagnostic needs finite repeated labels")
    if len({sum(r["seed"] == s for r in pairs) for s in paths}) != 1:
        raise ValueError("each path must supply the same number of opportunities")
    rows, folds = [], []
    for held_out in sorted(paths):
        train_pairs = [r for r in pairs if r["seed"] != held_out]
        held = [r for r in pairs if r["seed"] == held_out]
        train, query = compact_endpoints(train_pairs), compact_endpoints(held)
        seed = int(np.random.SeedSequence([root, held_out, 16_016]).generate_state(1)[0])
        predictions = {}
        for label in LABELS:
            prediction, diagnostics = fit_continuation(
                train, query, seed=seed, epochs=epochs, objective="paired_contrast",
                contrast_targets=label_targets(train_pairs, label),
            )
            predictions[label] = prediction[1::2] - prediction[::2]
            folds.append({"held_out_path": held_out, "label": label, **diagnostics})
        for i, pair in enumerate(held):
            repeats = np.asarray(pair["replicate_tail_advantages"])
            rows.append({"seed": held_out, "check_step": pair["check_step"],
                         "single_draw_label": float(repeats[0]), "replicates": len(repeats),
                         "repeat_mean": float(repeats.mean()),
                         "repeat_variance": float(repeats.var(ddof=1)),
                         **{f"{k}_prediction": float(v[i]) for k, v in predictions.items()}})
    return {"optimizer_seed": root, "branch_fit_seeds": list(fit_seeds),
            "trigger_eval_seeds": list(eval_seeds), "evaluation_paths_used": 0,
            "pairs": len(rows), "rows": rows, "folds": folds,
            "metrics": metrics(rows), "path_metrics": {
                str(s): metrics([r for r in rows if r["seed"] == s]) for s in sorted(paths)},
            "label_draws_per_training_opportunity": {"single_draw": 1, "averaged": samples.shape[1]},
            "cached_future_pair_labels": int(samples.size),
            "additional_primitive_steps": 0, "controller_training_iterations": 0,
            "critic_optimizer_steps": len(folds) * epochs}


def run_cell(args):
    cached = []
    for path, protocol in ((args.source_result, SOURCE_PROTOCOL),
                           (args.endpoint_result, ENDPOINT_PROTOCOL)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["status"] != "complete" or data["protocol"]["protocol_version"] != protocol
                or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
            raise ValueError("averaged-label source is not the completed registered root")
        cell = data["cells"][0]
        if (cell["branch_fit_seeds"] != args.branch_fit_seeds
                or cell["trigger_eval_seeds"] != args.trigger_eval_seeds):
            raise ValueError("averaged-label source seed roles differ")
        cached.append(cell)
    noise, source = cached
    if (source["pairs_per_seed"] != args.pairs_per_seed
            or any(sum(r["seed"] == s for r in noise["noise_rows"]) != args.noise_pairs_per_seed
                   for s in args.branch_fit_seeds)
            or any(len(r["replicate_tail_advantages"]) != args.future_replicates for r in noise["noise_rows"])):
        raise ValueError("averaged-label source sampling differs")
    pairs = join_cached_pairs(source["branch_fit_rows"], noise["noise_rows"])
    result = qualify(pairs, root=args.optimizer_seed, fit_seeds=args.branch_fit_seeds,
                     eval_seeds=args.trigger_eval_seeds)
    result["inherited_stage18_additional_primitive_steps"] = noise["additional_primitive_steps"]
    return result


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--endpoint-result", type=Path, required=True)
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    parser.add_argument("--noise-pairs-per-seed", type=int, default=2)
    parser.add_argument("--future-replicates", type=int, default=8)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete",
              "protocol": {"protocol_version": PROTOCOL_VERSION, "source_protocol": SOURCE_PROTOCOL,
                           "endpoint_protocol": ENDPOINT_PROTOCOL, "optimizer_seed": args.optimizer_seed,
                           "source_result": str(args.source_result), "endpoint_result": str(args.endpoint_result),
                           "labels": LABELS, "single_draw_replica_index": 0,
                           "value_epochs": VALUE_EPOCHS, "normalization": "original_training_endpoints_only",
                           "evidence_role": "cached_label_averaging_diagnostic_only",
                           "additional_primitive_steps": 0, "policy_deployment": False},
              "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
