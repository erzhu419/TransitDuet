"""Qualify paired continuation learning on cached branch-fit endpoints only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_continuation_credit import (
    PROTOCOL_VERSION as SOURCE_PROTOCOL, VALUE_EPOCHS, VALUE_WIDTH, fit_continuation,
)
from .pointmaze_goal_validation import _json_ready


PROTOCOL_VERSION = "pointmaze_paired_value_stage17_v1_qualification"
OBJECTIVES = ("absolute", "paired_contrast")


def endpoints(pairs):
    return [{**r[f"{arm}_endpoint"], "seed": r["seed"],
             "pair_key": (r["seed"], r["check_step"]), "arm": arm}
            for r in pairs for arm in ("now", "wait")]


def qualify(pairs, *, root, fit_seeds, eval_seeds, epochs=VALUE_EPOCHS):
    paths = set(fit_seeds)
    if len(paths) < 2 or paths.intersection(eval_seeds) or {r["seed"] for r in pairs} != paths:
        raise ValueError("paired-value qualification has wrong seed roles")
    if len({(r["seed"], r["check_step"]) for r in pairs}) != len(pairs):
        raise ValueError("duplicate paired-value opportunities")
    rows, folds = [], []
    for held_out in sorted(paths):
        train = endpoints([r for r in pairs if r["seed"] != held_out])
        held_pairs = [r for r in pairs if r["seed"] == held_out]
        query = endpoints(held_pairs)
        seed = int(np.random.SeedSequence([root, held_out, 16_016]).generate_state(1)[0])
        predictions = {}
        for objective in OBJECTIVES:
            prediction, diagnostics = fit_continuation(
                train, query, seed=seed, epochs=epochs, objective=objective,
            )
            predictions[objective] = prediction[1::2] - prediction[::2]
            folds.append({"held_out_path": held_out, **diagnostics})
        for i, row in enumerate(held_pairs):
            rows.append({
                "seed": held_out, "check_step": row["check_step"],
                "actual_tail_advantage": row["actual_tail_advantage"],
                "window_ise_advantage": row["window_ise_advantage"],
                "stage16_prediction": row["predicted_tail_advantage"],
                **{f"{key}_prediction": float(value[i]) for key, value in predictions.items()},
            })
    metrics = contrast_metrics(rows)
    return {
        "optimizer_seed": root, "branch_fit_seeds": list(fit_seeds),
        "trigger_eval_seeds": list(eval_seeds), "evaluation_paths_used": 0,
        "pairs": len(rows), "rows": rows, "folds": folds,
        "metrics": metrics, "path_metrics": {
            str(s): contrast_metrics([r for r in rows if r["seed"] == s]) for s in sorted(paths)
        },
        "qualification_gate_passed": all(
            metrics["paired_contrast_mse"] < metrics[key]
            for key in ("zero_mse", "absolute_mse", "stage16_mse")
        ),
        "additional_primitive_steps": 0,
        "critic_optimizer_steps": len(folds) * epochs,
    }


def contrast_metrics(rows):
    truth = np.asarray([r["actual_tail_advantage"] for r in rows])
    output = {"zero_mse": float(np.mean(truth ** 2))}
    for key in (*OBJECTIVES, "stage16"):
        prediction = np.asarray([r[f"{key}_prediction"] for r in rows])
        output[f"{key}_mse"] = float(np.mean((prediction - truth) ** 2))
    return output


def run_cell(args):
    data = json.loads(args.source_result.read_text(encoding="utf-8"))
    if (data["status"] != "complete" or data["protocol"]["protocol_version"] != SOURCE_PROTOCOL
            or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
        raise ValueError("qualification source is not the completed Stage-16 root")
    source = data["cells"][0]
    if (source["branch_fit_seeds"] != args.branch_fit_seeds
            or source["trigger_eval_seeds"] != args.trigger_eval_seeds
            or source["pairs_per_seed"] != args.pairs_per_seed):
        raise ValueError("qualification source seed roles or sampling differ")
    # Deployment rows stay in the source file; qualification only reads branch-fit data.
    return qualify(source["branch_fit_rows"], root=args.optimizer_seed,
                   fit_seeds=args.branch_fit_seeds, eval_seeds=args.trigger_eval_seeds)


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    args = parser.parse_args(argv)
    output = {
        "status": "dry_run" if args.dry_run else "complete",
        "protocol": {
            "protocol_version": PROTOCOL_VERSION, "source_protocol": SOURCE_PROTOCOL,
            "source_result": str(args.source_result), "optimizer_seed": args.optimizer_seed,
            "value_epochs": VALUE_EPOCHS, "value_hidden_dim": VALUE_WIDTH,
            "objectives": OBJECTIVES, "evidence_role": "paired_critic_qualification_only",
            "additional_primitive_steps": 0, "policy_deployment": False,
        },
        "cells": [] if args.dry_run else [run_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
