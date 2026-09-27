"""Split-future decision value of tail credit under frozen continuation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_contextual_pair_diagnostic import PROTOCOL_VERSION as PREDICTION_PROTOCOL
from .pointmaze_continuation_credit import PROTOCOL_VERSION as ENDPOINT_PROTOCOL
from .pointmaze_fresh_future_diagnostic import PROTOCOL_VERSION as FUTURE_PROTOCOL
from .pointmaze_goal_validation import _json_ready


PROTOCOL_VERSION = "pointmaze_tail_decision_stage24_v1_diagnostic"
FROZEN_MODELS = ("linear", "contextual", "random_context")
METHODS = ("oracle_tail", *FROZEN_MODELS)


def split_indices(count):
    if count < 4 or count % 2:
        raise ValueError("tail decision diagnostic requires two equal repeated-future halves")
    return list(range(count // 2)), list(range(count // 2, count))


def summarize(rows):
    n = len(rows)
    reference = np.asarray([r["short_window_now"] for r in rows], dtype=int)
    advantage = np.asarray([r["scoring_total_advantage_mean"] for r in rows])
    variance = np.asarray([r["scoring_total_mean_variance"] for r in rows])
    comparisons = {}
    for method in METHODS:
        action = np.asarray([r["now_choices"][method] for r in rows], dtype=int)
        difference = action - reference
        comparisons[method] = {
            "now_count": int(action.sum()), "switches_vs_short_window": int(np.count_nonzero(difference)),
            "mean_ise_benefit_vs_short_window": float(np.mean(difference * advantage)),
            "conditional_mc_standard_error": float(np.sqrt(np.sum(difference ** 2 * variance)) / n),
        }
    return {"opportunities": n, "short_window_now_count": int(reference.sum()), "comparisons": comparisons}


def score_direction(cases, selection_indices, scoring_indices):
    if set(selection_indices).intersection(scoring_indices):
        raise ValueError("oracle selection and scoring futures must be disjoint")
    rows = []
    for case in cases:
        w = case["window_advantage"]
        futures = np.asarray(case["tail_advantages"])
        selection_mean = float(futures[selection_indices].mean())
        scores = w + futures[scoring_indices]
        # ISE(wait)-ISE(now) is positive when choosing now reduces cost.
        choices = {"oracle_tail": w + selection_mean > 0,
                   **{m: w + case["predictions"][m] > 0 for m in FROZEN_MODELS}}
        rows.append({"seed": case["seed"], "check_step": case["check_step"],
                     "window_advantage": w, "selection_tail_mean": selection_mean,
                     "scoring_total_advantage_mean": float(scores.mean()),
                     "scoring_total_mean_variance": float(scores.var(ddof=1) / len(scores)),
                     "short_window_now": w > 0, "now_choices": choices})
    return {"selection_indices": selection_indices, "scoring_indices": scoring_indices,
            "rows": rows, "summary": summarize(rows),
            "path_summaries": {str(s): summarize([r for r in rows if r["seed"] == s])
                               for s in sorted({r["seed"] for r in rows})}}


def join_cases(endpoint, future, prediction, *, fit_seeds, opportunities_per_path, replicates):
    key = lambda r: (r["seed"], r["check_step"])
    old = {key(r): r for r in endpoint["branch_fit_rows"]}
    new = {key(r): r for r in future["rows"]}
    models = {key(r): r for r in prediction["rows"]}
    if (set(new) != set(models) or not set(new) <= set(old)
            or len(new) != len(future["rows"]) or len(models) != len(prediction["rows"])
            or {r[0] for r in new} != set(fit_seeds)
            or any(sum(s == path for s, _ in new) != opportunities_per_path for path in fit_seeds)):
        raise ValueError("tail decision diagnostic needs the exact frozen opportunities")
    cases = []
    for k in sorted(new):
        e, f, p = old[k], new[k], models[k]
        a = np.asarray(f["replicate_tail_advantages"])
        if a.shape != (replicates,) or not np.isfinite(a).all():
            raise ValueError("tail decision future budget differs")
        if (not np.isclose(e["full_ise_advantage"], e["window_ise_advantage"] + e["actual_tail_advantage"], rtol=1e-10, atol=1e-10)
                or e["now_upper_call_count"] != e["wait_upper_call_count"]):
            raise ValueError("window/tail credit identity or paired call budget differs")
        np.testing.assert_allclose([p["repeat_mean"], p["mean_variance"]],
                                   [a.mean(), a.var(ddof=1) / len(a)], rtol=1e-12, atol=1e-15)
        if p["predictions"]["linear"] != f["predictions"]["ridge_averaged"]:
            raise ValueError("tail decision frozen linear prediction differs")
        cases.append({"seed": k[0], "check_step": k[1], "window_advantage": e["window_ise_advantage"],
                      "tail_advantages": a, "predictions": {m: p["predictions"][m] for m in FROZEN_MODELS}})
    return cases


def run_cell(args):
    first, second = split_indices(args.future_replicates)
    cells = []
    for path, protocol in ((args.endpoint_result, ENDPOINT_PROTOCOL), (args.fresh_result, FUTURE_PROTOCOL),
                           (args.prediction_result, PREDICTION_PROTOCOL)):
        data = json.loads(path.read_text())
        if (data["status"] != "complete" or data["protocol"]["protocol_version"] != protocol
                or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
            raise ValueError("tail decision source is not the completed registered root")
        c = data["cells"][0]
        if c["branch_fit_seeds"] != args.branch_fit_seeds or c["trigger_eval_seeds"] != args.trigger_eval_seeds:
            raise ValueError("tail decision seed roles differ")
        cells.append(c)
    cases = join_cases(*cells, fit_seeds=args.branch_fit_seeds,
                       opportunities_per_path=args.opportunities_per_path, replicates=args.future_replicates)
    directions = {"first_to_second": score_direction(cases, first, second),
                  "second_to_first": score_direction(cases, second, first)}
    primary, reverse = (directions[k]["rows"] for k in ("first_to_second", "second_to_first"))
    return {"optimizer_seed": args.optimizer_seed, "branch_fit_seeds": args.branch_fit_seeds,
            "trigger_eval_seeds": args.trigger_eval_seeds, "directions": directions,
            "oracle_action_agreement_fraction": float(np.mean([
                a["now_choices"]["oracle_tail"] == b["now_choices"]["oracle_tail"] for a, b in zip(primary, reverse)])),
            "additional_primitive_steps": 0, "controller_training_iterations": 0,
            "critic_fits": 0, "policy_updates": 0, "evaluation_paths_used": 0}


def main(argv=None):
    parser = build_parser()
    for key in ("endpoint", "fresh", "prediction"):
        parser.add_argument(f"--{key}-result", type=Path, required=True)
    parser.add_argument("--opportunities-per-path", type=int, default=2)
    parser.add_argument("--future-replicates", type=int, default=64)
    args = parser.parse_args(argv)
    first, second = split_indices(args.future_replicates)
    output = {"status": "dry_run" if args.dry_run else "complete", "protocol": {
        "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
        "evidence_role": "retrospective_split_future_tail_decision_only", "policy_deployment": False,
        "short_window_is_observed_counterfactual": True, "scoring_labels_previously_seen": True,
        "primary_direction": "first_to_second", "primary_selection_indices": first,
        "primary_scoring_indices": second, "zero_tie_action": "wait_one_check",
        "new_qualification_gate": False, "additional_primitive_steps": 0,
        **{f"{k}_result": str(getattr(args, f"{k}_result")) for k in ("endpoint", "fresh", "prediction")}},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
