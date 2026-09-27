"""Score frozen continuation predictions on independent conditional futures."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch

from .pointmaze_adaptive_pair_diagnostic import rollout_intervention
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_continuation_credit import PROTOCOL_VERSION as ENDPOINT_PROTOCOL
from .pointmaze_deployed_pair_diagnostic import replay_source_controller
from .pointmaze_goal_validation import _json_ready
from .pointmaze_onecheck_advantage import require_factual_match
from .pointmaze_regularized_pair_diagnostic import PROTOCOL_VERSION as PREDICTION_PROTOCOL
from .pointmaze_state_noise_diagnostic import PROTOCOL_VERSION as NOISE_PROTOCOL, require_same_boundary


PROTOCOL_VERSION = "pointmaze_fresh_future_stage21_v1_diagnostic"
CANDIDATE = "ridge_averaged"
CONTROLS = ("zero", "stage19_averaged")
FAMILY_COMPARISONS = 4
FAMILY_ALPHA = 0.05
_WORKER = None


def fresh_seed(root, path, check, replica):
    return int(np.random.SeedSequence([root, path, check, replica, 21_021]).generate_state(1)[0])


def fixed_comparisons(rows):
    samples = np.asarray([r["replicate_tail_advantages"] for r in rows], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("fresh-future scoring needs finite repeated contrasts")
    n, k = samples.shape
    means = samples.mean(axis=1)
    correction = float(np.mean(samples.var(axis=1, ddof=1)) / k)
    predictions = {key: np.asarray([r["predictions"][key] for r in rows])
                   for key in (CANDIDATE, *CONTROLS)}
    raw = {key: float(np.mean((p - means) ** 2)) for key, p in predictions.items()}
    comparisons = {}
    for control in CONTROLS:
        p, b = predictions[CANDIDATE], predictions[control]
        # Paired squared-error differences cancel A^2 and the variance correction.
        differences = (b ** 2 - p ** 2)[:, None] - 2 * (b - p)[:, None] * samples
        mean = float(differences.mean())
        terms = differences.var(axis=1, ddof=1) / (k * n ** 2)
        variance = float(terms.sum())
        df = float(variance ** 2 / np.sum(terms ** 2 / (k - 1))) if variance > 0 else None
        se = float(np.sqrt(variance))
        critical = float(t.ppf(1 - FAMILY_ALPHA / (2 * FAMILY_COMPARISONS), df)) if df else 0.0
        comparisons[control] = {
            "control_minus_candidate_mse": mean, "mc_standard_error": se,
            "welch_degrees_of_freedom": df, "bonferroni_mc_interval": [mean - critical * se, mean + critical * se],
        }
    return {"opportunities": n, "replicates_per_opportunity": k,
            "mse_to_repeat_mean": raw, "finite_repeat_correction": correction,
            "conditional_mean_mse_estimate": {key: value - correction for key, value in raw.items()},
            "comparisons": comparisons,
            "point_gate_passed": all(c["control_minus_candidate_mse"] > 0 for c in comparisons.values()),
            "precision_gate_passed": all(c["bonferroni_mc_interval"][0] > 0 for c in comparisons.values())}


def reconstruction_steps(args):
    evaluations = sum((i + 1) % args.checkpoint_evaluation_interval == 0 or i == args.iterations - 1
                      for i in range(args.iterations))
    return {
        "training": args.iterations * len(args.train_seeds) * args.horizon,
        "selection": (1 + evaluations) * len(args.selection_seeds) * args.horizon,
        "untrained_canonical_and_fixed_evaluation": 3 * (len(args.branch_fit_seeds) + len(args.trigger_eval_seeds)) * args.horizon,
    }


def replay_steps(opportunities, replicates, horizon):
    return {"reference_replay": opportunities * horizon,
            "original_pair_replay": opportunities * 2 * horizon,
            "fresh_future_pair_replay": opportunities * replicates * 2 * horizon}


def load_cases(args):
    cells = []
    for path, protocol in ((args.endpoint_result, ENDPOINT_PROTOCOL),
                           (args.noise_result, NOISE_PROTOCOL),
                           (args.prediction_result, PREDICTION_PROTOCOL)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["status"] != "complete" or data["protocol"]["protocol_version"] != protocol
                or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
            raise ValueError("fresh-future input is not the completed registered root")
        cell = data["cells"][0]
        if (cell["branch_fit_seeds"] != args.branch_fit_seeds or cell["trigger_eval_seeds"] != args.trigger_eval_seeds):
            raise ValueError("fresh-future input seed roles differ")
        cells.append(cell)
    endpoints, noise, prediction = cells
    old = {(r["seed"], r["check_step"]): r for r in endpoints["branch_fit_rows"]}
    selected = {(r["seed"], r["check_step"]): r for r in noise["noise_rows"]}
    frozen = {(r["seed"], r["check_step"]): r for r in prediction["rows"]}
    if (set(selected) != set(frozen) or not set(selected) <= set(old)
            or len(selected) != len(noise["noise_rows"]) or len(frozen) != len(prediction["rows"])
            or endpoints["pairs_per_seed"] != args.pairs_per_seed
            or any(sum(s == path for s, _ in selected) != args.noise_pairs_per_seed for path in args.branch_fit_seeds)):
        raise ValueError("frozen prediction opportunities differ from the source cache")
    old_seeds = {s for r in selected.values() for s in r["future_seeds"]}
    fresh_seeds, cases = set(), []
    for key in sorted(selected):
        r, cached, p = old[key], selected[key], frozen[key]
        draws = np.asarray(cached["replicate_tail_advantages"])
        if (len(draws) != args.cached_future_replicates
                or not np.isclose(r["actual_tail_advantage"], cached["original_tail_advantage"], atol=1e-10, rtol=1e-10)
                or not np.isclose(p["repeat_mean"], draws.mean(), atol=1e-15, rtol=1e-12)):
            raise ValueError("frozen prediction labels differ from the source cache")
        seeds = [fresh_seed(args.optimizer_seed, *key, replica) for replica in range(args.future_replicates)]
        if len(set(seeds)) != len(seeds) or set(seeds) & (old_seeds | fresh_seeds):
            raise ValueError("fresh future seeds overlap existing draws")
        fresh_seeds.update(seeds)
        cases.append({"seed": key[0], "check_step": key[1], "endpoint_pair": r,
                      "future_seeds": seeds, "predictions": {"zero": 0.0, **{
                          name: p[f"{name}_prediction"] for name in (CANDIDATE, "stage19_averaged")}}})
    return cases


def init_worker(controller, predictor, args, time_scale):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = (controller, predictor, args, time_scale)


def sample_case(case):
    controller, predictor, args, time_scale = _WORKER
    seed, check = case["seed"], case["check_step"]
    options = dict(seed=seed, predictor=predictor, args=args, time_scale=time_scale)
    reference = rollout_intervention(controller, intervention_step=None, arm=None, **options)
    branch = dict(intervention_step=check, collect_value_trace=True, capture_endpoint_history=True, **options)
    originals = [rollout_intervention(controller, arm=arm, **branch) for arm in ("now", "wait_one_check")]
    if not np.array_equal(originals[0]["prefix"], originals[1]["prefix"]):
        raise RuntimeError("fresh-future intervention prefixes differ")
    require_factual_match(originals[0] if originals[0]["score"] >= predictor["threshold"] else originals[1], reference)
    for name, actual in zip(("now", "wait"), originals):
        a, b = actual["value_trace"][0], case["endpoint_pair"][f"{name}_endpoint"]
        if (a["step"] != b["step"] or not np.array_equal(a["features"], b["features"])
                or abs(a["cost_to_go"] - b["cost_to_go"]) > 1e-8):
            raise RuntimeError("reconstructed controller differs from frozen endpoints")
    contrasts = []
    for future in case["future_seeds"]:
        arms = [rollout_intervention(controller, arm=arm, continuation_seed=future, **branch)
                for arm in ("now", "wait_one_check")]
        for actual, original in zip(arms, originals):
            require_same_boundary(actual, original, step=check + time_scale.upper_period_steps)
        contrasts.append(arms[1]["value_trace"][0]["cost_to_go"] - arms[0]["value_trace"][0]["cost_to_go"])
    return {"seed": seed, "check_step": check, "predictions": case["predictions"],
            "future_seeds": case["future_seeds"], "replicate_tail_advantages": contrasts}


def run_cell(args):
    if args.workers < 1 or args.future_replicates < 2:
        raise ValueError("fresh-future run needs workers and repeated draws")
    cases = load_cases(args)
    print("reconstructing frozen Stage-12 controller once; no continuation critic fitting", flush=True)
    source, controller, time_scale = replay_source_controller(args)
    print("controller reconstruction complete; starting independent futures", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker,
                             initargs=(controller, source["trigger_predictor"], args, time_scale)) as pool:
        futures = [pool.submit(sample_case, case) for case in cases]
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"fresh-future states complete: {len(rows)}/{len(cases)}", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["check_step"]))
    return {"optimizer_seed": args.optimizer_seed, "branch_fit_seeds": args.branch_fit_seeds,
            "trigger_eval_seeds": args.trigger_eval_seeds, "opportunities": len(rows), "rows": rows,
            "workers": args.workers, "metrics": fixed_comparisons(rows),
            "path_metrics": {str(s): fixed_comparisons([r for r in rows if r["seed"] == s])
                             for s in args.branch_fit_seeds},
            "controller_selected_iteration": source["controller_selected_iteration"],
            "replayed_controller_training_iterations": args.iterations,
            "controller_reconstruction_primitive_steps": reconstruction_steps(args),
            "additional_replay_primitive_steps": replay_steps(len(rows), args.future_replicates, args.horizon),
            "continuation_critic_fits": 0}


def main(argv=None):
    parser = build_parser()
    for key in ("source", "endpoint", "noise", "prediction"):
        parser.add_argument(f"--{key}-result", type=Path, required=True)
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    parser.add_argument("--noise-pairs-per-seed", type=int, default=2)
    parser.add_argument("--cached-future-replicates", type=int, default=8)
    parser.add_argument("--future-replicates", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete", "protocol": {
        "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
        **{f"{k}_result": str(getattr(args, f"{k}_result")) for k in ("source", "endpoint", "noise", "prediction")},
        "future_seed_namespace": 21_021, "future_replicates": args.future_replicates,
        "candidate": CANDIDATE, "controls": CONTROLS, "family_comparisons": FAMILY_COMPARISONS,
        "family_alpha": FAMILY_ALPHA, "interval": "approximate_bonferroni_welch_t_conditional_mc",
        "evidence_role": "frozen_prediction_independent_future_precision_only",
        "policy_deployment": False, "continuation_critic_fits": 0},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
