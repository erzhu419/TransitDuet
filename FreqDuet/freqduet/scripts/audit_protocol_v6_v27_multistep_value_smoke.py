#!/usr/bin/env python3
"""Audit the preregistered non-effect V27 causal value smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_protocol_v6_v27_multistep_value_common import (
    CANDIDATES,
    CURRENT_MAIN,
    V13_ANCHOR,
    candidate_evaluation_checks,
    candidate_training_checks,
    control_evaluation_checks,
    load_aggregate,
    load_training_rows,
    strict_matrix_checks,
)


CONFIGS = [CURRENT_MAIN, V13_ANCHOR, *CANDIDATES]
TRAIN_SEEDS = [31903]
EVAL_SEEDS = [65903]
TRAIN_EPISODES = 8


def evaluate_v27_multistep_value_smoke(
    aggregate_dir: Path,
    log_roots: list[Path],
) -> dict[str, object]:
    manifest, per_eval = load_aggregate(aggregate_dir)
    strict_checks = strict_matrix_checks(
        manifest,
        per_eval,
        configs=CONFIGS,
        train_seeds=TRAIN_SEEDS,
        eval_seeds=EVAL_SEEDS,
        train_episodes=TRAIN_EPISODES,
        reference=V13_ANCHOR,
    )
    if not all(strict_checks.values()):
        raise ValueError(f"V27 smoke strict checks failed: {strict_checks}")

    training = load_training_rows(
        log_roots,
        train_seeds=TRAIN_SEEDS,
        train_episodes=TRAIN_EPISODES,
    )
    candidate_results = {}
    for candidate in CANDIDATES:
        training_checks, training_diagnostics = candidate_training_checks(
            training.loc[training["config"] == candidate].copy(),
            candidate=candidate,
            train_seeds=TRAIN_SEEDS,
            train_episodes=TRAIN_EPISODES,
        )
        evaluation_checks, evaluation_diagnostics = (
            candidate_evaluation_checks(
                per_eval.loc[per_eval["config"] == candidate].copy(),
                candidate=candidate,
                train_seeds=TRAIN_SEEDS,
                eval_seeds=EVAL_SEEDS,
            ))
        candidate_results[candidate] = {
            "training_checks": training_checks,
            "evaluation_checks": evaluation_checks,
            "training_diagnostics": training_diagnostics,
            "evaluation_diagnostics": evaluation_diagnostics,
            "passes": bool(
                all(training_checks.values())
                and all(evaluation_checks.values())),
        }

    control_checks = {
        control: control_evaluation_checks(
            per_eval.loc[per_eval["config"] == control].copy())
        for control in (CURRENT_MAIN, V13_ANCHOR)
    }
    passes = bool(
        all(result["passes"] for result in candidate_results.values())
        and all(all(checks.values()) for checks in control_checks.values()))
    return {
        "gate_version": "freqduet-v27-causal-multistep-value-smoke-v1",
        "status": "mechanical_pass" if passes else "no_pass",
        "effect_evidence": False,
        "formal_screen_authorized": passes,
        "strict_checks": strict_checks,
        "candidate_results": candidate_results,
        "control_checks": control_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument(
        "--logs-root", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_v27_multistep_value_smoke(
        args.aggregate_dir, args.logs_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_pass and not result["formal_screen_authorized"]:
        raise SystemExit("V27 smoke did not authorize the formal screen")
    print("DONE V27 causal-multistep-value smoke gate")


if __name__ == "__main__":
    main()
