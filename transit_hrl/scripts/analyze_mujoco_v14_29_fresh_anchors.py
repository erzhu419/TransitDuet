#!/usr/bin/env python3
"""Qualify the frozen v14.29 fresh MuJoCo anchor bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts import mujoco_v14_29_fresh_anchor_spec as spec


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _anchor_dir(run_name: str, environment: str, seed: int) -> Path:
    return (
        ROOT / "results" / str(run_name) / "anchors" / str(environment)
        / f"replicate_{int(seed)}"
    )


def _qualify_anchor(path: Path, environment: str, seed: int) -> dict[str, Any]:
    summary_path = path / "cell_summary.json"
    checkpoint_path = path / "checkpoint.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"incomplete v14.29 anchor: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "environment": str(environment),
        "optimizer_seed": int(seed),
        "method": "freq_hrl",
        "code_revision": spec.FROZEN_ALGORITHM_REVISION,
        "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "lower_action_router_mode": "causal_joint_band_projection",
        "lower_action_router_strength": 0.5,
        "lower_action_router_function_preserving": True,
        "checkpoint_selection_mode": spec.CHECKPOINT_SELECTION_MODE,
        "iterations": spec.PRETRAIN_ITERATIONS,
        "rollout_seed_roots": list(spec.PRETRAIN_SEEDS),
        "checkpoint_selection_seed_roots": list(
            spec.PRETRAIN_SELECTION_SEEDS
        ),
        "eval_seeds": list(spec.DEVELOPMENT_EVALUATION_SEEDS),
        "training_disturbance_modes": list(spec.TRAINING_DISTURBANCE_MODES),
        "evaluation_disturbance_modes": list(
            spec.EVALUATION_DISTURBANCE_MODES
        ),
        "steps": spec.STEPS,
        "upper_period": spec.UPPER_PERIOD,
        "lower_action_router_training_schedule": "constant",
        "source_identity_status": "verified",
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v14.29 anchor contract mismatch: {mismatches}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != str(summary.get("checkpoint_file_sha256", "")):
        raise ValueError(f"v14.29 anchor checkpoint digest mismatch: {path}")
    parameter_sha256 = str(summary.get("frozen_parameter_sha256", ""))
    selected_score = float(summary.get("checkpoint_selection_score", float("nan")))
    qualified = bool(
        len(parameter_sha256) == 64
        and bool(summary.get("checkpoint_has_eligible_selection"))
        and math.isfinite(selected_score)
        and int(summary.get("selected_checkpoint_iteration", -1))
        >= int(spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION)
        and int(summary.get("evaluation_row_count", 0))
        == len(spec.DEVELOPMENT_EVALUATION_SEEDS)
        * len(spec.EVALUATION_DISTURBANCE_MODES)
    )
    return {
        "environment": str(environment),
        "optimizer_seed": int(seed),
        "qualified": qualified,
        "checkpoint_file_sha256": checkpoint_sha256,
        "parameter_sha256": parameter_sha256,
        "selected_checkpoint_iteration": int(
            summary.get("selected_checkpoint_iteration", -1)
        ),
        "checkpoint_selection_score": selected_score,
        "evaluation_row_count": int(summary.get("evaluation_row_count", 0)),
    }


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    rows = [
        _qualify_anchor(_anchor_dir(run_name, environment, seed), environment, seed)
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]
    qualified = sum(row["qualified"] for row in rows)
    result = {
        "analysis_version": "mujoco_v14_29_fresh_anchor_qualification_v1",
        "status": (
            "fresh_anchor_bank_qualified"
            if qualified == len(rows) else "fresh_anchor_bank_not_qualified"
        ),
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "anchor_count": len(rows),
        "qualified_anchor_count": qualified,
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "anchors": rows,
    }
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "fresh_anchor_qualification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_name, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "qualified_anchor_count": result["qualified_anchor_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
