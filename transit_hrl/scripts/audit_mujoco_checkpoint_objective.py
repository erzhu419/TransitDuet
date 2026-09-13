#!/usr/bin/env python3
"""Summarize recorded checkpoint ranks on the server without loading policies."""

import argparse
import json
from pathlib import Path


def audit(run_directory):
    cells = []
    for path in sorted(Path(run_directory).glob("cells/*/*/replicate_*/training_history.json")):
        summary = json.loads((path.parent / "cell_summary.json").read_text())
        history = json.loads(path.read_text())
        selected_iteration = int(summary["selected_checkpoint_iteration"])
        selected = next(row for row in history if row["iteration"] == selected_iteration)
        eligible = [row for row in history if (
            row.get("checkpoint_evaluation_performed", False)
            and row["iteration"] >= summary["checkpoint_minimum_eligible_iteration"]
        )]
        best_return = max(eligible, key=lambda row: row["episode_return_mean"])
        best_step_reward = max(eligible, key=lambda row: row["reward_mean_mean"])
        cells.append({
            "environment": summary["environment"],
            "arm": path.parent.parent.name,
            "optimizer_seed": summary["optimizer_seed"],
            "score_contract": summary["checkpoint_score_contract"],
            "eligible_checkpoint_count": len(eligible),
            "selected_iteration": selected_iteration,
            "selected_step_reward": selected["reward_mean_mean"],
            "selected_episode_return": selected["episode_return_mean"],
            "selected_episode_length": selected["episode_length_mean"],
            "selected_matches_max_step_reward": (
                selected["reward_mean_mean"] == best_step_reward["reward_mean_mean"]
            ),
            "max_return_iteration": best_return["iteration"],
            "max_episode_return": best_return["episode_return_mean"],
            "max_return_episode_length": best_return["episode_length_mean"],
            "relative_selection_return_gap": (
                (best_return["episode_return_mean"] - selected["episode_return_mean"])
                / max(abs(selected["episode_return_mean"]), 1.0)
            ),
        })
    return {
        "evidence_role": "recorded_selection_history_diagnostic_only",
        "interpretation": (
            "Compares metrics on the existing selection paths. No checkpoint "
            "is loaded or changed. Gaps are not heldout performance gains."
        ),
        "cell_count": len(cells),
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
