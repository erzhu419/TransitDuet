"""CS top-venue experiment matrix for Freq-HRL.

The matrix records the eight experiments that are most likely to matter for
ML/RL/AI/data-mining reviewers.  It is both a manuscript checklist and a
machine-readable scheduler manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("transit_hrl/results")
DEFAULT_OUTPUT_DIR = Path("transit_hrl/results/cs_top_venue_experiment_matrix_latest")

ARTIFACTS = {
    "strong_learned": Path("strong_learned_baseline_validation_latest/summary.json"),
    "baseline": Path("baseline_ablation_matrix_latest/summary.json"),
    "agency": Path("agency_demand_onboard_coverage_latest/summary.json"),
    "order_book": Path("order_book_lobster_venue_grade_multisymbol/summary.json"),
    "claims": Path("top_journal_unified_matrix_latest/summary.json"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    out = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")).replace("|", "/") for field in fields) + " |")
    return out


def _summary_section(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def _status_for_strong_learned(summary: dict[str, Any]) -> str:
    ppo = str(summary.get("ppo_strong_baseline_status", "missing"))
    sac_td3 = str(summary.get("sac_td3_status", "registered_external_missing"))
    if ppo == "supported" and sac_td3 == "supported":
        return "supported"
    if ppo in {"supported", "partial"}:
        return "partial_ppo_supported"
    if ppo == "not_supported":
        return "not_supported"
    return "registered_executable"


def build_cs_top_venue_experiment_matrix(results_root: Path = DEFAULT_RESULTS_ROOT) -> dict[str, Any]:
    paths = {key: results_root / value for key, value in ARTIFACTS.items()}
    strong = _summary_section(_read_json(paths["strong_learned"]))
    baseline = _summary_section(_read_json(paths["baseline"]))
    agency = _summary_section(_read_json(paths["agency"]))
    order_book = _read_json(paths["order_book"])
    order_coverage = order_book.get("coverage", {}) if isinstance(order_book.get("coverage"), dict) else {}
    venue_pairs = int(order_coverage.get("venue_grade_l2_l3_session_pairs", 0) or 0)
    same_agency = str(agency.get("same_agency_native_control_status", "external_missing"))
    strong_status = _status_for_strong_learned(strong)
    scenario_count = int(strong.get("scenario_count", 0) or 0)
    sample_eff_status = "supported" if paths["strong_learned"].exists() and scenario_count > 0 else "registered_executable"
    rows = [
        {
            "id": "E1",
            "experiment": "strong learned RL baselines",
            "review_question": "Does Freq-HRL beat learned flat PPO and generic learned HRL under matched training?",
            "current_status": strong_status,
            "artifact": str(paths["strong_learned"]),
            "claim_gate": "PPO-family learned baselines need paired Sharpe/return/FocusScore; SAC/TD3 must remain explicit limitations until implemented.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.strong_learned_baseline_validation "
                "--output-dir transit_hrl/results/strong_learned_baseline_validation_latest"
            ),
            "paper_table": "main_baseline_table",
            "priority": 1,
            "scheduler_note": "Shardable with --num-shards N --shard-index K; merge shards with --merge-inputs.",
        },
        {
            "id": "E2",
            "experiment": "learned-baseline cross-stress regime",
            "review_question": "Do learned-policy results replicate across stationary, burst, persistent, and OOD regimes?",
            "current_status": "supported" if scenario_count >= 4 else "registered_executable",
            "artifact": str(paths["strong_learned"]),
            "claim_gate": "At least four stress regimes with paired learned-policy rows.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.strong_learned_baseline_validation "
                "--scenarios persistent_shift stationary_high_noise localized_burst ood_period "
                "--output-dir transit_hrl/results/strong_learned_baseline_validation_latest"
            ),
            "paper_table": "stress_generalization_table",
            "priority": 2,
            "scheduler_note": "Shardable over scenario/policy-mode pairs with --num-shards N --shard-index K.",
        },
        {
            "id": "E3",
            "experiment": "complete ablation main table",
            "review_question": "Are frequency routing, promotion, leakage, encoder, and plan-curve ablations all visible?",
            "current_status": str(baseline.get("claim_status", "registered_executable")),
            "artifact": str(paths["baseline"]),
            "claim_gate": "Baseline matrix must include heuristic ablations plus learned rows when present.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.baseline_ablation_matrix "
                "--output-dir transit_hrl/results/baseline_ablation_matrix_latest"
            ),
            "paper_table": "main_ablation_table",
            "priority": 3,
        },
        {
            "id": "E4",
            "experiment": "parameter-budget fair comparison",
            "review_question": "Are gains caused by frequency responsibility rather than more parameters?",
            "current_status": str(strong.get("parameter_budget_status", "registered_executable")),
            "artifact": str(paths["strong_learned"]).replace("summary.json", "parameter_budget.csv"),
            "claim_gate": "Freq-HRL, flat PPO, and generic HRL PPO must share state/action dimensions and parameter counts.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.strong_learned_baseline_validation "
                "--policy-modes freq_hrl flat_ppo generic_hrl_ppo "
                "--output-dir transit_hrl/results/strong_learned_baseline_validation_latest"
            ),
            "paper_table": "parameter_budget_appendix",
            "priority": 4,
            "scheduler_note": "Parameter-budget rows are emitted by every strong learned baseline shard and checked after merge.",
        },
        {
            "id": "E5",
            "experiment": "sensitivity and robustness",
            "review_question": "Does the result survive promotion/leakage/plan hyperparameter perturbations?",
            "current_status": "registered_executable",
            "artifact": "transit_hrl/results/sensitivity_robustness_matrix_latest/summary.json",
            "claim_gate": "Report stress-registered sensitivity; do not claim universal hyperparameter robustness.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.strong_learned_baseline_validation "
                "--scenarios persistent_shift promotion_recovery stationary_high_noise localized_burst ood_period "
                "--output-dir transit_hrl/results/sensitivity_robustness_matrix_latest"
            ),
            "paper_table": "robustness_appendix",
            "priority": 5,
        },
        {
            "id": "E6",
            "experiment": "runtime and sample efficiency",
            "review_question": "What is the wall-clock and environment-step cost of the learned method?",
            "current_status": sample_eff_status,
            "artifact": str(paths["strong_learned"]).replace("summary.json", "sample_efficiency.csv"),
            "claim_gate": "Report environment steps, iterations, elapsed seconds, and held-out objective proxy.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.strong_learned_baseline_validation "
                "--output-dir transit_hrl/results/strong_learned_baseline_validation_latest"
            ),
            "paper_table": "sample_efficiency_appendix",
            "priority": 6,
            "scheduler_note": "Sample-efficiency rows are emitted per scenario/policy-mode shard and merged with per-seed rows.",
        },
        {
            "id": "E7",
            "experiment": "same-agency real Transit",
            "review_question": "Is real AFC/APC/OD/onboard-load evidence one linked agency control loop?",
            "current_status": "supported" if same_agency == "supported" else "boundary_registered",
            "artifact": str(paths["agency"]),
            "claim_gate": "Current public truth-source coverage is not a full same-agency deployment loop unless the gate says supported.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.transit.agency_demand_onboard_coverage "
                "--output-dir transit_hrl/results/agency_demand_onboard_coverage_latest"
            ),
            "paper_table": "real_data_boundary_table",
            "priority": 7,
        },
        {
            "id": "E8",
            "experiment": "larger L2/L3 order-book replay",
            "review_question": "Does market evidence scale beyond fixture/sample replay?",
            "current_status": "strong_scale" if venue_pairs >= 20 else "partial_scale",
            "artifact": str(paths["order_book"]),
            "claim_gate": "Finance/data-mining venues should see at least 20 venue-grade symbol-session pairs or a limitation.",
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.order_book_large_replay_manifest_validation "
                "--manifest transit_hrl/data/order_book_large_manifest.csv "
                "--output-dir transit_hrl/results/order_book_large_replay_manifest_latest"
            ),
            "paper_table": "order_book_scale_appendix",
            "priority": 8,
        },
    ]
    blockers = [
        row["id"] for row in rows
        if row["current_status"] in {
            "registered_executable",
            "registered_external_missing",
            "not_supported",
        }
    ]
    return {
        "experiments": rows,
        "scheduler_manifest": [
            {
                "id": row["id"],
                "priority": row["priority"],
                "command": row["command"],
                "expected_artifact": row["artifact"],
                "paper_table": row["paper_table"],
            }
            for row in rows
        ],
        "summary": {
            "experiment_count": len(rows),
            "blockers": blockers,
            "blocker_count": len(blockers),
            "strong_learned_status": strong_status,
            "learned_cross_stress_scenarios": scenario_count,
            "same_agency_status": same_agency,
            "venue_grade_pairs": venue_pairs,
        },
        "boundary": (
            "This is an experiment readiness matrix, not a claim of completion. "
            "Rows only upgrade paper claims when their artifact gates pass."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "cs_experiment_matrix.csv", payload["experiments"])
    _write_csv(output_dir / "scheduler_manifest.csv", payload["scheduler_manifest"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# CS Top-Venue Experiment Matrix",
        "",
        payload["boundary"],
        "",
        f"- experiment count: `{payload['summary']['experiment_count']}`",
        f"- blocker count: `{payload['summary']['blocker_count']}`",
        f"- strong learned status: `{payload['summary']['strong_learned_status']}`",
        f"- learned stress scenarios: `{payload['summary']['learned_cross_stress_scenarios']}`",
        "",
        *_md_table(
            payload["experiments"],
            ["id", "experiment", "current_status", "claim_gate", "paper_table", "priority"],
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = build_cs_top_venue_experiment_matrix(results_root=args.results_root)
    write_outputs(args.output_dir, payload)
    print(
        "cs_top_venue_experiment_matrix "
        f"experiments={payload['summary']['experiment_count']} "
        f"blockers={payload['summary']['blocker_count']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
