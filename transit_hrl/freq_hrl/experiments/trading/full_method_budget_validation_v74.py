"""Merge and gate the source-bound Freq-HRL v7.4 budget ladder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from freq_hrl.experiments.reproducibility import is_hex_digest

from . import full_method_budget_plan_v74 as plan
from . import full_method_hpo_v7 as hpo


BUDGET_VALIDATION_PROTOCOL_VERSION = (
    "freq_hrl_v7_4_robust_checkpoint_budget_validation_v1"
)


def _cell_summary(path: Path) -> dict[str, Any]:
    return json.loads(
        (Path(path) / "cell_summary.json").read_text(encoding="utf-8")
    )


def _expected_keys(budget: int) -> set[tuple[str, str, int]]:
    return {
        (variant_id, candidate_id, int(seed))
        for variant_id, candidates in plan.REPRESENTATIVE_CANDIDATES.items()
        for candidate_id in candidates
        for seed in plan.DEFAULT_BUDGET_OPTIMIZER_SEEDS
    }


def _row_passes(row: dict[str, Any]) -> bool:
    return (
        float(row["trained_checkpoint_fraction"])
        >= plan.MIN_TRAINED_REPLICATE_FRACTION
        and float(row["validation_learning_gain_mean"])
        > plan.MIN_MEAN_VALIDATION_LEARNING_GAIN
        and float(row["checkpoint_plateau_replicate_fraction"])
        >= plan.MIN_PLATEAU_REPLICATE_FRACTION
    )


def summarize_budget_cells(
    input_dirs: Iterable[Path],
    *,
    expected_budgets: Iterable[int],
) -> dict[str, Any]:
    budgets = tuple(map(int, expected_budgets))
    if not budgets or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budget merge requires unique increasing budgets")
    if not set(budgets).issubset(plan.BUDGET_LADDER):
        raise ValueError("budget merge contains an unregistered budget")
    if set(plan.REPRESENTATIVE_CANDIDATES) != set(hpo.HPO_VARIANT_IDS):
        raise ValueError("budget plan does not cover every tuned model family")
    for variant_id, candidates in plan.REPRESENTATIVE_CANDIDATES.items():
        allowed = set(hpo.candidate_ids_for_variant(variant_id))
        if not set(candidates).issubset(allowed):
            raise ValueError(
                f"budget plan candidate drifted for {variant_id}"
            )
    directories = [Path(path) for path in input_dirs]
    by_budget: dict[int, list[Path]] = {budget: [] for budget in budgets}
    for directory in directories:
        summary = _cell_summary(directory)
        budget = int(summary["iterations"])
        if budget not in by_budget:
            raise ValueError(f"unexpected budget cell: {budget}")
        by_budget[budget].append(directory)

    leaderboard: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    budget_rows: dict[int, list[dict[str, Any]]] = {}
    for budget in budgets:
        summaries, rows, hf_rows, seen = hpo._load_validated_hpo_cells(
            by_budget[budget]
        )
        expected = _expected_keys(budget)
        if seen != expected:
            delta = sorted((expected - seen) | (seen - expected))
            raise ValueError(
                f"incomplete budget-{budget} matrix: {delta[:6]}"
            )
        hpo._validate_common_hpo_fields(summaries)
        if {int(summary["iterations"]) for summary in summaries} != {budget}:
            raise ValueError(f"budget-{budget} matrix mixed iteration counts")
        rows_for_budget = []
        for variant_id, candidates in plan.REPRESENTATIVE_CANDIDATES.items():
            for candidate_id in candidates:
                row = hpo._candidate_leaderboard_row(
                    variant_id=variant_id,
                    candidate_id=candidate_id,
                    replicates=list(plan.DEFAULT_BUDGET_OPTIMIZER_SEEDS),
                    summaries=summaries,
                    rows=rows,
                    hf_rows=hf_rows,
                )
                row["iteration_budget"] = budget
                row["budget_gate_status"] = (
                    "pass" if _row_passes(row) else "fail"
                )
                rows_for_budget.append(row)
                leaderboard.append(row)
        budget_rows[budget] = rows_for_budget
        all_summaries.extend(summaries)

    revisions = {str(row["code_revision"]).lower() for row in all_summaries}
    manifests = {
        str(row["source_manifest_sha256"]).lower() for row in all_summaries
    }
    source_statuses = {
        str(row["source_identity_status"]) for row in all_summaries
    }
    source_verified = (
        len(revisions) == 1
        and len(manifests) == 1
        and source_statuses == {"verified"}
        and is_hex_digest(next(iter(revisions)), length=40)
        and is_hex_digest(next(iter(manifests)), length=64)
    )
    eligible_by_budget = {
        budget: bool(rows) and all(_row_passes(row) for row in rows)
        for budget, rows in budget_rows.items()
    }
    mandatory_complete = set(plan.MANDATORY_BUDGETS).issubset(budgets)
    selected_iterations: int | None = None
    if not source_verified:
        status = "invalid_source_identity"
    elif not mandatory_complete:
        status = "awaiting_mandatory_budgets"
    elif eligible_by_budget.get(plan.MIN_FINAL_ITERATIONS, False):
        selected_iterations = plan.MIN_FINAL_ITERATIONS
        status = "budget_selected"
    elif max(plan.BUDGET_LADDER) not in budgets:
        status = "requires_256"
    elif eligible_by_budget.get(max(plan.BUDGET_LADDER), False):
        selected_iterations = max(plan.BUDGET_LADDER)
        status = "budget_selected"
    else:
        status = "blocked_budget_limited_at_256"

    plan_audit = plan.validate_plan()
    return {
        "status": status,
        "protocol_version": BUDGET_VALIDATION_PROTOCOL_VERSION,
        "budget_plan_version": plan.BUDGET_PLAN_VERSION,
        "budget_plan_sha256": plan_audit["sha256"],
        "selection_rule": plan.SELECTION_RULE,
        "evaluated_budgets": list(budgets),
        "mandatory_budgets_complete": mandatory_complete,
        "selected_iterations": selected_iterations,
        "budget_gate_by_iterations": {
            str(budget): (
                "pass" if eligible_by_budget[budget] else "fail"
            )
            for budget in budgets
        },
        "representative_row_count": len(leaderboard),
        "cell_count": len(all_summaries),
        "optimizer_seeds": list(plan.DEFAULT_BUDGET_OPTIMIZER_SEEDS),
        "source_identity_status": (
            "verified" if source_verified else "unregistered_or_incomplete"
        ),
        "code_revision": next(iter(revisions)) if len(revisions) == 1 else "",
        "source_manifest_sha256": (
            next(iter(manifests)) if len(manifests) == 1 else ""
        ),
        "leaderboard": leaderboard,
    }


def validate_budget_decision(payload: dict[str, Any]) -> dict[str, Any]:
    plan_audit = plan.validate_plan()
    if payload.get("status") != "budget_selected":
        raise ValueError("training budget has not been selected")
    if payload.get("protocol_version") != BUDGET_VALIDATION_PROTOCOL_VERSION:
        raise ValueError("budget validation protocol drifted")
    if (
        payload.get("budget_plan_version") != plan.BUDGET_PLAN_VERSION
        or payload.get("budget_plan_sha256") != plan_audit["sha256"]
    ):
        raise ValueError("budget plan commitment drifted")
    selected = int(payload.get("selected_iterations", 0))
    if selected not in plan.BUDGET_LADDER or selected < plan.MIN_FINAL_ITERATIONS:
        raise ValueError("selected iteration budget is not registered")
    if payload.get("budget_gate_by_iterations", {}).get(str(selected)) != "pass":
        raise ValueError("selected iteration budget failed its gate")
    if not bool(payload.get("mandatory_budgets_complete")):
        raise ValueError("mandatory budget ladder was not completed")
    if payload.get("source_identity_status") != "verified":
        raise ValueError("budget decision source identity is not verified")
    if not is_hex_digest(payload.get("code_revision"), length=40):
        raise ValueError("budget decision code revision is invalid")
    if not is_hex_digest(payload.get("source_manifest_sha256"), length=64):
        raise ValueError("budget decision source manifest is invalid")
    return {"status": "valid", "selected_iterations": selected}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_budget_decision(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(payload["leaderboard"])
    serializable = {key: value for key, value in payload.items() if key != "leaderboard"}
    _write_csv(output_dir / "budget_leaderboard.csv", rows)
    (output_dir / "budget_decision.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Freq-HRL v7.4 Training-Budget Decision",
        "",
        f"- status: `{serializable['status']}`",
        f"- evaluated budgets: `{serializable['evaluated_budgets']}`",
        f"- selected iterations: `{serializable['selected_iterations']}`",
        f"- plan SHA-256: `{serializable['budget_plan_sha256']}`",
        "",
        "A final HPO freeze is forbidden unless this decision is valid and "
        "source-identical to every final HPO cell.",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
