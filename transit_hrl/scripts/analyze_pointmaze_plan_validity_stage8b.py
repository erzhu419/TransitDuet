#!/usr/bin/env python3
"""Root-level analysis for paired keep/renew Stage-8B qualification."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.pointmaze_plan_validity_branching import (  # noqa: E402
    BRANCH_CATEGORIES,
    POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH,
    POINTMAZE_PLAN_VALIDITY_POLICY,
    POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
    PREDICTOR_NAMES,
)
from scripts.pointmaze_plan_validity_stage8b_spec import (  # noqa: E402
    OPTIMIZER_SEEDS,
    PREFLIGHT_OPTIMIZER_SEEDS,
    RUNTIME_EXPECTATIONS,
    seed_roles,
)


SEED_FIELDS = {
    "train": "train_seeds",
    "selection": "selection_seeds",
    "branch_fit": "branch_fit_seeds",
    "branch_eval": "branch_eval_seeds",
}


def _interval(
    values: Iterable[float],
    *,
    confidence: float,
    classify: bool = True,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("Stage-8B analysis requires finite root values")
    mean = float(np.mean(array))
    if array.size < 2:
        lower, upper = float("-inf"), float("inf")
    else:
        standard_error = float(stats.sem(array))
        if standard_error <= 1e-15:
            lower = upper = mean
        else:
            critical = float(stats.t.ppf(
                0.5 + float(confidence) / 2.0,
                df=array.size - 1,
            ))
            lower = mean - critical * standard_error
            upper = mean + critical * standard_error
    result = {
        "n": int(array.size),
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": float(confidence),
    }
    if classify:
        result["status"] = (
            "supported"
            if lower > 0.0
            else "contradicted" if upper < 0.0 else "inconclusive"
        )
    return result


def _cell_identity(cell: dict[str, Any]) -> int:
    root = int(cell.get("optimizer_seed", -1))
    if (
        root < 0
        or cell.get("policy") != POINTMAZE_PLAN_VALIDITY_POLICY
        or cell.get("protocol_version")
        != POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION
        or cell.get("algorithm_path")
        != POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH
        or cell.get("evidence_role")
        != "counterfactual_plan_validity_qualification"
        or cell.get("trigger_training")
        != "disabled_qualification_only"
        or cell.get("plan_validity_predictor_deployment") != "disabled"
    ):
        raise ValueError("Stage-8B cell contract is invalid")
    return root


def _validate_seed_contracts(cells: Iterable[dict[str, Any]]) -> None:
    expected_sizes: tuple[int, ...] | None = None
    owners: dict[int, tuple[int, str]] = {}
    for cell in cells:
        root = _cell_identity(cell)
        role_values = {
            role: tuple(map(int, cell.get(field, [])))
            for role, field in SEED_FIELDS.items()
        }
        if any(len(values) != len(set(values)) for values in role_values.values()):
            raise ValueError(f"Stage-8B seed role contains duplicates: {root}")
        role_sets = {role: set(values) for role, values in role_values.items()}
        sizes = tuple(len(role_sets[role]) for role in SEED_FIELDS)
        if any(size < 1 for size in sizes):
            raise ValueError(f"Stage-8B seed role is empty: {root}")
        if expected_sizes is None:
            expected_sizes = sizes
        elif sizes != expected_sizes:
            raise ValueError("Stage-8B seed role counts differ across roots")
        if sum(sizes) != len(set().union(*role_sets.values())):
            raise ValueError(f"Stage-8B seed roles overlap within root: {root}")
        for role, seeds in role_sets.items():
            for seed in seeds:
                previous = owners.setdefault(seed, (root, role))
                if previous != (root, role):
                    raise ValueError(
                        "Stage-8B seeds are reused across optimizer roots: "
                        f"{seed} belongs to {previous} and {(root, role)}"
                    )


def _validate_controller_rows(
    cell: dict[str, Any],
    *,
    roots_seen: set[int],
) -> float:
    root = _cell_identity(cell)
    if root in roots_seen:
        raise ValueError(f"duplicate Stage-8B optimizer root: {root}")
    roots_seen.add(root)
    evaluation_seeds = set(map(int, cell.get("branch_eval_seeds", [])))
    fit_seeds = set(map(int, cell.get("branch_fit_seeds", [])))
    if not evaluation_seeds or not fit_seeds or evaluation_seeds & fit_seeds:
        raise ValueError(f"Stage-8B branch seed split is invalid: {root}")

    controller_seeds = evaluation_seeds | fit_seeds
    trained_rows = cell.get("canonical_evaluation_rows", [])
    untrained_rows = cell.get("untrained_evaluation_rows", [])
    trained_all = {int(row["seed"]): row for row in trained_rows}
    untrained_all = {int(row["seed"]): row for row in untrained_rows}
    if (
        len(trained_rows) != len(controller_seeds)
        or len(untrained_rows) != len(controller_seeds)
        or len(trained_all) != len(trained_rows)
        or len(untrained_all) != len(untrained_rows)
        or set(trained_all) != controller_seeds
        or set(untrained_all) != controller_seeds
    ):
        raise ValueError(f"Stage-8B controller evaluation is incomplete: {root}")
    trained = {seed: trained_all[seed] for seed in evaluation_seeds}
    untrained = {seed: untrained_all[seed] for seed in evaluation_seeds}
    differences = []
    for seed in sorted(evaluation_seeds):
        final = trained[seed]
        initial = untrained[seed]
        if (
            float(final.get("protocol_valid", 0.0)) != 1.0
            or float(initial.get("protocol_valid", 0.0)) != 1.0
            or int(final.get("training_replicate_seed", -1)) != root
            or int(initial.get("training_replicate_seed", -1)) != root
        ):
            raise ValueError(
                f"Stage-8B controller row contract is invalid: {(root, seed)}"
            )
        differences.append(
            float(initial["tracking_squared_error_integral"])
            - float(final["tracking_squared_error_integral"])
        )
    return float(np.mean(differences))


def _validate_branch_rows(
    cell: dict[str, Any],
    *,
    field: str,
    split: str,
    seed_field: str,
    require_predictions: bool,
) -> list[dict[str, Any]]:
    root = _cell_identity(cell)
    evaluation_seeds = set(map(int, cell[seed_field]))
    rows = cell.get(field)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Stage-8B branch rows are missing: {root}")
    observed: set[tuple[int, str, int]] = set()
    schema = tuple(map(str, cell.get("branch_feature_names", [])))
    masks = {
        name: tuple(map(int, indices))
        for name, indices in cell.get("branch_feature_masks", {}).items()
    }
    path_manifest = cell.get("branch_path_manifest", {})
    if not schema or not masks or not {
        str(seed) for seed in evaluation_seeds
    }.issubset(path_manifest):
        raise ValueError(f"Stage-8B compact branch metadata is invalid: {root}")
    categories_by_seed: dict[int, set[str]] = {}
    category_counts_by_seed: dict[int, Counter[str]] = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        category = str(row.get("category", ""))
        step = int(row.get("opportunity_step", -1))
        identity = (seed, category, step)
        if (
            seed not in evaluation_seeds
            or category not in BRANCH_CATEGORIES
            or identity in observed
            or row.get("split") != split
            or row.get("protocol_version")
            != POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION
            or row.get("algorithm_path")
            != POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH
            or int(row.get("optimizer_seed", -1)) != root
            or not bool(row.get("protocol_valid"))
            or bool(row.get("candidate_feature_has_future_access", True))
            or bool(row.get("candidate_feature_has_regime_label", True))
            or not bool(row.get(
                "oracle_regime_used_only_by_diagnostic_predictor", False
            ))
            or int(row.get("keep_upper_calls_at_opportunity", -1)) != 0
            or int(row.get("renew_upper_calls_at_opportunity", -1)) != 1
            or int(row.get("downstream_upper_call_count_per_branch", -1))
            != 0
            or not bool(row.get("lower_controller_remains_closed_loop"))
            or abs(float(row.get("prefix_max_abs_difference", 1.0))) > 1e-10
            or abs(float(row.get("feature_max_abs_difference", 1.0))) > 1e-10
        ):
            raise ValueError(f"Stage-8B branch row is invalid: {identity}")
        observed.add(identity)
        categories_by_seed.setdefault(seed, set()).add(category)
        category_counts_by_seed.setdefault(seed, Counter())[category] += 1
        if len(row.get("causal_features", [])) != len(schema):
            raise ValueError(f"Stage-8B feature shape is invalid: {identity}")
        if any(
            not np.isfinite(float(row[key]))
            for key in (
                "keep_tracking_squared_error_integral",
                "renew_tracking_squared_error_integral",
                "renew_ise_advantage",
            )
        ):
            raise ValueError(f"Stage-8B branch metric is invalid: {identity}")
        expected = (
            float(row["keep_tracking_squared_error_integral"])
            - float(row["renew_tracking_squared_error_integral"])
        )
        if abs(expected - float(row["renew_ise_advantage"])) > 1e-12:
            raise ValueError(f"Stage-8B branch advantage changed: {identity}")
        if require_predictions:
            for predictor in PREDICTOR_NAMES:
                if not np.isfinite(float(row.get(f"prediction_{predictor}"))):
                    raise ValueError(
                        f"Stage-8B predictor output is invalid: {identity}"
                    )
        source = row.get("source_step")
        lag = row.get("lag_steps")
        if category == "neutral_matched":
            if source is not None or lag is not None:
                raise ValueError(f"Stage-8B neutral label is invalid: {identity}")
        elif int(source) + int(lag) != step:
            raise ValueError(f"Stage-8B event lag is invalid: {identity}")
        path = path_manifest[str(seed)]
        if (
            category.startswith("regime_")
            and int(source) not in path["regime_change_steps"]
        ) or (
            category.startswith("force_")
            and int(source) not in path["force_pulse_start_steps"]
        ) or (
            category.startswith("distractor_")
            and int(source) not in path["distractor_change_steps"]
        ):
            raise ValueError(f"Stage-8B event source is invalid: {identity}")
    if set(categories_by_seed) != evaluation_seeds or any(
        categories != set(BRANCH_CATEGORIES)
        for categories in categories_by_seed.values()
    ):
        raise ValueError(f"Stage-8B category coverage is incomplete: {root}")
    expected_count = int(cell.get("max_events_per_class", -1))
    if expected_count < 1 or any(
        set(counts.values()) != {expected_count}
        for counts in category_counts_by_seed.values()
    ):
        raise ValueError(f"Stage-8B category counts are unbalanced: {root}")
    return rows


def _selected_utility(
    rows: list[dict[str, Any]],
    *,
    predictor: str,
    selection_rate: float,
) -> float:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    utilities = []
    for seed_rows in grouped.values():
        count = max(1, int(round(float(selection_rate) * len(seed_rows))))
        selected = sorted(
            seed_rows,
            key=lambda row: (
                float(row[f"prediction_{predictor}"]),
                -int(row["opportunity_step"]),
            ),
            reverse=True,
        )[:count]
        utilities.append(float(np.mean([
            float(row["renew_ise_advantage"]) for row in selected
        ])))
    return float(np.mean(utilities))


def _root_summary(cell: dict[str, Any]) -> dict[str, float]:
    roots_seen: set[int] = set()
    learned = _validate_controller_rows(cell, roots_seen=roots_seen)
    fit_rows = _validate_branch_rows(
        cell,
        field="branch_fit_rows",
        split="predictor_fit",
        seed_field="branch_fit_seeds",
        require_predictions=False,
    )
    rows = _validate_branch_rows(
        cell,
        field="branch_evaluation_rows",
        split="qualification_eval",
        seed_field="branch_eval_seeds",
        require_predictions=True,
    )
    expected_manifest = {
        str(seed)
        for seed in (
            *map(int, cell["branch_fit_seeds"]),
            *map(int, cell["branch_eval_seeds"]),
        )
    }
    if set(cell.get("branch_path_manifest", {})) != expected_manifest:
        raise ValueError("Stage-8B path manifest has unexpected seeds")
    budget = cell.get("paired_branch_transition_budget", {})
    expected_fit = sum(
        int(row["keep_primitive_steps_replayed"])
        + int(row["renew_primitive_steps_replayed"])
        for row in fit_rows
    )
    expected_eval = sum(
        int(row["keep_primitive_steps_replayed"])
        + int(row["renew_primitive_steps_replayed"])
        for row in rows
    )
    if (
        int(budget.get("fit_primitive_steps_replayed", -1)) != expected_fit
        or int(budget.get("evaluation_primitive_steps_replayed", -1))
        != expected_eval
        or not bool(budget.get("counted_as_extra_supervision"))
    ):
        raise ValueError("Stage-8B paired-branch budget accounting changed")
    category_mean = {
        category: float(np.mean([
            float(row["renew_ise_advantage"])
            for row in rows
            if row["category"] == category
        ]))
        for category in BRANCH_CATEGORIES
    }
    target = np.asarray([
        float(row["renew_ise_advantage"]) for row in rows
    ], dtype=np.float64)
    predictions = {
        name: np.asarray([
            float(row[f"prediction_{name}"]) for row in rows
        ], dtype=np.float64)
        for name in PREDICTOR_NAMES
    }
    selection_rate = float(
        cell["plan_validity_predictor"]["predictors"]["causal_history"]
        ["evaluation"]["selection_rate"]
    )
    spearman = {
        name: float(stats.spearmanr(target, values).statistic)
        for name, values in predictions.items()
    }
    spearman = {
        name: value if np.isfinite(value) else 0.0
        for name, value in spearman.items()
    }
    utility = {
        name: _selected_utility(
            rows, predictor=name, selection_rate=selection_rate
        )
        for name in PREDICTOR_NAMES
    }
    return {
        "controller_learning_gain": learned,
        **{
            f"category_mean__{name}": value
            for name, value in category_mean.items()
        },
        "regime_250_vs_010": (
            category_mean["regime_lag_250ms"]
            - category_mean["regime_lag_010ms"]
        ),
        "regime_250_vs_force": (
            category_mean["regime_lag_250ms"]
            - category_mean["force_pulse_010ms"]
        ),
        "regime_250_vs_distractor": (
            category_mean["regime_lag_250ms"]
            - category_mean["distractor_change_010ms"]
        ),
        **{
            f"spearman__{name}": value for name, value in spearman.items()
        },
        **{
            f"selected_utility__{name}": value
            for name, value in utility.items()
        },
        "history_selected_utility_vs_plan_state": (
            utility["causal_history"] - utility["plan_state"]
        ),
        "oracle_regime_selected_utility_increment": (
            utility["causal_history_plus_regime"]
            - utility["causal_history"]
        ),
        "oracle_regime_spearman_increment": (
            spearman["causal_history_plus_regime"]
            - spearman["causal_history"]
        ),
    }


def analyze_stage8b(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
    expected_roots: Iterable[int] | None = None,
    expected_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    items = list(cells)
    if not items:
        raise ValueError("Stage-8B analysis requires cells")
    identities = [_cell_identity(cell) for cell in items]
    if len(set(identities)) != len(items):
        raise ValueError("Stage-8B optimizer roots must be unique")
    if expected_roots is not None and set(identities) != set(map(int, expected_roots)):
        raise ValueError("Stage-8B registered optimizer-root matrix is incomplete")
    runtime = {
        json.dumps(cell.get("runtime_versions"), sort_keys=True)
        for cell in items
    }
    if len(runtime) != 1 or "null" in runtime:
        raise ValueError("Stage-8B runtime versions differ or are missing")
    if expected_runtime is not None and any(
        cell.get("runtime_versions", {}).get(name) != version
        for cell in items
        for name, version in expected_runtime.items()
    ):
        raise ValueError("Stage-8B runtime does not match the frozen protocol")
    _validate_seed_contracts(items)

    by_root = {
        _cell_identity(cell): _root_summary(cell) for cell in items
    }
    metric_names = tuple(next(iter(by_root.values())))
    intervals = {
        name: _interval(
            (summary[name] for summary in by_root.values()),
            confidence=confidence,
        )
        for name in metric_names
    }
    checks = {
        "controller_learned": (
            intervals["controller_learning_gain"]["status"] == "supported"
        ),
        "renewal_has_value_after_250ms": (
            intervals["category_mean__regime_lag_250ms"]["status"]
            == "supported"
        ),
        "value_emerges_after_causal_consequence": (
            intervals["regime_250_vs_010"]["status"] == "supported"
        ),
        "regime_value_exceeds_force_pulse": (
            intervals["regime_250_vs_force"]["status"] == "supported"
        ),
        "regime_value_exceeds_distractor": (
            intervals["regime_250_vs_distractor"]["status"] == "supported"
        ),
        "causal_history_predicts_renewal_value": (
            intervals["spearman__causal_history"]["status"] == "supported"
        ),
        "causal_history_selection_has_positive_local_value": (
            intervals["selected_utility__causal_history"]["status"]
            == "supported"
        ),
        "history_selection_beats_current_plan_state": (
            intervals["history_selected_utility_vs_plan_state"]["status"]
            == "supported"
        ),
    }
    authorized = bool(all(checks.values()))
    return {
        "analysis_version": "pointmaze_plan_validity_stage8b_analysis_v2",
        "protocol_version": POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "independent_optimizer_root_count": len(items),
        "statistical_unit": "optimizer_seed_root",
        "primary_endpoint": "renew_ise_advantage_keep_minus_renew",
        "root_summaries": {
            str(root): values for root, values in sorted(by_root.items())
        },
        "intervals": intervals,
        "qualification_checks": checks,
        "stage9_authorized": authorized,
        "decision": (
            "plan_validity_trigger_development_authorized"
            if authorized else "stage9_not_authorized"
        ),
        "claim_boundary": (
            "paired local branch value and predictor ranking only; no "
            "deployed-trigger or closed-loop performance claim"
        ),
    }


def load_cells(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        if (
            payload.get("status") != "complete"
            or protocol.get("protocol_version")
            != POINTMAZE_PLAN_VALIDITY_PROTOCOL_VERSION
            or protocol.get("algorithm_path")
            != POINTMAZE_PLAN_VALIDITY_ALGORITHM_PATH
        ):
            raise ValueError(f"wrong or incomplete Stage-8B result: {path}")
        payload_cells = payload.get("cells", [])
        if len(payload_cells) != 1:
            raise ValueError(f"Stage-8B result must contain one cell: {path}")
        cells.extend(payload_cells)
    return cells


def render_report(analysis: dict[str, Any]) -> str:
    intervals = analysis["intervals"]
    rows = (
        ("controller_learning_gain", "controller learning ISE gain"),
        (
            "category_mean__regime_lag_250ms",
            "renew value at regime +250 ms",
        ),
        ("regime_250_vs_010", "regime +250 ms minus +10 ms"),
        ("regime_250_vs_force", "regime +250 ms minus force pulse"),
        (
            "regime_250_vs_distractor",
            "regime +250 ms minus distractor change",
        ),
        ("spearman__causal_history", "causal-history rank correlation"),
        (
            "selected_utility__causal_history",
            "causal-history selected local value",
        ),
        (
            "history_selected_utility_vs_plan_state",
            "history selection minus plan-state baseline",
        ),
    )
    lines = [
        "# PointMaze Counterfactual Plan-Validity Stage-8B",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        f"Decision: **{analysis['decision']}**",
        "",
        "| Registered quantity | Root mean [95% CI] | Status |",
        "|---|---:|---|",
    ]
    for key, label in rows:
        value = intervals[key]
        lines.append(
            f"| {label} | {value['mean']:.6f} "
            f"[{value['ci_lower']:.6f}, {value['ci_upper']:.6f}] "
            f"| {value['status']} |"
        )
    lines.extend((
        "",
        analysis["claim_boundary"] + ".",
        "",
    ))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = load_cells(args.inputs)
    roots = {int(cell["optimizer_seed"]) for cell in cells}
    if roots == set(PREFLIGHT_OPTIMIZER_SEEDS):
        expected_roots = PREFLIGHT_OPTIMIZER_SEEDS
        preflight = True
    elif roots == set(OPTIMIZER_SEEDS):
        expected_roots = OPTIMIZER_SEEDS
        preflight = False
    else:
        raise ValueError("Stage-8B inputs are not a complete registered matrix")
    for cell in cells:
        expected = seed_roles(int(cell["optimizer_seed"]))
        if preflight:
            expected = {name: values[:1] for name, values in expected.items()}
        if any(
            tuple(map(int, cell.get(SEED_FIELDS[role], []))) != values
            for role, values in expected.items()
        ):
            raise ValueError("Stage-8B seeds do not match the frozen protocol")
    analysis = analyze_stage8b(
        cells,
        confidence=args.confidence,
        expected_roots=expected_roots,
        expected_runtime=RUNTIME_EXPECTATIONS,
    )
    analysis["matrix"] = "preflight" if preflight else "formal_development"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(analysis), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
