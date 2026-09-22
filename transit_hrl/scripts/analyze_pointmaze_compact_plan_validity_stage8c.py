#!/usr/bin/env python3
"""Root-level analysis for Stage-8C compact plan-validity qualification."""

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

from freq_hrl.experiments.pointmaze_compact_plan_validity import (  # noqa: E402
    COMPACT_PLAN_VALIDITY_PREDICTORS,
    POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH,
    POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY,
    POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import (  # noqa: E402
    BRANCH_CATEGORIES,
)
from scripts.analyze_pointmaze_plan_validity_stage8b import (  # noqa: E402
    _interval,
    _selected_utility,
)
from scripts.pointmaze_compact_plan_validity_stage8c_spec import (  # noqa: E402
    OPTIMIZER_SEEDS,
    PREFLIGHT_OPTIMIZER_SEEDS,
    RIDGE_ALPHA_GRID,
    RUNTIME_EXPECTATIONS,
    seed_roles,
)


SEED_FIELDS = {
    "train": "train_seeds",
    "selection": "selection_seeds",
    "branch_fit": "branch_fit_seeds",
    "branch_eval": "branch_eval_seeds",
}
EXPECTED_FEATURE_COUNTS = {
    "current_compact_quadratic": 170,
    "causal_dynamic_quadratic": 170,
    "causal_validity_interactions": 39,
}


def _cell_identity(cell: dict[str, Any]) -> int:
    root = int(cell.get("optimizer_seed", -1))
    if (
        root < 0
        or cell.get("policy") != POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY
        or cell.get("protocol_version")
        != POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION
        or cell.get("algorithm_path")
        != POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH
        or cell.get("evidence_role")
        != "compact_plan_validity_predictor_qualification"
        or cell.get("trigger_training") != "disabled_qualification_only"
        or cell.get("plan_validity_predictor_deployment") != "disabled"
    ):
        raise ValueError("Stage-8C cell contract is invalid")
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
        if any(not values or len(values) != len(set(values))
               for values in role_values.values()):
            raise ValueError(f"Stage-8C seed role is empty or duplicated: {root}")
        role_sets = {role: set(values) for role, values in role_values.items()}
        sizes = tuple(len(role_sets[role]) for role in SEED_FIELDS)
        if expected_sizes is None:
            expected_sizes = sizes
        elif sizes != expected_sizes:
            raise ValueError("Stage-8C seed role counts differ across roots")
        if sum(sizes) != len(set().union(*role_sets.values())):
            raise ValueError(f"Stage-8C seed roles overlap within root: {root}")
        for role, seeds in role_sets.items():
            for seed in seeds:
                previous = owners.setdefault(seed, (root, role))
                if previous != (root, role):
                    raise ValueError(
                        "Stage-8C seeds are reused across optimizer roots: "
                        f"{seed} belongs to {previous} and {(root, role)}"
                    )


def _validate_controller_rows(cell: dict[str, Any]) -> float:
    root = _cell_identity(cell)
    fit_seeds = set(map(int, cell["branch_fit_seeds"]))
    eval_seeds = set(map(int, cell["branch_eval_seeds"]))
    controller_seeds = fit_seeds | eval_seeds
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
        raise ValueError(f"Stage-8C controller evaluation is incomplete: {root}")
    differences = []
    for seed in sorted(eval_seeds):
        trained = trained_all[seed]
        untrained = untrained_all[seed]
        if (
            float(trained.get("protocol_valid", 0.0)) != 1.0
            or float(untrained.get("protocol_valid", 0.0)) != 1.0
            or int(trained.get("training_replicate_seed", -1)) != root
            or int(untrained.get("training_replicate_seed", -1)) != root
        ):
            raise ValueError(
                f"Stage-8C controller row contract is invalid: {(root, seed)}"
            )
        differences.append(
            float(untrained["tracking_squared_error_integral"])
            - float(trained["tracking_squared_error_integral"])
        )
    return float(np.mean(differences))


def _validate_predictor_contract(cell: dict[str, Any]) -> None:
    root = _cell_identity(cell)
    predictor = cell.get("plan_validity_predictor", {})
    models = predictor.get("predictors", {})
    if (
        set(models) != set(COMPACT_PLAN_VALIDITY_PREDICTORS)
        or int(predictor.get("fit_group_count", -1))
        != len(cell["branch_fit_seeds"])
        or not bool(predictor.get(
            "selection_utility_is_local_counterfactual_not_closed_loop_return"
        ))
    ):
        raise ValueError(f"Stage-8C predictor suite is invalid: {root}")
    for name, expected_count in EXPECTED_FEATURE_COUNTS.items():
        item = models[name]
        feature_names = tuple(map(str, item.get("feature_names", [])))
        model = item.get("model", {})
        if (
            int(item.get("feature_count", -1)) != expected_count
            or len(feature_names) != expected_count
            or len(set(feature_names)) != expected_count
            or any(
                token in feature.lower()
                for feature in feature_names
                for token in ("distractor", "oracle", "regime", "event", "source", "lag")
            )
            or tuple(map(float, model.get("alpha_grid", [])))
            != tuple(map(float, RIDGE_ALPHA_GRID))
            or model.get("alpha_selection")
            != "leave_one_path_seed_out_mean_mse"
            or int(model.get("group_count", -1))
            != len(cell["branch_fit_seeds"])
            or float(model.get("alpha", -1.0)) not in RIDGE_ALPHA_GRID
        ):
            raise ValueError(
                f"Stage-8C predictor contract changed: {(root, name)}"
            )


def _validate_branch_rows(
    cell: dict[str, Any],
    *,
    field: str,
    split: str,
    seed_field: str,
    require_predictions: bool,
) -> list[dict[str, Any]]:
    root = _cell_identity(cell)
    seeds = set(map(int, cell[seed_field]))
    rows = cell.get(field)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Stage-8C branch rows are missing: {root}")
    schema = tuple(map(str, cell.get("branch_feature_names", [])))
    path_manifest = cell.get("branch_path_manifest", {})
    if not schema or not {str(seed) for seed in seeds}.issubset(path_manifest):
        raise ValueError(f"Stage-8C compact branch metadata is invalid: {root}")
    observed: set[tuple[int, str, int]] = set()
    counts: dict[int, Counter[str]] = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        category = str(row.get("category", ""))
        step = int(row.get("opportunity_step", -1))
        identity = (seed, category, step)
        if (
            seed not in seeds
            or category not in BRANCH_CATEGORIES
            or identity in observed
            or row.get("split") != split
            or row.get("protocol_version")
            != POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION
            or row.get("algorithm_path")
            != POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH
            or int(row.get("optimizer_seed", -1)) != root
            or not bool(row.get("protocol_valid"))
            or bool(row.get("candidate_feature_has_future_access", True))
            or bool(row.get("candidate_feature_has_regime_label", True))
            or bool(row.get("privileged_regime_context_present", True))
            or "oracle_regime_context" in row
            or int(row.get("keep_upper_calls_at_opportunity", -1)) != 0
            or int(row.get("renew_upper_calls_at_opportunity", -1)) != 1
            or int(row.get("downstream_upper_call_count_per_branch", -1)) != 0
            or not bool(row.get("lower_controller_remains_closed_loop"))
            or abs(float(row.get("prefix_max_abs_difference", 1.0))) > 1e-10
            or abs(float(row.get("feature_max_abs_difference", 1.0))) > 1e-10
            or len(row.get("causal_features", [])) != len(schema)
        ):
            raise ValueError(f"Stage-8C branch row is invalid: {identity}")
        observed.add(identity)
        counts.setdefault(seed, Counter())[category] += 1
        expected = (
            float(row["keep_tracking_squared_error_integral"])
            - float(row["renew_tracking_squared_error_integral"])
        )
        if (
            not np.isfinite(expected)
            or abs(expected - float(row["renew_ise_advantage"])) > 1e-12
        ):
            raise ValueError(f"Stage-8C branch advantage changed: {identity}")
        if require_predictions and any(
            not np.isfinite(float(row.get(f"prediction_{name}")))
            for name in COMPACT_PLAN_VALIDITY_PREDICTORS
        ):
            raise ValueError(f"Stage-8C predictor output is invalid: {identity}")
        source = row.get("source_step")
        lag = row.get("lag_steps")
        if category == "neutral_matched":
            if source is not None or lag is not None:
                raise ValueError(f"Stage-8C neutral label is invalid: {identity}")
        elif int(source) + int(lag) != step:
            raise ValueError(f"Stage-8C event lag is invalid: {identity}")
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
            raise ValueError(f"Stage-8C event source is invalid: {identity}")
    expected_count = int(cell.get("max_events_per_class", -1))
    if (
        set(counts) != seeds
        or expected_count < 1
        or any(set(counter) != set(BRANCH_CATEGORIES) for counter in counts.values())
        or any(set(counter.values()) != {expected_count} for counter in counts.values())
    ):
        raise ValueError(f"Stage-8C opportunity coverage is invalid: {root}")
    return rows


def _root_summary(cell: dict[str, Any]) -> dict[str, float]:
    learned = _validate_controller_rows(cell)
    _validate_predictor_contract(cell)
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
        str(seed) for seed in (
            *map(int, cell["branch_fit_seeds"]),
            *map(int, cell["branch_eval_seeds"]),
        )
    }
    if set(cell.get("branch_path_manifest", {})) != expected_manifest:
        raise ValueError("Stage-8C path manifest has unexpected seeds")
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
        raise ValueError("Stage-8C paired-branch budget accounting changed")

    target = np.asarray([
        float(row["renew_ise_advantage"]) for row in rows
    ], dtype=np.float64)
    predictions = {
        name: np.asarray([
            float(row[f"prediction_{name}"]) for row in rows
        ], dtype=np.float64)
        for name in COMPACT_PLAN_VALIDITY_PREDICTORS
    }
    selection_rate = float(
        cell["plan_validity_predictor"]["predictors"]
        ["causal_validity_interactions"]["evaluation"]["selection_rate"]
    )
    utility = {
        name: _selected_utility(rows, predictor=name, selection_rate=selection_rate)
        for name in COMPACT_PLAN_VALIDITY_PREDICTORS
    }
    spearman = {
        name: float(stats.spearmanr(target, prediction).statistic)
        for name, prediction in predictions.items()
    }
    if not all(np.isfinite(value) for value in spearman.values()):
        raise ValueError("Stage-8C rank correlation is non-finite")
    delayed = float(np.mean([
        row["renew_ise_advantage"]
        for row in rows
        if row["category"] == "regime_lag_250ms"
    ]))
    candidate = "causal_validity_interactions"
    current = "current_compact_quadratic"
    dynamic = "causal_dynamic_quadratic"
    return {
        "controller_learning_gain": learned,
        "delayed_renewal_value": delayed,
        **{f"selected_utility__{name}": value for name, value in utility.items()},
        **{f"spearman__{name}": value for name, value in spearman.items()},
        "candidate_utility_vs_current": utility[candidate] - utility[current],
        "candidate_utility_vs_dynamic": utility[candidate] - utility[dynamic],
        "candidate_spearman_vs_current": spearman[candidate] - spearman[current],
        "candidate_spearman_vs_dynamic": spearman[candidate] - spearman[dynamic],
    }


def analyze_stage8c(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
    expected_roots: Iterable[int] | None = None,
    expected_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    items = list(cells)
    if not items:
        raise ValueError("Stage-8C analysis requires cells")
    identities = [_cell_identity(cell) for cell in items]
    if len(set(identities)) != len(items):
        raise ValueError("Stage-8C optimizer roots must be unique")
    if expected_roots is not None and set(identities) != set(map(int, expected_roots)):
        raise ValueError("Stage-8C registered optimizer-root matrix is incomplete")
    runtimes = {
        json.dumps(cell.get("runtime_versions"), sort_keys=True) for cell in items
    }
    if len(runtimes) != 1 or "null" in runtimes:
        raise ValueError("Stage-8C runtime versions differ or are missing")
    if expected_runtime is not None and any(
        cell.get("runtime_versions", {}).get(name) != version
        for cell in items
        for name, version in expected_runtime.items()
    ):
        raise ValueError("Stage-8C runtime does not match the frozen protocol")
    _validate_seed_contracts(items)
    by_root = {_cell_identity(cell): _root_summary(cell) for cell in items}
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
        "delayed_renewal_has_value": (
            intervals["delayed_renewal_value"]["status"] == "supported"
        ),
        "candidate_rank_is_positive": (
            intervals["spearman__causal_validity_interactions"]["status"]
            == "supported"
        ),
        "candidate_selected_utility_is_positive": (
            intervals["selected_utility__causal_validity_interactions"]["status"]
            == "supported"
        ),
        "candidate_utility_beats_current_only": (
            intervals["candidate_utility_vs_current"]["status"] == "supported"
        ),
        "candidate_rank_beats_current_only": (
            intervals["candidate_spearman_vs_current"]["status"] == "supported"
        ),
        "candidate_rank_beats_generic_dynamic": (
            intervals["candidate_spearman_vs_dynamic"]["status"] == "supported"
        ),
    }
    authorized = bool(all(checks.values()))
    return {
        "analysis_version": "pointmaze_compact_plan_validity_stage8c_analysis_v1",
        "protocol_version": POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "independent_optimizer_root_count": len(items),
        "statistical_unit": "optimizer_seed_root",
        "primary_endpoint": (
            "causal_validity_interactions_selected_utility_minus_"
            "current_compact_quadratic"
        ),
        "root_summaries": {
            str(root): values for root, values in sorted(by_root.items())
        },
        "intervals": intervals,
        "qualification_checks": checks,
        "stage9_authorized": authorized,
        "decision": (
            "budgeted_trigger_development_authorized"
            if authorized else "stage9_not_authorized"
        ),
        "claim_boundary": (
            "held-out local branch-value prediction only; no deployed-trigger "
            "or closed-loop performance claim"
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
            != POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION
            or protocol.get("algorithm_path")
            != POINTMAZE_COMPACT_PLAN_VALIDITY_ALGORITHM_PATH
        ):
            raise ValueError(f"wrong or incomplete Stage-8C result: {path}")
        payload_cells = payload.get("cells", [])
        if len(payload_cells) != 1:
            raise ValueError(f"Stage-8C result must contain one cell: {path}")
        cells.extend(payload_cells)
    return cells


def render_report(analysis: dict[str, Any]) -> str:
    intervals = analysis["intervals"]
    rows = (
        ("controller_learning_gain", "controller learning ISE gain"),
        ("delayed_renewal_value", "renew value at regime +250 ms"),
        (
            "spearman__causal_validity_interactions",
            "candidate rank correlation",
        ),
        (
            "selected_utility__causal_validity_interactions",
            "candidate selected local value",
        ),
        ("candidate_utility_vs_current", "candidate utility minus current-only"),
        ("candidate_spearman_vs_current", "candidate rank minus current-only"),
        ("candidate_spearman_vs_dynamic", "candidate rank minus generic dynamic"),
        ("candidate_utility_vs_dynamic", "candidate utility minus generic dynamic"),
    )
    lines = [
        "# PointMaze Compact Plan-Validity Stage-8C",
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
    lines.extend(("", analysis["claim_boundary"] + ".", ""))
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
        raise ValueError("Stage-8C inputs are not a complete registered matrix")
    for cell in cells:
        expected = seed_roles(int(cell["optimizer_seed"]))
        if preflight:
            expected = {
                "train": expected["train"][:1],
                "selection": expected["selection"][:1],
                "branch_fit": expected["branch_fit"][:2],
                "branch_eval": expected["branch_eval"][:2],
            }
        if any(
            tuple(map(int, cell.get(SEED_FIELDS[role], []))) != values
            for role, values in expected.items()
        ):
            raise ValueError("Stage-8C seeds do not match the frozen protocol")
    analysis = analyze_stage8c(
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
