"""Build compact manuscript evidence from completed local results; no rollouts."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE9 = "pointmaze_budgeted_trigger_stage9_v2_confirmation_20260926_r1"
UNPRIV_RUN = "pointmaze_stage9_unprivileged_queries_20261118"
PARETO_RUN = "pointmaze_stage9_budget_pareto_20261118"
STRONG_BASELINES_RUN = "pointmaze_route_b_strong_baselines_20261118"
REACHER_POLYGONAL_RUN = "reacher_polygonal_route_b_confirmation_20261118"
CEM_PLANNER_RUN = "pointmaze_route_b_expensive_cem_planner_20261118"
REVIEW_CONTROLS_RUN = "pointmaze_review_controls_20261120"
ROOTS = (209011, 209023, 209037, 209049, 209061, 209073, 209089, 209101)
MODES = ("fixed", "balanced_jitter", "current_compact_quadratic", "causal_validity_interactions")
PAIRS = {
    "fixed_minus_candidate_ise": ("fixed", "causal_validity_interactions", "tracking_squared_error_integral"),
    "jitter_minus_candidate_ise": ("balanced_jitter", "causal_validity_interactions", "tracking_squared_error_integral"),
    "current_minus_candidate_ise": ("current_compact_quadratic", "causal_validity_interactions", "tracking_squared_error_integral"),
    "candidate_minus_fixed_return": ("causal_validity_interactions", "fixed", "episode_return"),
}
T7 = 2.3646242510102993


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def t7_interval(values: list[float]) -> dict:
    assert len(values) == 8
    mean = fmean(values)
    radius = T7 * stdev(values) / sqrt(8)
    return {"mean": mean, "CI95": [mean - radius, mean + radius]}


def stage9_input(root: int) -> Path:
    return PAPER / "data" / "stage9" / f"replicate_{root}.json"


def extract_stage9_inputs() -> None:
    """Keep only episode outcomes, saved call times and fit accounting."""
    trigger_fields = (
        "seed",
        "mode",
        "tracking_squared_error_integral",
        "episode_return",
        "upper_decision_count",
        "protocol_valid",
        "trigger_score_checks",
        "trigger_early_calls",
        "decision_steps",
    )
    outcome_fields = ("seed", "tracking_squared_error_integral")
    for root in ROOTS:
        source = Path("results") / STAGE9 / "cells/hrl_regime_history" / f"replicate_{root}/result.json"
        payload = read(ROOT / source)
        cell = payload["cells"][0]
        assert payload["status"] == "complete" and cell["optimizer_seed"] == root
        compact = {
            "original_source": source.as_posix(),
            **{
                key: cell[key]
                for key in ("optimizer_seed", "trigger_eval_seeds", "branch_fit_primitive_steps_replayed")
            },
            "trigger_evaluation_rows": [
                {key: row[key] for key in trigger_fields} for row in cell["trigger_evaluation_rows"]
            ],
            **{
                field: [{key: row[key] for key in outcome_fields} for row in cell[field]]
                for field in ("canonical_evaluation_rows", "untrained_evaluation_rows")
            },
            "policy_contract": {
                "bin_steps": cell["time_scale"]["upper_period_steps"],
                "max_offset_steps": cell["trigger_max_offset_steps"],
                "check_stride_steps": cell["trigger_check_stride_steps"],
                "branch_horizon_steps": round(
                    cell["branch_window_seconds"] / cell["time_scale"]["env_dt_seconds"]
                ),
                "candidate_features": len(cell["trigger_predictors"][MODES[-1]]["feature_names"]),
                "threshold_quantile": cell["trigger_threshold_quantile"],
                "fit_queries_per_root": len(cell["branch_fit_rows"]),
            },
        }
        path = stage9_input(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compact, indent=2) + "\n")


def stage9() -> dict:
    effects = {name: [] for name in (*PAIRS, "controller_learning_ise")}
    means = {mode: [] for mode in MODES}
    checks, early, branch_steps, paths, by_root = [], [], 0, [], {}
    mode_checks = {mode: [] for mode in MODES}
    example, policy_contract = None, None
    paths_favoring_candidate = 0
    for root in ROOTS:
        path = stage9_input(root)
        cell = read(path)
        assert cell["optimizer_seed"] == root
        seeds = cell["trigger_eval_seeds"]
        rows = cell["trigger_evaluation_rows"]
        indexed = {(r["seed"], r["mode"]): r for r in rows}
        assert len(seeds) == 16 and len(indexed) == len(rows) == 64
        assert all(r["upper_decision_count"] == 24 and r["protocol_valid"] for r in rows)
        paths_favoring_candidate += sum(
            indexed[s, "fixed"]["tracking_squared_error_integral"]
            > indexed[s, MODES[-1]]["tracking_squared_error_integral"] for s in seeds
        )
        for name, (after, before, field) in PAIRS.items():
            effects[name].append(fmean(indexed[s, after][field] - indexed[s, before][field] for s in seeds))
        trained = {r["seed"]: r for r in cell["canonical_evaluation_rows"]}
        untrained = {r["seed"]: r for r in cell["untrained_evaluation_rows"]}
        effects["controller_learning_ise"].append(
            fmean(
                untrained[s]["tracking_squared_error_integral"]
                - trained[s]["tracking_squared_error_integral"]
                for s in seeds
            )
        )
        for mode in MODES:
            means[mode].append(fmean(indexed[s, mode]["tracking_squared_error_integral"] for s in seeds))
            mode_checks[mode].extend(indexed[s, mode]["trigger_score_checks"] for s in seeds)
        if root == ROOTS[0]:
            example = {
                "root": root,
                "path_seed": seeds[0],
                "source": str(path.relative_to(ROOT)),
                "decision_steps": {mode: indexed[seeds[0], mode]["decision_steps"] for mode in MODES},
            }
            policy_contract = cell["policy_contract"]
        candidate = [indexed[s, MODES[-1]] for s in seeds]
        checks.extend(r["trigger_score_checks"] for r in candidate)
        early.extend(r["trigger_early_calls"] for r in candidate)
        branch_steps += cell["branch_fit_primitive_steps_replayed"]
        by_root[str(root)] = {name: values[-1] for name, values in effects.items()}
        paths.append(str(path.relative_to(ROOT)))
    intervals = {name: t7_interval(values) for name, values in effects.items()}
    return {
        "sources": paths,
        "statistical_unit": "optimizer_seed_root",
        "root_count": 8,
        "evaluation_paths": 128,
        "paths_favoring_candidate_over_fixed": paths_favoring_candidate,
        "paired_mode_rollouts": 512,
        "root_effects": by_root,
        "interval_method": "paired_root_Student_t_df7_two_sided95;unchanged_registered_endpoints",
        "contrasts": intervals,
        "mean_ise": {mode: fmean(v) for mode, v in means.items()},
        "upper_calls_per_episode_all_modes": 24,
        "candidate_score_checks_mean": fmean(checks),
        "candidate_score_checks_range": [min(checks), max(checks)],
        "candidate_predeadline_calls_mean": fmean(early),
        "branch_fit_primitive_steps_replayed_all_roots": branch_steps,
        "score_checks_by_mode": {mode: fmean(values) for mode, values in mode_checks.items()},
        "example_schedule": example,
        "policy_contract": policy_contract,
    }


def unprivileged_ablation() -> dict:
    path = ROOT / "results" / UNPRIV_RUN / "summary.json"
    summary = read(path)
    roots = summary["roots"]
    assert summary["root_count"] == len(roots) == 8
    by_root = {}
    contrast_specs = {
        "periodic_minus_unprivileged_uniform_ise": ("periodic", "unprivileged_uniform"),
        "jitter_minus_unprivileged_uniform_ise": ("jitter", "unprivileged_uniform"),
        "distance_minus_unprivileged_uniform_ise": ("distance_threshold", "unprivileged_uniform"),
        "privileged_minus_unprivileged_uniform_ise": ("privileged_stage9", "unprivileged_uniform"),
        "distance_minus_unprivileged_observable_ise": ("distance_threshold", "unprivileged_observable"),
    }
    series = {k: [] for k in contrast_specs}
    for r in roots:
        seed = str(r["optimizer_seed"])
        all_panel = r["panels"]["all"]
        by_root[seed] = {}
        for name, (ref, cand) in contrast_specs.items():
            val = all_panel[ref]["mean_ise"] - all_panel[cand]["mean_ise"]
            by_root[seed][name] = val
            series[name].append(val)
    contrasts = {name: t7_interval(vals) for name, vals in series.items()}
    return {
        "source": str(path.relative_to(ROOT)),
        "statistical_unit": "optimizer_seed_root",
        "root_count": 8,
        "evaluation_paths_per_root": 64,
        "paired_paths_total": 512,
        "mean_ise": summary["mean_ise"],
        "mean_return": summary["mean_return"],
        "mean_early_calls": summary["mean_early_calls"],
        "mean_score_checks": summary["mean_score_checks"],
        "contrasts": contrasts,
        "root_effects": by_root,
    }


def budget_pareto() -> dict:
    path = ROOT / "results" / PARETO_RUN / "summary.json"
    summary = read(path)
    roots = summary["roots"]
    assert summary["root_count"] == len(roots) == 8
    budgets = ("12", "15", "20", "24", "30", "40")
    frontier = {}
    for b in budgets:
        entry = summary["frontier"][b]
        dist_minus_unpriv = [
            r["budgets"][b]["methods"]["distance_threshold"]["mean_ise"]
            - r["budgets"][b]["methods"]["unprivileged_uniform"]["mean_ise"]
            for r in roots
        ]
        per_minus_unpriv = [
            r["budgets"][b]["methods"]["periodic"]["mean_ise"]
            - r["budgets"][b]["methods"]["unprivileged_uniform"]["mean_ise"]
            for r in roots
        ]
        frontier[b] = {
            "calls_per_episode": int(b),
            "mean_ise": entry["mean_ise"],
            "mean_return": entry["mean_return"],
            "mean_score_checks": entry["mean_score_checks"],
            "periodic_minus_unprivileged_ise": t7_interval(per_minus_unpriv),
            "distance_minus_unprivileged_ise": t7_interval(dist_minus_unpriv),
            "distance_minus_unprivileged_roots": {
                str(r["optimizer_seed"]): d for r, d in zip(roots, dist_minus_unpriv)
            },
            "relative_gain_vs_periodic": entry["contrasts"]["unprivileged_uniform_vs_periodic"]["relative_ise_gain"],
            "relative_gain_vs_distance": entry["contrasts"]["unprivileged_uniform_vs_distance_threshold"]["relative_ise_gain"],
        }

    def cross_contrast(cand_b: str, ref_b: str, ref: str = "periodic") -> dict:
        reference = [r["budgets"][ref_b]["methods"][ref]["mean_ise"] for r in roots]
        diffs = [
            baseline - r["budgets"][cand_b]["methods"]["unprivileged_uniform"]["mean_ise"]
            for r, baseline in zip(roots, reference)
        ]
        ci = t7_interval(diffs)
        return {
            **ci,
            "relative_gain": ci["mean"] / fmean(reference),
            "call_reduction_pct": 1.0 - int(cand_b) / int(ref_b),
            "favorable_roots": sum(d > 0.0 for d in diffs),
        }

    cross_savings = {
        "unprivileged_B20_vs_periodic_B24": cross_contrast("20", "24"),
        "unprivileged_B24_vs_periodic_B30": cross_contrast("24", "30"),
        "unprivileged_B30_vs_periodic_B40": cross_contrast("30", "40"),
        "unprivileged_B20_vs_distance_B24": cross_contrast("20", "24", "distance_threshold"),
        "unprivileged_B24_vs_distance_B30": cross_contrast("24", "30", "distance_threshold"),
    }
    return {
        "source": str(path.relative_to(ROOT)),
        "statistical_unit": "optimizer_seed_root",
        "root_count": 8,
        "evaluation_paths_per_root": 64,
        "paired_paths_per_budget": 512,
        "frontier": frontier,
        "cross_budget_call_savings": cross_savings,
    }


def pointmaze_strong_baselines() -> dict:
    path = ROOT / "results" / STRONG_BASELINES_RUN / "summary.json"
    summary = read(path)
    roots = summary["roots"]
    assert summary["root_count"] == len(roots) == 8
    contrast_specs = {
        "native_periodic_minus_ridge_ise": ("native_periodic_hrl", "ridge_unprivileged"),
        "shared_periodic_minus_ridge_ise": ("shared_periodic", "ridge_unprivileged"),
        "shared_jitter_minus_ridge_ise": ("shared_jitter", "ridge_unprivileged"),
        "temple_minus_ridge_ise": ("temple_gate", "ridge_unprivileged"),
        "distance_minus_ridge_ise": ("distance_threshold", "ridge_unprivileged"),
        "deliberation_oc_minus_ridge_ise": ("deliberation_option_critic", "ridge_unprivileged"),
        "generic_mlp_minus_ridge_ise": ("generic_mlp_unprivileged", "ridge_unprivileged"),
        "native_periodic_minus_generic_mlp_ise": ("native_periodic_hrl", "generic_mlp_unprivileged"),
        "shared_periodic_minus_generic_mlp_ise": ("shared_periodic", "generic_mlp_unprivileged"),
        "temple_minus_generic_mlp_ise": ("temple_gate", "generic_mlp_unprivileged"),
    }
    by_root = {}
    series = {k: [] for k in contrast_specs}
    for r in roots:
        seed = str(r["optimizer_seed"])
        all_panel = r["panels"]["all"]
        by_root[seed] = {}
        for name, (ref, cand) in contrast_specs.items():
            val = all_panel[ref]["mean_ise"] - all_panel[cand]["mean_ise"]
            by_root[seed][name] = val
            series[name].append(val)
    contrasts = {}
    for name, (ref, cand) in contrast_specs.items():
        ci = t7_interval(series[name])
        ref_mean = summary["mean_ise"][ref]
        contrasts[name] = {
            **ci,
            "relative_gain": ci["mean"] / ref_mean,
            "favorable_roots": sum(v > 0.0 for v in series[name]),
        }
    return {
        "source": str(path.relative_to(ROOT)),
        "statistical_unit": "optimizer_seed_root",
        "root_count": 8,
        "evaluation_paths_per_root": 64,
        "paired_paths_total": 512,
        "mean_ise": summary["mean_ise"],
        "mean_return": summary["mean_return"],
        "mean_early_calls": summary["mean_early_calls"],
        "contrasts": contrasts,
        "root_effects": by_root,
    }


def reacher_polygonal_confirmation() -> dict:
    path = ROOT / "results" / REACHER_POLYGONAL_RUN / "summary.json"
    summary = read(path)
    roots = summary["roots"]
    assert summary["root_count"] == len(roots) == 8
    contrast_specs = {
        "shared_jitter_minus_ridge_ise": ("shared_jitter", "ridge_unprivileged"),
        "best_clock_minus_ridge_ise": ("best_clock", "ridge_unprivileged"),
        "shared_periodic_minus_ridge_ise": ("shared_periodic", "ridge_unprivileged"),
        "temple_minus_ridge_ise": ("temple_gate", "ridge_unprivileged"),
        "distance_minus_ridge_ise": ("distance_threshold", "ridge_unprivileged"),
        "deliberation_oc_minus_ridge_ise": ("deliberation_option_critic", "ridge_unprivileged"),
        "native_periodic_minus_ridge_ise": ("native_periodic_hrl", "ridge_unprivileged"),
        "best_clock_minus_generic_mlp_ise": ("best_clock", "generic_mlp_unprivileged"),
        "shared_periodic_minus_generic_mlp_ise": ("shared_periodic", "generic_mlp_unprivileged"),
        "temple_minus_generic_mlp_ise": ("temple_gate", "generic_mlp_unprivileged"),
        "generic_mlp_minus_ridge_ise": ("generic_mlp_unprivileged", "ridge_unprivileged"),
    }
    by_root = {}
    series = {k: [] for k in contrast_specs}
    learning_vals = []
    for r in roots:
        seed = str(r["optimizer_seed"])
        all_panel = r["panels"]["all"]
        by_root[seed] = {}
        for name, (ref, cand) in contrast_specs.items():
            val = all_panel[ref]["mean_ise"] - all_panel[cand]["mean_ise"]
            by_root[seed][name] = val
            series[name].append(val)
        learning_vals.append(r["untrained_mean_ise"] - all_panel["shared_jitter"]["mean_ise"])
    contrasts = {}
    for name, (ref, cand) in contrast_specs.items():
        ci = t7_interval(series[name])
        ref_mean = summary["mean_ise"][ref]
        contrasts[name] = {
            **ci,
            "relative_gain": ci["mean"] / ref_mean,
            "favorable_roots": sum(v > 0.0 for v in series[name]),
        }
    nominal = {}
    for name, (ref, cand) in contrast_specs.items():
        if cand == "ridge_unprivileged" and ref != "generic_mlp_unprivileged":
            values = series[name]
            nominal[name] = float(2 * student_t.sf(abs(fmean(values)) / (stdev(values) / sqrt(8)), df=7))
    adjusted, running_max = {}, 0.0
    for rank, name in enumerate(sorted(nominal, key=nominal.get)):
        running_max = max(running_max, min(1.0, (len(nominal) - rank) * nominal[name]))
        adjusted[name] = running_max
    auth_roots = [r for r in roots if r["waypoint_authority_relative_gain"] > 0.05]
    low_roots = [r for r in roots if r["waypoint_authority_relative_gain"] <= 0.05]
    methods = tuple(summary["mean_ise"].keys())
    return {
        "source": str(path.relative_to(ROOT)),
        "plant": summary["plant"],
        "statistical_unit": "optimizer_seed_root",
        "root_count": 8,
        "evaluation_paths_per_root": 64,
        "paired_paths_total": 512,
        "controller_learning_ise": t7_interval(learning_vals),
        "mean_waypoint_authority_relative_gain": summary["mean_waypoint_authority_relative_gain"],
        "authority_stratum_counts": {
            "authority_gt_0p05_roots": len(auth_roots),
            "authority_le_0p05_roots": len(low_roots),
        },
        "ridge_baseline_ise_sensitivity": {"paired_two_sided_p": nominal, "holm_p": adjusted},
        "descriptive_ise_by_authority": {
            "authority_6_roots": {m: fmean(r["panels"]["all"][m]["mean_ise"] for r in auth_roots) for m in methods},
            "low_authority_2_roots": {m: fmean(r["panels"]["all"][m]["mean_ise"] for r in low_roots) for m in methods},
        },
        "mean_ise": summary["mean_ise"],
        "mean_return": summary["mean_return"],
        "mean_early_calls": summary["mean_early_calls"],
        "contrasts": contrasts,
        "root_effects": by_root,
    }


def expensive_cem_planner() -> dict:
    path = ROOT / "results" / CEM_PLANNER_RUN / "summary.json"
    summary = read(path)
    roots = summary["roots"]
    assert summary["root_count"] == len(roots) == 8
    budgets = ("12", "15", "20", "24", "30", "40")
    frontier = {}
    for b in budgets:
        entry = summary["frontier"][b]
        per_minus_unpriv = [
            r["budgets"][b]["methods"]["periodic"]["mean_ise"]
            - r["budgets"][b]["methods"]["unprivileged_uniform"]["mean_ise"]
            for r in roots
        ]
        dist_minus_unpriv = [
            r["budgets"][b]["methods"]["distance_threshold"]["mean_ise"]
            - r["budgets"][b]["methods"]["unprivileged_uniform"]["mean_ise"]
            for r in roots
        ]
        p_ms = entry["planner_ms_per_episode"]
        t_ms = entry["trigger_score_ms_per_episode"]
        upper_ms = {k: p_ms[k] + t_ms[k] for k in p_ms}
        frontier[b] = {
            "calls_per_episode": int(b),
            "mean_ise": entry["mean_ise"],
            "planner_ms_per_episode": p_ms,
            "trigger_score_ms_per_episode": t_ms,
            "upper_loop_ms_per_episode": upper_ms,
            "total_control_ms_per_episode": entry["total_control_ms_per_episode"],
            "mean_score_checks_unprivileged": fmean(
                r["budgets"][b]["methods"]["unprivileged_uniform"]["mean_score_checks"] for r in roots
            ),
            "periodic_minus_unprivileged_ise": t7_interval(per_minus_unpriv),
            "distance_minus_unprivileged_ise": t7_interval(dist_minus_unpriv),
            "relative_gain_vs_periodic": entry["contrasts"]["unprivileged_vs_periodic"]["relative_ise_gain"],
            "relative_gain_vs_distance": entry["contrasts"]["unprivileged_vs_distance_threshold"]["relative_ise_gain"],
        }

    def cem_cross(cand_method: str, cand_b: str, ref_b: str, ref: str = "periodic") -> dict:
        ise_diffs, plan_ms_diffs, upper_ms_diffs = [], [], []
        measured_ctrl_ms_diffs = []
        by_root_plan, by_root_upper = {}, {}
        for r in roots:
            seed = str(r["optimizer_seed"])
            ref_m = r["budgets"][ref_b]["methods"][ref]
            cand_m = r["budgets"][cand_b]["methods"][cand_method]
            ise_diffs.append(ref_m["mean_ise"] - cand_m["mean_ise"])
            p_diff = ref_m["planner_ms_per_episode"] - cand_m["planner_ms_per_episode"]
            u_diff = (ref_m["planner_ms_per_episode"] + ref_m["trigger_score_ms_per_episode"]) - (
                cand_m["planner_ms_per_episode"] + cand_m["trigger_score_ms_per_episode"]
            )
            plan_ms_diffs.append(p_diff)
            upper_ms_diffs.append(u_diff)
            measured_ctrl_ms_diffs.append(
                ref_m["total_control_ms_per_episode"] - cand_m["total_control_ms_per_episode"]
            )
            by_root_plan[seed] = p_diff
            by_root_upper[seed] = u_diff
        ref_ise = summary["frontier"][ref_b]["mean_ise"][ref]
        ref_plan = summary["frontier"][ref_b]["planner_ms_per_episode"][ref]
        ise_ci = t7_interval(ise_diffs)
        plan_ci = t7_interval(plan_ms_diffs)
        upper_ci = t7_interval(upper_ms_diffs)
        return {
            "ise_gain": {**ise_ci, "relative_gain": ise_ci["mean"] / ref_ise, "favorable_roots": sum(x > 0 for x in ise_diffs)},
            "planner_ms_saved": {
                **plan_ci,
                "relative_saved": plan_ci["mean"] / ref_plan,
                "favorable_roots": sum(x > 0 for x in plan_ms_diffs),
                "roots": by_root_plan,
            },
            "upper_loop_ms_saved": {
                **upper_ci,
                "favorable_roots": sum(x > 0 for x in upper_ms_diffs),
                "roots": by_root_upper,
            },
            "measured_control_ms_saved": t7_interval(measured_ctrl_ms_diffs),
        }

    cross_savings = {
        "unprivileged_B20_vs_periodic_B24": cem_cross("unprivileged_uniform", "20", "24"),
        "unprivileged_B24_vs_periodic_B30": cem_cross("unprivileged_uniform", "24", "30"),
        "unprivileged_B20_vs_periodic_B30": cem_cross("unprivileged_uniform", "20", "30"),
        "distance_B24_vs_periodic_B30": cem_cross("distance_threshold", "24", "30"),
        "unprivileged_B20_vs_distance_B24": cem_cross("unprivileged_uniform", "20", "24", "distance_threshold"),
        "unprivileged_B24_vs_distance_B30": cem_cross("unprivileged_uniform", "24", "30", "distance_threshold"),
    }
    return {
        "source": str(path.relative_to(ROOT)),
        "statistical_unit": "optimizer_seed_root",
        "root_count": 8,
        "evaluation_paths_per_root": 32,
        "paired_paths_per_budget": 256,
        "frontier": frontier,
        "cross_budget_savings": cross_savings,
    }


def pointmaze_review_controls() -> dict:
    path = Path("results") / REVIEW_CONTROLS_RUN / "summary.json"
    summary = read(ROOT / path)
    roots = summary["roots"]
    assert len(roots) == 8 and summary["phase"] == "formal"
    assert summary["calls_per_episode"] == 24 and summary["paths_per_root"] == 64
    assert {row["optimizer_seed"] for row in roots} == set(range(1_981_400_000, 1_981_480_000, 10_000))
    assert all(row["fit_rows"] == [192, 384] for row in roots)
    contrasts = {}
    for comparator, result in summary["contrasts"].items():
        gains = [row["mean_ise"][comparator] - row["mean_ise"]["ridge"] for row in roots]
        ci = t7_interval(gains)
        radius = float(student_t.ppf(0.9875, df=7) * stdev(gains) / sqrt(8))
        adjusted = [ci["mean"] - radius, ci["mean"] + radius]
        halves = result["half_gains"]
        supported = adjusted[0] > 0 and min(halves) > 0
        assert supported == result["supported"] and abs(fmean(halves) - ci["mean"]) < 1e-12
        contrasts[comparator] = {
            **ci, "CI97p5": adjusted, "half_gains": halves,
            "relative_gain": ci["mean"] / fmean(row["mean_ise"][comparator] for row in roots),
            "favorable_roots": sum(gain > 0 for gain in gains), "supported": supported,
        }
    return {"source": path.as_posix(), "root_count": 8, "paired_paths": 512,
            "calls_per_episode": 24, "contrasts": contrasts, "mean_ise": summary["mean_ise"],
            "supervision_steps_per_rule": sum(row["supervision_steps_per_rule"] for row in roots),
            "supervision_planner_calls_per_rule": sum(row["supervision_planner_calls_per_rule"] for row in roots),
            "evaluation_steps": 512 * 3 * 1200,
            "different_yoked_schedules": sum(row["different_yoked_schedules"] for row in roots)}


def extract_waiting_value_inputs() -> None:
    local = Path("results/plan_validity_local_episode_diagnostic_20261110/summary.json")
    matched = Path("results/plan_validity_target_credit_20261110/summary.json")
    payload = {
        "sources": [str(local), str(matched)],
        "local_credit_parts": {
            str(row["optimizer_seed"]): row["cross_future_attribution"]["mean_credit_parts"]
            for row in read(ROOT / local)["cells"] if row["task"] == "reacher_tracking"
        },
        "matched_episode_panels": {
            task: row["episode_panels"]["all"] for task, row in read(ROOT / matched)["task_means"].items()
        },
    }
    (PAPER / "data/waiting_value.json").write_text(json.dumps(payload, indent=2) + "\n")


def waiting_value_diagnostics() -> dict:
    path = Path("paper/data/waiting_value.json")
    inputs = read(ROOT / path)
    by_root = {}
    for root, credit in inputs["local_credit_parts"].items():
        by_root[root] = {
            "adaptive_50_step_gain": credit["local_advantage"] + credit["opportunity_gap"],
            "full_episode_gain": credit["full_advantage"],
        }
    target_comparison = {}
    for task, panel in inputs["matched_episode_panels"].items():
        means = panel["mean_ise"]
        target_comparison[task] = {
            "wait_over_local_relative_gain": 1 - means["W"] / means["L"],
            "wait_over_frozen_relative_gain": 1 - means["W"] / means["history"],
            "wait_minus_local_by_root": panel["root_differences_from_L"]["W"],
        }
    return {"source": str(path), "scope": "two_root_per_task_development",
            "ellipse_reacher_interventions": by_root, "matched_target_comparison": target_comparison}


def transit(run: str, pairs: tuple[str, ...]) -> dict:
    path = ROOT / "results" / run / "report/result.json"
    report = read(path)
    assert report["coverage_qualified"]
    return {
        "source": str(path.relative_to(ROOT)),
        "protocol": report["protocol"],
        "comparisons": {name: report["comparisons"][name] for name in pairs},
        "prefixes": report["test_prefixes"],
        "native_evaluation_episodes": report["evaluation_native_episodes"],
    }


def build() -> dict:
    pointmaze = stage9()
    assert round(pointmaze["contrasts"]["fixed_minus_candidate_ise"]["mean"], 6) == 0.505541
    assert round(pointmaze["contrasts"]["candidate_minus_fixed_return"]["mean"], 6) == 40.662456
    unpriv = unprivileged_ablation()
    pareto = budget_pareto()
    strong_b = pointmaze_strong_baselines()
    reacher_poly = reacher_polygonal_confirmation()
    cem = expensive_cem_planner()
    run304 = "native_transit_closed_loop_confirmation_stage304_20261021_r2"
    path304 = ROOT / "results" / run304 / "report/result.json"
    r304 = read(path304)
    assert r304["coverage_qualified"] and r304["independent_scene_units"] == 320
    audit_path = path304.with_name("descriptive_audit.json")
    audit = read(audit_path)
    events = audit["arm_diagnostics"]["service_only"]["fleet_events"]
    return {
        "paper_scope": "plan_validity_driven_replanning;not_domain_general_Freq_HRL",
        "rebuild_training_or_simulation": 0,
        "pointmaze_stage9_confirmation": pointmaze,
        "pointmaze_unprivileged_ablation": unpriv,
        "pointmaze_budget_pareto": pareto,
        "pointmaze_strong_baselines": strong_b,
        "reacher_polygonal_confirmation": reacher_poly,
        "pointmaze_expensive_cem_planner": cem,
        "pointmaze_review_controls": pointmaze_review_controls(),
        "waiting_value_development_diagnostics": waiting_value_diagnostics(),
        "transit_stage298_single_intervention": transit(
            "native_transit_service_confirmation_stage298_20261019_r1", ("joint_budgeted_vs_raw",)
        ),
        "transit_stage301_single_intervention": transit(
            "native_transit_service_comparison_stage301_20261020_r1",
            ("service_only_vs_raw", "joint_budgeted_vs_raw", "joint_budgeted_vs_service_only"),
        ),
        "transit_stage304_two_decisions": {
            "source": str(path304.relative_to(ROOT)),
            "statistical_unit": "complete_native_scene_episode",
            "independent_scene_units": 320,
            "descriptive_source": str(audit_path.relative_to(ROOT)),
            "descriptive": {
                "fleet_peak_increases": sum(e["arm_peak"] > e["raw_peak"] for e in events),
                "fleet_peak_decreases": sum(e["arm_peak"] < e["raw_peak"] for e in events),
                **audit["joint_vs_service"],
            },
            "comparisons": r304["comparisons"],
            "wait_retention_vs_base": r304["wait_retention_vs_base"],
            **{
                k: r304[k]
                for k in (
                    "service_closed_loop_supported",
                    "joint_closed_loop_supported",
                    "critic_incremental_cost_supported",
                )
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract-stage9",
        action="store_true",
        help="refresh compact inputs from the original eight local result files",
    )
    parser.add_argument("--extract-waiting-value", action="store_true",
                        help="refresh anonymized development inputs from the original result files")
    opts = parser.parse_args()
    if opts.extract_stage9:
        extract_stage9_inputs()
    if opts.extract_waiting_value:
        extract_waiting_value_inputs()
    evidence = build()
    output = PAPER / "plan_validity_evidence.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    pm = evidence["pointmaze_stage9_confirmation"]
    print(
        json.dumps(
            {
                "output": str(output),
                "Stage9": pm["contrasts"],
                "unprivileged_mean_ise": evidence["pointmaze_unprivileged_ablation"]["mean_ise"],
                "strong_baselines_ise": evidence["pointmaze_strong_baselines"]["mean_ise"],
                "reacher_polygonal_ise": evidence["reacher_polygonal_confirmation"]["mean_ise"],
                "cem_cross_savings": {
                    k: {"ise": v["ise_gain"]["mean"], "planner_ms": v["planner_ms_saved"]["mean"]}
                    for k, v in evidence["pointmaze_expensive_cem_planner"]["cross_budget_savings"].items()
                },
                "rebuild_training_or_simulation": 0,
            },
            indent=2,
        )
    )
