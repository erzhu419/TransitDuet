"""Build compact manuscript evidence from completed local results; no rollouts."""

import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE9 = "pointmaze_budgeted_trigger_stage9_v2_confirmation_20260926_r1"
ROOTS = (209011, 209023, 209037, 209049, 209061, 209073, 209089, 209101)
MODES = ("fixed", "balanced_jitter", "current_compact_quadratic", "causal_validity_interactions")
PAIRS = {
    "fixed_minus_candidate_ise": ("fixed", "causal_validity_interactions", "tracking_squared_error_integral"),
    "jitter_minus_candidate_ise": ("balanced_jitter", "causal_validity_interactions", "tracking_squared_error_integral"),
    "current_minus_candidate_ise": ("current_compact_quadratic", "causal_validity_interactions", "tracking_squared_error_integral"),
    "candidate_minus_fixed_return": ("causal_validity_interactions", "fixed", "episode_return"),
}


def read(path):
    return json.loads(path.read_text())


def stage9():
    effects = {name: [] for name in (*PAIRS, "controller_learning_ise")}
    means = {mode: [] for mode in MODES}
    checks, early, branch_steps, paths, by_root = [], [], 0, [], {}
    for root in ROOTS:
        path = ROOT / "results" / STAGE9 / "cells/hrl_regime_history" / f"replicate_{root}/result.json"
        payload = read(path)
        cell = payload["cells"][0]
        assert payload["status"] == "complete" and cell["optimizer_seed"] == root
        seeds = cell["trigger_eval_seeds"]
        rows = cell["trigger_evaluation_rows"]
        indexed = {(r["seed"], r["mode"]): r for r in rows}
        assert len(seeds) == 16 and len(indexed) == len(rows) == 64
        assert all(r["upper_decision_count"] == 24 and r["protocol_valid"] for r in rows)
        for name, (after, before, field) in PAIRS.items():
            effects[name].append(fmean(indexed[s, after][field] - indexed[s, before][field] for s in seeds))
        trained = {r["seed"]: r for r in cell["canonical_evaluation_rows"]}
        untrained = {r["seed"]: r for r in cell["untrained_evaluation_rows"]}
        effects["controller_learning_ise"].append(fmean(
            untrained[s]["tracking_squared_error_integral"] - trained[s]["tracking_squared_error_integral"] for s in seeds))
        for mode in MODES:
            means[mode].append(fmean(indexed[s, mode]["tracking_squared_error_integral"] for s in seeds))
        candidate = [indexed[s, MODES[-1]] for s in seeds]
        checks.extend(r["trigger_score_checks"] for r in candidate)
        early.extend(r["trigger_early_calls"] for r in candidate)
        branch_steps += cell["branch_fit_primitive_steps_replayed"]
        by_root[str(root)] = {name: values[-1] for name, values in effects.items()}
        paths.append(str(path.relative_to(ROOT)))
    # The frozen confirmation has eight roots: t(.975, 7), not a path-level CI.
    t7 = 2.3646242510102993
    intervals = {}
    for name, values in effects.items():
        mean = fmean(values)
        radius = t7 * stdev(values) / sqrt(8)
        intervals[name] = {"mean": mean, "CI95": [mean-radius, mean+radius]}
    return {"sources": paths, "statistical_unit": "optimizer_seed_root", "root_count": 8,
        "evaluation_paths": 128, "paired_mode_rollouts": 512, "root_effects": by_root,
        "interval_method": "paired_root_Student_t_df7_two_sided95;unchanged_registered_endpoints",
        "contrasts": intervals, "mean_ise": {mode: fmean(v) for mode, v in means.items()},
        "upper_calls_per_episode_all_modes": 24, "candidate_score_checks_mean": fmean(checks),
        "candidate_score_checks_range": [min(checks), max(checks)],
        "candidate_predeadline_calls_mean": fmean(early),
        "branch_fit_primitive_steps_replayed_all_roots": branch_steps}


def transit(run, pairs):
    path = ROOT / "results" / run / "report/result.json"
    report = read(path)
    assert report["coverage_qualified"]
    data = {"source": str(path.relative_to(ROOT)), "protocol": report["protocol"],
        "comparisons": {name: report["comparisons"][name] for name in pairs},
        "prefixes": report["test_prefixes"], "native_evaluation_episodes": report["evaluation_native_episodes"]}
    return data


def build():
    pointmaze = stage9()
    # Six-decimal values in the completed official report detect sign/unit transcription errors.
    assert round(pointmaze["contrasts"]["fixed_minus_candidate_ise"]["mean"], 6) == .505541
    assert round(pointmaze["contrasts"]["candidate_minus_fixed_return"]["mean"], 6) == 40.662456
    run304 = "native_transit_closed_loop_confirmation_stage304_20261021_r2"
    path304 = ROOT / "results" / run304 / "report/result.json"
    r304 = read(path304)
    assert r304["coverage_qualified"] and r304["independent_scene_units"] == 320
    return {"paper_scope": "plan_validity_driven_replanning;not_domain_general_Freq_HRL",
        "new_training_or_simulation": 0, "pointmaze_stage9_confirmation": pointmaze,
        "transit_stage298_single_intervention": transit(
            "native_transit_service_confirmation_stage298_20261019_r1", ("joint_budgeted_vs_raw",)),
        "transit_stage301_single_intervention": transit(
            "native_transit_service_comparison_stage301_20261020_r1",
            ("service_only_vs_raw", "joint_budgeted_vs_raw", "joint_budgeted_vs_service_only")),
        "transit_stage304_two_decisions": {"source": str(path304.relative_to(ROOT)),
            "statistical_unit": "complete_native_scene_episode", "independent_scene_units": 320,
            "comparisons": r304["comparisons"], "wait_retention_vs_base": r304["wait_retention_vs_base"],
            **{k: r304[k] for k in ("service_closed_loop_supported", "joint_closed_loop_supported", "critic_incremental_cost_supported")}}}


if __name__ == "__main__":
    evidence = build()
    output = PAPER / "plan_validity_evidence.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    pm = evidence["pointmaze_stage9_confirmation"]
    print(json.dumps({"output": str(output), "Stage9": pm["contrasts"],
        "score_checks_mean": pm["candidate_score_checks_mean"],
        "branch_fit_steps": pm["branch_fit_primitive_steps_replayed_all_roots"],
        "new_training_or_simulation": 0}, indent=2))
