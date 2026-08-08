#!/usr/bin/env python3
"""Assemble the current FreqDuet paper table/figure/appendix package.

The experiment tree intentionally keeps raw logs and candidate results in their
native directories. This script creates a stable paper-facing bundle with the
tables, figures, manifest snapshot, and negative-results appendix needed for
writing and review.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_submission_gate import require_submission_ready


DEFAULT_MANIFEST = ROOT / "paper_manifest.yaml"
DEFAULT_OUT = ROOT / "results_freqduet" / "paper_package" / "current"
PACKAGE_CONFIG_EXPERIMENTS = {
    "paper_ablation_v1_ep200_wu10_4domain_60seed",
    "paper_external_classical_v1_ep200_wu10_4domain_60seed",
    "paper_broad_generalization_v1_ep100_wu10_60seed",
}


def copy_file(src: Path, dst: Path, copied: list[dict], missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"src": str(src), "dst": str(dst)})


def copy_tree_files(
    src_dir: Path,
    dst_dir: Path,
    copied: list[dict],
    missing: list[str],
    optional: bool = False,
) -> None:
    if not src_dir.exists():
        if not optional:
            missing.append(str(src_dir))
        return
    files = sorted(path for path in src_dir.rglob("*") if path.is_file())
    if not files:
        if not optional:
            missing.append(str(src_dir))
        return
    for src in files:
        copy_file(src, dst_dir / src.relative_to(src_dir), copied, missing)


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path,
              copied: list[dict], missing: list[str],
              optional: bool = False) -> None:
    if not src_dir.exists():
        if not optional:
            missing.append(str(src_dir))
        return
    files = sorted(src_dir.glob(pattern))
    if not files:
        if not optional:
            missing.append(f"{src_dir}/{pattern}")
        return
    for src in files:
        copy_file(src, dst_dir / src.name, copied, missing)


def read_manifest(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def experiment(manifest: dict, key: str) -> dict:
    return manifest.get("experiments", {}).get(key, {})


def copy_core_tables(manifest: dict, out_dir: Path,
                     copied: list[dict], missing: list[str]) -> None:
    table_specs = []

    paper_ablation = experiment(manifest, "paper_ablation_v1_ep200_wu10_4domain_60seed")
    table_specs.extend([
        ("paper_ablation_v1_ep200_60seed_per_seed.csv",
         paper_ablation.get("per_seed_csv")),
        ("paper_ablation_v1_ep200_60seed_summary.csv",
         paper_ablation.get("summary_csv")),
        ("paper_ablation_v1_ep200_60seed_method_summary.csv",
         Path(paper_ablation.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if paper_ablation.get("paper_summary_dir") else None),
        ("paper_ablation_v1_ep200_60seed_paired_deltas.csv",
         Path(paper_ablation.get("paper_summary_dir", "")) / "paper_matrix_paired_deltas.csv"
         if paper_ablation.get("paper_summary_dir") else None),
    ])

    paper_external = experiment(
        manifest, "paper_external_classical_v1_ep200_wu10_4domain_60seed")
    table_specs.extend([
        ("paper_external_classical_v1_ep200_60seed_per_seed.csv",
         paper_external.get("per_seed_csv")),
        ("paper_external_classical_v1_ep200_60seed_summary.csv",
         paper_external.get("summary_csv")),
        ("paper_external_classical_v1_ep200_60seed_paired_deltas.csv",
         Path(paper_external.get("comparison_vs_main_dir", ""))
         / "external_baseline_paired_deltas.csv"
         if paper_external.get("comparison_vs_main_dir") else None),
    ])

    paper_broad = experiment(
        manifest, "paper_broad_generalization_v1_ep100_wu10_60seed")
    table_specs.extend([
        ("paper_broad_generalization_v1_ep100_60seed_per_seed.csv",
         paper_broad.get("per_seed_csv")),
        ("paper_broad_generalization_v1_ep100_60seed_summary.csv",
         paper_broad.get("summary_csv")),
        ("paper_broad_generalization_v1_ep100_60seed_completion_audit.csv",
         paper_broad.get("completion_audit_csv")),
        ("paper_broad_generalization_v1_ep100_60seed_method_summary.csv",
         Path(paper_broad.get("paper_summary_dir", "")) / "broad_method_summary.csv"
         if paper_broad.get("paper_summary_dir") else None),
        ("paper_broad_generalization_v1_ep100_60seed_paired_deltas.csv",
         paper_broad.get("paired_deltas_csv")),
    ])

    paper_trace = experiment(manifest, "paper_trace_diag_v1_4domain_3seed")
    table_specs.extend([
        ("paper_trace_diag_v1_4domain_3seed_per_seed.csv",
         paper_trace.get("per_seed_csv")),
        ("paper_trace_diag_v1_4domain_3seed_summary.csv",
         paper_trace.get("summary_csv")),
        ("paper_trace_diag_v1_4domain_3seed_audit.csv",
         paper_trace.get("trace_audit_csv")),
    ])

    promoted = experiment(manifest, "promoted_longtrain_ep200_wu10")
    table_specs.extend([
        ("longtrain_promoted_per_seed.csv", promoted.get("per_seed_csv")),
        ("longtrain_promoted_summary.csv", promoted.get("summary_csv")),
        ("longtrain_promoted_method_summary.csv",
         Path(promoted.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if promoted.get("paper_summary_dir") else None),
        ("longtrain_promoted_paired_deltas.csv",
         Path(promoted.get("paper_summary_dir", "")) / "paper_matrix_paired_deltas.csv"
         if promoted.get("paper_summary_dir") else None),
    ])

    detseed_v1 = experiment(
        manifest, "detseed_cfaction_domainbest_v1_ep200_wu10_4domain_20seed")
    table_specs.extend([
        ("detseed_cfaction_domainbest_v1_ep200_per_seed.csv",
         detseed_v1.get("per_seed_csv")),
        ("detseed_cfaction_domainbest_v1_ep200_summary.csv",
         detseed_v1.get("summary_csv")),
        ("detseed_cfaction_domainbest_v1_ep200_vs_current_main.csv",
         detseed_v1.get("comparison_vs_current_main")),
        ("detseed_cfaction_domainbest_v1_ep200_vs_fixed_headway.csv",
         detseed_v1.get("comparison_vs_fixed_headway")),
        ("detseed_current_main_ep200_vs_fixed_headway.csv",
         detseed_v1.get("comparison_current_main_vs_fixed_headway")),
    ])

    detseed_fixed = experiment(
        manifest, "detseed_external_fixed_headway_ep200_wu10_4domain_20seed")
    table_specs.extend([
        ("detseed_external_fixed_headway_ep200_per_seed.csv",
         detseed_fixed.get("per_seed_csv")),
        ("detseed_external_fixed_headway_ep200_summary.csv",
         detseed_fixed.get("summary_csv")),
    ])

    current_tb = experiment(manifest, "final_matrix_current_terminalbias_ep100_wu10_4domain_20seed")
    table_specs.extend([
        ("final_terminalbias_ep100_per_seed.csv", current_tb.get("per_seed_csv")),
        ("final_terminalbias_ep100_summary.csv", current_tb.get("summary_csv")),
        ("final_terminalbias_ep100_method_summary.csv",
         Path(current_tb.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if current_tb.get("paper_summary_dir") else None),
        ("final_terminalbias_ep100_paired_deltas.csv",
         Path(current_tb.get("paper_summary_dir", "")) / "paper_matrix_paired_deltas.csv"
         if current_tb.get("paper_summary_dir") else None),
        ("final_terminalbias_ep100_vs_external_classical.csv",
         current_tb.get("comparison_vs_external_classical")),
    ])

    current_tb200 = experiment(manifest, "final_matrix_current_terminalbias_ep200_wu10_4domain_20seed")
    table_specs.extend([
        ("final_terminalbias_ep200_per_seed.csv", current_tb200.get("per_seed_csv")),
        ("final_terminalbias_ep200_summary.csv", current_tb200.get("summary_csv")),
        ("final_terminalbias_ep200_method_summary.csv",
         Path(current_tb200.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if current_tb200.get("paper_summary_dir") else None),
        ("final_terminalbias_ep200_paired_deltas.csv",
         Path(current_tb200.get("paper_summary_dir", "")) / "paper_matrix_paired_deltas.csv"
         if current_tb200.get("paper_summary_dir") else None),
        ("final_terminalbias_ep200_vs_external_classical.csv",
         current_tb200.get("comparison_vs_external_classical_ep200")),
    ])

    freeze100 = experiment(manifest, "freeze100_ep200_wu10_4domain_20seed")
    table_specs.extend([
        ("freeze100_ep200_per_seed.csv", freeze100.get("per_seed_csv")),
        ("freeze100_ep200_summary.csv", freeze100.get("summary_csv")),
        ("freeze100_ep200_method_summary.csv",
         Path(freeze100.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if freeze100.get("paper_summary_dir") else None),
        ("freeze100_ep200_vs_current.csv", freeze100.get("comparison_vs_current_ep200")),
        ("freeze100_ep200_vs_external_classical.csv", freeze100.get("comparison_vs_external_ep200")),
    ])

    current_freeze100 = experiment(
        manifest, "final_matrix_current_freeze100_ep200_wu10_4domain_20seed")
    table_specs.extend([
        ("final_current_freeze100_ep200_per_seed.csv",
         current_freeze100.get("per_seed_csv")),
        ("final_current_freeze100_ep200_summary.csv",
         current_freeze100.get("summary_csv")),
        ("final_current_freeze100_ep200_method_summary.csv",
         Path(current_freeze100.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if current_freeze100.get("paper_summary_dir") else None),
        ("final_current_freeze100_ep200_paired_deltas.csv",
         Path(current_freeze100.get("paper_summary_dir", "")) / "paper_matrix_paired_deltas.csv"
         if current_freeze100.get("paper_summary_dir") else None),
        ("final_current_freeze100_ep200_vs_internal.csv",
         current_freeze100.get("comparison_vs_internal")),
        ("final_current_freeze100_ep200_vs_external_classical.csv",
         current_freeze100.get("comparison_vs_external_ep200")),
        ("final_current_freeze100_ep200_vs_previous_terminalbias.csv",
         current_freeze100.get("comparison_vs_previous_terminalbias_ep200")),
    ])

    external_ep200 = experiment(manifest, "external_baselines_ep200_wu10_4domain_20seed")
    table_specs.extend([
        ("external_baselines_ep200_per_seed.csv", external_ep200.get("per_seed_csv")),
        ("external_baselines_ep200_summary.csv", external_ep200.get("summary_csv")),
    ])

    gen = experiment(manifest, "heldout_generalization_ep100_wu10")
    table_specs.extend([
        ("generalization_per_seed.csv", gen.get("per_seed_csv")),
        ("generalization_summary.csv", gen.get("summary_csv")),
    ])

    broad = experiment(manifest, "broad_generalization_ep100_wu10")
    table_specs.extend([
        ("broad_generalization_per_seed.csv", broad.get("per_seed_csv")),
        ("broad_generalization_summary.csv", broad.get("summary_csv")),
        ("broad_generalization_completion_audit.csv", broad.get("completion_audit_csv")),
        ("broad_generalization_method_summary.csv",
         Path(broad.get("paper_summary_dir", "")) / "broad_method_summary.csv"
         if broad.get("paper_summary_dir") else None),
        ("broad_generalization_paired_deltas.csv", broad.get("paired_deltas_csv")),
    ])

    external = experiment(manifest, "external_baselines_promoted_ep100")
    table_specs.extend([
        ("external_baselines_per_seed.csv", external.get("per_seed_csv")),
        ("external_baselines_summary.csv", external.get("summary_csv")),
    ])

    transitduet_like = experiment(manifest, "transitduet_like_baseline_ep200")
    table_specs.extend([
        ("transitduet_like_per_seed.csv", transitduet_like.get("per_seed_csv")),
        ("transitduet_like_summary.csv", transitduet_like.get("summary_csv")),
        ("transitduet_like_paired_deltas.csv", transitduet_like.get("paired_deltas_csv")),
    ])

    sumorl = experiment(manifest, "sumorl_style_holdrl_baseline_ep100_wu10")
    table_specs.extend([
        ("sumorl_holdrl_per_seed.csv", sumorl.get("per_seed_csv")),
        ("sumorl_holdrl_summary.csv", sumorl.get("summary_csv")),
        ("sumorl_holdrl_method_summary.csv",
         Path(sumorl.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if sumorl.get("paper_summary_dir") else None),
        ("sumorl_holdrl_vs_current_main_ep100.csv",
         Path(sumorl.get("comparison_dir_vs_current_main_ep100_reference", "")) / "candidate_paired_deltas.csv"
         if sumorl.get("comparison_dir_vs_current_main_ep100_reference") else None),
        ("sumorl_holdrl_vs_external_classical.csv",
         Path(sumorl.get("comparison_dir_vs_external_classical", "")) / "external_baseline_paired_deltas.csv"
         if sumorl.get("comparison_dir_vs_external_classical") else None),
    ])

    sumorl_rawhist = experiment(manifest, "sumorl_rawhist_holdrl_baseline_ep100_wu10")
    table_specs.extend([
        ("sumorl_rawhist_holdrl_per_seed.csv", sumorl_rawhist.get("per_seed_csv")),
        ("sumorl_rawhist_holdrl_summary.csv", sumorl_rawhist.get("summary_csv")),
        ("sumorl_rawhist_holdrl_method_summary.csv",
         Path(sumorl_rawhist.get("paper_summary_dir", "")) / "paper_matrix_method_summary.csv"
         if sumorl_rawhist.get("paper_summary_dir") else None),
        ("sumorl_rawhist_holdrl_vs_current_main_ep100.csv",
         Path(sumorl_rawhist.get("comparison_dir_vs_current_main_ep100_reference", "")) / "candidate_paired_deltas.csv"
         if sumorl_rawhist.get("comparison_dir_vs_current_main_ep100_reference") else None),
        ("sumorl_rawhist_holdrl_vs_external_classical.csv",
         Path(sumorl_rawhist.get("comparison_dir_vs_external_classical", "")) / "external_baseline_paired_deltas.csv"
         if sumorl_rawhist.get("comparison_dir_vs_external_classical") else None),
        ("sumorl_rawhist_holdrl_vs_plain_sumorl.csv",
         Path(sumorl_rawhist.get("comparison_dir_vs_plain_sumorl", "")) / "candidate_paired_deltas.csv"
         if sumorl_rawhist.get("comparison_dir_vs_plain_sumorl") else None),
    ])

    real_profiles = experiment(manifest, "external_afc_apc_profile_audit_v1")
    table_specs.extend([
        ("external_afc_apc_source_coverage.csv",
         real_profiles.get("source_coverage_csv")),
        ("external_afc_apc_aggregate_profile.csv",
         real_profiles.get("aggregate_profile_csv")),
        ("external_afc_apc_profile_alignment.csv",
         real_profiles.get("profile_alignment_csv")),
    ])

    real_truth = experiment(manifest, "external_od_onboard_truth_audit_v1")
    table_specs.extend([
        ("external_truth_source_coverage.csv",
         real_truth.get("source_coverage_csv")),
        ("external_truth_claim_boundaries.csv",
         real_truth.get("claim_boundaries_csv")),
        ("mbta_onboard_route_targets.csv",
         real_truth.get("mbta_route_targets_csv")),
        ("mbta_hourly_board_alight_load.csv",
         real_truth.get("mbta_hourly_profile_csv")),
        ("mta_od_sample_top_pairs.csv",
         real_truth.get("mta_od_top_pairs_csv")),
        ("mta_od_sample_hourly_profile.csv",
         real_truth.get("mta_od_hourly_profile_csv")),
    ])

    same_network = experiment(manifest, "mbta_same_network_calibration_audit_v1")
    table_specs.extend([
        ("mbta_same_network_source_coverage.csv",
         same_network.get("source_coverage_csv")),
        ("mbta_same_network_claim_boundaries.csv",
         same_network.get("claim_boundaries_csv")),
        ("mbta_same_network_overlap_summary.csv",
         same_network.get("overlap_summary_csv")),
        ("mbta_route111_apc_gtfs_profile.csv",
         same_network.get("focus_profile_csv")),
        ("mbta_apc_top_routes.csv",
         same_network.get("top_routes_csv")),
        ("mbta_apc_route_stop_gtfs_overlap.csv",
         same_network.get("route_stop_overlap_csv")),
    ])

    mta_bus_time = experiment(manifest, "mta_bus_time_offline_cache_v1")
    table_specs.extend([
        ("mta_bus_time_source_coverage.csv",
         mta_bus_time.get("source_coverage_csv")),
        ("mta_bus_time_claim_boundaries.csv",
         mta_bus_time.get("claim_boundaries_csv")),
        ("mta_bus_time_agencies.csv",
         mta_bus_time.get("agencies_csv")),
        ("mta_bus_time_routes.csv",
         mta_bus_time.get("routes_csv")),
        ("mta_bus_time_stops.csv",
         mta_bus_time.get("stops_csv")),
        ("mta_bus_time_route_stop_sequences.csv",
         mta_bus_time.get("route_stop_sequences_csv")),
        ("mta_bus_time_vehicle_snapshot_meta.csv",
         mta_bus_time.get("vehicle_snapshot_meta_csv")),
        ("mta_bus_time_vehicle_snapshots.csv",
         mta_bus_time.get("vehicle_snapshots_csv")),
    ])

    route_day = experiment(manifest, "route_day_heldout_readiness_v1")
    table_specs.extend([
        ("route_family_coverage.csv",
         route_day.get("route_family_coverage_csv")),
        ("service_day_split_protocol.csv",
         route_day.get("service_day_split_protocol_csv")),
        ("route_day_claim_boundaries.csv",
         route_day.get("claim_boundaries_csv")),
    ])

    for name, src in table_specs:
        required = name.startswith("paper_")
        if not src:
            if required:
                missing.append(f"manifest table source missing for {name}")
            continue
        src_path = rel(src)
        if not src_path.exists() and not required:
            continue
        copy_file(src_path, out_dir / "tables" / name, copied, missing)


def copy_figures(manifest: dict, out_dir: Path,
                 copied: list[dict], missing: list[str]) -> None:
    figures = manifest.get("figures", {})
    for key, item in figures.items():
        if "out_dir" not in item:
            continue
        required = bool(item.get("required", False))
        src_dir = rel(item["out_dir"])
        dst_dir = out_dir / "figures" / key
        copy_glob(src_dir, "*.png", dst_dir, copied, missing, optional=not required)
        copy_glob(src_dir, "*.pdf", dst_dir, copied, missing, optional=not required)
        copy_glob(src_dir, "*.csv", dst_dir / "source_data", copied, missing,
                  optional=not required)
        copy_glob(
            src_dir,
            "*.json",
            dst_dir / "source_data",
            copied,
            missing,
            optional=True,
        )


def copy_manuscript_notes(manifest: dict, out_dir: Path,
                          copied: list[dict], missing: list[str]) -> None:
    notes = manifest.get("manuscript_notes", {})
    for key, item in notes.items():
        src = item.get("path")
        if not src:
            missing.append(f"manifest manuscript note source missing for {key}")
            continue
        copy_file(rel(src), out_dir / "manuscript_notes" / Path(src).name,
                  copied, missing)


def copy_config_snapshots(manifest: dict, out_dir: Path,
                          copied: list[dict], missing: list[str]) -> None:
    for key, item in manifest.get("experiments", {}).items():
        if key not in PACKAGE_CONFIG_EXPERIMENTS and not item.get("package_required"):
            continue
        config_dir = item.get("config_dir")
        if not config_dir:
            config_set_name = item.get("config_set")
            if not config_set_name:
                continue
            config_set = manifest.get("config_sets", {}).get(config_set_name)
            if not config_set:
                missing.append(f"manifest config_set missing for {key}: {config_set_name}")
                continue
            dst_dir = out_dir / "configs" / str(config_set_name)
            for config_name in sorted(collect_config_names(config_set)):
                src = ROOT / "configs_freqduet" / f"{config_name}.yaml"
                copy_file(src, dst_dir / src.name, copied, missing)
            continue
        dst_dir = out_dir / "configs" / Path(config_dir).name
        copy_glob(rel(config_dir), "*.yaml", dst_dir, copied, missing)


def collect_config_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, str):
        name = value[:-5] if value.endswith(".yaml") else value
        if Path(name).name.startswith("F_freqduet_"):
            names.add(name)
    elif isinstance(value, dict):
        if "configs" in value:
            return collect_config_names(value["configs"])
        for child in value.values():
            names.update(collect_config_names(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            names.update(collect_config_names(child))
    return names


def copy_paper_scripts(manifest: dict, out_dir: Path,
                       copied: list[dict], missing: list[str]) -> None:
    for item in manifest.get("paper_scripts", []):
        copy_file(rel(item), out_dir / "scripts" / Path(item).name, copied, missing)


def copy_paper_curation(manifest: dict, out_dir: Path,
                        copied: list[dict], missing: list[str]) -> None:
    for key, item in manifest.get("paper_curation", {}).items():
        src = item.get("out_dir")
        if not src:
            if item.get("required", False):
                missing.append(f"manifest paper curation source missing for {key}")
            continue
        copy_tree_files(
            rel(src),
            out_dir / "curation" / key,
            copied,
            missing,
            optional=not bool(item.get("required", False)),
        )


def copy_data_sources(manifest: dict, out_dir: Path,
                      copied: list[dict], missing: list[str]) -> None:
    for key, item in manifest.get("data_sources", {}).items():
        for src in item.get("files", []):
            src_path = rel(src)
            src_rel = Path(src)
            if src_rel.parts and src_rel.parts[0] == "data":
                dst_rel = Path(*src_rel.parts[1:])
            else:
                dst_rel = Path(key) / src_rel.name
            copy_file(src_path, out_dir / "data_sources" / dst_rel, copied, missing)


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):+.4f}"


def summarize_candidate(comparison_dir: Path) -> list[str]:
    path = comparison_dir / "candidate_paired_deltas.csv"
    if not path.exists():
        path = comparison_dir / "broad_candidate_paired_deltas.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "metric" not in df.columns:
        return []
    rows = df[df["metric"].astype(str).eq("composite")].copy()
    lines = []
    for _, row in rows.iterrows():
        domain = str(row.get("domain", row.get("scenario", "unknown")))
        delta = _fmt(row.get("delta_candidate_minus_baseline"))
        lo = _fmt(row.get("delta_ci95_lo"))
        hi = _fmt(row.get("delta_ci95_hi"))
        win = row.get("candidate_win_rate")
        win_text = "NA" if pd.isna(win) else f"{float(win):.3f}"
        lines.append(
            f"- `{domain}`: delta {delta}, 95% CI [{lo}, {hi}], win {win_text}"
        )
    return lines


def build_negative_appendix(manifest: dict, out_dir: Path,
                            copied: list[dict], missing: list[str]) -> None:
    experiments = manifest.get("experiments", {})
    negative_keys = []
    for key, item in experiments.items():
        status = str(item.get("status", ""))
        note = str(item.get("note", ""))
        if ("negative" in status or "not_promoted" in status
                or "do not promote" in note):
            negative_keys.append(key)

    lines = [
        "# FreqDuet Negative Results Appendix",
        "",
        "This appendix is generated from `paper_manifest.yaml`. Deltas are",
        "`candidate - promoted main`; lower composite is better.",
        "",
    ]
    for key in negative_keys:
        item = experiments[key]
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"- status: `{item.get('status', 'unknown')}`")
        if item.get("config_set"):
            lines.append(f"- config_set: `{item['config_set']}`")
        if item.get("comparison_dir"):
            lines.append(f"- comparison_dir: `{item['comparison_dir']}`")
        if item.get("comparison_dirs"):
            lines.append(f"- comparison_dirs: `{item['comparison_dirs']}`")
        if item.get("note"):
            lines.append(f"- conclusion: {item['note']}")
        comparison_dirs = []
        if item.get("comparison_dir"):
            comparison_dirs.append((key, item["comparison_dir"]))
        if isinstance(item.get("comparison_dirs"), dict):
            comparison_dirs.extend(item["comparison_dirs"].items())
        elif isinstance(item.get("comparison_dirs"), list):
            comparison_dirs.extend((Path(x).name, x) for x in item["comparison_dirs"])
        for label, comparison_dir in comparison_dirs:
            summary = summarize_candidate(rel(comparison_dir))
            if summary:
                lines.append("")
                if len(comparison_dirs) > 1:
                    lines.append(f"### {label}")
                    lines.append("")
                lines.extend(summary)
                copy_glob(
                    rel(comparison_dir),
                    "*.csv",
                    out_dir / "appendix" / "negative_result_tables" / key / str(label),
                    copied,
                    missing,
                )
        lines.append("")

    appendix = out_dir / "appendix" / "negative_results_appendix.md"
    appendix.parent.mkdir(parents=True, exist_ok=True)
    appendix.write_text("\n".join(lines), encoding="utf-8")
    copied.append({"src": "generated", "dst": str(appendix)})


def write_readme(out_dir: Path, manifest: dict, copied: list[dict],
                 missing: list[str]) -> None:
    text = [
        "# FreqDuet Paper Package",
        "",
        f"- manifest version: `{manifest.get('version', 'unknown')}`",
        "- canonical 60-seed paper tables use the `paper_*_60seed_*.csv` names",
        "- `tables/`: final seed-level and summary CSVs",
        "- `figures/`: decomposer and mechanism figures with source data",
        "- `data_sources/`: small public AFC/APC profile caches and README files",
        "- `configs/`: generated paper config snapshots",
        "- `scripts/`: paper-facing run, sync, summarize, and plotting scripts",
        "- `manuscript_notes/`: method framing and realism/data evidence notes",
        "- `curation/`: selected main-table/main-figure manifests and copied panels",
        "- `appendix/negative_results_appendix.md`: failed candidate summary",
        "- `package_manifest.json`: copied and missing artifacts",
        "",
        f"Copied artifacts: {len(copied)}",
        f"Missing artifacts: {len(missing)}",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Append/update files in an existing package instead of rebuilding it.",
    )
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Explicitly reproduce a package whose manifest is on hold.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    manifest = read_manifest(manifest_path)
    require_submission_ready(
        manifest, allow_historical=args.allow_historical)
    if out_dir.exists() and not args.no_clean:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    missing: list[str] = []

    copy_file(manifest_path, out_dir / "paper_manifest.yaml", copied, missing)
    copy_core_tables(manifest, out_dir, copied, missing)
    copy_figures(manifest, out_dir, copied, missing)
    copy_manuscript_notes(manifest, out_dir, copied, missing)
    copy_config_snapshots(manifest, out_dir, copied, missing)
    copy_data_sources(manifest, out_dir, copied, missing)
    copy_paper_scripts(manifest, out_dir, copied, missing)
    copy_paper_curation(manifest, out_dir, copied, missing)
    build_negative_appendix(manifest, out_dir, copied, missing)
    write_readme(out_dir, manifest, copied, missing)

    payload = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "copied": copied,
        "missing": missing,
    }
    with (out_dir / "package_manifest.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote {out_dir}")
    print(f"copied={len(copied)} missing={len(missing)}")
    if missing:
        print("missing:")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
