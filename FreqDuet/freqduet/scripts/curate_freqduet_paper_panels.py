#!/usr/bin/env python3
"""Create a concise paper-facing table/figure curation manifest.

The full paper package intentionally preserves many source tables and failed
candidate records. This script selects a smaller main-paper panel set and
copies the exact source artifacts into a stable curation directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_submission_gate import (
    read_submission_manifest,
    require_submission_ready,
)


DEFAULT_OUT = ROOT / "results_freqduet" / "paper_curation" / "current"
DEFAULT_MANIFEST = ROOT / "paper_manifest.yaml"


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def copy_artifact(src: Path | None, dst: Path, missing: list[str]) -> str:
    if src is None or not src.exists():
        missing.append(str(dst))
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def table_specs() -> list[dict[str, object]]:
    package = ROOT / "results_freqduet/paper_package/current/tables"
    return [
        {
            "panel_id": "Table 1",
            "role": "main external baseline result",
            "title": "FreqDuet V1 vs fixed-headway, rule-holding, and rule-MPC",
            "source_candidates": [
                package / "paper_external_classical_v1_ep200_60seed_paired_deltas.csv",
                ROOT
                / "results_freqduet/paper_external_classical_v1_ep200_wu10_4domain_60seed/compare_main_vs_external_classical/external_baseline_paired_deltas.csv",
            ],
            "output": "main_tables/table1_external_classical_paired_deltas.csv",
            "claim": "statistically tied with strong fixed-headway; strongly better than rule-holding and rule-MPC",
        },
        {
            "panel_id": "Table 2",
            "role": "mechanism ablation",
            "title": "Leakage-control and frequency-decomposition ablation",
            "source_candidates": [
                package / "paper_ablation_v1_ep200_60seed_paired_deltas.csv",
                ROOT
                / "results_freqduet/paper_ablation_v1_ep200_wu10_4domain_60seed/paper_summary/paper_matrix_paired_deltas.csv",
            ],
            "output": "main_tables/table2_internal_ablation_paired_deltas.csv",
            "claim": "noleakage is decisively worse; other internal controls are close",
        },
        {
            "panel_id": "Table 3",
            "role": "broad held-out perturbation robustness",
            "title": "Demand-noise, OD-shift, and rush-shift generalization",
            "source_candidates": [
                package / "paper_broad_generalization_v1_ep100_60seed_paired_deltas.csv",
                ROOT
                / "results_freqduet/paper_broad_generalization_v1_ep100_wu10_60seed/paper_summary/broad_paired_deltas.csv",
            ],
            "output": "main_tables/table3_broad_generalization_paired_deltas.csv",
            "claim": "robust perturbation performance; leakage-control necessity remains the strongest broad claim",
        },
        {
            "panel_id": "Table 4",
            "role": "route/day realism and next validation protocol",
            "title": "Route-family and service-day held-out readiness",
            "source_candidates": [
                package / "route_family_coverage.csv",
                ROOT
                / "results_freqduet/route_day_heldout_readiness/v1/route_family_coverage.csv",
            ],
            "output": "main_tables/table4_route_family_coverage.csv",
            "claim": "route/day split metadata are packaged; completed route/day policy matrix remains future work",
        },
        {
            "panel_id": "Extended Data Table 1",
            "role": "TransitDuet lineage baseline",
            "title": "Closest preserved TransitDuet-family no-frequency baseline",
            "source_candidates": [
                package / "transitduet_like_paired_deltas.csv",
                ROOT
                / "results_freqduet/transitduet_like_baseline_ep200/transitduet_like_paired_deltas.csv",
            ],
            "output": "supplementary_tables/edtable1_transitduet_like_paired_deltas.csv",
            "claim": "closest preserved no-frequency TransitDuet-family control is locked and bounded",
        },
        {
            "panel_id": "Extended Data Table 2",
            "role": "claim boundary",
            "title": "External-data and route/day claim boundaries",
            "source_candidates": [
                package / "route_day_claim_boundaries.csv",
                ROOT
                / "results_freqduet/route_day_heldout_readiness/v1/route_day_claim_boundaries.csv",
            ],
            "output": "supplementary_tables/edtable2_route_day_claim_boundaries.csv",
            "claim": "safe and unsafe route/day claims are explicitly separated",
        },
    ]


def figure_specs() -> list[dict[str, object]]:
    figures = ROOT / "results_freqduet/paper_package/current/figures"
    return [
        {
            "panel_id": "Figure 1a",
            "role": "frequency decomposer validation",
            "title": "Synthetic LF/HF component recovery",
            "source_candidates": [
                figures / "decomposer_validation_paper_v1_60seed/decomposer_synthetic_components.png",
                ROOT
                / "results_freqduet/decomposer_validation/paper_v1_60seed/decomposer_synthetic_components.png",
            ],
            "output": "main_figures/fig1a_decomposer_synthetic_components.png",
            "claim": "harmonic causal decomposer separates low- and high-frequency demand components",
        },
        {
            "panel_id": "Figure 1b",
            "role": "trace alignment",
            "title": "Decomposer trace alignment",
            "source_candidates": [
                figures / "decomposer_validation_paper_v1_60seed/decomposer_trace_alignment.png",
                ROOT
                / "results_freqduet/decomposer_validation/paper_v1_60seed/decomposer_trace_alignment.png",
            ],
            "output": "main_figures/fig1b_decomposer_trace_alignment.png",
            "claim": "simulator traces expose interpretable LF/HF separation",
        },
        {
            "panel_id": "Figure 2a",
            "role": "main/mechanism performance overview",
            "title": "Domain-method mechanism bars",
            "source_candidates": [
                figures / "mechanism_paper_ablation_v1_ep200_60seed/mechanism_domain_method_bars.png",
                ROOT
                / "results_freqduet/mechanism_figures/paper_ablation_v1_ep200_60seed/mechanism_domain_method_bars.png",
            ],
            "output": "main_figures/fig2a_mechanism_domain_method_bars.png",
            "claim": "main, frequency controls, and leakage failure are visible by domain",
        },
        {
            "panel_id": "Figure 2b",
            "role": "HF residual control mechanism",
            "title": "HF energy to holding response",
            "source_candidates": [
                figures / "mechanism_paper_v1_trace_alignment/mechanism_hf_energy_to_holding.png",
                ROOT
                / "results_freqduet/mechanism_figures/paper_v1_trace_alignment/mechanism_hf_energy_to_holding.png",
            ],
            "output": "main_figures/fig2b_hf_energy_to_holding.png",
            "claim": "HF residuals align with lower holding/wait responses",
        },
        {
            "panel_id": "Figure 3a",
            "role": "real demand-profile audit",
            "title": "External AFC/APC profile overlay",
            "source_candidates": [
                figures / "external_afc_apc_profile_audit_v1/external_afc_apc_profile_overlay.png",
                ROOT
                / "results_freqduet/real_afc_apc_profile_audit/v1/external_afc_apc_profile_overlay.png",
            ],
            "output": "main_figures/fig3a_external_afc_apc_profile_overlay.png",
            "claim": "paper package is grounded against public passenger-count demand shapes",
        },
        {
            "panel_id": "Figure 3b",
            "role": "same-agency load-profile audit",
            "title": "MBTA Route 111 APC load profile",
            "source_candidates": [
                figures / "mbta_same_network_calibration_audit_v1/mbta_route111_apc_load_profile.png",
                ROOT
                / "results_freqduet/mbta_same_network_calibration_audit/v1/mbta_route111_apc_load_profile.png",
            ],
            "output": "main_figures/fig3b_mbta_route111_apc_load_profile.png",
            "claim": "MBTA APC route/stop load targets are structurally usable for future route/day validation",
        },
        {
            "panel_id": "Extended Data Figure 1",
            "role": "action/state spectral diagnostic",
            "title": "Action and state spectrum",
            "source_candidates": [
                figures / "mechanism_paper_ablation_v1_ep200_60seed/mechanism_action_state_spectrum.png",
                ROOT
                / "results_freqduet/mechanism_figures/paper_ablation_v1_ep200_60seed/mechanism_action_state_spectrum.png",
            ],
            "output": "supplementary_figures/edfig1_action_state_spectrum.png",
            "claim": "state/action spectra remain available as mechanism source evidence",
        },
    ]


def write_claim_map(out_dir: Path) -> None:
    text = """# FreqDuet Paper Claim-Evidence Map

## One-sentence argument

In OD-driven bus holding control, FreqDuet shows that frequency-separated
hierarchical control with leakage prevention is mechanistically traceable and
competitive with a strong fixed-headway policy, supported by 60-seed simulation
matrices, external classical baselines, mechanism traces, and public AFC/APC/AVL
realism audits, while same-day field calibration and route/day policy
generalization remain explicit future work.

## Claim Discipline

- Main performance: write "statistically tied with strong fixed-headway and
  significantly better than rule-holding/rule-MPC", not "dominates fixed-headway".
- Mechanism: write "leakage control is necessary and HF residuals align with
  lower holding/wait responses", not "the trace proves field-causal deployment
  gains".
- Realism data: write "external public AFC/APC/AVL data support realism audits
  and route/day protocol readiness", not "same-day AFC/APC/AVL/OD field
  calibration is complete".
- Phase 4: write "bounded executable terminal-dispatch timetable control is
  implemented", not "learned first-stop/terminal launch value control is
  validated".

## Main Presentation

- Table 1: external classical baseline comparison.
- Table 2: frequency/leakage ablation.
- Table 3: broad demand perturbation generalization.
- Table 4: route/day held-out readiness and claim boundary.
- Figure 1: decomposer validation and trace alignment.
- Figure 2: mechanism and HF-to-holding alignment.
- Figure 3: external data realism and same-agency load-profile audit.
"""
    (out_dir / "claim_evidence_map.md").write_text(text, encoding="utf-8")


def build_manifest(specs: list[dict[str, object]], out_dir: Path, missing: list[str]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        source = first_existing([Path(p) for p in spec["source_candidates"]])
        dst = out_dir / str(spec["output"])
        copied_to = copy_artifact(source, dst, missing)
        rows.append(
            {
                "panel_id": spec["panel_id"],
                "role": spec["role"],
                "title": spec["title"],
                "source": "" if source is None else str(source),
                "curated_artifact": copied_to,
                "claim": spec["claim"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Explicitly curate a package whose manifest is on hold.",
    )
    args = parser.parse_args()

    manifest = read_submission_manifest(args.manifest)
    require_submission_ready(
        manifest, allow_historical=args.allow_historical)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    table_manifest = build_manifest(table_specs(), out_dir, missing)
    figure_manifest = build_manifest(figure_specs(), out_dir, missing)
    table_manifest.to_csv(out_dir / "paper_table_manifest.csv", index=False)
    figure_manifest.to_csv(out_dir / "paper_panel_manifest.csv", index=False)
    write_claim_map(out_dir)
    with (out_dir / "paper_curation_manifest.json").open("w") as f:
        json.dump(
            {
                "out_dir": str(out_dir),
                "tables": len(table_manifest),
                "figures": len(figure_manifest),
                "missing": missing,
            },
            f,
            indent=2,
        )

    print(f"wrote {out_dir}")
    print(f"tables={len(table_manifest)} figures={len(figure_manifest)} missing={len(missing)}")
    if missing:
        print("missing:")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
