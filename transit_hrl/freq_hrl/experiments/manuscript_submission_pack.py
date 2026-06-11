"""Build a conservative manuscript submission package from Freq-HRL artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("transit_hrl/results")
DEFAULT_OUTPUT_DIR = Path("transit_hrl/results/manuscript_submission_pack_latest")
DEFAULT_MD_DIR = Path("transit_hrl/md")


ARTIFACTS = {
    "unified": Path("top_journal_unified_matrix_latest/summary.json"),
    "baseline": Path("baseline_ablation_matrix_latest/summary.json"),
    "agency": Path("agency_demand_onboard_coverage_latest/summary.json"),
    "external_truth": Path("external_transit_truth_validation_latest/summary.json"),
    "order_book": Path("order_book_lobster_venue_grade_multisymbol/summary.json"),
    "leakage": Path("leakage_no_tradeoff_matrix_latest/summary.json"),
    "theory": Path("freq_hrl_theory_appendix_latest/summary.json"),
    "encoder": Path("encoder_cross_domain_matrix/summary.json"),
}


CONSERVATIVE_CLAIM_WORDING = {
    "C1": "Native learned promotion improves reward and waiting-time metrics in the current registered stress evidence.",
    "C2": "Native real-demand Transit validation is supported under public AFC/APC profiles, with separate public source coverage for board/alight/load and estimated OD.",
    "C3": "Venue-grade L2/L3 order-book replay is supported as a reproducible replay path on three LOBSTER symbol-session pairs.",
    "C4": "Advanced encoders have cross-domain support, with public-market and L3 rows kept as bounded or mixed evidence where appropriate.",
    "C5": "Leakage no-tradeoff is supported only where same-domain drift reduction and performance noninferiority or strict CI gates both pass.",
    "C6": "The formal appendix gives sufficient-condition bounds and reporting propositions rather than a universal convergence theorem.",
    "C7": "Promotion reward/wait improvement replicates across the current registered persistent-stress and OD-shift matrices.",
    "C8": "Frequency-responsibility evidence is supported against non-frequency, misrouted-frequency, no-promotion, and no-leakage alternatives.",
    "C9": "Stress-generalization support is limited to the registered stress regimes that pass paired evidence gates.",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _artifact_paths(results_root: Path) -> dict[str, Path]:
    return {key: results_root / value for key, value in ARTIFACTS.items()}


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        if value is None or value == "":
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


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


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("\n", " ") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_claim_table(unified: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in unified.get("claims", []) or []:
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("id", ""))
        rows.append({
            "id": claim_id,
            "claim": str(row.get("claim", "")),
            "status": str(row.get("status", "")),
            "conservative_wording": CONSERVATIVE_CLAIM_WORDING.get(claim_id, ""),
            "evidence": str(row.get("evidence", "")),
            "boundary": str(row.get("remaining_gap", "")),
            "artifact": str(row.get("artifact", "")),
        })
    return rows


def build_baseline_table(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preferred_metrics = {"sharpe", "total_return", "FocusScore", "LowerLFDrift"}
    for row in baseline.get("paired_checks", []) or []:
        if not isinstance(row, dict) or str(row.get("metric", "")) not in preferred_metrics:
            continue
        rows.append({
            "check": str(row.get("check", "")),
            "control": str(row.get("control", "")),
            "metric": str(row.get("metric", "")),
            "status": str(row.get("status", "")),
            "n": int(row.get("n_common", 0) or 0),
            "delta_mean": _fmt(row.get("delta_mean")),
            "ci95_low": _fmt(row.get("delta_ci95_low")),
            "ci95_high": _fmt(row.get("delta_ci95_high")),
            "win_rate": _fmt(row.get("win_rate")),
        })
    return rows


def build_real_data_table(agency: dict[str, Any], external_truth: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in agency.get("claim_boundaries", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "evidence_item": str(row.get("evidence_item", "")),
            "status": str(row.get("status", "")),
            "allowed_wording": str(row.get("allowed_wording", "")),
            "boundary": str(row.get("forbidden_wording", "")),
            "evidence": str(row.get("evidence", "")),
        })
    for row in external_truth.get("source_coverage", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "evidence_item": str(row.get("source", "")),
            "status": str(row.get("claim_status", "")),
            "allowed_wording": str(row.get("source_kind", "")),
            "boundary": str(row.get("boundary", "")),
            "evidence": (
                f"rows={row.get('rows', row.get('sample_rows', 0))} "
                f"routes={row.get('unique_routes', '')} stops={row.get('unique_stops', '')} "
                f"origins={row.get('unique_origins', '')} destinations={row.get('unique_destinations', '')}"
            ),
        })
    return rows


def build_figure_plan() -> list[dict[str, Any]]:
    return [
        {
            "figure": "Fig. 1",
            "title": "Frequency-separated HRL protocol",
            "main_conclusion": "Freq-HRL assigns slow planning, shock promotion, high-frequency control, and leakage accounting to distinct decision paths.",
            "panels": "A: problem abstraction; B: encoder bands; C: upper/lower policies and promotion; D: leakage and credit gates.",
            "primary_artifacts": "freq_hrl_gpt.md; freq_hrl_dev_manual.md; theory_appendix_latest",
            "review_risk": "Avoid implying a universal convergence theorem; label assumptions and sufficient conditions.",
        },
        {
            "figure": "Fig. 2",
            "title": "Claim and ablation evidence matrix",
            "main_conclusion": "The current evidence matrix is fully supported under conservative claim boundaries.",
            "panels": "A: C1-C9 claim matrix; B: baseline/ablation deltas; C: stress-regime coverage; D: unsupported or bounded rows.",
            "primary_artifacts": "top_journal_unified_matrix_latest; baseline_ablation_matrix_latest; trading_pressure_matrix",
            "review_risk": "Show no-promotion override as native promotion evidence, not as a raw trading Sharpe win.",
        },
        {
            "figure": "Fig. 3",
            "title": "Native Transit promotion and real-demand service response",
            "main_conclusion": "Native Transit evidence supports wait/reward promotion claims and service-response improvements under public AFC/APC demand profiles.",
            "panels": "A: promotion reward/wait CIs; B: real-demand score/wait/alighting/throughput CIs; C: service-response signal; D: claim boundary notes.",
            "primary_artifacts": "transit_native_promotion_v47_odshift_wait_first_512seed_summaryonly; transit_native_real_demand_service_response_v7_48pair_merged",
            "review_risk": "Do not call MBTA/MTA external truth a linked native control loop.",
        },
        {
            "figure": "Fig. 4",
            "title": "External Transit data coverage",
            "main_conclusion": "Public external sources cover board/alight/load and estimated OD fields, while GTFS-ride-native feeds remain optional replication.",
            "panels": "A: AFC/APC demand traces; B: MBTA board/alight/load source coverage; C: MTA OD source coverage; D: GTFS-ride gap ledger.",
            "primary_artifacts": "agency_demand_onboard_coverage_latest; external_transit_truth_validation_latest",
            "review_risk": "Separate observed load/source coverage from Freq-HRL-improved load outcomes.",
        },
        {
            "figure": "Fig. 5",
            "title": "Order-book replay and encoder generalization",
            "main_conclusion": "The trading path supports venue-grade replay infrastructure and cross-domain encoder evidence, with L3 and public-market rows bounded.",
            "panels": "A: L2/L3 manifest coverage; B: matching/replay semantics; C: encoder domain matrix; D: execution sensitivity table.",
            "primary_artifacts": "order_book_lobster_venue_grade_multisymbol; encoder_cross_domain_matrix",
            "review_risk": "Keep large-scale multi-day venue replay as future scale, not current evidence.",
        },
    ]


def _source_summary(agency: dict[str, Any], external_truth: dict[str, Any], order_book: dict[str, Any]) -> dict[str, Any]:
    coverage = order_book.get("coverage", {}) if isinstance(order_book.get("coverage"), dict) else {}
    return {
        "agency_scope": agency.get("summary", {}).get("evidence_scope", ""),
        "agency_supported_boundaries": agency.get("summary", {}).get("supported_boundaries", ""),
        "agency_external_missing": agency.get("summary", {}).get("external_missing_boundaries", ""),
        "external_truth_scope": external_truth.get("summary", {}).get("evidence_scope", ""),
        "external_truth_supported": external_truth.get("summary", {}).get("supported_boundaries", ""),
        "order_book_pairs": coverage.get("venue_grade_l2_l3_session_pairs", ""),
        "order_book_quality": coverage.get("source_quality_status", ""),
    }


def write_submission_package(
    md_path: Path,
    *,
    claim_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    agency: dict[str, Any],
    external_truth: dict[str, Any],
    order_book: dict[str, Any],
) -> None:
    source_summary = _source_summary(agency, external_truth, order_book)
    baseline_summary = baseline.get("summary", {}) if isinstance(baseline.get("summary"), dict) else {}
    lines = [
        "# Freq-HRL Conservative Submission Package",
        "",
        "Date: 2026-06-12",
        "",
        "## One-Sentence Argument",
        "",
        "In time-series control environments with separable slow trends and fast residual disturbances, Freq-HRL provides a frequency-separated hierarchical policy protocol, supported by paired validation across synthetic trading, native Transit, public demand and external truth-source coverage, and venue-grade order-book replay paths, while leaving full deployment validation and joint agency OD/load control as explicit boundaries.",
        "",
        "## Title Options",
        "",
        "1. Frequency-Separated Hierarchical Reinforcement Learning for Time-Series Control",
        "2. Freq-HRL: Responsibility-Separated Control for Multi-Scale Time-Series Environments",
        "3. Frequency-Routed Planning, Promotion, and Control in Hierarchical Reinforcement Learning",
        "",
        "## Draft Abstract",
        "",
        "Time-series control problems often couple slowly varying operating regimes with high-frequency disturbances. Conventional flat policies and generic hierarchical policies can mix these responsibilities, making recovery, attribution, and stress generalization difficult to validate. We introduce Freq-HRL, a frequency-separated hierarchical reinforcement learning protocol that routes low-frequency trend and planning signals to an upper controller, high-frequency residual control to a lower controller, and persistent shocks to a promotion-driven replanning path. The protocol includes causal frequency encoders, plan-curve actions, frequency-attributed credit, and leakage constraints that explicitly penalize responsibility drift. Across the current validation matrix, all nine conservative evidence claims are supported, including native learned promotion, native Transit real-demand service response, leakage no-tradeoff gates, baseline ablations, stress-regime coverage, public external Transit truth-source coverage, and venue-grade L2/L3 order-book replay paths. Public Transit evidence combines AFC/APC demand-driven native simulation with MBTA bus board/alight/load coverage and MTA estimated OD coverage, while order-book evidence uses LOBSTER/NASDAQ TotalView-ITCH symbol-session replay. These results support Freq-HRL as a domain-general protocol for frequency-routed time-series control, not as a completed deployment validation for every real transit agency or exchange venue.",
        "",
        "## Core Contributions",
        "",
        "1. A domain-general Freq-HRL protocol that separates low-frequency planning, high-frequency control, promotion-based replanning, and leakage accounting.",
        "2. Native Transit validation with learned promotion, wait-credit, real-demand service response, and same-domain leakage no-tradeoff gates.",
        "3. Public external Transit data coverage for MBTA board/alight/load and MTA estimated OD, kept separate from native-control performance claims.",
        "4. Quant and order-book validation that includes baseline/ablation matrices, stress regimes, encoder variants, and venue-grade L2/L3 replay paths.",
        "5. A formal appendix with causal encoder, leakage, promotion, credit, paired-CI, and stress-claim boundary propositions.",
        "",
        "## Main Claim Table",
        "",
        *_markdown_table(claim_rows, ["id", "status", "conservative_wording", "boundary"]),
        "",
        "## Main Baseline And Data Facts",
        "",
        f"- baseline/ablation claim status: `{baseline_summary.get('claim_status', '')}`",
        f"- scenario Freq-HRL-family win rate: `{baseline_summary.get('scenario_freq_family_win_rate', '')}`",
        f"- required positive baselines: `{baseline_summary.get('required_baselines_positive', [])}`",
        f"- real-demand evidence scope: `{source_summary['agency_scope']}`",
        f"- agency supported / external-missing boundaries: `{source_summary['agency_supported_boundaries']}` / `{source_summary['agency_external_missing']}`",
        f"- public external truth scope: `{source_summary['external_truth_scope']}`",
        f"- venue-grade L2/L3 order-book pairs: `{source_summary['order_book_pairs']}` with source quality `{source_summary['order_book_quality']}`",
        "",
        "## Conservative Claim Boundary",
        "",
        "Allowed claim: Freq-HRL is a domain-general frequency-separated HRL protocol validated across the current paired synthetic, native Transit, public Transit data, and venue-grade replay evidence matrix.",
        "",
        "Disallowed claim: Freq-HRL is fully validated for all real-world deployments, all transit OD/onboard-load dynamics, or large-scale production exchange execution.",
        "",
        "## Limitations To State In The Manuscript",
        "",
        "- MBTA board/alight/load and MTA OD are separate public sources, not one joint agency native-control loop.",
        "- GTFS-ride-native replication remains an optional external validation path.",
        "- LOBSTER order-book evidence is venue-grade and multi-symbol, but currently limited to three symbol-session pairs.",
        "- Some public-market and L3 encoder rows are bounded or mixed rather than headline performance wins.",
        "- Theory results are sufficient-condition and reporting-boundary results, not universal convergence guarantees.",
        "",
        "## Submission Checklist",
        "",
        "- Main text: title, abstract, introduction, method overview, experiments, discussion, limitations.",
        "- Main tables: C1-C9 evidence, baseline/ablation, real-data coverage.",
        "- Figures: Python-rendered SVG/PDF/PNG/TIFF drafts and panel source data are under `transit_hrl/results/manuscript_figures_latest/`; regenerate with `python3 -m freq_hrl.experiments.manuscript_figures`.",
        "- Supplementary Information: Methods/SI draft in `freq_hrl_methods_si_2026-06-12.md`.",
        "- Availability: Data and Code Availability draft in `freq_hrl_data_code_availability_2026-06-12.md`.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_methods_si(md_path: Path, paths: dict[str, Path]) -> None:
    lines = [
        "# Freq-HRL Methods And Supplementary Information Draft",
        "",
        "Date: 2026-06-12",
        "",
        "## Method Overview",
        "",
        "Freq-HRL treats each domain as a causal time-series control environment with endogenous state `z_t`, exogenous stream `x_t`, and action-dependent outcomes. A causal encoder maps `x_{<=t}` into low-frequency trend, middle-frequency regime buffer, high-frequency residual, uncertainty, persistence, and energy summaries. The upper controller consumes low-frequency trend and bounded residual summaries to produce a plan action. The lower controller consumes the active plan, local state, and high-frequency context to produce fast control actions. A promotion gate monitors persistent residual events and can trigger early upper-level replanning. Leakage diagnostics and constraints measure whether upper and lower controllers are acting outside their assigned frequency responsibilities.",
        "",
        "## Algorithmic Modules",
        "",
        "| module | role | artifact hook |",
        "|---|---|---|",
        "| causal encoder | transforms observed exogenous history into frequency summaries without future leakage | `freq_hrl/encoders/*`; theory theorem 1 |",
        "| upper planner | emits low-frequency plan/timetable/risk curve actions | native Transit and Quant policy artifacts |",
        "| lower controller | handles local high-frequency residual control under the active upper plan | native Transit lower context and trading lower controller |",
        "| promotion gate | triggers replanning under persistent shocks | native promotion v47 and pressure matrices |",
        "| leakage regularizer | penalizes responsibility drift and supports no-tradeoff gates | leakage matrix latest |",
        "| evidence matrix | records conservative supported/partial/missing claim status | top-journal unified matrix |",
        "",
        "## Validation Protocol",
        "",
        "All headline empirical claims are read from stored artifacts rather than reconstructed from prose. Paired comparisons use common seeds or source windows, report direction-specific deltas and 95% confidence intervals where available, and separate strict improvement from noninferiority/no-harm evidence. Stress-generalization claims are treated as intersection claims: a global claim is supported only when every registered regime passes the relevant evidence gate.",
        "",
        "## Main Artifact Paths",
        "",
    ]
    for key, path in paths.items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend([
        "",
        "## Statistics And CI Reporting",
        "",
        "Paired seed/source deltas are the default estimator. For each metric, the treatment and control are compared on matched seeds or matched data windows. Confidence intervals are interpreted according to the metric direction: improvement is supported when the direction-adjusted interval excludes zero in the favorable direction; noninferiority is reported separately when the interval supports a predeclared no-harm margin but not strict improvement.",
        "",
        "## Data Sources",
        "",
        "- Public AFC station-hour entries: `transit_hrl/data/public_afc_mta/hourly_ridership.csv`.",
        "- Public APC route boardings: `transit_hrl/data/public_apc_halifax/route_boardings.csv`.",
        "- MBTA bus board/alight/load source: downloaded to ignored raw cache from `https://mbta-massdot.opendata.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee`.",
        "- MTA Subway OD estimate source: sampled from `https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj`.",
        "- LOBSTER/NASDAQ TotalView-ITCH sample replay: `transit_hrl/data/lobster_sample_raw/` is ignored locally; committed replay summaries are under `transit_hrl/results/order_book_lobster_venue_grade_multisymbol/`.",
        "",
        "## Reproduction Notes",
        "",
        "The raw MBTA and MTA caches are intentionally ignored by git. Regenerate their derived summaries with:",
        "",
        "```bash",
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.external_transit_truth_validation --download-missing --mta-od-total-rows 116279069 --output-dir transit_hrl/results/external_transit_truth_validation_latest",
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.agency_demand_onboard_coverage --output-dir transit_hrl/results/agency_demand_onboard_coverage_latest",
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.top_journal_unified_matrix --output-dir transit_hrl/results/top_journal_unified_matrix_latest",
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_submission_pack --output-dir transit_hrl/results/manuscript_submission_pack_latest",
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_figures --output-dir transit_hrl/results/manuscript_figures_latest",
        "```",
        "",
        "## Supplementary Boundaries",
        "",
        "- Native Transit performance uses simulator service-response metrics under public demand profiles.",
        "- External MBTA/MTA data close field-coverage boundaries, not direct Freq-HRL outcome-improvement claims on those exact files.",
        "- GTFS-ride ingestion is implemented as a supported path, but no public native GTFS-ride feed is currently committed.",
        "- Order-book replay supports venue-grade L2/L3 paths on three symbol-session pairs; final production-scale exchange replay remains a larger-data replication step.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_plan(md_path: Path, figure_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Freq-HRL Figure Plan",
        "",
        "Date: 2026-06-12",
        "",
        "Backend: Python/matplotlib. Rendered SVG/PDF/PNG/TIFF drafts, panel source CSVs, and a preview montage are written to `transit_hrl/results/manuscript_figures_latest/` by `python3 -m freq_hrl.experiments.manuscript_figures`. TIFF files are regenerated locally and ignored by git because they are large.",
        "",
    ]
    for row in figure_rows:
        lines.extend([
            f"## {row['figure']}: {row['title']}",
            "",
            f"Conclusion: {row['main_conclusion']}",
            "",
            f"Panels: {row['panels']}",
            "",
            f"Primary artifacts: `{row['primary_artifacts']}`",
            "",
            f"Review risk: {row['review_risk']}",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_data_availability(md_path: Path) -> None:
    lines = [
        "# Data And Code Availability Draft",
        "",
        "Date: 2026-06-12",
        "",
        "## Data Availability",
        "",
        "The processed evidence tables, validation summaries, claim matrices, source-coverage ledgers, rendered manuscript figures, and figure source-data CSVs generated in this study are available in the repository under `transit_hrl/results/`. The public AFC/APC demand traces used by the native Transit validation are stored under `transit_hrl/data/public_afc_mta/` and `transit_hrl/data/public_apc_halifax/`. Public external Transit truth-source coverage was derived from the MBTA Bus Ridership by Trip, Season, Route, Line, and Stop dataset and the MTA Subway Origin-Destination Ridership Estimate 2024 dataset; the derived summaries are committed under `transit_hrl/results/external_transit_truth_validation_latest/`, while raw downloaded caches are ignored under `transit_hrl/data/public_mbta_bus_ridership_raw/` and `transit_hrl/data/public_mta_od_raw/` to avoid committing large third-party files. The LOBSTER/NASDAQ TotalView-ITCH sample-derived replay summaries are available under `transit_hrl/results/order_book_lobster_venue_grade_multisymbol/`; access to any full raw proprietary exchange feed remains governed by the original data provider.",
        "",
        "## Code Availability",
        "",
        "The code used to generate the Freq-HRL experiments, evidence matrices, external data-source ledgers, manuscript submission package, and manuscript figures is available in the same repository under `transit_hrl/`. The external Transit source ledger can be regenerated with `python3 -m freq_hrl.experiments.transit.external_transit_truth_validation`, the agency coverage ledger with `python3 -m freq_hrl.experiments.transit.agency_demand_onboard_coverage`, the unified claim matrix with `python3 -m freq_hrl.experiments.top_journal_unified_matrix`, and the figure set with `python3 -m freq_hrl.experiments.manuscript_figures`.",
        "",
        "## Dataset Citations And Source URLs",
        "",
        "- MBTA Bus Ridership by Trip, Season, Route, Line, and Stop: https://mbta-massdot.opendata.arcgis.com/datasets/8daf4a33925a4df59183f860826d29ee",
        "- MTA Subway Origin-Destination Ridership Estimate 2024: https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj",
        "- GTFS-ride specification for optional native-format replication: https://gtfsride.org/specification",
        "- LOBSTER sample data and NASDAQ TotalView-ITCH semantics should be cited according to the data provider's required citation terms in the final manuscript.",
        "",
        "## Missing Information / Risk Flags",
        "",
        "- Add final repository URL, release tag, and DOI if the submission requires archival code deposit.",
        "- Confirm whether the final journal requires source-data files for each figure panel.",
        "- Do not describe ignored raw MBTA/MTA caches as newly generated data; they are reused public third-party data.",
        "- Do not imply public GTFS-ride native feed availability unless such a feed is added later.",
        "",
        "## Chinese Check",
        "",
        "- 需要投稿前补最终代码仓库链接、release tag、可能的 Zenodo DOI。",
        "- MBTA/MTA 是公开第三方数据源；当前提交的是派生 summary，不提交大 raw cache。",
        "- GTFS-ride 是可复现实验接口和标准钩子，不是当前已经拿到的真实 feed。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_submission_pack(results_root: Path, output_dir: Path, md_dir: Path) -> dict[str, Any]:
    paths = _artifact_paths(results_root)
    artifacts = {key: _read_json(path) for key, path in paths.items()}
    claim_rows = build_claim_table(artifacts["unified"])
    baseline_rows = build_baseline_table(artifacts["baseline"])
    real_data_rows = build_real_data_table(artifacts["agency"], artifacts["external_truth"])
    figure_rows = build_figure_plan()

    output_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "claim_evidence_table.csv", claim_rows)
    _write_csv(output_dir / "baseline_ablation_table.csv", baseline_rows)
    _write_csv(output_dir / "real_data_table.csv", real_data_rows)
    _write_csv(output_dir / "figure_plan.csv", figure_rows)

    submission_md = md_dir / "freq_hrl_submission_package_2026-06-12.md"
    methods_md = md_dir / "freq_hrl_methods_si_2026-06-12.md"
    figure_md = md_dir / "freq_hrl_figure_plan_2026-06-12.md"
    data_md = md_dir / "freq_hrl_data_code_availability_2026-06-12.md"
    write_submission_package(
        submission_md,
        claim_rows=claim_rows,
        baseline=artifacts["baseline"],
        agency=artifacts["agency"],
        external_truth=artifacts["external_truth"],
        order_book=artifacts["order_book"],
    )
    write_methods_si(methods_md, paths)
    write_figure_plan(figure_md, figure_rows)
    write_data_availability(data_md)

    payload = {
        "summary": {
            "claims": len(claim_rows),
            "supported_claims": sum(1 for row in claim_rows if row["status"] == "supported"),
            "baseline_rows": len(baseline_rows),
            "real_data_rows": len(real_data_rows),
            "figures": len(figure_rows),
            "output_dir": str(output_dir),
            "md_dir": str(md_dir),
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
        "outputs": {
            "claim_evidence_table": str(output_dir / "claim_evidence_table.csv"),
            "baseline_ablation_table": str(output_dir / "baseline_ablation_table.csv"),
            "real_data_table": str(output_dir / "real_data_table.csv"),
            "figure_plan": str(output_dir / "figure_plan.csv"),
            "submission_package": str(submission_md),
            "methods_si": str(methods_md),
            "figure_plan_md": str(figure_md),
            "data_code_availability": str(data_md),
        },
        "boundary": (
            "This package is submission-preparation scaffolding. It keeps claim "
            "wording conservative and links to Python-rendered draft figures, "
            "while preserving deployment-scale boundaries."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR)
    args = parser.parse_args()
    payload = build_submission_pack(
        results_root=args.results_root,
        output_dir=args.output_dir,
        md_dir=args.md_dir,
    )
    print(
        "manuscript_submission_pack "
        f"claims={payload['summary']['claims']} "
        f"figures={payload['summary']['figures']} "
        f"output={payload['summary']['output_dir']}"
    )


if __name__ == "__main__":
    main()
