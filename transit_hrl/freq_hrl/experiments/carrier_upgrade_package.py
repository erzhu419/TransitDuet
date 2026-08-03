"""Build the Freq-HRL carrier-upgrade package.

The package turns the current prototype evidence into a frozen manuscript and
reproducibility scaffold. It does not run expensive experiments; it records the
contracts, claim boundaries, audit checks, baseline manifest, data scale-up plan,
proof obligations, and reproduction commands needed to move from a validated
prototype to a journal-grade platform.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.cs_top_venue_experiment_matrix import build_cs_top_venue_experiment_matrix
from freq_hrl.experiments.theory_appendix import build_theory_payload
from freq_hrl.core.shared_core_audit import audit_shared_training_core
from freq_hrl.core.spec import (
    default_spec,
    validate_claim_freeze,
    validate_shared_core_paths,
)


DATE_TAG = "2026-06-27"
DEFAULT_RESULTS_ROOT = Path("transit_hrl/results")
DEFAULT_OUTPUT_DIR = Path("transit_hrl/results/carrier_upgrade_package_latest")
DEFAULT_MD_DIR = Path("transit_hrl/md")
DEFAULT_SOURCE_ROOT = Path(".")


ARTIFACTS = {
    "claims": Path("top_journal_unified_matrix_latest/claims.csv"),
    "baseline_checks": Path("baseline_ablation_matrix_latest/paired_checks.csv"),
    "baseline_summary": Path("baseline_ablation_matrix_latest/summary.json"),
    "promotion_checks": Path("transit_native_promotion_v47_odshift_wait_first_512seed_summaryonly/paired_checks.csv"),
    "real_demand_checks": Path("transit_native_real_demand_service_response_v7_48pair_merged/paired_checks.csv"),
    "leakage_checks": Path("leakage_no_tradeoff_matrix_latest/paired_checks.csv"),
    "encoder_checks": Path("encoder_cross_domain_matrix_latest/paired_checks.csv"),
    "order_book_summary": Path("order_book_lobster_venue_grade_multisymbol/summary.json"),
    "external_truth_summary": Path("external_transit_truth_validation_latest/summary.json"),
    "agency_summary": Path("agency_demand_onboard_coverage_latest/summary.json"),
    "theory_summary": Path("freq_hrl_theory_appendix_latest/summary.json"),
    "cs_experiment_summary": Path("cs_top_venue_experiment_matrix_latest/summary.json"),
}


MD_NAMES = {
    "carrier_plan": f"freq_hrl_carrier_upgrade_plan_{DATE_TAG}.md",
    "algorithm_spec": f"freq_hrl_algorithm_spec_{DATE_TAG}.md",
    "shared_core": f"freq_hrl_shared_core_audit_{DATE_TAG}.md",
    "baseline": f"freq_hrl_baseline_manifest_{DATE_TAG}.md",
    "real_data": f"freq_hrl_real_data_scaleup_plan_{DATE_TAG}.md",
    "theory": f"freq_hrl_theory_proof_appendix_{DATE_TAG}.md",
    "manuscript": f"freq_hrl_full_manuscript_draft_{DATE_TAG}.md",
    "cs_venue": f"freq_hrl_cs_top_venue_strategy_{DATE_TAG}.md",
    "cs_experiments": f"freq_hrl_cs_top_venue_experiment_matrix_{DATE_TAG}.md",
    "repro": f"freq_hrl_reproducibility_package_{DATE_TAG}.md",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        values = [
            str(row.get(field, "")).replace("\n", " ").replace("|", "/")
            for field in fields
        ]
        out.append("| " + " | ".join(values) + " |")
    return out


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        if value in ("", None):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _artifact_paths(results_root: Path) -> dict[str, Path]:
    return {key: results_root / value for key, value in ARTIFACTS.items()}


def build_claim_freeze(claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in claim_rows:
        claim_id = str(row.get("id", ""))
        out.append({
            "claim_id": claim_id,
            "status": str(row.get("status", "")),
            "claim": str(row.get("claim", "")),
            "frozen_boundary": str(row.get("boundary", row.get("remaining_gap", ""))),
            "evidence_artifact": str(row.get("artifact", "")),
            "allowed_wording": _allowed_wording(
                claim_id,
                str(row.get("claim", "")),
                status=str(row.get("status", "missing")),
            ),
            "disallowed_wording": _disallowed_wording(claim_id),
        })
    return out


def _allowed_wording(claim_id: str, claim: str, *, status: str) -> str:
    if status == "partial":
        return (
            f"The frozen evidence partially supports '{claim}'; only subchecks "
            "explicitly marked supported in the raw-only claim matrix may be stated."
        )
    if status == "not_supported":
        return f"The frozen evidence does not support '{claim}'; report it as an unresolved target."
    if status == "missing":
        return f"No eligible frozen evidence supports '{claim}'; report the evidence as missing."
    allowed = {
        "C1": "Native learned promotion improves reward/wait under the registered native stress artifact.",
        "C2": "Native public AFC/APC demand service-response improves score, wait, alighting, and throughput in the current validation loop.",
        "C3": "Venue-grade L2/L3 replay infrastructure is supported on the current LOBSTER/NASDAQ TotalView-ITCH symbol sessions.",
        "C4": "Advanced encoder paths have cross-domain support under bounded public-market and L3 caveats.",
        "C5": "Leakage no-tradeoff is supported where same-domain drift reduction and performance gates both pass.",
        "C6": "The formal appendix gives sufficient-condition results for the protocol claims.",
        "C7": "Promotion improvement replicates across the current registered persistent and OD-shift stress matrices.",
        "C8": "Baseline and ablation evidence supports frequency responsibility over non-frequency and misrouted alternatives.",
        "C9": "Synthetic stress coverage is supported for the registered stationary, burst, persistent, and OOD regimes.",
    }
    return allowed.get(claim_id, claim)


def _disallowed_wording(claim_id: str) -> str:
    disallowed = {
        "C1": "Do not claim learned promotion is universally superior under every deployment stress.",
        "C2": "Do not claim one joint agency APC/AFC/OD/onboard-load control deployment.",
        "C3": "Do not claim production-scale exchange execution is solved.",
        "C4": "Do not claim every advanced encoder dominates in every domain.",
        "C5": "Do not claim no-tradeoff outside domains passing both drift and performance gates.",
        "C6": "Do not claim a universal convergence theorem.",
        "C7": "Do not claim all possible stress regimes are covered.",
        "C8": "Do not claim frequency features alone are the contribution.",
        "C9": "Do not extrapolate to unregistered stress families.",
    }
    return disallowed.get(claim_id, "Do not overstate beyond the artifact boundary.")


def build_shared_core_audit(source_root: Path) -> list[dict[str, Any]]:
    checks = [
        (
            "shared data contracts",
            "transit_hrl/freq_hrl/core/types.py",
            "ExogenousBin and FrequencyFeatures keep domain adapters outside the core.",
        ),
        (
            "causal encoder interface",
            "transit_hrl/freq_hrl/encoders/base.py",
            "Encoders expose causal low/mid/high frequency summaries.",
        ),
        (
            "promotion gate",
            "transit_hrl/freq_hrl/core/promotion_gate.py",
            "Persistent high-frequency energy can trigger high-level replanning.",
        ),
        (
            "leakage accounting",
            "transit_hrl/freq_hrl/core/leakage.py",
            "Upper HF power and lower LF drift are measured as responsibility leakage.",
        ),
        (
            "dual actor-critic core",
            "transit_hrl/freq_hrl/rl/training.py",
            "Dual-level PPO training loop is domain-agnostic through rollout adapters.",
        ),
        (
            "Transit native adapter",
            "transit_hrl/freq_transitduet/runner_v3.py",
            "Transit runner consumes Freq-HRL configs and native wait/promotion metrics.",
        ),
        (
            "Transit native full config",
            "transit_hrl/freq_transitduet/configs_freqduet/T_freqhrl_native_full.yaml",
            "Native Transit instantiation of the protocol.",
        ),
        (
            "Trading policy adapter",
            "transit_hrl/freq_hrl/policies/ac_trading.py",
            "Quant/trading policy path uses the same frequency-responsibility protocol.",
        ),
        (
            "Order-book replay adapter",
            "transit_hrl/freq_hrl/experiments/top_journal_unified_matrix.py",
            "Order-book evidence is pulled into the same claim matrix.",
        ),
    ]
    rows = []
    for item, rel, role in checks:
        exists = (source_root / rel).exists()
        rows.append({
            "audit_item": item,
            "status": "supported" if exists else "missing",
            "path": rel,
            "role": role,
            "next_upgrade": "Keep interface frozen; domain code may only enter through adapters." if exists else "Add or restore the missing interface.",
        })
    return rows


def build_baseline_manifest(
    baseline_rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    baseline_summary = dict(baseline_summary or {})
    learned_manifest = {
        str(row.get("baseline", "")): row
        for row in baseline_summary.get("learned_baseline_manifest", []) or []
        if isinstance(row, dict)
    }
    required = [
        ("vanilla_rl", "flat non-hierarchical baseline", "current"),
        ("hrl_raw", "generic HRL with raw state", "current"),
        ("raw_history", "raw demand/history feature baseline", "current"),
        ("freq_single_policy", "frequency features without HRL responsibility split", "current"),
        ("allfreq_alllayers", "all frequency bands visible to all layers", "current"),
        ("swapped", "misrouted LF/HF responsibility ablation", "current"),
        ("no_promotion", "promotion removed", "current_with_native_override"),
        ("no_leakage", "leakage regularization removed", "current"),
        ("lf_upper_only", "upper gets LF without lower HF completion", "current_boundary"),
        ("hf_lower_only", "lower gets HF without full upper protocol", "current_boundary"),
        ("flat_ppo", "strong flat learned-policy baseline", "registered_strong_learned"),
        ("flat_sac", "off-policy continuous-control baseline", "registered_strong_learned"),
        ("flat_td3", "deterministic actor-critic baseline", "registered_strong_learned"),
        ("generic_hrl_ppo", "non-frequency learned HRL baseline", "registered_strong_learned"),
    ]
    support: dict[str, dict[str, Any]] = {}
    for row in baseline_rows:
        check = str(row.get("check", ""))
        if not check.startswith("freq_hrl_vs_"):
            continue
        rest = check[len("freq_hrl_vs_"):]
        metric = str(row.get("metric", ""))
        suffix = f"_{metric}"
        if not rest.endswith(suffix):
            continue
        baseline = rest[: -len(suffix)]
        item = support.setdefault(baseline, {})
        item[metric] = row
    rows = []
    for baseline, purpose, tier in required:
        metrics = support.get(baseline, {})
        headline = [
            metrics.get(metric, {}).get("status", "missing")
            for metric in ("sharpe", "total_return", "FocusScore")
        ]
        learned = learned_manifest.get(baseline, {})
        headline_status = (
            "supported" if headline and all(s == "supported" for s in headline)
            else ("missing" if not metrics else "partial")
        )
        if learned:
            evidence_status = str(learned.get("evidence_status", headline_status))
            headline_status = evidence_status
        rows.append({
            "baseline": baseline,
            "purpose": purpose,
            "tier": tier,
            "headline_status": headline_status,
            "sharpe_delta": _fmt_float(metrics.get("sharpe", {}).get("delta_mean")),
            "return_delta": _fmt_float(metrics.get("total_return", {}).get("delta_mean")),
            "focus_delta": _fmt_float(metrics.get("FocusScore", {}).get("delta_mean")),
            "n_common": metrics.get("sharpe", {}).get("n_common", ""),
            "paper_role": (
                "main_table" if tier.startswith("current")
                else str(learned.get("paper_role", "must_complete_or_limit"))
            ),
        })
    return rows


def build_data_scaleup_manifest(
    external_truth: dict[str, Any],
    agency: dict[str, Any],
    order_book: dict[str, Any],
) -> list[dict[str, Any]]:
    coverage = order_book.get("coverage", {}) if isinstance(order_book.get("coverage"), dict) else {}
    agency_summary = agency.get("summary", {}) if isinstance(agency.get("summary"), dict) else {}
    truth_summary = external_truth.get("summary", {}) if isinstance(external_truth.get("summary"), dict) else {}
    same_agency_control = str(
        agency_summary.get("same_agency_native_control_status", "")
        or "external_missing"
    )
    field_complete = str(
        agency_summary.get("field_complete_data_status", "")
        or "external_missing"
    )
    return [
        {
            "domain": "Transit",
            "upgrade": "same-agency AFC/APC/OD/onboard-load native control loop",
            "current_status": same_agency_control,
            "current_evidence": (
                f"scope={agency_summary.get('evidence_scope', '')}; "
                f"field_complete={field_complete}"
            ),
            "minimum_next_dataset": "one public or partnered agency feed with board, alight, onboard load, OD, and GTFS/GTFS-ride alignment",
            "claim_boundary": "current public MBTA/MTA sources are truth-source coverage, not one linked deployment loop",
        },
        {
            "domain": "Transit",
            "upgrade": "GTFS-ride native replication",
            "current_status": "external_missing",
            "current_evidence": str(agency_summary.get("external_missing_boundaries", "")),
            "minimum_next_dataset": "GTFS-ride board/alight/load/OD feed or a reproducible converter from agency AVL/APC exports",
            "claim_boundary": "optional replication path until a native feed is available",
        },
        {
            "domain": "Transit",
            "upgrade": "public external truth-source scale",
            "current_status": "supported",
            "current_evidence": str(truth_summary.get("evidence_scope", "")),
            "minimum_next_dataset": "retain MBTA board/alight/load and MTA OD scripts with checksums and row-count audit",
            "claim_boundary": "truth-source coverage, not direct Freq-HRL control on those raw files",
        },
        {
            "domain": "Market",
            "upgrade": "multi-symbol multi-session L2/L3 replay",
            "current_status": "supported_path",
            "current_evidence": f"venue_pairs={coverage.get('venue_grade_l2_l3_session_pairs', '')}; quality={coverage.get('source_quality_status', '')}",
            "minimum_next_dataset": "at least 20 symbol-session pairs across volatile, quiet, trend, and reversal sessions",
            "claim_boundary": "current path is venue-grade replay infrastructure, not production exchange execution",
        },
        {
            "domain": "Market",
            "upgrade": "execution sensitivity",
            "current_status": "partial_supported",
            "current_evidence": f"latency_bins={coverage.get('latency_bins', '')}; modes={coverage.get('execution_modes', '')}",
            "minimum_next_dataset": "latency, slippage, queue-ahead, cancel/replace, and transaction-cost sweeps",
            "claim_boundary": "alpha and execution robustness must be reported separately",
        },
    ]


def build_proof_manifest(theory_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    theory_payload = dict(theory_payload or {})
    theorem_rows = [
        dict(row) for row in theory_payload.get("theorems", []) or []
        if isinstance(row, dict)
    ]
    if theorem_rows:
        return [
            {
                "proof_item": f"{row.get('id', '')}: {row.get('title', '')}".strip(": "),
                "status": "formalized_statement",
                "statement": str(row.get("statement", "")),
                "paper_use": str(row.get("diagnostic", "")),
                "assumption_count": len(row.get("assumptions", []) or []),
                "limitation": str(row.get("limitation", "")),
            }
            for row in theorem_rows
        ]
    return [
        {
            "proof_item": "causal encoder lemma",
            "status": "formalized_skeleton",
            "statement": "If E_phi only consumes x_{<=t}, no policy action can depend on future exogenous observations through the encoder.",
            "paper_use": "guards against lookahead leakage in all frequency features",
        },
        {
            "proof_item": "frequency responsibility proposition",
            "status": "formalized_skeleton",
            "statement": "Under band-separated exogenous drivers and bounded cross-band covariance, routing LF features to upper and HF residuals to lower reduces cross-level credit variance.",
            "paper_use": "turns frequency decomposition from feature engineering into an HRL responsibility principle",
        },
        {
            "proof_item": "leakage bound",
            "status": "formalized_skeleton",
            "statement": "With an LPF penalty on cumulative lower actions and an HPF penalty on upper actions, responsibility leakage is bounded by the constraint budget plus optimization residual.",
            "paper_use": "supports the no-tradeoff boundary when performance gates also pass",
        },
        {
            "proof_item": "promotion detection tradeoff",
            "status": "formalized_skeleton",
            "statement": "Persistence thresholds induce an explicit false-positive/false-negative tradeoff between early replanning and shock overreaction.",
            "paper_use": "explains why promotion claims are stress-registered rather than universal",
        },
        {
            "proof_item": "paired-CI claim rule",
            "status": "formalized_skeleton",
            "statement": "A claim is supported only when paired deltas pass direction-aware confidence gates over matched seeds or source windows.",
            "paper_use": "connects statistical evidence to claim boundaries",
        },
    ]


def build_repro_commands() -> list[dict[str, Any]]:
    return [
        {
            "stage": "unit_tests",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m unittest discover -s transit_hrl/tests",
            "output": "all local tests",
            "expected": "OK",
        },
        {
            "stage": "claim_matrix",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.top_journal_unified_matrix --output-dir transit_hrl/results/top_journal_unified_matrix_latest",
            "output": "claims.csv, report.md, summary.json",
            "expected": "9 conservative claims with explicit boundaries",
        },
        {
            "stage": "external_truth_raw_cache",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.external_transit_truth_validation --download-missing --mta-od-total-rows 116279069 --output-dir transit_hrl/results/external_transit_truth_validation_latest",
            "output": "public MBTA/MTA derived summaries; raw caches remain ignored",
            "expected": "supported public board/alight/load and estimated OD coverage",
        },
        {
            "stage": "agency_coverage",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.transit.agency_demand_onboard_coverage --output-dir transit_hrl/results/agency_demand_onboard_coverage_latest",
            "output": "source_coverage.csv, claim_boundaries.csv, deployment_data_gate.csv",
            "expected": "field coverage and same-agency native-control boundary ledger",
        },
        {
            "stage": "baseline_manifest",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.baseline_ablation_matrix --output-dir transit_hrl/results/baseline_ablation_matrix_latest",
            "output": "paired baseline and ablation checks",
            "expected": "frequency-responsibility baselines supported where registered",
        },
        {
            "stage": "strong_learned_baselines",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.trading.strong_learned_baseline_validation --output-dir transit_hrl/results/strong_learned_baseline_validation_latest",
            "output": "learned PPO-family per-seed rows, paired checks, parameter budget, and sample-efficiency tables",
            "expected": "matched-budget Freq-HRL PPO vs flat PPO and generic HRL PPO evidence; SAC/TD3 remain explicit registered baselines until implemented",
        },
        {
            "stage": "cs_top_venue_experiment_matrix",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.cs_top_venue_experiment_matrix --output-dir transit_hrl/results/cs_top_venue_experiment_matrix_latest",
            "output": "1-8 experiment matrix and scheduler manifest",
            "expected": "reviewer-facing experiment ledger with commands, gates, and claim boundaries",
        },
        {
            "stage": "sensitivity_robustness",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.trading.sensitivity_robustness_matrix --output-dir transit_hrl/results/sensitivity_robustness_matrix_latest",
            "output": "profile-wise robustness rows and paired noninferiority checks",
            "expected": "stress-registered plan/leakage/constraint/recenter sensitivity boundaries",
        },
        {
            "stage": "figures",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_figures --output-dir transit_hrl/results/manuscript_figures_latest",
            "output": "SVG/PDF/PNG/TIFF figures and source_data CSVs",
            "expected": "five manuscript figures",
        },
        {
            "stage": "submission_pack",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.manuscript_submission_pack --output-dir transit_hrl/results/manuscript_submission_pack_latest",
            "output": "conservative submission package",
            "expected": "claim tables, methods/SI, data availability",
        },
        {
            "stage": "theory_appendix",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.theory_appendix --output-dir transit_hrl/results/freq_hrl_theory_appendix_latest",
            "output": "formal theorem/proposition appendix",
            "expected": "sufficient-condition and reporting-boundary statements",
        },
        {
            "stage": "carrier_upgrade",
            "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m freq_hrl.experiments.carrier_upgrade_package --output-dir transit_hrl/results/carrier_upgrade_package_latest",
            "output": "carrier upgrade md and manifests",
            "expected": "frozen spec, audit, baseline, data, theory, manuscript, reproducibility package",
        },
    ]


def _seed_summary(path: Path, *, artifact: str, role: str) -> dict[str, Any]:
    data = _read_json(path)
    seeds = data.get("seeds", []) if isinstance(data, dict) else []
    seeds = [int(seed) for seed in seeds if str(seed).strip()]
    stride = 0
    if len(seeds) >= 2:
        diffs = sorted({b - a for a, b in zip(seeds, seeds[1:])})
        stride = diffs[0] if len(diffs) == 1 else 0
    return {
        "artifact": artifact,
        "role": role,
        "path": str(path),
        "status": "present" if path.exists() else "missing",
        "seed_count": len(seeds),
        "first_seed": seeds[0] if seeds else "",
        "last_seed": seeds[-1] if seeds else "",
        "seed_stride": stride,
        "boundary": "scheduler seed ledger; raw shard logs are not committed",
    }


def build_scheduler_seed_manifest(results_root: Path) -> list[dict[str, Any]]:
    return [
        _seed_summary(
            results_root / "transit_native_promotion_v47_odshift_wait_first_512seed_summaryonly/summary.json",
            artifact="native_promotion_v47_odshift_wait_first_512seed",
            role="C1/C7 native learned-promotion evidence",
        ),
        _seed_summary(
            results_root / "transit_native_real_demand_service_response_v7_48pair_merged/summary.json",
            artifact="native_real_demand_service_response_v7_48pair",
            role="C2 native real-demand service-response evidence",
        ),
    ]


def build_repro_artifact_manifest(
    results_root: Path,
    output_dir: Path,
    scheduler_seed_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    figure_source_dir = results_root / "manuscript_figures_latest/source_data"
    figure_sources = sorted(figure_source_dir.glob("*.csv")) if figure_source_dir.exists() else []
    rows = [
        {
            "artifact": "claim_matrix",
            "path": str(results_root / "top_journal_unified_matrix_latest/summary.json"),
            "kind": "evidence_table",
            "status": "present" if (results_root / "top_journal_unified_matrix_latest/summary.json").exists() else "missing",
            "count": "",
            "reproduce_stage": "claim_matrix",
            "commit_policy": "commit compact summary and claim tables",
        },
        {
            "artifact": "figure_source_data",
            "path": str(figure_source_dir),
            "kind": "source_data",
            "status": "present" if figure_sources else "missing",
            "count": len(figure_sources),
            "reproduce_stage": "figures",
            "commit_policy": "commit source_data CSVs and lightweight figure drafts",
        },
        {
            "artifact": "scheduler_seed_manifest",
            "path": str(output_dir / "scheduler_seed_manifest.csv"),
            "kind": "seed_ledger",
            "status": "present" if scheduler_seed_manifest else "missing",
            "count": sum(int(row.get("seed_count", 0) or 0) for row in scheduler_seed_manifest),
            "reproduce_stage": "claim_matrix",
            "commit_policy": "commit compact seed ledger, not raw scheduler scratch",
        },
        {
            "artifact": "cs_top_venue_experiment_matrix",
            "path": str(output_dir / "cs_top_venue_experiment_matrix.csv"),
            "kind": "experiment_ledger",
            "status": "present" if (output_dir / "cs_top_venue_experiment_matrix.csv").exists() else "generated_by_carrier",
            "count": "",
            "reproduce_stage": "cs_top_venue_experiment_matrix",
            "commit_policy": "commit compact reviewer-facing ledger and scheduler manifest",
        },
        {
            "artifact": "external_transit_raw_cache",
            "path": "transit_hrl/data/public_mbta_bus_ridership_raw; transit_hrl/data/public_mta_od_raw",
            "kind": "ignored_raw_cache",
            "status": "regenerate_with_command",
            "count": "",
            "reproduce_stage": "external_truth_raw_cache",
            "commit_policy": "do not commit raw third-party cache",
        },
        {
            "artifact": "carrier_validation_json",
            "path": str(output_dir / "spec_validation.json"),
            "kind": "machine_check",
            "status": "present",
            "count": 1,
            "reproduce_stage": "carrier_upgrade",
            "commit_policy": "commit compact validation JSON",
        },
    ]
    return rows


def build_cs_top_venue_readiness(
    *,
    baseline_summary: dict[str, Any],
    agency: dict[str, Any],
    order_book: dict[str, Any],
    theory_summary: dict[str, Any],
    shared_core_validation: dict[str, Any],
    claim_freeze: list[dict[str, Any]],
    cs_experiment_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Reviewer-facing readiness matrix for CS top venues."""
    cs_experiment_summary = dict(cs_experiment_summary or {})
    cs_exp_meta = (
        cs_experiment_summary.get("summary", {})
        if isinstance(cs_experiment_summary.get("summary"), dict)
        else {}
    )
    baseline_meta = baseline_summary.get("summary", {}) if isinstance(baseline_summary.get("summary"), dict) else {}
    agency_meta = agency.get("summary", {}) if isinstance(agency.get("summary"), dict) else {}
    order_coverage = order_book.get("coverage", {}) if isinstance(order_book.get("coverage"), dict) else {}
    theorem_count = len(theory_summary.get("theorems", []) or [])
    claims_supported = sum(1 for row in claim_freeze if row.get("status") == "supported")
    learned_status = str(baseline_meta.get("strong_learned_baseline_status", "registered_missing"))
    same_agency_status = str(agency_meta.get("same_agency_native_control_status", "external_missing"))
    venue_pairs = int(order_coverage.get("venue_grade_l2_l3_session_pairs", 0) or 0)
    return [
        {
            "review_axis": "single_algorithm_claim",
            "cs_expectation": "One crisp algorithmic contribution, not a domain engineering story.",
            "current_status": "paper_ready_with_boundary" if claims_supported >= 9 else "partial",
            "evidence": f"supported_claims={claims_supported}; frozen_spec={default_spec().version}",
            "next_action": "Lead with frequency-responsibility routing; keep FreqDuet as prior domain origin.",
            "venue_risk": "Rejected as incremental feature engineering if routing/credit/leakage are not foregrounded.",
        },
        {
            "review_axis": "shared_training_core",
            "cs_expectation": "Quant and Transit must look like adapters around one algorithm.",
            "current_status": str(shared_core_validation.get("status", "missing")),
            "evidence": (
                f"checked_core_files={shared_core_validation.get('core_boundary', {}).get('checked_files', 0)}; "
                f"adapter_entries={len(shared_core_validation.get('adapter_evidence', []) or [])}"
            ),
            "next_action": "In the paper, show one pseudocode block and two domain-adapter instantiations.",
            "venue_risk": "If this reads like two systems, ML/RL reviewers will discount generality.",
        },
        {
            "review_axis": "strong_rl_baselines",
            "cs_expectation": "Flat PPO/SAC/TD3 and generic HRL baselines must be explicit.",
            "current_status": "blocker" if learned_status in {"registered_missing", "missing"} else learned_status,
            "evidence": (
                f"strong_learned_baseline_status={learned_status}; "
                f"experiment_matrix_status={cs_exp_meta.get('strong_learned_status', 'missing')}"
            ),
            "next_action": "Run or clearly limit flat PPO/SAC/TD3 and generic HRL; do not count heuristic baselines as learned RL.",
            "venue_risk": "Top ML/RL venues will treat missing learned baselines as a core empirical weakness.",
        },
        {
            "review_axis": "ablation_completeness",
            "cs_expectation": "Ablate frequency routing, promotion, leakage, encoder, and plan curve separately.",
            "current_status": str(baseline_meta.get("claim_status", "missing")),
            "evidence": (
                f"required_positive={baseline_meta.get('required_baselines_positive', [])}; "
                f"scenario_win_rate={baseline_meta.get('scenario_freq_family_win_rate', '')}"
            ),
            "next_action": "Keep the main ablation table compact; move long stress matrices to appendix.",
            "venue_risk": "A reviewer may accept bounded claims if ablations are clean and pre-registered.",
        },
        {
            "review_axis": "real_world_evidence",
            "cs_expectation": "Real data must support generality without overclaiming deployment.",
            "current_status": "paper_ready_with_boundary",
            "evidence": (
                f"scope={agency_meta.get('evidence_scope', '')}; "
                f"same_agency_control={same_agency_status}"
            ),
            "next_action": "State public truth-source coverage; do not claim same-agency deployment validation.",
            "venue_risk": "Overclaiming Transit data linkage is more dangerous than calling it a limitation.",
        },
        {
            "review_axis": "order_book_scale",
            "cs_expectation": "Market evidence should be reproducible and not rely on tiny toy fixtures.",
            "current_status": "partial_scale" if venue_pairs < 20 else "strong_scale",
            "evidence": (
                f"venue_grade_pairs={venue_pairs}; "
                f"quality={order_coverage.get('source_quality_status', '')}"
            ),
            "next_action": "For a finance/data-mining venue, expand to more symbol-session pairs; for ML/RL, keep as secondary evidence.",
            "venue_risk": "A finance-heavy reviewer may ask for much larger venue-grade replay.",
        },
        {
            "review_axis": "theory_scope",
            "cs_expectation": "Formal statements should support the method but avoid false convergence claims.",
            "current_status": "paper_ready_with_boundary" if theorem_count >= 9 else "partial",
            "evidence": f"formal_theorems_or_propositions={theorem_count}",
            "next_action": "Frame theory as causality/leakage/promotion/CI reporting bounds, not universal RL convergence.",
            "venue_risk": "The theory is adequate for an empirical ML systems paper, not a theory-track claim.",
        },
        {
            "review_axis": "reproducibility",
            "cs_expectation": "Artifacts, seeds, source data, and raw-cache commands must be auditable.",
            "current_status": "paper_ready_with_boundary",
            "evidence": "carrier package emits claim, baseline, data, proof, seed, and reproducibility manifests",
            "next_action": "Attach artifact checklist in appendix and avoid committing raw third-party caches.",
            "venue_risk": "Usually a strength if the claim wording remains conservative.",
        },
    ]


def write_cs_venue_strategy(path: Path, rows: list[dict[str, Any]]) -> None:
    blocker_rows = [row for row in rows if row.get("current_status") == "blocker"]
    lines = [
        "# Freq-HRL CS Top-Venue Strategy",
        "",
        f"Date: {DATE_TAG}",
        "",
        "Target: CS top conference/journal review standards, especially ML/RL/AI/data-mining reviewers who will judge novelty, baselines, ablations, statistical evidence, and reproducibility before domain details.",
        "",
        "## Submission Positioning",
        "",
        "Main thesis: frequency-responsibility routing is an algorithmic principle for time-series HRL. FreqDuet is the origin story and Transit is one domain adapter; Freq-HRL must be presented as the general method.",
        "",
        "Recommended framing: submit as an empirical ML/RL method with strong systems-style reproducibility, not as a Transit-only paper. The paper should lead with one algorithm box, one interface contract, one claim matrix, and two domain instantiations.",
        "",
        "## Readiness Matrix",
        "",
        *_md_table(rows, ["review_axis", "cs_expectation", "current_status", "evidence", "next_action", "venue_risk"]),
        "",
        "## Current Blocking Items",
        "",
    ]
    if blocker_rows:
        for row in blocker_rows:
            lines.append(f"- {row['review_axis']}: {row['next_action']}")
    else:
        lines.append("- No hard blocker is registered by the current matrix; remaining risks are claim-boundary and scale risks.")
    lines.extend([
        "",
        "## Venue Fit",
        "",
        "- NeurIPS/ICML/ICLR style: strongest if flat PPO/SAC/TD3 and generic HRL baselines are completed; otherwise position as a bounded empirical method and expect baseline objections.",
        "- KDD/WebConf/AAAI/IJCAI style: stronger if real-data and cross-domain evidence are foregrounded, with cautious claims about deployment.",
        "- JMLR/TMLR style: requires a cleaner algorithmic narrative, strong reproducibility, and a larger appendix; results can be broader but must be statistically disciplined.",
        "- Transportation or fintech journals: evidence depth matters more than algorithmic minimalism; same-agency Transit and larger L2/L3 replay become higher priority.",
        "",
        "## Non-Negotiable Writing Rule",
        "",
        "Do not claim production deployment, universal encoder dominance, or universal RL convergence. The publishable claim is domain-general frequency-responsibility routing with conservative empirical support.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cs_experiments(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        dict(row) for row in payload.get("experiments", []) or []
        if isinstance(row, dict)
    ]
    summary = (
        payload.get("summary", {})
        if isinstance(payload.get("summary"), dict)
        else {}
    )
    lines = [
        "# Freq-HRL CS Top-Venue Experiment Matrix",
        "",
        f"Date: {DATE_TAG}",
        "",
        "Purpose: make the eight reviewer-critical experiments explicit, executable, and claim-gated. This prevents the manuscript from silently counting protocol smoke tests as top-venue empirical evidence.",
        "",
        f"- experiment count: `{summary.get('experiment_count', len(rows))}`",
        f"- blocker count: `{summary.get('blocker_count', '')}`",
        f"- strong learned status: `{summary.get('strong_learned_status', '')}`",
        f"- learned cross-stress scenarios: `{summary.get('learned_cross_stress_scenarios', '')}`",
        f"- same-agency status: `{summary.get('same_agency_status', '')}`",
        f"- venue-grade pairs: `{summary.get('venue_grade_pairs', '')}`",
        "",
        *_md_table(rows, ["id", "experiment", "current_status", "claim_gate", "artifact", "paper_table", "priority"]),
        "",
        "## Scheduler Use",
        "",
        "The companion CSV `cs_top_venue_scheduler_manifest.csv` contains the command for each row. Commands may be dispatched on CPU nodes; raw third-party data remain outside the committed repository unless explicitly licensed and compact.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_carrier_plan(path: Path, claim_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Freq-HRL Carrier Upgrade Plan",
        "",
        f"Date: {DATE_TAG}",
        "",
        "Purpose: turn the current Freq-HRL validated prototype into a journal-grade research platform. The central claim stays narrow: frequency decomposition is a hierarchical control-responsibility principle, not a feature trick.",
        "",
        "## Seven-Step Upgrade",
        "",
        "1. Freeze the algorithm definition: interfaces, causal encoder contract, promotion rule, leakage accounting, claim gates.",
        "2. Prove that Quant/Transit use the same training core: only domain adapters may differ.",
        "3. Build a strong baseline and ablation manifest: flat RL, generic HRL, raw history, frequency-feature-only, swapped routing, no-promotion, no-leakage.",
        "4. Push real data one level deeper: same-agency Transit loop and multi-session venue-grade L2/L3 market replay.",
        "5. Convert theory scaffolding into theorem/proposition statements with explicit assumptions.",
        "6. Rewrite the full manuscript around one claim: frequency-responsibility routing improves time-series HRL.",
        "7. Ship a reproducibility package: claim matrix, figure source data, scheduler seed manifest, raw-cache regeneration commands.",
        "",
        "## Frozen Claim Count",
        "",
        f"Current conservative claim rows: {len(claim_rows)}.",
        "",
        "This plan deliberately keeps deployment-scale and production-execution claims outside the main text until the corresponding external data loops are closed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_algorithm_spec(path: Path, claim_freeze: list[dict[str, Any]]) -> None:
    lines = [
        "# Freq-HRL Frozen Algorithm Specification",
        "",
        f"Date: {DATE_TAG}",
        "",
        "## Definition",
        "",
        "Freq-HRL is a frequency-responsibility protocol for hierarchical reinforcement learning in environments driven by non-stationary exogenous time series. A causal encoder decomposes the exogenous stream into low-frequency trend, middle-frequency regime buffer, and high-frequency residual. The upper controller owns slow plan variables; the lower controller owns fast residual correction; promotion transfers persistent high-frequency shocks into upper-level replanning; leakage penalties prevent either layer from acting outside its frequency responsibility.",
        "",
        "Machine-checkable contract: `transit_hrl/freq_hrl/core/spec.py`. The carrier package writes `spec_validation.json` so the frozen C1-C9 claim ledger and shared-core path audit can be verified without reading prose.",
        "",
        "## Frozen Interface",
        "",
        "| component | required contract |",
        "|---|---|",
        "| Exogenous stream | time-stamped, causal, no future observations |",
        "| Causal encoder | emits x_low, x_mid, x_high, low forecast, uncertainty, high energy, persistence |",
        "| Upper policy | consumes low trend, low forecast, uncertainty, promotion signal, leakage feedback, and endogenous plan state |",
        "| Lower policy | consumes active upper plan, local endogenous state, high residual, middle buffer, and shock age |",
        "| Promotion gate | fires only on persistent residual evidence and records false-positive/false-negative boundary |",
        "| Leakage accounting | measures upper high-frequency power and lower low-frequency drift |",
        "| Claim gate | paired, direction-aware CI or explicit noninferiority rule |",
        "",
        "## Non-Negotiable Invariants",
        "",
        "1. The encoder must be causal: no x_{t+1:T} can enter a policy decision at time t.",
        "2. Frequency features are not sufficient. The experiment must test routing responsibility, not only richer observations.",
        "3. Promotion is a replanning mechanism, not a free improvement label.",
        "4. No-tradeoff is domain-local: drift reduction and performance noninferiority must pass in the same domain.",
        "5. Domain-general means shared core plus domain adapters, not copy-pasted domain algorithms.",
        "",
        "## Frozen Claims",
        "",
        *_md_table(claim_freeze, ["claim_id", "status", "allowed_wording", "disallowed_wording"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shared_core(
    path: Path,
    rows: list[dict[str, Any]],
    shared_core_validation: dict[str, Any],
) -> None:
    adapter_rows = list(shared_core_validation.get("adapter_evidence", []))
    boundary = dict(shared_core_validation.get("core_boundary", {}) or {})
    lines = [
        "# Freq-HRL Shared-Core Audit",
        "",
        f"Date: {DATE_TAG}",
        "",
        "Audit question: do Transit and Quant evidence paths instantiate one Freq-HRL core, or two unrelated implementations? The answer should be reviewed through explicit adapter boundaries.",
        "",
        "Machine-checkable audits: `transit_hrl/results/carrier_upgrade_package_latest/spec_validation.json` validates shared-core artifact paths; `transit_hrl/results/carrier_upgrade_package_latest/shared_core_validation.json` checks that core/encoder/RL modules do not import domain code and that Quant/Transit adapters use the shared training entries.",
        "",
        *_md_table(rows, ["audit_item", "status", "path", "role", "next_upgrade"]),
        "",
        "## Source Boundary Audit",
        "",
        f"- status: `{shared_core_validation.get('status', 'unknown')}`",
        f"- checked core files: `{boundary.get('checked_files', 0)}`",
        f"- boundary violations: `{len(boundary.get('violations', []))}`",
        "",
        *_md_table(adapter_rows, ["adapter", "status", "required_symbol", "role", "evidence"]),
        "",
        "## Reviewer-Facing Boundary",
        "",
        (
            "The v2 shared-core claim is supported."
            if shared_core_validation.get("status") == "supported"
            else "The v2 shared-core migration is incomplete: trading uses the asynchronous SMDP kernel, while Transit surrogate/native adapters still use legacy joint-PPO entries. This is a paper blocker, not a supported shared-core claim."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_baseline(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Freq-HRL Baseline And Ablation Manifest",
        "",
        f"Date: {DATE_TAG}",
        "",
        "Purpose: separate genuine frequency-responsibility gains from gains caused by more parameters, more history, or more features.",
        "",
        *_md_table(rows, ["baseline", "tier", "headline_status", "sharpe_delta", "return_delta", "focus_delta", "n_common", "paper_role"]),
        "",
        "## Main-Table Rule",
        "",
        "The manuscript main table should include all `current` rows. `registered_strong_learned` rows (flat PPO/SAC/TD3 and generic learned HRL) must either be completed with paired evidence or explicitly moved to limitations; they cannot be silently counted as supported.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_real_data(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Freq-HRL Real-Data Scale-Up Plan",
        "",
        f"Date: {DATE_TAG}",
        "",
        "Purpose: move from public truth-source coverage and venue-grade replay paths toward deployment-grade external validation.",
        "",
        *_md_table(rows, ["domain", "upgrade", "current_status", "current_evidence", "minimum_next_dataset", "claim_boundary"]),
        "",
        "## Claim Discipline",
        "",
        "Current evidence supports protocol validation with public external truth-source coverage. It does not yet support a full deployment claim for one transit agency or production exchange venue.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_theory(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Freq-HRL Theory And Proof Appendix Skeleton",
        "",
        f"Date: {DATE_TAG}",
        "",
        "The target is not a universal convergence theorem. The target is a defensible set of sufficient-condition statements that support the method's claim boundaries.",
        "",
        *_md_table(rows, ["proof_item", "status", "statement", "paper_use", "assumption_count", "limitation"]),
        "",
        "## Suggested Assumptions",
        "",
        "A1. The exogenous process admits a causal approximate band decomposition with bounded reconstruction residual.",
        "A2. The upper action affects low-frequency plan variables more directly than high-frequency residual dynamics.",
        "A3. The lower action affects high-frequency correction more directly than long-horizon plan variables, up to measurable leakage.",
        "A4. Paired experiment seeds or source windows are exchangeable enough for direction-aware CI gates.",
        "",
        "## Proof Strategy",
        "",
        "1. Prove no-lookahead from encoder causality.",
        "2. Bound cross-level credit variance under band-routed observations.",
        "3. Bound responsibility leakage under upper-HPF and lower-LPF penalties.",
        "4. Derive promotion threshold false-positive and false-negative tradeoffs.",
        "5. Connect empirical paired-CI gates to conservative claim wording.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript(path: Path, claim_freeze: list[dict[str, Any]]) -> None:
    supported = sum(row.get("status") == "supported" for row in claim_freeze)
    lines = [
        "# Frequency-Separated Hierarchical Reinforcement Learning For Time-Series Control",
        "",
        f"Draft date: {DATE_TAG}",
        "",
        "## Abstract",
        "",
        "Many control problems are driven by exogenous time series that mix slow regime structure with fast residual disturbances. We introduce Freq-HRL, a frequency-separated hierarchical reinforcement learning protocol with causal routing, promotion-triggered replanning, and leakage accounting. The v2 evidence policy admits only observed raw outcomes to headline claim status; deterministic outcome projections remain sensitivity analyses. The current package is therefore a research implementation under confirmatory validation, not a completed domain-general performance result.",
        "",
        "## 1. Introduction",
        "",
        "The paper's core claim is narrow: frequency decomposition is not merely a representation trick; it is a control-responsibility principle for HRL. The low-frequency component should primarily shape plans, the high-frequency component should primarily shape local corrections, and persistent high-frequency evidence should trigger controlled replanning. The manuscript should not drift into a deployment paper or a universal RL-convergence paper.",
        "",
        "## Claim Boundary",
        "",
        "Allowed claim: Freq-HRL implements and evaluates frequency-responsibility routing under explicit raw-outcome validation boundaries.",
        "",
        "Disallowed claims: full same-agency Transit OD/onboard-load deployment validation, production exchange execution, universal encoder dominance, and universal nonconvex actor-critic convergence.",
        "",
        "## 2. Method",
        "",
        "Freq-HRL consists of a causal spectral encoder, upper planner, lower residual controller, promotion gate, leakage accounting, and paired claim-gating protocol. Domain adapters provide endogenous state and rollout semantics; the core interface remains domain-free.",
        "",
        "## 3. Experiments",
        "",
        "Experiments are organized around claim boundaries rather than isolated metrics: baseline/ablation evidence, native Transit promotion, public real-demand service response, leakage no-tradeoff, advanced encoders, stress regimes, and order-book replay infrastructure.",
        "",
        "## 4. Results",
        "",
        f"The raw-only matrix currently supports {supported} of {len(claim_freeze)} registered claims.",
        "",
        *_md_table(claim_freeze, ["claim_id", "status", "allowed_wording"]),
        "",
        "## 5. Discussion And Limitations",
        "",
        "The current evidence supports implementation and bounded mechanism claims. Raw native improvement, matched learned baselines, large real replay, and verified theory remain open.",
        "",
        "## Figure Plan",
        "",
        "Fig. 1: Frequency-separated protocol. Fig. 2: Claim and ablation matrix. Fig. 3: Transit promotion and real-demand service response. Fig. 4: External Transit data coverage. Fig. 5: Order-book replay and encoder generalization. SI: scheduler seeds, data scripts, paired-CI rules, and proof details.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repro(
    path: Path,
    commands: list[dict[str, Any]],
    artifact_manifest: list[dict[str, Any]],
    scheduler_seed_manifest: list[dict[str, Any]],
) -> None:
    lines = [
        "# Freq-HRL Reproducibility Package",
        "",
        f"Date: {DATE_TAG}",
        "",
        "This package records the commands required to regenerate the current claim matrix, manuscript tables, figures, and carrier-upgrade artifacts. Raw third-party caches remain ignored; regeneration commands must download or rebuild them explicitly.",
        "",
        *_md_table(commands, ["stage", "command", "output", "expected"]),
        "",
        "## Reproducibility Artifacts",
        "",
        *_md_table(artifact_manifest, ["artifact", "kind", "status", "count", "reproduce_stage", "commit_policy"]),
        "",
        "## Scheduler Seed Ledger",
        "",
        *_md_table(scheduler_seed_manifest, ["artifact", "role", "status", "seed_count", "first_seed", "last_seed", "seed_stride", "boundary"]),
        "",
        "## Artifact Policy",
        "",
        "Commit compact summaries, claim tables, figure source data, and manuscript figures. Do not commit large raw third-party files, scheduler scratch shards, or generated TIFFs unless required by a journal submission portal.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_carrier_upgrade_package(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    md_dir: Path = DEFAULT_MD_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, Any]:
    paths = _artifact_paths(results_root)
    claim_rows = _read_csv(paths["claims"])
    baseline_rows = _read_csv(paths["baseline_checks"])
    baseline_summary = _read_json(paths["baseline_summary"])
    external_truth = _read_json(paths["external_truth_summary"])
    agency = _read_json(paths["agency_summary"])
    order_book = _read_json(paths["order_book_summary"])
    theory_summary = _read_json(paths["theory_summary"]) or build_theory_payload(results_root)
    cs_experiment_payload = (
        _read_json(paths["cs_experiment_summary"])
        or build_cs_top_venue_experiment_matrix(results_root)
    )

    claim_freeze = build_claim_freeze(claim_rows)
    shared_core = build_shared_core_audit(source_root)
    shared_core_validation = audit_shared_training_core(source_root)
    baseline_manifest = build_baseline_manifest(baseline_rows, baseline_summary)
    data_scaleup = build_data_scaleup_manifest(external_truth, agency, order_book)
    proof_manifest = build_proof_manifest(theory_summary)
    repro_commands = build_repro_commands()
    scheduler_seed_manifest = build_scheduler_seed_manifest(results_root)
    repro_artifacts = build_repro_artifact_manifest(
        results_root,
        output_dir,
        scheduler_seed_manifest,
    )
    cs_top_venue_readiness = build_cs_top_venue_readiness(
        baseline_summary=baseline_summary,
        agency=agency,
        order_book=order_book,
        theory_summary=theory_summary,
        shared_core_validation=shared_core_validation,
        claim_freeze=claim_freeze,
        cs_experiment_summary=cs_experiment_payload,
    )
    cs_experiment_rows = [
        dict(row) for row in cs_experiment_payload.get("experiments", []) or []
        if isinstance(row, dict)
    ]
    cs_scheduler_rows = [
        dict(row) for row in cs_experiment_payload.get("scheduler_manifest", []) or []
        if isinstance(row, dict)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "claim_freeze.csv", claim_freeze)
    _write_csv(output_dir / "shared_core_audit.csv", shared_core)
    _write_csv(output_dir / "baseline_manifest.csv", baseline_manifest)
    _write_csv(output_dir / "data_scaleup_manifest.csv", data_scaleup)
    _write_csv(output_dir / "proof_manifest.csv", proof_manifest)
    _write_csv(output_dir / "reproducibility_commands.csv", repro_commands)
    _write_csv(output_dir / "scheduler_seed_manifest.csv", scheduler_seed_manifest)
    _write_csv(output_dir / "reproducibility_artifact_manifest.csv", repro_artifacts)
    _write_csv(output_dir / "cs_top_venue_readiness.csv", cs_top_venue_readiness)
    _write_csv(output_dir / "cs_top_venue_experiment_matrix.csv", cs_experiment_rows)
    _write_csv(output_dir / "cs_top_venue_scheduler_manifest.csv", cs_scheduler_rows)

    spec_validation = {
        "frozen_spec": default_spec().to_mapping(),
        "claim_freeze": validate_claim_freeze(claim_freeze),
        "shared_core": validate_shared_core_paths(shared_core, source_root=source_root),
    }

    md_paths = {
        key: md_dir / name
        for key, name in MD_NAMES.items()
    }
    write_carrier_plan(md_paths["carrier_plan"], claim_rows)
    write_algorithm_spec(md_paths["algorithm_spec"], claim_freeze)
    write_shared_core(md_paths["shared_core"], shared_core, shared_core_validation)
    write_baseline(md_paths["baseline"], baseline_manifest)
    write_real_data(md_paths["real_data"], data_scaleup)
    write_theory(md_paths["theory"], proof_manifest)
    write_manuscript(md_paths["manuscript"], claim_freeze)
    write_cs_venue_strategy(md_paths["cs_venue"], cs_top_venue_readiness)
    write_cs_experiments(md_paths["cs_experiments"], cs_experiment_payload)
    write_repro(md_paths["repro"], repro_commands, repro_artifacts, scheduler_seed_manifest)

    summary = {
        "date": DATE_TAG,
        "claims": len(claim_freeze),
        "supported_claims": sum(1 for row in claim_freeze if row.get("status") == "supported"),
        "component_inventory_present": sum(1 for row in shared_core if row.get("status") == "supported"),
        "component_inventory_total": len(shared_core),
        "baseline_rows": len(baseline_manifest),
        "data_scaleup_rows": len(data_scaleup),
        "proof_rows": len(proof_manifest),
        "repro_commands": len(repro_commands),
        "repro_artifacts": len(repro_artifacts),
        "scheduler_seed_rows": len(scheduler_seed_manifest),
        "cs_top_venue_readiness_rows": len(cs_top_venue_readiness),
        "cs_top_venue_experiment_rows": len(cs_experiment_rows),
        "cs_top_venue_blockers": [
            row["review_axis"]
            for row in cs_top_venue_readiness
            if row.get("current_status") == "blocker"
        ],
        "cs_top_venue_experiment_blockers": (
            cs_experiment_payload.get("summary", {}).get("blockers", [])
            if isinstance(cs_experiment_payload.get("summary"), dict)
            else []
        ),
        "output_dir": str(output_dir),
        "md_dir": str(md_dir),
        "documents": {key: str(path) for key, path in md_paths.items()},
        "manifests": {
            "claim_freeze": str(output_dir / "claim_freeze.csv"),
            "shared_core_audit": str(output_dir / "shared_core_audit.csv"),
            "baseline_manifest": str(output_dir / "baseline_manifest.csv"),
            "data_scaleup_manifest": str(output_dir / "data_scaleup_manifest.csv"),
            "proof_manifest": str(output_dir / "proof_manifest.csv"),
            "reproducibility_commands": str(output_dir / "reproducibility_commands.csv"),
            "reproducibility_artifact_manifest": str(output_dir / "reproducibility_artifact_manifest.csv"),
            "scheduler_seed_manifest": str(output_dir / "scheduler_seed_manifest.csv"),
            "cs_top_venue_readiness": str(output_dir / "cs_top_venue_readiness.csv"),
            "cs_top_venue_experiment_matrix": str(output_dir / "cs_top_venue_experiment_matrix.csv"),
            "cs_top_venue_scheduler_manifest": str(output_dir / "cs_top_venue_scheduler_manifest.csv"),
            "spec_validation": str(output_dir / "spec_validation.json"),
            "shared_core_validation": str(output_dir / "shared_core_validation.json"),
        },
        "spec_validation": {
            "version": spec_validation["frozen_spec"]["version"],
            "claim_freeze_status": spec_validation["claim_freeze"]["status"],
            "shared_core_path_inventory_status": spec_validation["shared_core"]["status"],
        },
        "shared_core_validation": {
            "status": shared_core_validation["status"],
            "checked_files": shared_core_validation["core_boundary"]["checked_files"],
            "adapter_entries": len(shared_core_validation["adapter_evidence"]),
            "boundary_violations": len(shared_core_validation["core_boundary"]["violations"]),
        },
        "boundary": (
            "Carrier upgrade package freezes the protocol and manuscript plan; it "
            "does not substitute for future same-agency Transit deployment or "
            "large multi-session market replay."
        ),
    }
    with (output_dir / "spec_validation.json").open("w", encoding="utf-8") as f:
        json.dump(spec_validation, f, indent=2)
    with (output_dir / "shared_core_validation.json").open("w", encoding="utf-8") as f:
        json.dump(shared_core_validation, f, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()
    summary = build_carrier_upgrade_package(
        results_root=args.results_root,
        output_dir=args.output_dir,
        md_dir=args.md_dir,
        source_root=args.source_root,
    )
    print(
        "carrier_upgrade_package "
        f"claims={summary['claims']} "
        f"docs={len(summary['documents'])} "
        f"output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
