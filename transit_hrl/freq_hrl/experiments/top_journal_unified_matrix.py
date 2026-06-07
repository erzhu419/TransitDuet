"""Unified top-journal evidence matrix for Freq-HRL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = {
    "native_promotion_v24": Path("transit_hrl/results/transit_native_promotion_pressure_guarded_wait_v24_2048seed_merged/summary.json"),
    "native_promotion_v21": Path("transit_hrl/results/transit_native_promotion_reward_guarded_projected_wait_v21_8192seed_w32x6_merged/summary.json"),
    "native_real_demand_v2": Path("transit_hrl/results/transit_native_real_demand_waitaware_v2_24seed_merged_drift/summary.json"),
    "order_book_manifest": Path("transit_hrl/results/scheduler_order_book_large_replay_manifest_fixture_smoke/summary.json"),
    "encoder_matrix": Path("transit_hrl/results/scheduler_encoder_cross_domain_matrix/summary.json"),
    "leakage_matrix": Path("transit_hrl/results/scheduler_leakage_no_tradeoff_matrix/summary.json"),
    "theory_appendix": Path("transit_hrl/results/scheduler_freq_hrl_theory_appendix/summary.json"),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _check_by_metric(data: dict[str, Any] | None, metric: str) -> dict[str, Any]:
    if not data:
        return {}
    return next((row for row in data.get("paired_checks", []) or [] if row.get("metric") == metric), {})


def _supported(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")) == "supported"


def _positive(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")) in {"supported", "positive_mixed", "noninferiority_supported"}


def _status_from_flags(*, present: bool, supported: bool, partial: bool) -> str:
    if not present:
        return "missing"
    if supported:
        return "supported"
    if partial:
        return "partial"
    return "not_supported"


def build_unified_matrix(results_root: Path) -> dict[str, Any]:
    artifacts = {
        key: _read_json(results_root / path.relative_to("transit_hrl/results"))
        for key, path in DEFAULT_ARTIFACTS.items()
    }
    paths = {
        key: str(results_root / path.relative_to("transit_hrl/results"))
        for key, path in DEFAULT_ARTIFACTS.items()
    }

    promotion = artifacts["native_promotion_v24"] or artifacts["native_promotion_v21"]
    promotion_reward = _check_by_metric(promotion, "ep_reward")
    promotion_wait = _check_by_metric(promotion, "avg_wait_min")
    if not promotion_wait:
        promotion_wait = _check_by_metric(promotion, "native_avg_board_wait_min")
    real = artifacts["native_real_demand_v2"]
    real_score = _check_by_metric(real, "control_score")
    real_reward = _check_by_metric(real, "ep_reward")
    real_wait = _check_by_metric(real, "native_avg_board_wait_min")
    real_alighted = _check_by_metric(real, "native_alighted_pax")
    order_book = artifacts["order_book_manifest"] or {}
    encoder = artifacts["encoder_matrix"] or {}
    leakage = artifacts["leakage_matrix"] or {}
    theory = artifacts["theory_appendix"] or {}

    encoder_domains = encoder.get("domain_summary", []) if isinstance(encoder, dict) else []
    encoder_supported_domains = [
        row.get("domain") for row in encoder_domains
        if int(row.get("supported", 0)) > 0
    ]
    leakage_verdicts = leakage.get("domain_verdicts", []) if isinstance(leakage, dict) else []
    leakage_supported_domains = [
        row.get("domain") for row in leakage_verdicts
        if row.get("verdict") == "no_tradeoff_supported"
    ]
    theory_examples = theory.get("examples", {}) if isinstance(theory, dict) else {}

    claims = [
        {
            "id": "C1",
            "claim": "Native learned promotion improves reward and wait",
            "status": _status_from_flags(
                present=bool(promotion),
                supported=_supported(promotion_reward) and _supported(promotion_wait),
                partial=_supported(promotion_reward) or _positive(promotion_wait),
            ),
            "evidence": (
                f"reward={promotion_reward.get('status', 'missing')} "
                f"wait={promotion_wait.get('status', 'missing')}"
            ),
            "remaining_gap": "Wait CI must be supported together with reward in the same native run.",
            "artifact": paths["native_promotion_v24"] if artifacts["native_promotion_v24"] else paths["native_promotion_v21"],
        },
        {
            "id": "C2",
            "claim": "Native real AFC/APC demand improves score/reward without wait/alighting loss",
            "status": _status_from_flags(
                present=bool(real),
                supported=_supported(real_score) and _supported(real_reward) and _supported(real_wait) and _positive(real_alighted),
                partial=_supported(real_score) and _supported(real_reward) and _positive(real_wait),
            ),
            "evidence": (
                f"score={real_score.get('status', 'missing')} "
                f"reward={real_reward.get('status', 'missing')} "
                f"wait={real_wait.get('status', 'missing')} "
                f"alighted={real_alighted.get('status', 'missing')}"
            ),
            "remaining_gap": "Alighting throughput and wait CI still need supported native real-demand evidence.",
            "artifact": paths["native_real_demand_v2"],
        },
        {
            "id": "C3",
            "claim": "Large L2/L3 order-book replay path exists",
            "status": _status_from_flags(
                present=bool(order_book),
                supported=bool(order_book.get("coverage", {}).get("l2_files", 0)) and bool(order_book.get("coverage", {}).get("l3_files", 0)),
                partial=bool(order_book),
            ),
            "evidence": str(order_book.get("coverage", {})),
            "remaining_gap": "Replace fixture manifest with larger real venue L2/L3 feeds.",
            "artifact": paths["order_book_manifest"],
        },
        {
            "id": "C4",
            "claim": "Advanced encoder evidence spans Quant and Transit",
            "status": _status_from_flags(
                present=bool(encoder_domains),
                supported=len(encoder_supported_domains) >= 4,
                partial=bool(encoder_supported_domains),
            ),
            "evidence": f"supported_domains={encoder_supported_domains}",
            "remaining_gap": "Public market needs paired multi-window CIs; L3 remains mixed.",
            "artifact": paths["encoder_matrix"],
        },
        {
            "id": "C5",
            "claim": "Leakage no-tradeoff holds beyond surrogate",
            "status": _status_from_flags(
                present=bool(leakage_verdicts),
                supported=len(leakage_supported_domains) >= 2,
                partial=bool(leakage_supported_domains),
            ),
            "evidence": f"no_tradeoff_domains={leakage_supported_domains}",
            "remaining_gap": "Native real-demand needs LowerLFDrift metrics and alighting-safe improvement.",
            "artifact": paths["leakage_matrix"],
        },
        {
            "id": "C6",
            "claim": "Formal theory appendix covers main protocol claims",
            "status": _status_from_flags(
                present=bool(theory),
                supported="primal_dual_avg_violation_bound_example" in theory_examples,
                partial=bool(theory_examples),
            ),
            "evidence": f"examples={sorted(theory_examples.keys())}",
            "remaining_gap": "Turn proof sketches into polished manuscript appendix text with assumptions near theorem statements.",
            "artifact": paths["theory_appendix"],
        },
    ]
    supported = sum(1 for row in claims if row["status"] == "supported")
    partial = sum(1 for row in claims if row["status"] == "partial")
    return {
        "claims": claims,
        "summary": {
            "claims": len(claims),
            "supported": supported,
            "partial": partial,
            "not_supported_or_missing": len(claims) - supported - partial,
        },
        "artifacts": paths,
        "boundary": "Unified matrix records current evidence quality; it is not itself a performance validation run.",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
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


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "claims.csv", payload["claims"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Freq-HRL Unified Top-Journal Evidence Matrix",
        "",
        payload["boundary"],
        "",
        f"- supported: `{payload['summary']['supported']}`",
        f"- partial: `{payload['summary']['partial']}`",
        f"- not supported or missing: `{payload['summary']['not_supported_or_missing']}`",
        "",
        "| id | claim | status | evidence | remaining gap |",
        "|---|---|---|---|---|",
    ]
    for row in payload["claims"]:
        lines.append(
            f"| {row['id']} | {row['claim']} | {row['status']} "
            f"| {row['evidence']} | {row['remaining_gap']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("transit_hrl/results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/top_journal_unified_matrix"),
    )
    args = parser.parse_args()
    payload = build_unified_matrix(Path(args.results_root))
    write_outputs(Path(args.output_dir), payload)
    print(
        "top_journal_unified_matrix "
        f"supported={payload['summary']['supported']} partial={payload['summary']['partial']}"
    )


if __name__ == "__main__":
    main()
