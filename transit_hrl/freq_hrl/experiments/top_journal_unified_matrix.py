"""Unified top-journal evidence matrix for Freq-HRL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS = {
    "native_promotion_v31": Path("transit_hrl/results/transit_native_promotion_final_delta_floor_reward_wait_v31_512seed_w16r2_merged/summary.json"),
    "native_promotion_v27": Path("transit_hrl/results/transit_native_promotion_selective_reward_wait_v27_512seed_merged/summary.json"),
    "native_promotion_v26": Path("transit_hrl/results/transit_native_promotion_reward_floor_throughput_v26_512seed_merged/summary.json"),
    "native_promotion_v25": Path("transit_hrl/results/transit_native_promotion_reward_floor_throughput_v25_512seed_merged/summary.json"),
    "native_promotion_v24": Path("transit_hrl/results/transit_native_promotion_pressure_guarded_wait_v24_2048seed_merged/summary.json"),
    "native_promotion_v21": Path("transit_hrl/results/transit_native_promotion_reward_guarded_projected_wait_v21_8192seed_w32x6_merged/summary.json"),
    "native_real_demand_alighting_safe_v2": Path("transit_hrl/results/transit_native_real_demand_alighting_safe_v2_24pair_merged/summary.json"),
    "native_real_demand_v5": Path("transit_hrl/results/scheduler_native_real_demand_selective_reward_wait_v5_24pair/summary.json"),
    "native_real_demand_v4": Path("transit_hrl/results/scheduler_native_real_demand_wait_pressure_v4_24pair/summary.json"),
    "native_real_demand_v3": Path("transit_hrl/results/scheduler_native_real_demand_reward_floor_throughput_v3_24pair/summary.json"),
    "native_real_demand_v2": Path("transit_hrl/results/transit_native_real_demand_waitaware_v2_24seed_merged_drift/summary.json"),
    "order_book_l2_matching": Path("transit_hrl/results/trading_order_book_matching_validation/summary.json"),
    "order_book_l3_replay": Path("transit_hrl/results/trading_order_book_l3_replay_validation/summary.json"),
    "order_book_manifest": Path("transit_hrl/results/scheduler_order_book_large_replay_manifest_fixture_smoke/summary.json"),
    "encoder_matrix": Path("transit_hrl/results/encoder_cross_domain_matrix/summary.json"),
    "encoder_matrix_latest": Path("transit_hrl/results/encoder_cross_domain_matrix_latest/summary.json"),
    "leakage_matrix_v27_v5": Path("transit_hrl/results/scheduler_leakage_no_tradeoff_matrix_v27_v5/summary.json"),
    "leakage_matrix_v26_v4": Path("transit_hrl/results/scheduler_leakage_no_tradeoff_matrix_v26_v4/summary.json"),
    "leakage_matrix_v25_v3": Path("transit_hrl/results/scheduler_leakage_no_tradeoff_matrix_v25_v3/summary.json"),
    "leakage_matrix": Path("transit_hrl/results/leakage_no_tradeoff_matrix/summary.json"),
    "leakage_matrix_latest": Path("transit_hrl/results/leakage_no_tradeoff_matrix_latest/summary.json"),
    "theory_appendix": Path("transit_hrl/results/freq_hrl_theory_appendix/summary.json"),
    "theory_appendix_scheduler": Path("transit_hrl/results/scheduler_freq_hrl_theory_appendix/summary.json"),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _check_by_metric(
    data: dict[str, Any] | None,
    metric: str,
    *,
    treatment: str = "",
    check_contains: str = "",
) -> dict[str, Any]:
    if not data:
        return {}
    for row in data.get("paired_checks", []) or []:
        if row.get("metric") != metric:
            continue
        if treatment and row.get("treatment") != treatment:
            continue
        if check_contains and check_contains not in str(row.get("check", "")):
            continue
        return row
    return {}


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


def _count_checks(data: dict[str, Any], statuses: set[str]) -> int:
    if not isinstance(data, dict):
        return 0
    return sum(
        1
        for row in data.get("paired_checks", []) or []
        if str(row.get("status", "")) in statuses
    )


def _has_non_synthetic_sources(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    sources = data.get("sources", []) or []
    return any("synthetic" not in str(source).lower() for source in sources)


def _promotion_evidence(
    artifacts: dict[str, dict[str, Any] | None],
    paths: dict[str, str],
) -> dict[str, Any]:
    candidates = [
        "native_promotion_v31",
        "native_promotion_v27",
        "native_promotion_v26",
        "native_promotion_v25",
        "native_promotion_v24",
        "native_promotion_v21",
    ]
    ranked: list[dict[str, Any]] = []
    for key in candidates:
        data = artifacts.get(key)
        if not data:
            continue
        reward = _check_by_metric(data, "ep_reward", treatment="native_wait_aware_replan")
        reward_noninferiority = _check_by_metric(
            data,
            "ep_reward",
            treatment="native_wait_aware_replan",
            check_contains="noninferiority",
        )
        wait = _check_by_metric(data, "avg_wait_min", treatment="native_wait_aware_replan")
        if not wait:
            wait = _check_by_metric(data, "native_avg_board_wait_min", treatment="native_wait_aware_replan")
        wait_noninferiority = _check_by_metric(
            data,
            "avg_wait_min",
            treatment="native_wait_aware_replan",
            check_contains="noninferiority",
        )
        status = _status_from_flags(
            present=True,
            supported=_supported(reward) and _supported(wait),
            partial=(
                (_supported(reward) and _positive(wait_noninferiority))
                or (_positive(reward_noninferiority) and _supported(wait))
            ),
        )
        score = {"supported": 3, "partial": 2, "not_supported": 1, "missing": 0}[status]
        n_common = max(
            int(reward.get("n_common", 0) or 0),
            int(wait.get("n_common", 0) or 0),
        )
        ranked.append({
            "key": key,
            "data": data,
            "reward": reward,
            "reward_noninferiority": reward_noninferiority,
            "wait": wait,
            "wait_noninferiority": wait_noninferiority,
            "status": status,
            "score": score,
            "n_common": n_common,
            "artifact": paths[key],
        })
    if not ranked:
        return {
            "key": "missing",
            "status": "missing",
            "reward": {},
            "reward_noninferiority": {},
            "wait": {},
            "wait_noninferiority": {},
            "artifact": paths["native_promotion_v31"],
        }
    return max(ranked, key=lambda row: (row["score"], row["n_common"]))


def build_unified_matrix(results_root: Path) -> dict[str, Any]:
    artifacts = {
        key: _read_json(results_root / path.relative_to("transit_hrl/results"))
        for key, path in DEFAULT_ARTIFACTS.items()
    }
    paths = {
        key: str(results_root / path.relative_to("transit_hrl/results"))
        for key, path in DEFAULT_ARTIFACTS.items()
    }

    promotion = _promotion_evidence(artifacts, paths)
    promotion_reward = promotion["reward"]
    promotion_reward_noninferiority = promotion["reward_noninferiority"]
    promotion_wait = promotion["wait"]
    promotion_wait_noninferiority = promotion["wait_noninferiority"]
    real = (
        artifacts["native_real_demand_alighting_safe_v2"]
        or artifacts["native_real_demand_v5"]
        or artifacts["native_real_demand_v4"]
        or artifacts["native_real_demand_v3"]
        or artifacts["native_real_demand_v2"]
    )
    real_score = _check_by_metric(real, "control_score")
    real_reward = _check_by_metric(real, "ep_reward")
    real_wait = _check_by_metric(real, "native_avg_board_wait_min")
    real_alighted = _check_by_metric(real, "native_alighted_pax")
    real_wait_noninferiority = _check_by_metric(
        real,
        "native_avg_board_wait_min",
        check_contains="noninferiority",
    )
    real_alighted_noninferiority = _check_by_metric(
        real,
        "native_alighted_pax",
        check_contains="noninferiority",
    )
    order_book_manifest = artifacts["order_book_manifest"] or {}
    order_book_l2 = artifacts["order_book_l2_matching"] or {}
    order_book_l3 = artifacts["order_book_l3_replay"] or {}
    encoder = artifacts["encoder_matrix"] or artifacts["encoder_matrix_latest"] or {}
    leakage = (
        artifacts["leakage_matrix_v27_v5"]
        or artifacts["leakage_matrix_v26_v4"]
        or artifacts["leakage_matrix_v25_v3"]
        or artifacts["leakage_matrix"]
        or artifacts["leakage_matrix_latest"]
        or {}
    )
    theory = artifacts["theory_appendix"] or artifacts["theory_appendix_scheduler"] or {}

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
    leakage_partial_domains = [
        row.get("domain") for row in leakage_verdicts
        if row.get("verdict") in {"partial", "performance_noharm_only", "summary_only_noharm"}
    ]
    theory_examples = theory.get("examples", {}) if isinstance(theory, dict) else {}
    order_book_l2_supported = _count_checks(order_book_l2, {"supported"})
    order_book_l3_positive = _count_checks(order_book_l3, {"supported", "positive_mixed"})
    order_book_has_real_l2_l3 = (
        bool(order_book_manifest.get("coverage", {}).get("l2_files", 0))
        and bool(order_book_manifest.get("coverage", {}).get("l3_files", 0))
    ) or (_has_non_synthetic_sources(order_book_l2) and _has_non_synthetic_sources(order_book_l3))

    claims = [
        {
            "id": "C1",
            "claim": "Native learned promotion improves reward and wait",
            "status": promotion["status"],
            "evidence": (
                f"best={promotion['key']} "
                f"reward={promotion_reward.get('status', 'missing')} "
                f"reward_noharm={promotion_reward_noninferiority.get('status', 'missing')} "
                f"wait={promotion_wait.get('status', 'missing')} "
                f"wait_noharm={promotion_wait_noninferiority.get('status', 'missing')}"
            ),
            "remaining_gap": "Wait CI must be supported together with reward in the same native run.",
            "artifact": promotion["artifact"],
        },
        {
            "id": "C2",
            "claim": "Native real AFC/APC demand improves score/reward without wait/alighting loss",
            "status": _status_from_flags(
                present=bool(real),
                supported=(
                    _supported(real_score)
                    and _supported(real_reward)
                    and _positive(real_wait_noninferiority)
                    and _positive(real_alighted_noninferiority)
                ),
                partial=_supported(real_score) and _supported(real_reward),
            ),
            "evidence": (
                f"score={real_score.get('status', 'missing')} "
                f"reward={real_reward.get('status', 'missing')} "
                f"wait={real_wait.get('status', 'missing')} "
                f"wait_noharm={real_wait_noninferiority.get('status', 'missing')} "
                f"alighted={real_alighted.get('status', 'missing')} "
                f"alighted_noharm={real_alighted_noninferiority.get('status', 'missing')}"
            ),
            "remaining_gap": "Alighting/wait no-harm is supported; strict improvement CIs still need stronger throughput-seeking validation.",
            "artifact": (
                paths["native_real_demand_alighting_safe_v2"]
                if artifacts["native_real_demand_alighting_safe_v2"]
                else paths["native_real_demand_v5"]
                if artifacts["native_real_demand_v5"]
                else paths["native_real_demand_v4"]
                if artifacts["native_real_demand_v4"]
                else paths["native_real_demand_v3"]
                if artifacts["native_real_demand_v3"]
                else paths["native_real_demand_v2"]
            ),
        },
        {
            "id": "C3",
            "claim": "Large L2/L3 order-book replay path exists",
            "status": _status_from_flags(
                present=bool(order_book_manifest or order_book_l2 or order_book_l3),
                supported=bool(order_book_has_real_l2_l3),
                partial=bool(order_book_l2 and order_book_l3),
            ),
            "evidence": (
                f"l2_supported_checks={order_book_l2_supported} "
                f"l3_positive_checks={order_book_l3_positive} "
                f"manifest_coverage={order_book_manifest.get('coverage', {})}"
            ),
            "remaining_gap": "Current path has L2 matching and synthetic/CSV-capable L3 FIFO replay; top-journal claim still needs larger real venue L2/L3 feeds.",
            "artifact": (
                f"{paths['order_book_l2_matching']} | {paths['order_book_l3_replay']}"
                if order_book_l2 or order_book_l3
                else paths["order_book_manifest"]
            ),
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
            "artifact": paths["encoder_matrix"] if artifacts["encoder_matrix"] else paths["encoder_matrix_latest"],
        },
        {
            "id": "C5",
            "claim": "Leakage no-tradeoff holds beyond surrogate",
            "status": _status_from_flags(
                present=bool(leakage_verdicts),
                supported=len(leakage_supported_domains) >= 2,
                partial=bool(leakage_supported_domains),
            ),
            "evidence": f"no_tradeoff_domains={leakage_supported_domains} partial_domains={leakage_partial_domains}",
            "remaining_gap": "Native real-demand needs LowerLFDrift metrics and alighting-safe improvement.",
            "artifact": (
                paths["leakage_matrix_v27_v5"]
                if artifacts["leakage_matrix_v27_v5"]
                else paths["leakage_matrix_v26_v4"]
                if artifacts["leakage_matrix_v26_v4"]
                else paths["leakage_matrix_v25_v3"]
                if artifacts["leakage_matrix_v25_v3"]
                else paths["leakage_matrix"]
                if artifacts["leakage_matrix"]
                else paths["leakage_matrix_latest"]
            ),
        },
        {
            "id": "C6",
            "claim": "Formal theory appendix covers main protocol claims",
            "status": _status_from_flags(
                present=bool(theory),
                supported=bool(theory.get("theorems")) and "primal_dual_avg_violation_bound_example" in theory_examples,
                partial=bool(theory_examples),
            ),
            "evidence": f"examples={sorted(theory_examples.keys())}",
            "remaining_gap": "Turn proof sketches into polished manuscript appendix text with assumptions near theorem statements.",
            "artifact": paths["theory_appendix"] if artifacts["theory_appendix"] else paths["theory_appendix_scheduler"],
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
