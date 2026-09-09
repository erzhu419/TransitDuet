#!/usr/bin/env python3
"""Render source-bound Protocol V6 method, mechanism, and realism figures."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_external_afc_apc_profiles import (
    DEFAULT_AFC,
    DEFAULT_APC,
    DEFAULT_OD,
    build_alignment,
    load_afc,
    load_apc,
    load_freqduet_od,
)
from build_freqduet_protocol_v6_evidence_package import (
    CONFIRMED_SOURCE_CONFIG,
    NOGUARD_REFERENCE,
    PAPER_CONTROLLER,
    PROTOCOL,
    config_lineage,
    refresh_package_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT / "results_freqduet" / "paper_package" / "protocol_v6_current_best"
)
DEFAULT_FORMATS = ("svg", "pdf", "tiff", "png")
FIGURE_WIDTH_MM = 183.0

TEAL = "#187C73"
DEEP_TEAL = "#0E5E58"
ORANGE = "#D97732"
AMBER = "#B9832F"
BLUE = "#356B9A"
GRAY = "#68717A"
LIGHT_GRAY = "#D9DEE2"
PALE_TEAL = "#E6F2F0"
PALE_ORANGE = "#FAEDE4"
PALE_BLUE = "#E9F0F7"
PALE_GRAY = "#F2F4F5"
BLACK = "#202428"

MECHANISM_METRICS = (
    (
        "holding_vehicle_seconds_per_launched_trip",
        "Vehicle holding",
        "s per launched trip",
        1.0,
    ),
    (
        "holding_passenger_min_per_generated",
        "Passenger holding",
        "min per generated passenger",
        1.0,
    ),
    (
        "fleet_denied_trip_rate",
        "Denied trips",
        "percentage-point difference",
        100.0,
    ),
    (
        "terminal_dispatch_execution_error_abs_mean_s",
        "Terminal execution error",
        "absolute error difference (s)",
        1.0,
    ),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_snapshot_config(config_root: Path) -> tuple[dict[str, Any], list[str]]:
    path = config_root / "configs_freqduet" / f"{PAPER_CONTROLLER}.yaml"
    lineage = config_lineage(path, config_root)
    resolved: dict[str, Any] = {}
    labels: list[str] = []
    for item in lineage:
        payload = yaml.safe_load(item.read_text()) or {}
        payload.pop("_extends", None)
        resolved = _deep_merge(resolved, payload)
        labels.append(str(item.relative_to(config_root)))
    return resolved, labels


def nested(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"method contract lacks {path}")
        value = value[key]
    return value


def validate_method_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol.version": PROTOCOL,
        "env.fleet_inventory_mode": "fixed_pool",
        "env.observation_contract": "deployable_apc_avl_v4",
        "env.service_start_hour": 6,
        "env.service_end_hour": 19,
        "env.clearance_time_s": 14400,
        "upper.fleet_mode": "fixed",
        "upper.N_fleet": 12,
        "upper.algorithm_id": "pessimistic_ensemble_sac_v4",
        "upper.ensemble_size": 10,
        "upper.timetable_planner.enable": True,
        "upper.timetable_planner.terminal_schedule_mode": "exact_headway_curve",
        "upper.timetable_planner.terminal_dispatch": True,
        "upper.timetable_planner.replan_interval_s": 900.0,
        "upper.timetable_planner.horizon_s": 2700.0,
        "upper.timetable_planner.headway_budget_mode": "rolling_zero_sum_delta_v6",
        "frequency.enable": True,
        "frequency.method": "harmonic",
        "frequency.use_historical_prior": True,
        "frequency.bin_sec": 60.0,
        "frequency.fourier_K": 4,
        "frequency.harmonic_forgetting": 0.9995,
        "frequency.harmonic_prior_var": 0.01,
        "frequency.harmonic_ridge": 0.01,
        "frequency.forecast_horizon_s": 1800.0,
        "frequency.upper_mode": "low",
        "frequency.lower_mode": "high",
        "frequency.promotion.enable": False,
        "leakage.enable": False,
        "lower.algorithm_id": "pessimistic_ensemble_sac_lagrangian_v4",
        "lower.ensemble_size": 10,
        "lower.causal_holding_guard.enable": False,
        "lower.causal_departure_regularity.enable": True,
        "lower.causal_departure_regularity.evidence_mode": (
            "pre_action_departure_v6"
        ),
        "lower.causal_departure_regularity.objective_mode": (
            "avl_two_sided_incremental_reward"
        ),
        "lower.causal_departure_regularity.reward_weight": 2.0,
        "lower.causal_departure_regularity.tolerance_fraction": 0.02,
        "lower.causal_departure_regularity.cost_cap": 0.25,
        "objective.wait_metric": "restricted",
        "objective.weights.wait": 1.0,
        "objective.weights.fleet": 1.0,
        "objective.weights.headway": 1.0,
        "objective.weights.unserved": 5.0,
        "objective.weights.incomplete_service": 5.0,
    }
    observed = {path: nested(config, path) for path in expected}
    mismatches = {
        path: {"expected": expected[path], "observed": observed[path]}
        for path in expected
        if observed[path] != expected[path]
    }
    if mismatches:
        raise ValueError(f"current method contract drifted: {mismatches}")

    required_context = {
        "load",
        "capacity",
        "queue",
        "speed_residual",
        "shock_age",
        "schedule_slack",
        "regularity_hold_target_norm",
        "regularity_hold_target_valid",
    }
    context_features = list(nested(config, "frequency.lower_context.features"))
    if not required_context.issubset(context_features):
        missing = sorted(required_context - set(context_features))
        raise ValueError(f"compact APC/AVL context lacks {missing}")
    action_bins = [float(value) for value in nested(config, "lower.action_bins")]
    if action_bins != [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]:
        raise ValueError("current lower action alphabet drifted")

    return {
        "paper_controller": PAPER_CONTROLLER,
        "protocol": PROTOCOL,
        "service_start_hour": int(nested(config, "env.service_start_hour")),
        "service_end_hour": int(nested(config, "env.service_end_hour")),
        "clearance_time_s": float(nested(config, "env.clearance_time_s")),
        "fleet_size": int(nested(config, "upper.N_fleet")),
        "historical_prior": True,
        "frequency_method": nested(config, "frequency.method"),
        "harmonic_period_s": float(nested(config, "frequency.harmonic_period_s")),
        "fourier_k": int(nested(config, "frequency.fourier_K")),
        "harmonic_forgetting": float(
            nested(config, "frequency.harmonic_forgetting")
        ),
        "harmonic_prior_var": float(
            nested(config, "frequency.harmonic_prior_var")
        ),
        "harmonic_ridge": float(nested(config, "frequency.harmonic_ridge")),
        "bin_sec": float(nested(config, "frequency.bin_sec")),
        "forecast_horizon_s": float(
            nested(config, "frequency.forecast_horizon_s")
        ),
        "upper_frequency_authority": nested(config, "frequency.upper_mode"),
        "lower_frequency_authority": nested(config, "frequency.lower_mode"),
        "replan_interval_s": float(
            nested(config, "upper.timetable_planner.replan_interval_s")
        ),
        "planning_horizon_s": float(
            nested(config, "upper.timetable_planner.horizon_s")
        ),
        "headway_budget_mode": nested(
            config, "upper.timetable_planner.headway_budget_mode"
        ),
        "upper_delta_min_s": float(
            nested(config, "upper.timetable_planner.delta_min_s")
        ),
        "upper_delta_max_s": float(
            nested(config, "upper.timetable_planner.delta_max_s")
        ),
        "terminal_shift_min_s": float(
            nested(config, "upper.timetable_planner.terminal_shift_min_s")
        ),
        "terminal_shift_max_s": float(
            nested(config, "upper.timetable_planner.terminal_shift_max_s")
        ),
        "terminal_dispatch": True,
        "action_bins_s": action_bins,
        "uses_last_action_feature": bool(
            nested(config, "lower.use_last_action_feature")
        ),
        "lower_context_features": context_features,
        "regularity_objective": nested(
            config, "lower.causal_departure_regularity.objective_mode"
        ),
        "regularity_reward_weight": float(
            nested(config, "lower.causal_departure_regularity.reward_weight")
        ),
        "regularity_tolerance_fraction": float(
            nested(config, "lower.causal_departure_regularity.tolerance_fraction")
        ),
        "regularity_cost_cap": float(
            nested(config, "lower.causal_departure_regularity.cost_cap")
        ),
        "objective_wait_metric": nested(config, "objective.wait_metric"),
        "service_cost_weights": {
            name: float(nested(config, f"objective.weights.{name}"))
            for name in (
                "wait",
                "fleet",
                "headway",
                "unserved",
                "incomplete_service",
            )
        },
        "upper_algorithm": nested(config, "upper.algorithm_id"),
        "upper_hidden_dim": int(nested(config, "upper.hidden_dim")),
        "upper_ensemble_size": int(nested(config, "upper.ensemble_size")),
        "upper_learning_rate": float(nested(config, "upper.lr")),
        "upper_discount": float(nested(config, "upper.gamma")),
        "upper_batch_size": int(nested(config, "upper.batch_size")),
        "upper_updates_per_episode": int(
            nested(config, "upper.updates_per_episode")
        ),
        "lower_algorithm": nested(config, "lower.algorithm_id"),
        "lower_hidden_dim": int(nested(config, "lower.hidden_dim")),
        "lower_ensemble_size": int(nested(config, "lower.ensemble_size")),
        "lower_learning_rate": float(nested(config, "lower.lr")),
        "lower_dual_learning_rate": float(nested(config, "lower.lambda_lr")),
        "lower_discount": float(nested(config, "lower.gamma")),
        "lower_batch_size": int(nested(config, "lower.batch_size")),
        "lower_updates_per_episode": int(
            nested(config, "lower.updates_per_episode")
        ),
        "upper_warmup_episodes": int(
            nested(config, "coupling.upper_warmup_eps")
        ),
        "legacy_holding_guard_enabled": False,
        "promotion_enabled": False,
        "leakage_penalty_enabled": False,
    }


def read_one_pair(path: Path, candidate: str, reference: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("candidate") == candidate and row.get("reference") == reference
        ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one paired row for {candidate} vs {reference}, found {len(rows)}"
        )
    return rows[0]


def mechanism_rows(package_dir: Path) -> list[dict[str, Any]]:
    phases = (
        (
            "V8 (40 episodes)",
            package_dir / "source_artifacts" / "v8" / "frozen_paired_deltas.csv",
            CONFIRMED_SOURCE_CONFIG,
            24,
        ),
        (
            "V9 (200 episodes)",
            package_dir / "source_artifacts" / "v9" / "frozen_paired_deltas.csv",
            PAPER_CONTROLLER,
            64,
        ),
    )
    result: list[dict[str, Any]] = []
    for phase, path, candidate, expected_n in phases:
        row = read_one_pair(path, candidate, NOGUARD_REFERENCE)
        if int(row["n_pairs"]) != expected_n:
            raise ValueError(f"{phase} mechanism sample size drifted")
        for source_metric, label, unit, scale in MECHANISM_METRICS:
            prefix = f"delta_{source_metric}"
            required = [f"{prefix}_mean", f"{prefix}_ci_low", f"{prefix}_ci_high"]
            if any(not row.get(key) for key in required):
                raise ValueError(f"{phase} lacks mechanism metric {source_metric}")
            result.append({
                "phase": phase,
                "candidate": candidate,
                "reference": NOGUARD_REFERENCE,
                "metric": source_metric,
                "label": label,
                "unit": unit,
                "scale": scale,
                "mean": float(row[f"{prefix}_mean"]) * scale,
                "ci95_low": float(row[f"{prefix}_ci_low"]) * scale,
                "ci95_high": float(row[f"{prefix}_ci_high"]) * scale,
                "n_pairs": expected_n,
                "difference": "confirmed_policy_minus_protocol_reference",
                "lower_is_better": True,
            })
    return result


def prepare_realism_data(
    afc_path: Path,
    apc_path: Path,
    od_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    afc, afc_coverage = load_afc(afc_path)
    apc, apc_coverage = load_apc(apc_path)
    od, od_coverage = load_freqduet_od(od_path)
    profile = pd.concat([afc, apc, od], ignore_index=True)
    hourly = (
        profile.groupby(["source", "hour_floor"], as_index=False)["demand"]
        .sum()
        .rename(columns={"hour_floor": "hour"})
    )
    totals = hourly.groupby("source")["demand"].transform("sum")
    hourly["share"] = hourly["demand"] / totals
    coverage = pd.DataFrame([afc_coverage, apc_coverage, od_coverage])
    alignment = build_alignment(profile)
    return hourly, coverage, alignment


def read_balanced_cache_manifest(
    afc_path: Path,
    apc_path: Path,
) -> dict[str, Any] | None:
    if afc_path.parent.resolve() != apc_path.parent.resolve():
        return None
    path = afc_path.parent / "derivation_manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("manifest_version") != (
        "freqduet-balanced-external-profile-cache-v1"
    ):
        raise ValueError("Figure 5 balanced-cache manifest version drifted")
    sources = payload.get("sources", {})
    expected = {
        "public_afc_mta": afc_path.name,
        "public_apc_halifax": apc_path.name,
    }
    observed = {
        source: sources.get(source, {}).get("output_file")
        for source in expected
    }
    if observed != expected:
        raise ValueError("Figure 5 balanced-cache files do not match manifest")
    return payload


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: tuple[str, ...],
) -> list[str]:
    outputs: list[str] = []
    for extension in formats:
        path = out_dir / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {}
        if extension == "tiff":
            kwargs = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}
        elif extension == "png":
            kwargs = {"dpi": 300}
        fig.savefig(path, **kwargs)
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.018,
        y + height - 0.045,
        title,
        fontsize=6.8,
        fontweight="bold",
        color=BLACK,
        va="top",
    )
    ax.text(
        x + 0.018,
        y + height - 0.10,
        body,
        fontsize=5.6,
        color=BLACK,
        linespacing=1.3,
        va="top",
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GRAY,
    style: str = "-",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.15,
        linestyle=style,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def method_figure(
    contract: Mapping[str, Any],
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_MM / 25.4, 105.0 / 25.4))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.02, 0.64),
        0.17,
        0.24,
        "Historical demand",
        "OD intensity profile\ninitializes harmonic prior",
        PALE_GRAY,
        GRAY,
    )
    add_box(
        ax,
        (0.02, 0.24),
        0.17,
        0.24,
        "Online observations",
        "Causal APC arrivals\nSame-time AVL state",
        PALE_GRAY,
        GRAY,
    )
    add_box(
        ax,
        (0.235, 0.36),
        0.21,
        0.35,
        "Causal harmonic split",
        (
            f"{contract['bin_sec']:.0f} s count bins; RLS Fourier K="
            f"{contract['fourier_k']}\n\nLF: level, slope, forecast\n"
            "HF: innovation, energy"
        ),
        PALE_BLUE,
        BLUE,
    )
    add_box(
        ax,
        (0.49, 0.63),
        0.215,
        0.27,
        "Upper LF state",
        (
            "Demand level and slope\n"
            f"{contract['forecast_horizon_s'] / 60:.0f} min causal forecast\n"
            "OD structure\nFleet readiness"
        ),
        PALE_TEAL,
        TEAL,
    )
    add_box(
        ax,
        (0.755, 0.63),
        0.225,
        0.27,
        "Executable headway plan",
        (
            f"{contract['replan_interval_s'] / 60:.0f} min replanning\n"
            f"{contract['planning_horizon_s'] / 60:.0f} min horizon\n"
            "Rolling zero-sum deltas\nTerminal launch execution"
        ),
        PALE_TEAL,
        TEAL,
    )
    add_box(
        ax,
        (0.49, 0.24),
        0.215,
        0.27,
        "Lower HF + APC/AVL state",
        "Local innovation + energy\nLoad, queue, speed residual\nForward/follower gaps",
        PALE_ORANGE,
        ORANGE,
    )
    add_box(
        ax,
        (0.755, 0.24),
        0.225,
        0.27,
        "Discrete stop holding",
        "Actions: 0, 5, 10, 15,\n20, 30, or 45 s\nPrevious action in state",
        PALE_ORANGE,
        ORANGE,
    )

    add_arrow(ax, (0.19, 0.76), (0.235, 0.61), BLUE)
    add_arrow(ax, (0.19, 0.36), (0.235, 0.49), BLUE)
    add_arrow(ax, (0.445, 0.61), (0.49, 0.76), TEAL)
    add_arrow(ax, (0.445, 0.47), (0.49, 0.37), ORANGE)
    add_arrow(ax, (0.705, 0.765), (0.755, 0.765), TEAL)
    add_arrow(ax, (0.705, 0.375), (0.755, 0.375), ORANGE)
    add_arrow(ax, (0.867, 0.63), (0.867, 0.515), GRAY, "--")
    ax.text(0.878, 0.565, "target headway", fontsize=5.2, color=GRAY, va="center")

    objective = FancyBboxPatch(
        (0.49, 0.035),
        0.49,
        0.12,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=GRAY,
        facecolor="white",
    )
    ax.add_patch(objective)
    ax.text(
        0.507,
        0.125,
        "Pre-action causal objective",
        fontsize=6.8,
        fontweight="bold",
        color=BLACK,
        va="top",
    )
    ax.text(
        0.507,
        0.087,
        "Freeze forward/follower gaps before action.\nReward two-sided regularity change; legacy guard, promotion, and leakage are off.",
        fontsize=5.4,
        color=BLACK,
        va="top",
    )
    add_arrow(ax, (0.87, 0.24), (0.82, 0.155), GRAY, "--")
    ax.text(0.49, 0.935, "LOW-FREQUENCY AUTHORITY", color=DEEP_TEAL,
            fontsize=6.4, fontweight="bold")
    ax.text(0.49, 0.545, "HIGH-FREQUENCY AUTHORITY", color=ORANGE,
            fontsize=6.4, fontweight="bold")
    return save_figure(
        fig,
        out_dir,
        "fig1_protocol_v6_method",
        formats,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def mechanism_figure(
    rows: list[dict[str, Any]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH_MM / 25.4, 106.0 / 25.4),
        layout="constrained",
    )
    phase_order = ("V8 (40 episodes)", "V9 (200 episodes)")
    colors = (TEAL, AMBER)
    for index, (ax, metric_spec) in enumerate(zip(axes.flat, MECHANISM_METRICS)):
        metric, title, unit, _ = metric_spec
        selected = [next(
            row for row in rows
            if row["phase"] == phase and row["metric"] == metric
        ) for phase in phase_order]
        for y, row, color in zip((1, 0), selected, colors):
            mean = float(row["mean"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=5.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                ecolor=color,
                elinewidth=1.3,
                capsize=2.8,
                zorder=3,
            )
        ax.axvline(0.0, color=BLACK, linestyle="--", linewidth=0.8)
        ax.set_yticks((1, 0), ("V8, n=24", "V9, n=64"))
        ax.set_ylim(-0.55, 1.55)
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.set_xlabel(f"Current policy - protocol reference ({unit})")
        ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_label(ax, chr(ord("a") + index))
    return save_figure(
        fig,
        out_dir,
        "fig4_protocol_v6_physical_outcomes",
        formats,
    )


def realism_figure(
    profile: pd.DataFrame,
    coverage: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    labels = {
        "freqduet_od": "FreqDuet OD input",
        "public_afc_mta": "MTA AFC subset",
        "public_apc_halifax": "Halifax APC subset",
    }
    colors = {
        "freqduet_od": BLUE,
        "public_afc_mta": ORANGE,
        "public_apc_halifax": TEAL,
    }
    order = ("freqduet_od", "public_afc_mta", "public_apc_halifax")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_MM / 25.4, 82.0 / 25.4),
        gridspec_kw={"width_ratios": (1.5, 1.0)},
        layout="constrained",
    )
    ax = axes[0]
    for source in order:
        group = profile[profile["source"].eq(source)].sort_values("hour")
        ax.plot(
            group["hour"],
            group["share"],
            marker="o",
            linewidth=1.8,
            markersize=3.4,
            color=colors[source],
            label=labels[source],
        )
    ax.axvspan(6, 19, color=PALE_GRAY, zorder=0)
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.set_xticks((0, 6, 12, 18, 24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Share of each source's daily profile")
    ax.set_title("Hourly demand shape", loc="left", fontweight="bold", pad=7)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=6.1)
    panel_label(ax, "a")

    windows = (
        ("06-10", 6, 10, TEAL),
        ("10-15", 10, 15, BLUE),
        ("15-20", 15, 20, ORANGE),
        ("Other", None, None, GRAY),
    )
    ax = axes[1]
    left = np.zeros(len(order), dtype=float)
    y = np.arange(len(order))
    for window, lo, hi, color in windows:
        values = []
        for source in order:
            group = profile[profile["source"].eq(source)]
            if lo is None:
                value = 1.0 - sum(
                    float(group.loc[
                        (group["hour"] >= start) & (group["hour"] < stop),
                        "share",
                    ].sum())
                    for _, start, stop, _ in windows[:3]
                )
            else:
                value = float(group.loc[
                    (group["hour"] >= lo) & (group["hour"] < hi), "share"
                ].sum())
            values.append(max(value, 0.0))
        ax.barh(y, values, left=left, color=color, height=0.54, label=window)
        left += np.asarray(values)

    coverage_by_source = coverage.set_index("source")
    ylabels = []
    for source in order:
        row = coverage_by_source.loc[source]
        unit = "origins" if source == "freqduet_od" else (
            "complexes" if source == "public_afc_mta" else "routes"
        )
        ylabels.append(
            f"{labels[source]}\n{int(row['series_count'])} {unit}"
        )
    ax.set_yticks(y, ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of daily demand")
    ax.set_title("Demand-period composition", loc="left", fontweight="bold", pad=7)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.34), fontsize=6)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "b")
    return save_figure(
        fig,
        out_dir,
        "fig5_protocol_v6_external_realism",
        formats,
    )


def write_notes(
    out_dir: Path,
    coverage: pd.DataFrame,
    contract: Mapping[str, Any],
) -> None:
    coverage_by_source = coverage.set_index("source")
    mta = coverage_by_source.loc["public_afc_mta"]
    halifax = coverage_by_source.loc["public_apc_halifax"]
    od = coverage_by_source.loc["freqduet_od"]
    (out_dir / "supporting_captions.md").write_text(f"""# Protocol V6 Supporting Figure Captions

## Figure 1 | Causal frequency-to-authority architecture of the current controller

Historical OD intensities initialize a recursive harmonic demand prior, while
online APC arrivals update the filter causally in {contract['bin_sec']:.0f}-s bins.
Low-frequency level, slope, and forecast features enter the upper policy, which
replans an executable terminal headway curve every
{contract['replan_interval_s'] / 60:.0f} min over a
{contract['planning_horizon_s'] / 60:.0f}-min horizon under a rolling zero-sum
headway budget. Station-local high-frequency innovations and
compact same-time APC/AVL context enter the lower policy, which selects from
seven discrete holding actions between 0 and 45 s. The two-sided regularity
reward uses forward and follower departure gaps frozen before the action. In
the current confirmed configuration, the legacy holding guard, promotion, and
leakage penalty are disabled.

## Figure 4 | Paired physical outcomes of the current policy

Points show mean paired differences between the full current policy and the
Protocol V6 reference configuration; bars show 95% crossed-bootstrap confidence
intervals. Negative values favor the current policy. V8 contains 24 paired
rollouts and V9 contains 64. The current policy differs from the reference by
both compact APC/AVL context and the two-sided departure-regularity objective,
so these panels describe the combined policy's physical behavior rather than
an isolated legacy-guard effect.

## Figure 5 | External passenger-count demand-shape audit

Panel a compares separately normalized hourly demand shapes from the FreqDuet
OD input, a complete-day subset of the bounded public MTA AFC cache
({int(mta['rows'])} source rows; {int(mta['profile_units'])} station-complex
days), and a complete-route subset of the bounded public Halifax APC cache
({int(halifax['rows'])} source rows; {int(halifax['series_count'])} routes and
{int(halifax['profile_units'])} route-days). Panel b summarizes the corresponding
demand-period shares; the FreqDuet input contains {int(od['series_count'])}
origin series. The balanced-cache derivation excludes incomplete pagination
fragments. This remains a descriptive audit across unmatched systems and dates,
not a population estimate, same-day calibration, field-policy evaluation, or
evidence of deployed control benefit.
""")
    (out_dir / "supporting_figure_qa.md").write_text("""# Protocol V6 Supporting Figure QA

- Core conclusion: the current policy implements causal LF upper planning and
  HF/local APC-AVL lower holding; its physical outcomes are reported without
  relabelling the failed V9 gate; external data support realism only.
- Evidence chain: Figure 1 resolves the packaged current-controller config;
  Figure 4 reads V8/V9 paired source artifacts; Figure 5 reads balanced complete
  subsets derived from the tracked public AFC/APC caches and the local OD input.
- Backend: Python with matplotlib only.
- Final size: 183 mm double-column width; 105, 106, and 82 mm heights.
- Integrity: the legacy holding guard, promotion, and leakage penalty are
  explicitly shown as disabled; V8 and V9 are not pooled; Figure 4 is labelled
  as a combined-policy comparison; Figure 5 makes no field-effect claim.
- Exports: editable-text SVG, TrueType-text PDF, 600 dpi LZW TIFF, and 300 dpi
  PNG review render.
""")


def build_supporting_figures(
    package_dir: Path,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    *,
    config_root: Path | None = None,
    afc_path: Path = DEFAULT_AFC,
    apc_path: Path = DEFAULT_APC,
    od_path: Path = DEFAULT_OD,
) -> dict[str, Any]:
    unsupported = set(formats) - set(DEFAULT_FORMATS)
    if unsupported or not formats:
        raise ValueError(f"unsupported or empty figure formats: {sorted(unsupported)}")
    status = json.loads((package_dir / "evidence_status.json").read_text())
    if status.get("protocol") != PROTOCOL:
        raise ValueError("supporting figures are not bound to Protocol V6")
    if status.get("paper_controller") != PAPER_CONTROLLER:
        raise ValueError("supporting figures use a different controller")
    if status.get("submission_ready") is not False:
        raise ValueError("supporting figures must retain the submission hold")

    snapshot_root = config_root or package_dir / "configs"
    config, lineage = resolve_snapshot_config(snapshot_root)
    contract = validate_method_contract(config)
    contract["config_lineage"] = lineage
    physical_rows = mechanism_rows(package_dir)
    profile, coverage, alignment = prepare_realism_data(
        afc_path, apc_path, od_path
    )
    cache_manifest = read_balanced_cache_manifest(afc_path, apc_path)

    out_dir = package_dir / "figures"
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    figure1 = method_figure(contract, out_dir, formats)
    figure4 = mechanism_figure(physical_rows, out_dir, formats)
    figure5 = realism_figure(profile, coverage, out_dir, formats)

    (source_dir / "figure1_method_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    save_rows(source_dir / "figure4_physical_outcomes.csv", physical_rows)
    profile.to_csv(source_dir / "figure5_hourly_profiles.csv", index=False)
    coverage.to_csv(source_dir / "figure5_source_coverage.csv", index=False)
    alignment.to_csv(source_dir / "figure5_profile_alignment.csv", index=False)
    figure5_sources = [
        "figures/source_data/figure5_hourly_profiles.csv",
        "figures/source_data/figure5_source_coverage.csv",
        "figures/source_data/figure5_profile_alignment.csv",
    ]
    if cache_manifest is not None:
        cache_manifest_name = "figure5_cache_derivation_manifest.json"
        (source_dir / cache_manifest_name).write_text(
            json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n"
        )
        figure5_sources.append(f"figures/source_data/{cache_manifest_name}")
    write_notes(out_dir, coverage, contract)

    manifest = {
        "manifest_version": "freqduet-protocol-v6-supporting-figures-v1",
        "protocol": PROTOCOL,
        "paper_controller": PAPER_CONTROLLER,
        "backend": "python-matplotlib",
        "submission_ready": False,
        "figures": {
            "figure_1": {
                "stem": "fig1_protocol_v6_method",
                "width_mm": FIGURE_WIDTH_MM,
                "height_mm": 105.0,
                "outputs": figure1,
                "source_data": ["figures/source_data/figure1_method_contract.json"],
            },
            "figure_4": {
                "stem": "fig4_protocol_v6_physical_outcomes",
                "width_mm": FIGURE_WIDTH_MM,
                "height_mm": 106.0,
                "outputs": figure4,
                "source_data": ["figures/source_data/figure4_physical_outcomes.csv"],
            },
            "figure_5": {
                "stem": "fig5_protocol_v6_external_realism",
                "width_mm": FIGURE_WIDTH_MM,
                "height_mm": 82.0,
                "outputs": figure5,
                "source_data": figure5_sources,
                "claim_boundary": "descriptive realism audit only",
            },
        },
    }
    (out_dir / "supporting_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    refresh_package_manifest(package_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--afc-csv", type=Path, default=DEFAULT_AFC)
    parser.add_argument("--apc-csv", type=Path, default=DEFAULT_APC)
    parser.add_argument("--freqduet-od", type=Path, default=DEFAULT_OD)
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated subset of svg,pdf,tiff,png",
    )
    args = parser.parse_args()
    formats = tuple(
        value.strip().lower() for value in args.formats.split(",") if value.strip()
    )
    manifest = build_supporting_figures(
        args.package_dir.resolve(),
        formats,
        config_root=args.config_root.resolve() if args.config_root else None,
        afc_path=args.afc_csv.resolve(),
        apc_path=args.apc_csv.resolve(),
        od_path=args.freqduet_od.resolve(),
    )
    print(json.dumps({
        "status": "protocol_v6_supporting_figures_complete",
        "package_dir": str(args.package_dir.resolve()),
        "figure_count": len(manifest["figures"]),
        "formats": list(formats),
        "submission_ready": manifest["submission_ready"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
