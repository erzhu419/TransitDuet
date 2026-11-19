#!/usr/bin/env python3
"""Generate publication-grade vector PDF figures for the TR-C FreqDuet manuscript.

Usage:
  python3 generate_figures.py --stage pre   # Fig. 1 (Architecture) & Fig. 2 (Corridors & Harmonic-RLS)
  python3 generate_figures.py --stage full  # All 5 figures from verified V500/V501/V502 JSON summaries
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "freqduet"))
from frequency.demand_frequency import DemandFrequencyTracker

V500_DIR = REPO_ROOT / "experiments" / "v500_pathb_headroom_gate"
V501_DIR = REPO_ROOT / "experiments" / "v501_pathb_b1_stability"
V502_DIR = REPO_ROOT / "experiments" / "v502_pathb_trc_matrix"

# Colorblind-safe publication palette (Okabe-Ito / Nature / Elsevier style)
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_TEAL = "#009E73"
C_PURPLE = "#CC79A7"
C_GOLD = "#E69F00"
C_SKY = "#56B4E9"
C_GREY = "#595959"
C_LIGHT = "#F4F6F9"
C_DARK = "#1E293B"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 9.8,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / stem}.pdf", flush=True)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: List[str],
    *,
    edge: str = C_DARK,
    face: str = "#FFFFFF",
    header_bg: str | None = None,
    title_color: str = "#FFFFFF",
    fontsize: float = 8.0,
) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.15, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(box)
    if header_bg is not None:
        hh = min(0.052, h * 0.28)
        hdr = FancyBboxPatch(
            (x, y + h - hh), w, hh,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.0, edgecolor="none", facecolor=header_bg, zorder=3,
        )
        ax.add_patch(hdr)
        ax.text(
            x + w * 0.5, y + h - hh * 0.52, title,
            ha="center", va="center", fontsize=fontsize + 0.5,
            fontweight="bold", color=title_color, zorder=4,
        )
        body_top = y + h - hh - 0.016
    else:
        ax.text(
            x + w * 0.5, y + h - 0.024, title,
            ha="center", va="top", fontsize=fontsize + 0.4,
            fontweight="bold", color=edge, zorder=4,
        )
        body_top = y + h - 0.052

    step_y = (body_top - (y + 0.014)) / max(len(lines), 1)
    for i, ln in enumerate(lines):
        ax.text(
            x + 0.014, body_top - (i + 0.55) * step_y, ln,
            ha="left", va="center", fontsize=fontsize, color=C_DARK, zorder=4,
        )


def draw_arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    label: str = "",
    *,
    color: str = C_DARK,
    rad: float = 0.0,
    ls: str = "-",
    lx_off: float = 0.0,
    ly_off: float = 0.014,
    fontsize: float = 7.5,
) -> None:
    arr = FancyArrowPatch(
        p0, p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=5.5,head_width=3.4",
        linewidth=1.25, linestyle=ls, color=color, zorder=5,
    )
    ax.add_patch(arr)
    if label:
        mx = 0.5 * (p0[0] + p1[0]) + lx_off
        my = 0.5 * (p0[1] + p1[1]) + ly_off
        ax.text(
            mx, my, label,
            ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
            zorder=6,
        )


def make_fig1_architecture() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(13.4, 6.3))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Left column: Physical Corridor & Causal APC-RLS Decomposition
    draw_box(
        ax, 0.015, 0.56, 0.255, 0.39,
        "Fixed-Pool Corridor",
        [
            r"$\bullet$ Fleet $N_{\mathrm{fleet}}$, cap. $C_{\mathrm{bus}}$",
            r"$\bullet$ Daily quota $M_d$ trips/dir",
            r"$\bullet$ APC boardings $b_{v,s}$",
            r"$\bullet$ Causal AVL $(g^{\mathrm{fwd}}, g^{\mathrm{bwd}})$",
            r"$\bullet$ Terminal pool $\mathcal{I}_d(t)$",
        ],
        edge=C_DARK, face="#F8FAFC", header_bg=C_DARK, fontsize=8.0,
    )

    draw_box(
        ax, 0.015, 0.05, 0.255, 0.41,
        "Headway-Norm Harmonic-RLS",
        [
            r"$\bullet$ Divide boardings by observed headway",
            r"   and hold the rate between bus visits",
            r"$\bullet$ Fourier-RLS ($K=4, \gamma_f=0.998$):",
            r"   LF: $\hat{\lambda}^{\mathrm{LF}}_{s,d}(t+\tau) + \mathrm{slope}$",
            r"   HF: smoothed pre-update innovation",
            r"$\bullet$ Persistent shock promotion",
        ],
        edge=C_TEAL, face="#F0FDF4", header_bg=C_TEAL, fontsize=7.8,
    )

    # Center-top: Upper Slow-Clock Ensemble SAC
    draw_box(
        ax, 0.345, 0.61, 0.325, 0.34,
        r"Upper Slow-Clock SAC ($\Delta T_U = 900\,\mathrm{s}$)",
        [
            r"$\bullet$ State $s^U_k \in \mathbb{R}^{21}$: LF lead, $\sqrt{\Lambda_{\mathrm{rem}}/\Lambda_{\mathrm{cur}}}$,",
            r"   quota balance, ready pool, floor $H_{\min}$",
            r"$\bullet$ Joint plan $a^U_k = (\rho_k, \delta_k, \beta_k)$ ($|\mathcal{A}_U|=30$)",
            r"$\bullet$ Quota-conserving launch projection",
            r"$\bullet$ Shaped interval reward $r^U_k$",
        ],
        edge=C_BLUE, face="#EFF6FF", header_bg=C_BLUE, fontsize=7.8,
    )

    # Center-middle: Turnaround Budget Ledger
    draw_box(
        ax, 0.345, 0.33, 0.325, 0.21,
        "Turnaround Budget Ledger (60 s)",
        [
            r"$\bullet$ FIFO slack $\hat{\sigma}_{v,m} = \tau^{\mathrm{sched}}_{\pi(v)} - \widehat{\mathrm{ETA}}_v - \tau_{\mathrm{buf}}$",
            r"$\bullet$ Budget $B_{v,m} \leftarrow \min(B_{v,m}, \beta_k [\hat{\sigma}_{v,m}]_+)$",
            r"$\bullet$ Express guard if $\hat{\Lambda}^{\mathrm{LF}}_{\bar{d}} > 1.35\,\hat{\Lambda}^{\mathrm{LF}}_d$",
        ],
        edge=C_PURPLE, face="#FDF2F8", header_bg=C_PURPLE, fontsize=7.8,
    )

    # Center-bottom: Lower Constrained Ensemble SAC
    draw_box(
        ax, 0.345, 0.04, 0.325, 0.22,
        r"Lower Constrained SAC ($u_e \in \{0,\dots,25\}\,\mathrm{s}$)",
        [
            r"$\bullet$ State: budget, gaps, load, station queue, HF",
            r"$\bullet$ Hard mask $\mathcal{M}(B_{v,m})$ + zero-hold-adv. $Q_\psi$",
            r"$\bullet$ Passenger-time proxy and CMDP dual $\lambda_L$",
        ],
        edge=C_ORANGE, face="#FFF7ED", header_bg=C_ORANGE, fontsize=7.7,
    )

    # Right column: Execution and comparators
    draw_box(
        ax, 0.740, 0.53, 0.245, 0.42,
        "Closed-Loop Execution",
        [
            r"1. Fixed planned trip roster",
            r"   sweeper $\tau^{\mathrm{sched}}_{M_d}\geq T_{\mathrm{end}}$",
            r"2. Dispatch requires a ready bus",
            r"   and eligible plan time",
            r"3. Holds debit the trip budget",
            r"   $\sum_{e \in m} u_e \leq B_{v,m}(t_0)$",
            r"   with ETA-based updates",
        ],
        edge=C_DARK, face="#F8FAFC", header_bg="#334155", fontsize=7.8,
    )

    draw_box(
        ax, 0.740, 0.05, 0.245, 0.41,
        "Five Policy Representations",
        [
            r"$\bullet$ Shared reward formula and ledger",
            r"$\bullet$ Matched $(21,20)$-dim arms:",
            r"   - Aligned (LF$\to$U, HF$\to$L)",
            r"   - Raw APC History",
            r"   - All-Band Unseparated",
            r"   - Swapped (HF$\to$U, LF$\to$L)",
            r"   - Coarse APC (NoFreq-HRL)",
        ],
        edge=C_GOLD, face="#FFFBEB", header_bg="#B45309", fontsize=7.7,
    )

    # Arrows
    draw_arrow(ax, (0.142, 0.56), (0.142, 0.46), "APC + $\\Delta t$", color=C_TEAL, lx_off=0.052, ly_off=0.0)
    draw_arrow(ax, (0.270, 0.34), (0.345, 0.74), "LF + 20m\nforecast", color=C_BLUE, rad=-0.12, lx_off=-0.010, ly_off=0.04)
    draw_arrow(ax, (0.270, 0.15), (0.345, 0.15), "HF shock\n+ promo", color=C_ORANGE, ly_off=0.032)
    draw_arrow(ax, (0.508, 0.61), (0.508, 0.54), r"Plan $(H_{d,m}, \beta_k)$", color=C_BLUE, lx_off=0.062, ly_off=0.0)
    draw_arrow(ax, (0.508, 0.33), (0.508, 0.26), r"Budget $B_{v,m}$", color=C_PURPLE, lx_off=0.055, ly_off=0.0)
    draw_arrow(ax, (0.670, 0.78), (0.740, 0.78), "Timetable", color=C_BLUE, ly_off=0.025)
    draw_arrow(ax, (0.670, 0.16), (0.740, 0.60), "Hold $u_e$", color=C_ORANGE, rad=-0.10, lx_off=0.020, ly_off=-0.02)
    draw_arrow(ax, (0.270, 0.31), (0.345, 0.42), "LF", color=C_TEAL, ly_off=0.015)
    draw_arrow(ax, (0.270, 0.78), (0.345, 0.80), "AVL + Pool", color=C_DARK, ly_off=0.026)

    save_fig(fig, "fig1_architecture")


def make_fig2_corridors_decomposition() -> None:
    setup_style()
    tidal_meta = json.loads((V500_DIR / "corridors" / "tidal_bimodal" / "corridor_meta.json").read_text())
    mbta_meta = json.loads((V500_DIR / "corridors" / "mbta_route111_wkdy" / "corridor_meta.json").read_text())

    hours = np.arange(6, 20, dtype=np.float64)
    t_up = np.array([tidal_meta["hourly_summary"][f"{int(h):02d}:00:00"]["up_pax"] for h in hours], dtype=np.float64)
    t_dn = np.array([tidal_meta["hourly_summary"][f"{int(h):02d}:00:00"]["down_pax"] for h in hours], dtype=np.float64)
    m_up = np.array([mbta_meta["hourly_summary"][f"{int(h):02d}:00:00"]["inbound_up_pax"] for h in hours], dtype=np.float64)
    m_dn = np.array([mbta_meta["hourly_summary"][f"{int(h):02d}:00:00"]["outbound_down_pax"] for h in hours], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))

    # Panel (a): Synthetic Bimodal Tidal Corridor
    ax = axes[0]
    ax.plot(hours + 0.5, t_up, "-o", color=C_BLUE, lw=1.8, ms=4.0, label="Dir A (Inbound AM peak)")
    ax.plot(hours + 0.5, t_dn, "-s", color=C_ORANGE, lw=1.8, ms=3.8, label="Dir B (Outbound PM peak)")
    ax.fill_between(hours + 0.5, t_up, t_dn, where=(t_up >= t_dn), color=C_BLUE, alpha=0.10)
    ax.fill_between(hours + 0.5, t_up, t_dn, where=(t_dn > t_up), color=C_ORANGE, alpha=0.10)
    ax.set_title("(a) Bimodal Tidal Corridor (15,660 pax/day)")
    ax.set_xlabel("Hour of Service Day")
    ax.set_ylabel("Directional Demand (pax/h)")
    ax.set_xlim(6.0, 20.0)
    ax.set_ylim(200, 1620)
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20])
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(loc="upper right", frameon=True)

    # Panel (b): Boston MBTA Route 111 GTFS/APC Corridor
    ax = axes[1]
    ax.plot(hours + 0.5, m_up, "-o", color=C_BLUE, lw=1.8, ms=4.0, label="Inbound (Woodlawn $\\to$ Haymarket)")
    ax.plot(hours + 0.5, m_dn, "-s", color=C_ORANGE, lw=1.8, ms=3.8, label="Outbound (Haymarket $\\to$ Woodlawn)")
    ax.fill_between(hours + 0.5, m_up, m_dn, where=(m_up >= m_dn), color=C_BLUE, alpha=0.10)
    ax.fill_between(hours + 0.5, m_up, m_dn, where=(m_dn > m_up), color=C_ORANGE, alpha=0.10)
    ax.set_title("(b) MBTA-informed (9,330 pax/day)")
    ax.set_xlabel("Hour of Service Day")
    ax.set_ylabel("Directional Demand (pax/h)")
    ax.set_xlim(6.0, 20.0)
    ax.set_ylim(0, 1090)
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20])
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(loc="upper right", frameon=True, fontsize=7.2)

    # Panel (c): Causal Headway-Normalized Harmonic-RLS vs Naive 1-min Bin Jensen Bias
    ax = axes[2]
    rng = np.random.default_rng(20261118)
    mins = np.arange(0, 840, dtype=np.float64)
    hr_grid = 6.0 + mins / 60.0
    # True underlying station boarding rate (pax/min) with AM peak + 15-min localized HF surge at 11:15
    base_curve = np.interp(hr_grid, hours + 0.5, t_up / (18.0 * 60.0) * 2.1)
    true_rate = base_curve.copy()
    surge_mask = (mins >= 315) & (mins < 330)
    true_rate[surge_mask] *= 2.85

    # Feed both filters the same boarding events, never the latent rate.
    key = (1, True)
    prior_theta = np.zeros(10)
    initial_rate = 1.0
    prior_theta[0] = math.log1p(initial_rate)

    def tracker():
        return DemandFrequencyTracker(
            update_interval_s=20, bin_sec=60, method="harmonic",
            harmonic_period_s=50400, fourier_k=4, harmonic_forgetting=0.998,
            harmonic_prior_var=0.35, harmonic_prior={"local": {key: prior_theta}},
            promotion_enable=False,
        )

    normalized = tracker()
    archived = tracker()
    raw = tracker()
    last_arr = 0
    cur_est = initial_rate
    scaled_lf, lf_est, naive_lf, hf_pos = [], [], [], []
    for m in range(len(mins)):
        before = cur_est
        boarded = 0.0
        if m > 0 and m % 6 == 0:
            dt = m - last_arr
            boarded = float(rng.poisson(float(true_rate[last_arr:m].sum())))
            cur_est = 0.75 * (boarded / dt) + 0.25 * cur_est
            last_arr = m
        for tick in range(3):
            rate = before if tick < 2 else cur_est
            normalized.update({key: rate / 3})
            archived.update({key: rate})
            raw.update({key: boarded if tick == 2 else 0.0})
        observed = archived._get_local_state(key)
        scaled_lf.append(observed.low)
        lf_est.append(normalized._get_local_state(key).low)
        naive_lf.append(raw._get_local_state(key).low)
        hf_pos.append(max(0.0, observed.high))

    ax.plot(hr_grid, lf_est, color=C_BLUE, lw=1.5, label="LF: unit-consistent rate")
    ax.plot(hr_grid, scaled_lf, color=C_PURPLE, lw=1.5, label="LF: archived feed")
    ax.plot(hr_grid, naive_lf, "--", color=C_GREY, lw=1.2, label="LF: raw minute counts")
    ax.fill_between(hr_grid, 0.0, hf_pos, color=C_ORANGE, alpha=0.40, label="HF+: archived feed")
    ax.set_title("(c) Online Harmonic Update (Toy Data)")
    ax.set_xlabel("Hour of Service Day")
    ax.set_ylabel("Filter Output (rate scale)")
    ax.set_xlim(6.0, 20.0)
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18, 20])
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(loc="upper right", frameon=True, fontsize=7.1)

    fig.tight_layout()
    save_fig(fig, "fig2_corridors_decomposition")


def make_post_experiment_figures() -> None:
    setup_style()
    v501_rev = V501_DIR / "review03"
    v501_agg = json.loads((v501_rev / "aggregate" / "results.json").read_text())
    v502_agg = json.loads((V502_DIR / "review" / "aggregate" / "results.json").read_text())

    # --- Fig. 3: Training Stability Curves (ep1..ep200) & Frozen Milestone Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    corridors = [("tidal_bimodal", "Bimodal Tidal Corridor"), ("mbta_route111_wkdy", "MBTA Route 111 Corridor")]

    for idx, (c_key, c_title) in enumerate(corridors):
        ax = axes[idx]
        # Load all 6 V501 Aligned seeds + all 4 V502 Abl_NoBudget seeds
        aligned_curves = []
        for p in sorted(v501_rev.glob(f"train_{c_key}_s*/results.json")):
            js = json.loads(p.read_text())
            aligned_curves.append([row["journey_s"] for row in js["train_curve"]])
        nobud_curves = []
        for p in sorted((V502_DIR / "review").glob(f"learned_{c_key}_Abl_NoBudget_s*/results.json")):
            js = json.loads(p.read_text())
            nobud_curves.append([row["journey_s"] for row in js["train_curve"]])

        def _smooth(mat: np.ndarray, w: int = 9) -> tuple[np.ndarray, np.ndarray]:
            kern = np.ones(w) / w
            sm = np.array([np.convolve(row, kern, mode="valid") for row in mat])
            mean = sm.mean(axis=0)
            sem = sm.std(axis=0, ddof=1) / math.sqrt(max(len(sm), 1))
            return mean, sem

        w = 9
        xs = np.arange(w, 201)
        al_m, al_se = _smooth(np.asarray(aligned_curves, dtype=np.float64), w=w)
        nb_m, nb_se = _smooth(np.asarray(nobud_curves, dtype=np.float64), w=w)

        ax.plot(xs, al_m, color=C_BLUE, lw=2.0, label="FreqDuet (Turnaround-Budgeted)")
        ax.fill_between(xs, al_m - al_se, al_m + al_se, color=C_BLUE, alpha=0.18)
        ax.plot(xs, nb_m, color=C_ORANGE, lw=1.8, ls="--", label="Abl-NoBudget (Unbudgeted HRL)")
        ax.fill_between(xs, nb_m - nb_se, nb_m + nb_se, color=C_ORANGE, alpha=0.15)

        a0_val = v501_agg["by_corridor"][c_key]["baselines"]["A0_fixed_nohold"]["restricted_journey_s_per_pax"]
        a1_val = v501_agg["by_corridor"][c_key]["baselines"]["A1_fixed_bilateral"]["restricted_journey_s_per_pax"]
        ax.axhline(a0_val, color=C_GREY, ls=":", lw=1.2, label="Fixed-NoHold")
        ax.axhline(a1_val, color="#B91C1C", ls="-.", lw=1.1, label="Bilateral-Hold")
        ax.set_title(f"({'ab'[idx]}) {c_title}")
        ax.set_xlabel("Training Service Day (Episode)")
        ax.set_ylabel("Journey Time (s/pax)")
        ax.grid(True, alpha=0.25, ls="--")
        ax.legend(loc="upper right", frameon=True, fontsize=7.2)

    # Panel (c): Frozen Validation Checkpoints ep40 -> ep100 -> ep200
    ax = axes[2]
    tags = ["ep40", "ep100", "ep200"]
    x_pos = np.arange(len(tags))
    width = 0.34
    for i, (c_key, c_lbl, col) in enumerate([
        ("tidal_bimodal", "Bimodal Tidal", C_BLUE),
        ("mbta_route111_wkdy", "MBTA Route 111", C_TEAL),
    ]):
        base_a1 = v501_agg["by_corridor"][c_key]["baselines"]["A1_fixed_bilateral"]["restricted_journey_s_per_pax"]
        deltas = [
            v501_agg["by_corridor"][c_key]["b1_milestones"][t]["across_seed_mean"]["restricted_journey_s_per_pax"] - base_a1
            for t in tags
        ]
        stds = [
            v501_agg["by_corridor"][c_key]["b1_milestones"][t]["across_seed_std"]["restricted_journey_s_per_pax"]
            for t in tags
        ]
        ax.bar(x_pos + (i - 0.5) * width, deltas, width=width, yerr=stds, capsize=3, color=col, alpha=0.88, label=c_lbl)
    ax.axhline(0.0, color=C_DARK, lw=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Episode 40", "Episode 100", "Episode 200"])
    ax.set_ylim(-59.0, 4.0)
    ax.set_ylabel("Journey Delta vs. Bilateral Hold (s/pax)")
    ax.set_title("(c) Frozen Checkpoint Means (6 Seeds)")
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.legend(loc="lower left", frameon=True, fontsize=7.4)

    fig.tight_layout()
    save_fig(fig, "fig3_stability_curves")

    # --- Fig. 4: 4-Way Frequency Band Ablation & Pareto Trade-Off Frontier ---
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    freq_arms = [
        ("FreqDuet_Aligned", "Aligned\n(Ours)", C_BLUE),
        ("Abl_AllBand", "All-\nBand", C_SKY),
        ("Abl_RawHistory", "Raw\nHistory", C_GOLD),
        ("B5_TransitDuet_NoFreq", "No-Freq\nHRL", C_GREY),
        ("Abl_Swapped", "Swapped\nBands", C_ORANGE),
    ]
    short_titles = {
        "tidal_bimodal": "Tidal-Bimodal",
        "mbta_route111_wkdy": "MBTA-111",
    }

    for idx, (c_key, _c_title) in enumerate(corridors):
        ax = axes[idx]
        methods_d = v502_agg["by_corridor"][c_key]["confirm_16d"]["methods"]
        b0_j = methods_d["B0_Fixed_NoHold"]["mean"]["restricted_journey_s_per_pax"]
        vals = [methods_d[m]["mean"]["restricted_journey_s_per_pax"] - b0_j for m, _, _ in freq_arms]
        errs = [methods_d[m]["seed_std"]["restricted_journey_s_per_pax"] for m, _, _ in freq_arms]
        cols = [col for _, _, col in freq_arms]
        lbls = [lb for _, lb, _ in freq_arms]
        xp = np.arange(len(freq_arms))
        bars = ax.bar(xp, vals, yerr=errs, capsize=3, color=cols, alpha=0.88, width=0.58)
        ax.axhline(0.0, color=C_DARK, lw=0.9, ls="--")
        ax.set_xticks(xp)
        ax.set_xticklabels(lbls, fontsize=7.6)
        ax.set_ylabel("Journey Delta vs. Fixed-NoHold (s/pax)")
        ax.set_title(f"({'ab'[idx]}) Band Ablation: {short_titles[c_key]}")
        ax.grid(True, axis="y", alpha=0.25, ls="--")
        if c_key == "tidal_bimodal":
            ax.set_ylim(-40.5, 7.0)
            pad = 1.1
        else:
            ax.set_ylim(-20.2, 4.2)
            pad = 0.55
        for b, v, e in zip(bars, vals, errs):
            y_txt = (v - e - pad) if v < 0 else (v + e + pad)
            va = "top" if v < 0 else "bottom"
            ax.text(
                b.get_x() + b.get_width() * 0.5, y_txt, f"{v:+.1f}s",
                ha="center", va=va, fontsize=7.2, fontweight="bold", color=C_DARK,
            )

    # Panel (c): Journey Time vs Excess Waiting Time Pareto Frontier on MBTA Route 111
    ax = axes[2]
    m_mbta = v502_agg["by_corridor"]["mbta_route111_wkdy"]["confirm_16d"]["methods"]
    pareto_pts = [
        ("B1_Daganzo2009", "Forward-Hold", "#7F1D1D", "^", (6, -3), "left"),
        ("B2_Berrebi2018", "Bilateral-Hold", "#DC2626", "v", (6, -2), "left"),
        ("B4_FlatSAC", "Flat-SAC", C_PURPLE, "P", (-6, 3), "right"),
        ("B0_Fixed_NoHold", "Fixed-NoHold", C_GREY, "s", (6, 3), "left"),
        ("B5_TransitDuet_NoFreq", "NoFreq-HRL", C_SKY, "o", (6, 1), "left"),
        ("Abl_NoBudget", "Abl-NoBudget", C_ORANGE, "X", (-6, -2), "right"),
        ("FreqDuet_Aligned", "FreqDuet (Ours)", C_BLUE, "*", (7, -3), "left"),
        ("B3_EconMPC", "Rolling-Heuristic", C_GOLD, "D", (-6, -4), "right"),
    ]
    for key, name, col, mk, offset, ha_align in pareto_pts:
        ewt = m_mbta[key]["mean"]["ewt_s"]
        jrn = m_mbta[key]["mean"]["restricted_journey_s_per_pax"]
        sz = 135 if key == "FreqDuet_Aligned" else 52
        ax.scatter(ewt, jrn, color=col, marker=mk, s=sz, zorder=4, edgecolors="white", linewidth=0.5)
        ax.annotate(
            name, (ewt, jrn), xytext=offset, textcoords="offset points",
            ha=ha_align, va="center", fontsize=7.2,
            fontweight=("bold" if key == "FreqDuet_Aligned" else "normal"), color=col,
        )
    ax.set_xlim(3.1, 14.8)
    ax.set_ylim(1122.0, 1222.0)
    ax.set_xlabel("Excess Waiting Time, EWT (s) [$\\leftarrow$ better]")
    ax.set_ylabel("Journey Time (s/pax) [$\\downarrow$ better]")
    ax.set_title("(c) MBTA-111 Regularity\u2013Journey Trade-off")
    ax.grid(True, alpha=0.25, ls="--")

    fig.tight_layout()
    save_fig(fig, "fig4_frequency_budget_ablation")

    # --- Fig. 5: Out-of-Distribution Generalization Across Regimes ---
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.65))
    suites = [
        ("confirm_16d", "Nominal\n(16d)"),
        ("ood_peak_shift", "Peak Shift\n($\\pm 1$h)"),
        ("ood_hf_surge", "1.8x HF\nSurge"),
        ("ood_congestion", "Traffic\nCongestion"),
    ]
    comp_methods = [
        ("FreqDuet_Aligned", "FreqDuet (Proposed)", C_BLUE),
        ("B3_EconMPC", "Rolling-Heuristic", C_GOLD),
        ("Abl_AllBand", "Abl-AllBand (Unseparated)", C_SKY),
        ("B5_TransitDuet_NoFreq", "NoFreq-HRL", C_PURPLE),
        ("B2_Berrebi2018", "Bilateral-Hold", C_ORANGE),
    ]

    for idx, (c_key, c_title) in enumerate(corridors):
        ax = axes[idx]
        xp = np.arange(len(suites))
        w_bar = 0.155
        for mi, (m_key, m_lbl, col) in enumerate(comp_methods):
            deltas = []
            for s_key, _ in suites:
                md = v502_agg["by_corridor"][c_key][s_key]["methods"]
                b0 = md["B0_Fixed_NoHold"]["mean"]["restricted_journey_s_per_pax"]
                deltas.append(md[m_key]["mean"]["restricted_journey_s_per_pax"] - b0)
            ax.bar(xp + (mi - 2.0) * w_bar, deltas, width=w_bar, color=col, alpha=0.89, label=m_lbl)
        ax.axhline(0.0, color=C_DARK, lw=0.9, ls="--")
        ax.set_xticks(xp)
        ax.set_xticklabels([sl for _, sl in suites])
        ax.set_ylabel("Journey Delta vs. Fixed-NoHold (s/pax)")
        ax.set_title(f"({'ab'[idx]}) Perturbation Tests: {c_title}")
        ax.grid(True, axis="y", alpha=0.25, ls="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=5, frameon=True, fontsize=7.6, columnspacing=1.1,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    save_fig(fig, "fig5_ood_generalization")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre", "full"), default="pre")
    args = parser.parse_args()
    make_fig1_architecture()
    make_fig2_corridors_decomposition()
    if args.stage == "full":
        make_post_experiment_figures()


if __name__ == "__main__":
    main()
