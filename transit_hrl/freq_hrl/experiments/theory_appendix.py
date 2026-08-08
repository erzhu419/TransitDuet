"""Generate a compact theory appendix for Freq-HRL diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def shaped_return_deviation_bound(leakage_weight: float, leakage_costs: list[float]) -> float:
    """Bound |sum r - sum r'| for r' = r - lambda * leakage."""
    return float(max(leakage_weight, 0.0) * sum(max(float(cost), 0.0) for cost in leakage_costs))


def finite_sample_mean_ci_radius(*, sample_std: float, n: int, z_value: float = 1.96) -> float:
    """Normal-approximation half-width for a paired mean delta CI."""
    n_safe = max(int(n), 1)
    return float(max(float(sample_std), 0.0) * max(float(z_value), 0.0) / math.sqrt(n_safe))


def hierarchical_credit_residual_bound(
    *,
    total_credit: list[float],
    upper_credit: list[float],
    lower_credit: list[float],
) -> float:
    """Bound the additive credit mismatch by the L1 residual."""
    n = min(len(total_credit), len(upper_credit), len(lower_credit))
    if n <= 0:
        return 0.0
    return float(sum(
        abs(float(total_credit[i]) - float(upper_credit[i]) - float(lower_credit[i]))
        for i in range(n)
    ))


def promotion_false_positive_bound(
    *,
    window_bins: int,
    persistence_ratio: float,
    event_probability: float,
) -> float:
    """Hoeffding upper bound for stationary false promotion events.

    The promotion gate fires when the trailing residual-event share exceeds
    `persistence_ratio`. If stationary noise exceeds the threshold with
    probability p < rho, P(mean >= rho) <= exp(-2 n (rho - p)^2).
    """
    n = max(int(window_bins), 1)
    rho = float(min(max(persistence_ratio, 0.0), 1.0))
    p = float(min(max(event_probability, 0.0), 1.0))
    if p >= rho:
        return 1.0
    return float(math.exp(-2.0 * n * (rho - p) ** 2))


def promotion_detection_delay_bound(
    *,
    update_interval_s: float,
    window_bins: int,
    persistence_ratio: float,
) -> float:
    """Worst-case delay when every new residual event is above threshold."""
    n = max(int(window_bins), 1)
    rho = float(min(max(persistence_ratio, 0.0), 1.0))
    required = max(1, int(math.ceil(rho * n)))
    return float(max(update_interval_s, 0.0) * max(required, n))


def primal_dual_average_violation_bound(
    *,
    dual_radius: float,
    step_size: float,
    horizon: int,
    gradient_bound: float,
) -> float:
    """O(1/sqrt(T)) average-violation style bound for bounded dual updates.

    This is the standard projected subgradient bookkeeping term used in the
    appendix as a weak constrained-convergence argument, not a claim that the
    nonconvex actor-critic objective is globally optimized.
    """
    t_safe = max(int(horizon), 1)
    eta = max(float(step_size), 1e-12)
    radius = max(float(dual_radius), 0.0)
    grad = max(float(gradient_bound), 0.0)
    return float((radius ** 2) / (2.0 * eta * t_safe) + 0.5 * eta * (grad ** 2))


def conditional_no_tradeoff_margin(
    *,
    baseline_advantage: float,
    leakage_penalty_budget: float,
    constraint_slack: float,
) -> float:
    """Sufficient performance margin after leakage and constraint costs.

    A positive value is a sufficient no-tradeoff condition for the simplified
    appendix bookkeeping: the treatment's pre-constraint advantage exceeds the
    worst-case leakage shaping budget and any constraint slack consumed.
    """
    return float(baseline_advantage) - max(float(leakage_penalty_budget), 0.0) - max(float(constraint_slack), 0.0)


def stress_claim_coverage_fraction(*, supported_regimes: int, required_regimes: int) -> float:
    """Fraction of pre-registered stress regimes with supported evidence."""
    required = max(int(required_regimes), 1)
    supported = min(max(int(supported_regimes), 0), required)
    return float(supported / required)


def responsibility_reconstruction_error(
    *,
    upper_policy: list[float],
    raw_lower: list[float],
    transferred_lf: list[float],
) -> float:
    """Maximum action error after equal-and-opposite responsibility transfer."""

    if not (
        len(upper_policy) == len(raw_lower) == len(transferred_lf)
        and len(upper_policy) > 0
    ):
        raise ValueError("responsibility vectors must be non-empty and aligned")
    return float(max(
        abs(
            (float(upper_policy[index]) + float(transferred_lf[index]))
            + (float(raw_lower[index]) - float(transferred_lf[index]))
            - (float(upper_policy[index]) + float(raw_lower[index]))
        )
        for index in range(len(upper_policy))
    ))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _check(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("check") == name), {})


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def build_theorem_rows(examples: dict[str, Any]) -> list[dict[str, Any]]:
    """Paper-facing theorem/proof rows for the Freq-HRL appendix."""
    return [
        {
            "id": "Theorem 1",
            "title": "Causal Frequency Features Are Nonanticipative",
            "statement": (
                "For every decision time t, the feature vector emitted by a causal "
                "Freq-HRL encoder is measurable with respect to the observations "
                "available up to t."
            ),
            "assumptions": [
                "The domain adapter appends an exogenous bin only after that bin has occurred.",
                "The encoder update is a deterministic or seeded-random function of the previous encoder state and the current bin.",
                "The feature extractor does not use backward smoothing, centered windows, or future timestamps.",
            ],
            "proof": (
                "Use induction on the number of processed bins. The initial encoder "
                "state is fixed or seeded independently of future observations. If the "
                "state before bin k is a function only of bins 1 through k-1, then the "
                "next state is a function only of that state and bin k. Therefore the "
                "features after bin k are functions only of bins 1 through k. Mapping "
                "k to decision time t gives the nonanticipativity claim."
            ),
            "limitation": (
                "This is an information-flow guarantee. It does not claim that a chosen "
                "encoder is statistically optimal for every domain."
            ),
            "diagnostic": "Causal encoder tests cover EMA, Fourier, state-space, Haar/adaptive wavelet, and neural/PINN state-space paths.",
            "example": "",
        },
        {
            "id": "Theorem 2",
            "title": "Leakage-Shaped Return Gap Is Budgeted",
            "statement": (
                "For shaped rewards r'_t = r_t - lambda L_t with lambda >= 0 and "
                "causal leakage cost L_t >= 0, the absolute episode-return gap is "
                "bounded by lambda sum_t L_t."
            ),
            "assumptions": [
                "Leakage is computed from same-trajectory upper and lower action effects.",
                "The leakage multiplier is nonnegative.",
                "Task return and shaped return are evaluated on the same rollout.",
            ],
            "proof": (
                "Summing the shaped reward gives sum_t r'_t = sum_t r_t - lambda "
                "sum_t L_t. Since lambda and L_t are nonnegative, the shaped return "
                "is no larger than task return and the exact difference is lambda "
                "sum_t L_t. Any enforced leakage budget B therefore bounds the "
                "distortion by lambda B."
            ),
            "limitation": (
                "The bound controls reward-shaping distortion and responsibility "
                "violations; it is not a guarantee that stronger leakage penalties are "
                "performance-neutral."
            ),
            "diagnostic": "Leakage matrices report drift reduction and no-tradeoff gates for Transit and Trading variants.",
            "example": f"Example bound with lambda=0.30: {_fmt(examples['leakage_bound_example'])}.",
        },
        {
            "id": "Theorem 3",
            "title": "Stationary Promotion False Positives Are Exponentially Controlled",
            "statement": (
                "If stationary residual-threshold events have conditional probability "
                "p < rho and promotion requires a trailing-window event share of at "
                "least rho over n bins, then the false-promotion probability is at "
                "most exp(-2 n (rho - p)^2)."
            ),
            "assumptions": [
                "The gate uses only a finite causal residual-event window.",
                "Stationary residual events are bounded Bernoulli indicators with rate at most p.",
                "The detector promotes when the window mean exceeds rho.",
            ],
            "proof": (
                "The promotion statistic is the empirical mean of bounded event "
                "indicators in the trailing window. Under the stationary null, its "
                "expectation is at most p. Hoeffding's inequality bounds the "
                "probability that this mean exceeds rho by exp(-2 n (rho - p)^2)."
            ),
            "limitation": (
                "The bound is conservative and assumes a stationary null. It should be "
                "reported together with empirical promotion false-positive sweeps."
            ),
            "diagnostic": "Promotion sweep and persistent-stress recovery validations test the empirical tradeoff.",
            "example": (
                "Example n=10, rho=0.35, p=0.10: "
                f"{_fmt(examples['promotion_false_positive_bound_example'], digits=6)}."
            ),
        },
        {
            "id": "Theorem 4",
            "title": "Persistent-Shock Promotion Delay Is Window-Bounded",
            "statement": (
                "If every residual event after a regime shift exceeds the promotion "
                "threshold, the causal trailing-window gate promotes within one full "
                "persistence window."
            ),
            "assumptions": [
                "The gate updates every fixed interval.",
                "Promotion requires a finite number of positive residual events in the trailing window.",
                "After the shift, each new event in the window is positive.",
            ],
            "proof": (
                "After one full window, all entries in the trailing window are "
                "post-shift positive events, so the event share equals one and exceeds "
                "any rho <= 1. The implementation's conservative bound reports the "
                "full window duration, which avoids any future-looking detection."
            ),
            "limitation": (
                "Real shocks can be intermittent. In that case the false-negative and "
                "delay behavior depends on the post-shift event rate and threshold."
            ),
            "diagnostic": "Persistent-stress native promotion runs report replan counts, wait deltas, and recovery metrics.",
            "example": f"Example delay bound: {_fmt(examples['promotion_detection_delay_bound_s'], digits=1)}s.",
        },
        {
            "id": "Theorem 5",
            "title": "Hierarchical Wait-Credit Residual Bounds Attribution Error",
            "statement": (
                "Let c_t be total causal passenger-wait credit, and let c_t^U and "
                "c_t^L be the upper and lower frequency-attributed credits on the "
                "same rollout. The episode attribution error is bounded by "
                "sum_t |c_t - c_t^U - c_t^L|."
            ),
            "assumptions": [
                "Total, upper, and lower credits are computed from the same causal rollout.",
                "The policy losses consume only credits available at their decision times.",
                "The validation harness logs or reconstructs the residual term.",
            ],
            "proof": (
                "At each step, the attribution mismatch is exactly the absolute "
                "residual |c_t - c_t^U - c_t^L|. Summing the nonnegative per-step "
                "mismatches over the episode gives the stated L1 upper bound on total "
                "credit-assignment error."
            ),
            "limitation": (
                "Small residuals certify attribution consistency, not necessarily "
                "that the resulting learned policy globally improves wait time."
            ),
            "diagnostic": "Native wait-credit and real-demand control validations report reward/wait/alighting deltas and should keep residual columns in OD/onboard-load runs.",
            "example": f"Example residual bound: {_fmt(examples['credit_residual_bound_example'])}.",
        },
        {
            "id": "Theorem 6",
            "title": "Paired CI Width Shrinks at the Seed-Count Rate",
            "statement": (
                "For paired seed/source deltas with empirical standard deviation s "
                "and n independent pairs, the normal-approximation confidence "
                "half-width is z s / sqrt(n)."
            ),
            "assumptions": [
                "Treatment and control are paired by seed or source window.",
                "The paired deltas have finite variance.",
                "The z value matches the reported two-sided confidence level.",
            ],
            "proof": (
                "The paired estimator is the sample mean of the deltas. Its standard "
                "error is s / sqrt(n). Multiplication by the normal critical value z "
                "gives the reported half-width."
            ),
            "limitation": (
                "The statement is an evidence-width calculation. It does not remove "
                "bias from nonrepresentative stress regimes or public-data samples."
            ),
            "diagnostic": "The unified matrix records n_common and CI status for every paired claim.",
            "example": (
                "Example s=0.18, n=36, z=1.96: "
                f"{_fmt(examples['paired_ci_radius_example'])}."
            ),
        },
        {
            "id": "Theorem 7",
            "title": "Projected Primal-Dual Leakage Updates Control Average Excess",
            "statement": (
                "For bounded projected dual variables and bounded constraint samples, "
                "the standard projected-subgradient bookkeeping term for average "
                "constraint excess is O(1 / sqrt(T)) when the dual step is chosen on "
                "the 1 / sqrt(T) scale."
            ),
            "assumptions": [
                "The dual variable is projected onto a bounded nonnegative interval.",
                "Constraint samples are uniformly bounded.",
                "The actor update uses the current multiplier times the causal constraint excess.",
            ],
            "proof": (
                "Apply the standard projected subgradient inequality to the one-dimensional "
                "dual update. Summing over T steps and dividing by T yields the radius "
                "term divided by eta T plus eta times the squared gradient bound. "
                "Choosing eta proportional to 1 / sqrt(T) gives the stated average "
                "excess rate."
            ),
            "limitation": (
                "This is a constraint-control argument for the multiplier path. It is "
                "not a global convergence theorem for nonconvex actor-critic training."
            ),
            "diagnostic": "Dual PPO validation reports leakage budget, multiplier direction, and no-tradeoff status.",
            "example": (
                "Example average-violation bookkeeping term: "
                f"{_fmt(examples['primal_dual_avg_violation_bound_example'])}."
            ),
        },
        {
            "id": "Proposition 8",
            "title": "Leakage No-Tradeoff Requires Positive Slack",
            "statement": (
                "A leakage-constrained Freq-HRL variant can claim no-tradeoff only "
                "when its paired task advantage exceeds the leakage shaping budget "
                "and any consumed constraint slack on the same validation domain."
            ),
            "assumptions": [
                "Treatment and control are evaluated on paired seeds or source windows.",
                "The task metric is the same metric used for the no-tradeoff claim.",
                "Leakage penalty and constraint-slack budgets are reported on the same rollout family.",
            ],
            "proof": (
                "Let Delta be the unpenalized paired task advantage and let B be "
                "the worst-case performance budget consumed by leakage shaping and "
                "constraint slack. The shaped constrained advantage is at least "
                "Delta - B by the return-gap bookkeeping in Theorem 2 and the "
                "nonnegative slack budget. If Delta - B is positive, the treatment "
                "remains no worse than the control under this sufficient condition."
            ),
            "limitation": (
                "This is only a sufficient condition. If the margin is nonpositive, "
                "empirical no-tradeoff may still occur, but it must be supported by "
                "paired CIs rather than by this bookkeeping argument alone."
            ),
            "diagnostic": "Leakage no-tradeoff matrices must report drift reduction and task noninferiority on the same native or trading domain.",
            "example": (
                "Example no-tradeoff margin: "
                f"{_fmt(examples['conditional_no_tradeoff_margin_example'])}."
            ),
        },
        {
            "id": "Proposition 9",
            "title": "Stress-Generalization Claims Are Intersection Claims",
            "statement": (
                "A global stress-generalization claim over a pre-registered set of "
                "regimes is supported only if every required regime has paired "
                "evidence for the stated metric; otherwise the valid claim is the "
                "intersection of regimes that pass the evidence gate."
            ),
            "assumptions": [
                "The stress-regime set is declared before selecting the headline evidence.",
                "Each regime uses paired treatment/control validation with the same claim metric.",
                "The reporting layer records missing, inconclusive, and not-supported regimes separately.",
            ],
            "proof": (
                "Let R be the required regime set and S be the subset whose paired "
                "evidence passes the claim gate. The statement 'the method improves "
                "under all regimes in R' is the conjunction over every r in R. A "
                "single missing or failed conjunct makes the global statement "
                "unsupported. The strongest evidence-valid statement is therefore "
                "restricted to S, with R minus S reported as a boundary."
            ),
            "limitation": (
                "This proposition is a reporting rule rather than a statistical "
                "power theorem. It prevents overclaiming but does not decide how many "
                "seeds are needed within each regime."
            ),
            "diagnostic": "The unified matrix C9 lists required, supported, missing, and not-supported pressure regimes.",
            "example": (
                "Example coverage fraction for four supported of five regimes: "
                f"{_fmt(examples['stress_claim_coverage_fraction_example'])}."
            ),
        },
        {
            "id": "Proposition 10",
            "title": "Causal Responsibility Transfer Preserves Nominal Action",
            "statement": (
                "Let p_k be a lower low-frequency estimate available before upper "
                "boundary k. Assign u'_k = u_k + p_k and l'_t = l_t - p_k "
                "through that macro interval. The assignment is nonanticipative and "
                "u'_k + l'_t = u_k + l_t at every lower step."
            ),
            "assumptions": [
                "The transferred estimate p_k uses only lower commands observed before boundary k.",
                "The same effective transfer is added to the upper contribution and subtracted from the lower contribution.",
                "The actuator receives a deterministic function of the summed upper and lower contributions plus the same disturbance.",
            ],
            "proof": (
                "Because p_k is computed from the filter state immediately before "
                "boundary k, it is measurable with respect to the available history "
                "and is therefore nonanticipative. Algebraically, (u_k + p_k) + "
                "(l_t - p_k) = u_k + l_t componentwise. Applying the same actuator "
                "clipping and disturbance map to equal nominal sums yields equal "
                "executed actions for fixed raw policy outputs."
            ),
            "limitation": (
                "The proposition is a mechanism-level invariance. Retraining changes "
                "policy states and learned raw outputs, so empirical reward "
                "noninferiority and leakage reduction still require paired gates."
            ),
            "diagnostic": (
                "MuJoCo v10 reports raw and responsibility actions, transfer "
                "saturation, and per-path ResponsibilityReconstructionRMS."
            ),
            "example": (
                "Example maximum reconstruction error: "
                f"{_fmt(examples['responsibility_reconstruction_error_example'], digits=12)}."
            ),
        },
        {
            "id": "Proposition 11",
            "title": "Canonical Policy State Preserves Unconstrained Training Paths",
            "statement": (
                "Consider additive and causal-transfer controllers with matched "
                "initial parameters, random-number streams, environment seeds, "
                "and optimizer updates. If the actor and reward critic consume "
                "only a decomposition-invariant canonical state, the actuator "
                "executes the canonical raw action sum, and the responsibility "
                "cost is inactive, then both controllers have identical raw "
                "actions, environment trajectories, reward updates, and learned "
                "actor/reward-critic parameters at every training iteration."
            ),
            "assumptions": [
                "Canonical policy states are equal whenever raw policy outputs and environment histories are equal.",
                "The actuator computes one canonical raw sum rather than re-adding separately rounded responsibility components.",
                "Responsibility-specific state is confined to an inactive or separately optimized cost critic with no reward-path parameter sharing.",
                "Simulation, sampling, minibatch ordering, and optimizer operations are deterministic under the matched random streams.",
            ],
            "proof": (
                "At the initial step, parameters, canonical states, and random "
                "draws match, so raw actions match. The canonical actuator map "
                "then gives identical executed actions, rewards, and next "
                "environment states. By induction this holds for every step of "
                "the rollout. The resulting reward trajectories, log "
                "probabilities, advantages, and minibatch order are identical, "
                "so deterministic actor and reward-critic optimizer updates are "
                "identical. Repeating the argument over training iterations "
                "establishes pathwise equality."
            ),
            "limitation": (
                "The result does not cover active responsibility constraints, "
                "shared cost/reward parameters, nondeterministic kernels, or a "
                "policy state that exposes responsibility-specific variables. "
                "Safe constrained branches still require empirical return and "
                "drift gates."
            ),
            "diagnostic": (
                "MuJoCo v11 requires paired no-leakage checkpoint hashes and "
                "held-out raw-action, return, and raw-drift differences within "
                "the registered numerical tolerance."
            ),
            "example": (
                "The registered v11 preflight gate requires paired return and "
                "raw-action differences no larger than 1e-8."
            ),
        },
    ]


def build_theory_payload(results_root: Path) -> dict[str, Any]:
    checks = read_csv_rows(results_root / "freq_hrl_paper_diagnostics" / "statistical_checks.csv")
    examples = {
        "leakage_bound_example": shaped_return_deviation_bound(
            leakage_weight=0.30,
            leakage_costs=[0.12, 0.08, 0.05, 0.04],
        ),
        "promotion_false_positive_bound_example": promotion_false_positive_bound(
            window_bins=10,
            persistence_ratio=0.35,
            event_probability=0.10,
        ),
        "promotion_detection_delay_bound_s": promotion_detection_delay_bound(
            update_interval_s=60.0,
            window_bins=10,
            persistence_ratio=0.35,
        ),
        "paired_ci_radius_example": finite_sample_mean_ci_radius(
            sample_std=0.18,
            n=36,
        ),
        "primal_dual_avg_violation_bound_example": primal_dual_average_violation_bound(
            dual_radius=2.0,
            step_size=0.05,
            horizon=400,
            gradient_bound=1.0,
        ),
        "credit_residual_bound_example": hierarchical_credit_residual_bound(
            total_credit=[1.0, 0.6, 0.2],
            upper_credit=[0.7, 0.4, 0.1],
            lower_credit=[0.2, 0.2, 0.1],
        ),
        "conditional_no_tradeoff_margin_example": conditional_no_tradeoff_margin(
            baseline_advantage=0.18,
            leakage_penalty_budget=0.07,
            constraint_slack=0.03,
        ),
        "stress_claim_coverage_fraction_example": stress_claim_coverage_fraction(
            supported_regimes=4,
            required_regimes=5,
        ),
        "responsibility_reconstruction_error_example": (
            responsibility_reconstruction_error(
                upper_policy=[0.2, -0.7],
                raw_lower=[0.5, -0.1],
                transferred_lf=[0.12, -0.08],
            )
        ),
    }
    cited_checks = {
        "transit_learned_promotion_wait": _check(checks, "transit_learned_promotion_wait_vs_interval"),
        "native_learned_gate_reward": _check(checks, "transit_native_learned_gate_reward_vs_interval"),
        "native_learned_gate_wait": _check(checks, "transit_native_learned_gate_wait_vs_interval"),
        "real_demand_control_objective": _check(checks, "transit_real_demand_control_objective_vs_base"),
        "real_demand_control_wait": _check(checks, "transit_real_demand_control_wait_vs_base"),
        "trading_leakage_constraint": _check(checks, "trading_constraint_lower_lf"),
        "transit_leakage_constraint": _check(checks, "transit_constraint_lower_lf"),
    }
    return {
        "formal_objects": [
            "causal exogenous stream x_t",
            "endogenous environment state z_t",
            "causal spectral encoder E_phi(x_<=t)",
            "upper low-frequency plan policy pi_U",
            "lower high-frequency controller pi_L",
            "promotion gate g_promote",
            "action-effect leakage cost L_t",
            "frequency-attributed passenger wait credit c_t^wait",
            "paired seed/source estimator delta d_i",
        ],
        "assumptions": [
            "A1: the encoder reads only current and past exogenous bins.",
            "A2: the upper action remains active across multiple lower decisions unless a scheduled or promoted replan occurs.",
            "A3: leakage costs are nonnegative and computed causally from action effects.",
            "A4: under stationary noise, residual-threshold events are conditionally bounded by a Bernoulli rate p.",
            "A5: paired validation compares treatment/control on the same seed and source window.",
            "A6: frequency credit residuals are explicitly measurable from the same causal rollout.",
            "A7: constrained updates use bounded nonnegative dual variables and bounded constraint samples.",
            "A8: global stress-generalization claims declare their required regime set before selecting headline artifacts.",
            "A9: responsibility transfer is computed before an upper boundary and applied equal-and-oppositely to upper and lower action contributions.",
            "A10: canonical actor/reward-critic state and actuator reconstruction are responsibility-mode invariant when proving unconstrained path equality.",
        ],
        "theorems": build_theorem_rows(examples),
        "examples": examples,
        "cited_checks": cited_checks,
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Freq-HRL Theory Appendix",
        "",
        "## Formal Setup",
        "",
        "Freq-HRL assumes an endogenous state `z_t`, an exogenous time-series stream `x_t`, and a causal encoder `E_phi(x_<=t)` that emits low-frequency trend, middle-frequency regime buffer, high-frequency residual, uncertainty, energy, and persistence summaries.",
        "",
        "The upper policy `pi_U` consumes low-frequency trend/forecast plus bounded high-frequency summaries and emits a plan action. The lower policy `pi_L` consumes the active upper plan, local endogenous state, and high/middle-frequency residual context and emits high-frequency control actions.",
        "",
        "## Assumptions",
        "",
    ]
    for item in payload["assumptions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Theorems", ""])
    for theorem in payload["theorems"]:
        lines.extend([
            f"### {theorem['id']}: {theorem['title']}",
            "",
            f"Statement: {theorem['statement']}",
            "",
            "Assumptions:",
        ])
        for assumption in theorem["assumptions"]:
            lines.append(f"- {assumption}")
        lines.extend([
            "",
            f"Proof: {theorem['proof']}",
            "",
            f"Limitation: {theorem['limitation']}",
            "",
            f"Diagnostics: {theorem['diagnostic']}",
        ])
        if theorem.get("example"):
            lines.extend(["", f"Numeric example: {theorem['example']}"])
        lines.append("")
    lines.extend([
        "## Empirical Anchors",
        "",
        "| check | status | delta CI95 |",
        "|---|---|---:|",
    ])
    for name, row in payload["cited_checks"].items():
        if not row:
            continue
        lines.append(
            f"| {name} "
            f"| {row.get('status', 'missing')} "
            f"| {row.get('delta_ci95_low', 'NA')} to {row.get('delta_ci95_high', 'NA')} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "These results formalize the Freq-HRL protocol claims. They do not replace large-scale performance validation: native Transit under true onboard-load/alighting/OD dynamics, learned native promotion reward/wait CIs, and deeper order-book feeds still need broader seed and data coverage.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("transit_hrl/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/freq_hrl_theory_appendix"))
    args = parser.parse_args()
    payload = build_theory_payload(args.results_root)
    write_outputs(args.output_dir, payload)
    print(f"wrote {args.output_dir}")
    print(
        "theory_appendix "
        f"fp_bound={payload['examples']['promotion_false_positive_bound_example']:.6f} "
        f"delay_s={payload['examples']['promotion_detection_delay_bound_s']:.1f}"
    )


if __name__ == "__main__":
    main()
