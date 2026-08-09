"""Generate the scope-limited formal appendix for Freq-HRL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


FORMAL_SCOPE_VERSION = "freq_hrl_formal_scope_v3"


def shaped_return_deviation_bound(
    leakage_weight: float,
    leakage_costs: list[float],
) -> float:
    """Return the exact nonnegative shaping gap for a finite trajectory."""

    weight = max(float(leakage_weight), 0.0)
    return float(weight * sum(max(float(cost), 0.0) for cost in leakage_costs))


def finite_sample_mean_ci_radius(
    *,
    sample_std: float,
    n: int,
    z_value: float = 1.96,
) -> float:
    """Return a normal-approximation radius; this is not a finite-sample bound."""

    n_safe = max(int(n), 1)
    return float(
        max(float(sample_std), 0.0)
        * max(float(z_value), 0.0)
        / math.sqrt(n_safe)
    )


def hierarchical_credit_residual_bound(
    *,
    total_credit: list[float],
    upper_credit: list[float],
    lower_credit: list[float],
) -> float:
    """Return the L1 residual of an aligned additive credit decomposition."""

    if not (
        len(total_credit) == len(upper_credit) == len(lower_credit)
    ):
        raise ValueError("credit vectors must be aligned")
    return float(sum(
        abs(float(total) - float(upper) - float(lower))
        for total, upper, lower in zip(
            total_credit,
            upper_credit,
            lower_credit,
        )
    ))


def promotion_false_positive_bound(
    *,
    window_bins: int,
    persistence_ratio: float,
    event_probability: float,
) -> float:
    """Bound a null promotion-window crossing under conditional mean control.

    Let X_i be adapted Bernoulli indicators with
    E[X_i | F_{i-1}] <= p. For rho > p, the conditional Hoeffding bound gives
    P(n^{-1} sum_i X_i >= rho) <= exp(-2 n (rho-p)^2).
    """

    n = max(int(window_bins), 1)
    rho = float(min(max(persistence_ratio, 0.0), 1.0))
    probability = float(min(max(event_probability, 0.0), 1.0))
    if probability >= rho:
        return 1.0
    return float(math.exp(-2.0 * n * (rho - probability) ** 2))


def promotion_detection_delay_bound(
    *,
    update_interval_s: float,
    window_bins: int,
    persistence_ratio: float,
) -> float:
    """Bound persistence-gate delay from reset under all-active observations."""

    del persistence_ratio
    n = max(int(window_bins), 1)
    return float(max(update_interval_s, 0.0) * n)


def promotion_warm_window_delay_bound(
    *,
    update_interval_s: float,
    window_bins: int,
    persistence_ratio: float,
) -> float:
    """Bound delay after a shift when the trailing window is already full."""

    n = max(int(window_bins), 1)
    rho = float(min(max(persistence_ratio, 0.0), 1.0))
    required = max(1, int(math.ceil(rho * n)))
    return float(max(update_interval_s, 0.0) * required)


def projected_dual_regret_term(
    *,
    dual_radius: float,
    step_size: float,
    horizon: int,
    gradient_bound: float,
) -> float:
    """Return the projected-dual comparator term averaged over a horizon."""

    horizon_safe = max(int(horizon), 1)
    eta = max(float(step_size), 1e-12)
    radius = max(float(dual_radius), 0.0)
    gradient = max(float(gradient_bound), 0.0)
    return float(
        radius ** 2 / (2.0 * eta * horizon_safe)
        + 0.5 * eta * gradient ** 2
    )


def conditional_no_tradeoff_margin(
    *,
    baseline_advantage: float,
    leakage_penalty_budget: float,
    constraint_slack: float,
) -> float:
    """Return a sufficient accounting margin, not an RL convergence result."""

    return float(baseline_advantage) - max(
        float(leakage_penalty_budget), 0.0
    ) - max(float(constraint_slack), 0.0)


def stress_claim_coverage_fraction(
    *,
    supported_regimes: int,
    required_regimes: int,
) -> float:
    """Return the covered fraction of a predeclared stress-regime set."""

    required = max(int(required_regimes), 1)
    supported = min(max(int(supported_regimes), 0), required)
    return float(supported / required)


def responsibility_reconstruction_error(
    *,
    upper_policy: list[float],
    raw_lower: list[float],
    transferred_lf: list[float],
) -> float:
    """Return nominal-action error after equal-and-opposite transfer."""

    if not (
        len(upper_policy) == len(raw_lower) == len(transferred_lf)
        and upper_policy
    ):
        raise ValueError("responsibility vectors must be non-empty and aligned")
    return float(max(
        abs(
            (float(upper) + float(transfer))
            + (float(lower) - float(transfer))
            - (float(upper) + float(lower))
        )
        for upper, lower, transfer in zip(
            upper_policy,
            raw_lower,
            transferred_lf,
        )
    ))


def ideal_transfer_relative_leakage_reduction(
    *,
    lower_lf_norm: float,
    transfer_error_norm: float,
) -> float:
    """Return 1 - ||e||^2/||P_L l||^2 for ideal complementary bands."""

    lower_norm = max(float(lower_lf_norm), 0.0)
    error_norm = max(float(transfer_error_norm), 0.0)
    if lower_norm <= 0.0:
        raise ValueError("lower LF norm must be positive")
    return float(1.0 - (error_norm / lower_norm) ** 2)


def lower_router_frequency_response_power(
    *,
    alpha: float,
    angular_frequency: float,
    strength: float = 1.0,
) -> float:
    """Return the zero-state EMA-router power gain at one frequency.

    For the unclipped router

        b_{t+1} = (1-alpha) b_t + alpha z_t,
        e_t = z_t - beta b_t,

    the transfer function from latent proposal z to effective action e is
    H(q)=[1-(1-alpha(1-beta))q^-1]/[1-(1-alpha)q^-1].
    """

    smoothing = float(alpha)
    routing = float(strength)
    omega = float(angular_frequency)
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not math.isfinite(omega):
        raise ValueError("angular_frequency must be finite")
    if not 0.0 <= routing <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    cosine = math.cos(omega)
    numerator_coefficient = 1.0 - smoothing * (1.0 - routing)
    numerator = (
        1.0
        + numerator_coefficient ** 2
        - 2.0 * numerator_coefficient * cosine
    )
    denominator = (
        1.0
        + (1.0 - smoothing) ** 2
        - 2.0 * (1.0 - smoothing) * cosine
    )
    return float(numerator / denominator)


def lower_router_constant_transient(
    *,
    latent_magnitude: float,
    alpha: float,
    step: int,
    strength: float = 1.0,
) -> float:
    """Return |e_t| for a constant proposal and a zero router baseline."""

    smoothing = float(alpha)
    routing = float(strength)
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if int(step) < 0:
        raise ValueError("step must be non-negative")
    if not 0.0 <= routing <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    return float(
        abs(float(latent_magnitude))
        * (
            (1.0 - routing)
            + routing * (1.0 - smoothing) ** int(step)
        )
    )


def physical_power_excess_upper_bound(
    *,
    action_limit: float,
    rms_budget: float,
) -> float:
    """Bound a convex-low-pass power excess for clipped actions."""

    limit = float(action_limit)
    budget = float(rms_budget)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("action_limit must be positive and finite")
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("rms_budget must be positive and finite")
    return float(max(limit * limit - budget * budget, 0.0))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _check(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("check") == name), {})


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def build_numeric_examples() -> dict[str, float]:
    """Build deterministic examples used to sanity-check statement direction."""

    return {
        "leakage_bound_example": shaped_return_deviation_bound(
            0.30,
            [0.12, 0.08, 0.05, 0.04],
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
        "promotion_warm_window_delay_bound_s": promotion_warm_window_delay_bound(
            update_interval_s=60.0,
            window_bins=10,
            persistence_ratio=0.35,
        ),
        "paired_ci_radius_example": finite_sample_mean_ci_radius(
            sample_std=0.18,
            n=36,
        ),
        "projected_dual_regret_term_example": projected_dual_regret_term(
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
        "ideal_transfer_relative_leakage_reduction_example": (
            ideal_transfer_relative_leakage_reduction(
                lower_lf_norm=0.5,
                transfer_error_norm=0.2,
            )
        ),
        "lower_router_dc_power_gain_example": (
            lower_router_frequency_response_power(
                alpha=0.10,
                angular_frequency=0.0,
            )
        ),
        "lower_router_nyquist_power_gain_example": (
            lower_router_frequency_response_power(
                alpha=0.10,
                angular_frequency=math.pi,
            )
        ),
        "partial_router_dc_power_gain_example": (
            lower_router_frequency_response_power(
                alpha=0.10,
                angular_frequency=0.0,
                strength=0.10,
            )
        ),
        "lower_router_constant_transient_step_32_example": (
            lower_router_constant_transient(
                latent_magnitude=1.0,
                alpha=0.10,
                step=32,
            )
        ),
        "physical_power_excess_upper_bound_example": (
            physical_power_excess_upper_bound(
                action_limit=1.0,
                rms_budget=0.05,
            )
        ),
    }


def build_formal_statement_rows(
    examples: dict[str, float],
) -> list[dict[str, Any]]:
    """Return machine-readable formal statements with explicit claim classes."""

    return [
        {
            "id": "F1",
            "kind": "lemma",
            "title": "Causal encoder nonanticipativity",
            "statement": (
                "At decision time t, a recursively updated causal encoder emits "
                "features measurable with respect to observations available by t."
            ),
            "assumptions": [
                "Bins enter the adapter only after their event times.",
                "The update reads only prior encoder state, the current bin, and randomness independent of future bins.",
                "Feature extraction uses no centered window or backward smoother.",
            ],
            "proof": (
                "Induct on processed bins. The initial state is future-independent; "
                "measurability is preserved by each adapted state update and feature map."
            ),
            "limitation": "This is an information-flow result, not encoder optimality.",
            "diagnostic": "Prefix-invariance tests cover every registered encoder family.",
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F2",
            "kind": "identity",
            "title": "Leakage-shaped return gap",
            "statement": (
                "For r'_t=r_t-lambda L_t, lambda>=0, and L_t>=0 on one "
                "trajectory, G-G'=lambda sum_t L_t."
            ),
            "assumptions": [
                "Task and shaped returns use the same finite trajectory.",
                "The multiplier and leakage costs are nonnegative.",
            ],
            "proof": "Sum the per-step reward definition and cancel the task rewards.",
            "limitation": "The identity does not imply performance-neutral regularization.",
            "diagnostic": "Raw task return must be reported separately from shaped return.",
            "example": _fmt(examples["leakage_bound_example"]),
            "verification_status": "algebraically_verified",
        },
        {
            "id": "F3",
            "kind": "proposition",
            "title": "Band-projected responsibility transfer reduces ideal leakage",
            "statement": (
                "Let P_L and P_H be complementary orthogonal projectors and define "
                "leakage J(u,l)=||P_H u||^2+||P_L l||^2. For a causal p in the "
                "low-band subspace, set u'=u+p and l'=l-p. Then u'+l'=u+l, "
                "P_H u'=P_H u, and J(u',l')=||P_H u||^2+||P_L l-p||^2. "
                "If ||P_L l-p||<=kappa||P_L l|| with kappa<1, leakage falls by "
                "at least (1-kappa^2)||P_L l||^2."
            ),
            "assumptions": [
                "The analysis uses finite-dimensional complementary orthogonal band projectors.",
                "The transferred estimate is low-band and available before activation.",
                "Leakage is measured in responsibility space before actuator saturation.",
            ],
            "proof": (
                "Equal-and-opposite transfer preserves the sum. Since P_H p=0, "
                "upper HF leakage is unchanged; lower LF leakage becomes the squared "
                "estimation residual. Substitute the kappa bound."
            ),
            "limitation": (
                "Causal practical filters are approximate projectors; v13 separately "
                "tests raw behavioral drift and upper HF power."
            ),
            "diagnostic": "Report transfer error, reconstruction RMS, and both responsibility and raw drift.",
            "example": _fmt(
                examples["ideal_transfer_relative_leakage_reduction_example"]
            ),
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F4",
            "kind": "lemma",
            "title": "Nominal action invariance under equal-and-opposite transfer",
            "statement": (
                "Adding the same causal transfer to upper responsibility and "
                "subtracting it from lower responsibility preserves their nominal sum."
            ),
            "assumptions": [
                "The same transfer vector is used in both responsibility paths.",
                "The actuator receives a deterministic function of the summed contribution.",
            ],
            "proof": "Componentwise, (u+p)+(l-p)=u+l.",
            "limitation": "Retraining can change raw policy outputs and therefore needs paired evaluation.",
            "diagnostic": "ResponsibilityReconstructionRMS checks finite-precision equality.",
            "example": _fmt(
                examples["responsibility_reconstruction_error_example"],
                digits=12,
            ),
            "verification_status": "algebraically_verified",
        },
        {
            "id": "F5",
            "kind": "proposition",
            "title": "Pathwise equivalence with an inactive responsibility constraint",
            "statement": (
                "Matched additive and transfer controllers have identical raw actions, "
                "trajectories, reward updates, and actor/reward-critic parameters when "
                "their canonical policy state and actuator are decomposition-invariant, "
                "the responsibility constraint is inactive, and all random streams and "
                "deterministic optimizer operations are matched."
            ),
            "assumptions": [
                "Initial parameters, environment seeds, sampling, and minibatch order match.",
                "Responsibility-only variables do not enter or share parameters with the reward path.",
                "The numerical backend is deterministic for the matched operations.",
            ],
            "proof": (
                "Induct over rollout steps using nominal-action invariance, then over "
                "optimizer iterations using identical reward data, losses, and updates."
            ),
            "limitation": "Active constraints, shared critics, or nondeterministic kernels break the premise.",
            "diagnostic": "Matched no-leakage checkpoint hashes test the registered implementation path.",
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F6",
            "kind": "proposition",
            "title": "Conditional false-promotion concentration",
            "statement": (
                "For adapted Bernoulli residual events X_i satisfying "
                "E[X_i|F_{i-1}]<=p<rho, the probability that an n-bin event share "
                "reaches rho is at most exp(-2n(rho-p)^2)."
            ),
            "assumptions": [
                "The null event indicators are adapted and bounded in [0,1].",
                "Their conditional means are uniformly bounded by p.",
                "Additional gate conditions can only remove promotion events.",
            ],
            "proof": (
                "Apply the conditional Hoeffding exponential-moment bound to the "
                "supermartingale differences and optimize the Chernoff parameter."
            ),
            "limitation": "The claim fails without conditional null-rate control and is conservative under dependence.",
            "diagnostic": "Empirical false-positive sweeps must accompany the bound.",
            "example": _fmt(
                examples["promotion_false_positive_bound_example"],
                digits=6,
            ),
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F7",
            "kind": "proposition",
            "title": "Persistence-component detection delay",
            "statement": (
                "If every post-shift residual event is active, the persistence "
                "component of an n-bin causal gate crosses within n updates from "
                "reset and within ceil(rho n) updates once the window is initialized."
            ),
            "assumptions": [
                "The gate updates at a fixed interval.",
                "Every post-shift residual event exceeds the event threshold.",
                "No additional regime, strength, age, or cooldown condition blocks activation.",
            ],
            "proof": (
                "A reset gate requires n observations to satisfy its full-window guard. "
                "In a full pre-shift window, each active update replaces at most one "
                "inactive event, so ceil(rho n) replacements suffice."
            ),
            "limitation": "Intermittent shocks and learned gate conditions require empirical delay analysis.",
            "diagnostic": "Report both gate opportunities and realized replans by stress regime.",
            "example": (
                f"reset={_fmt(examples['promotion_detection_delay_bound_s'], 1)}s; "
                f"warm={_fmt(examples['promotion_warm_window_delay_bound_s'], 1)}s"
            ),
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F8",
            "kind": "identity",
            "title": "Frequency-credit accounting residual",
            "statement": (
                "For aligned total, upper, and lower credits, the episode additive "
                "attribution mismatch equals sum_t |c_t-c_t^U-c_t^L|."
            ),
            "assumptions": [
                "All credit streams refer to the same causal rollout and time index.",
                "No reward item is silently duplicated or omitted outside the residual.",
            ],
            "proof": "This follows from the definition of the per-step additive residual.",
            "limitation": "Small accounting residual does not imply wait-time improvement.",
            "diagnostic": "Native Transit runs should retain explicit credit residuals.",
            "example": _fmt(examples["credit_residual_bound_example"]),
            "verification_status": "definitionally_verified",
        },
        {
            "id": "F9",
            "kind": "lemma",
            "title": "Projected dual comparator inequality",
            "statement": (
                "For lambda_{t+1}=Pi_[0,R](lambda_t+eta g_t), |g_t|<=G, and "
                "any comparator lambda in [0,R], the average dual-regret term is "
                "bounded by R^2/(2 eta T)+eta G^2/2."
            ),
            "assumptions": [
                "The multiplier is projected onto a fixed bounded interval.",
                "Constraint excess samples are uniformly bounded.",
                "The statement concerns the dual sequence only.",
            ],
            "proof": (
                "Projection nonexpansiveness gives a one-step squared-distance "
                "inequality. Rearrange it, sum over T, telescope distances, and use "
                "|g_t|<=G."
            ),
            "limitation": (
                "Without primal optimality or a Slater-type condition, this is not an "
                "average constraint-violation or actor-critic convergence theorem."
            ),
            "diagnostic": "Report multiplier direction and empirical budget violation separately.",
            "example": _fmt(examples["projected_dual_regret_term_example"]),
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F10",
            "kind": "sufficient_condition",
            "title": "No-tradeoff accounting margin",
            "statement": (
                "If a paired task advantage exceeds the stated leakage-distortion "
                "budget plus consumed constraint slack, the resulting accounting "
                "margin is positive."
            ),
            "assumptions": [
                "All terms use one metric and one paired rollout family.",
                "The two budgets are valid upper bounds on performance distortion.",
            ],
            "proof": "Subtract both nonnegative budgets from the paired advantage.",
            "limitation": "This sufficient condition cannot replace a paired empirical noninferiority gate.",
            "diagnostic": "Native no-tradeoff claims require drift and task gates on the same data.",
            "example": _fmt(examples["conditional_no_tradeoff_margin_example"]),
            "verification_status": "accounting_identity",
        },
        {
            "id": "F11",
            "kind": "proposition",
            "title": "Causal partial EMA lower router has an exact response",
            "statement": (
                "For the unclipped zero-state recursion b_{t+1}=(1-alpha)b_t+"
                "alpha z_t and e_t=z_t-beta b_t, 0<alpha<=1 and 0<=beta<=1, "
                "the transfer function is H(q)=[1-(1-alpha(1-beta))q^-1]/"
                "[1-(1-alpha)q^-1]. Hence H(1)=1-beta and the DC power gain is "
                "(1-beta)^2. A constant proposal c produces "
                "|e_t|=|c|[(1-beta)+beta(1-alpha)^t] from a zero baseline."
            ),
            "assumptions": [
                "The router is in causal_ema_high_pass mode with fixed strength beta and actuator clipping is inactive.",
                "The analysis uses the zero-state linear recursion implemented by the router.",
                "Each action component is analyzed independently before the actuator map.",
            ],
            "proof": (
                "Take the one-sided z transform of the baseline recursion to obtain "
                "B(q)/Z(q)=alpha q^-1/[1-(1-alpha)q^-1]. Subtract beta times "
                "this ratio from one, evaluate on the unit circle, and square the modulus. "
                "For z_t=c, solve the scalar recurrence directly."
            ),
            "limitation": (
                "The exact spectral formula is pre-clipping. Clipping is audited "
                "separately because a nonlinear projection can create new harmonics."
            ),
            "diagnostic": (
                "Report latent and effective lower LF drift, router-removed RMS, "
                "and router clip rate on every held-out path."
            ),
            "example": (
                f"DC power gain={_fmt(examples['lower_router_dc_power_gain_example'], 8)}; "
                f"10% router DC power gain={_fmt(examples['partial_router_dc_power_gain_example'], 6)}; "
                f"Nyquist power gain={_fmt(examples['lower_router_nyquist_power_gain_example'], 6)}; "
                f"constant residual at t=32={_fmt(examples['lower_router_constant_transient_step_32_example'], 6)}"
            ),
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F12",
            "kind": "proposition",
            "title": "Exposed router state preserves an augmented Markov description",
            "statement": (
                "If the base controlled process is Markov in s_t and the router "
                "baseline b_t is updated deterministically from (b_t,z_t), then "
                "the augmented process (s_t,b_t) is Markov. Supplying b_t to the "
                "lower policy therefore prevents this router from introducing "
                "unobserved controller state."
            ),
            "assumptions": [
                "The base transition kernel is Markov in the declared environment state and executed action.",
                "The router update is deterministic and uses no future proposal or observation.",
                "The policy observation contains the complete router baseline used for the current route.",
            ],
            "proof": (
                "Conditioned on (s_t,b_t) and z_t, the effective action is fixed; "
                "the base kernel determines s_{t+1}, and the router recursion "
                "determines b_{t+1}. No earlier history enters either transition."
            ),
            "limitation": (
                "This isolates router-state observability only; other encoders or "
                "partially observed environments can still require recurrent state."
            ),
            "diagnostic": "The registered lower-policy input schema must include router.context.",
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F13",
            "kind": "lemma",
            "title": "Actuator clipping is bounded and nonexpansive",
            "statement": (
                "Componentwise projection C_A onto [-A,A]^d satisfies "
                "||C_A(x)-C_A(y)||_2<=||x-y||_2 and ||C_A(x)||_infinity<=A."
            ),
            "assumptions": [
                "The router uses componentwise Euclidean projection onto a fixed box.",
                "The action limit A is positive and finite.",
            ],
            "proof": (
                "Scalar interval projection is monotone with slope in [0,1]. "
                "Apply the scalar inequality componentwise and sum squared terms."
            ),
            "limitation": (
                "Nonexpansiveness does not preserve the linear high-pass spectrum; "
                "the preregistered clip-rate gate bounds reliance on this nonlinear regime."
            ),
            "diagnostic": "Reject candidates whose held-out router clip-rate CI exceeds the registered tolerance.",
            "verification_status": "proved_under_stated_assumptions",
        },
        {
            "id": "F14",
            "kind": "proposition",
            "title": "Physical leakage excess gives a bounded dual-update scale",
            "statement": (
                "Let each effective action component lie in [-A,A] and let its "
                "low-frequency estimate be a convex combination of prior effective "
                "actions. Then P_t=d^-1||ell_t||_2^2<=A^2 and the physical cost "
                "g_t=[P_t-beta^2]_+ is at most [A^2-beta^2]_+. Consequently a "
                "projected update lambda_{t+1}=Pi_[0,R](lambda_t+eta g_t) has "
                "one-step increase at most eta[A^2-beta^2]_+."
            ),
            "assumptions": [
                "Effective actions are clipped to a fixed componentwise limit A.",
                "The reported low-frequency estimate is a convex moving average of those actions.",
                "The RMS budget beta and dual step eta are positive and finite.",
            ],
            "proof": (
                "Convexity preserves each component's [-A,A] bound, so averaging "
                "their squares gives P_t<=A^2. Monotonicity of the positive-part "
                "operator yields the cost bound; projection cannot increase the "
                "unprojected positive update."
            ),
            "limitation": (
                "This is a numerical-scale and feasibility bound, not a guarantee "
                "that the learned policy satisfies the budget or preserves return."
            ),
            "diagnostic": (
                "Report physical power, power budget, power excess, dual saturation, "
                "and raw task return; do not infer success from multiplier motion."
            ),
            "example": _fmt(
                examples["physical_power_excess_upper_bound_example"],
                digits=6,
            ),
            "verification_status": "proved_under_stated_assumptions",
        },
    ]


def build_reporting_rules(examples: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "id": "R1",
            "kind": "reporting_approximation",
            "title": "Normal paired-mean interval width",
            "statement": "The large-sample normal half-width is z*s/sqrt(n).",
            "assumptions": [
                "Pairs are independent sampling units with finite variance.",
                "A normal approximation is appropriate for the stated use.",
            ],
            "limitation": (
                "This is not an exact finite-sample theorem; primary analyses use "
                "the frozen bootstrap or test specified by each protocol."
            ),
            "example": _fmt(examples["paired_ci_radius_example"]),
        },
        {
            "id": "R2",
            "kind": "reporting_rule",
            "title": "Stress claims are conjunctions over declared regimes",
            "statement": (
                "A claim over every regime in a predeclared set is supported only if "
                "the registered gate passes in every required regime."
            ),
            "assumptions": [
                "The regime set and metric are declared before outcome selection.",
                "Missing and failed regimes remain visible.",
            ],
            "limitation": "This controls wording, not within-regime statistical power.",
            "example": _fmt(examples["stress_claim_coverage_fraction_example"]),
        },
    ]


def build_theory_payload(results_root: Path) -> dict[str, Any]:
    checks = read_csv_rows(
        results_root / "freq_hrl_paper_diagnostics" / "statistical_checks.csv"
    )
    examples = build_numeric_examples()
    cited_checks = {
        "transit_learned_promotion_wait": _check(
            checks,
            "transit_learned_promotion_wait_vs_interval",
        ),
        "native_learned_gate_reward": _check(
            checks,
            "transit_native_learned_gate_reward_vs_interval",
        ),
        "native_learned_gate_wait": _check(
            checks,
            "transit_native_learned_gate_wait_vs_interval",
        ),
        "real_demand_control_objective": _check(
            checks,
            "transit_real_demand_control_objective_vs_base",
        ),
        "real_demand_control_wait": _check(
            checks,
            "transit_real_demand_control_wait_vs_base",
        ),
        "trading_leakage_constraint": _check(
            checks,
            "trading_constraint_lower_lf",
        ),
        "transit_leakage_constraint": _check(
            checks,
            "transit_constraint_lower_lf",
        ),
    }
    statements = build_formal_statement_rows(examples)
    return {
        "schema_version": FORMAL_SCOPE_VERSION,
        "proof_verification_status": "internal_scope_audit_pass",
        "independent_proof_verification": False,
        "formal_statements": statements,
        "reporting_rules": build_reporting_rules(examples),
        "formal_statement_counts": {
            kind: sum(row["kind"] == kind for row in statements)
            for kind in sorted({row["kind"] for row in statements})
        },
        "examples": examples,
        "cited_checks": cited_checks,
        "claim_boundary": (
            "The appendix proves finite-horizon causal, algebraic, projection, "
            "concentration, router-frequency, and dual-sequence statements under "
            "explicit assumptions. "
            "It does not prove global nonconvex actor-critic convergence, universal "
            "encoder optimality, or deployment-wide performance."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    lines = [
        "# Freq-HRL Scope-Limited Formal Appendix",
        "",
        f"Schema: `{payload['schema_version']}`",
        "",
        "Verification status: internal scope audit only; independent proof review is pending.",
        "",
        "## Formal Statements",
        "",
    ]
    for row in payload["formal_statements"]:
        lines.extend([
            f"### {row['id']} ({row['kind']}): {row['title']}",
            "",
            f"Statement: {row['statement']}",
            "",
            "Assumptions:",
        ])
        lines.extend(f"- {assumption}" for assumption in row["assumptions"])
        lines.extend([
            "",
            f"Proof: {row['proof']}",
            "",
            f"Limitation: {row['limitation']}",
            "",
            f"Diagnostic: {row['diagnostic']}",
        ])
        if row.get("example"):
            lines.extend(["", f"Numeric check: {row['example']}"])
        lines.append("")

    lines.extend(["## Reporting Rules", ""])
    for row in payload["reporting_rules"]:
        lines.extend([
            f"### {row['id']} ({row['kind']}): {row['title']}",
            "",
            f"Rule: {row['statement']}",
            "",
            f"Limitation: {row['limitation']}",
            "",
        ])

    lines.extend([
        "## Empirical Anchors",
        "",
        "Diagnostics test implementation premises; they are not proofs.",
        "",
        "| check | status | delta CI95 |",
        "| --- | --- | ---: |",
    ])
    for name, row in payload["cited_checks"].items():
        if row:
            lines.append(
                f"| {name} | {row.get('status', 'missing')} | "
                f"{row.get('delta_ci95_low', 'NA')} to "
                f"{row.get('delta_ci95_high', 'NA')} |"
            )
    lines.extend(["", "## Claim Boundary", "", payload["claim_boundary"]])
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("transit_hrl/results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/freq_hrl_theory_appendix"),
    )
    args = parser.parse_args()
    payload = build_theory_payload(args.results_root)
    write_outputs(args.output_dir, payload)
    print(f"wrote {args.output_dir}")
    print(
        "theory_appendix "
        f"formal_statements={len(payload['formal_statements'])} "
        f"verification={payload['proof_verification_status']}"
    )


if __name__ == "__main__":
    main()
