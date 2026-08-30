import numpy as np
import pytest

from freq_hrl.core import (
    CausalAuditAlignedGaugeFixer,
    CausalGaugeFixer,
    CausalMacroHoldAuditGaugeFixer,
    CausalStreamingAuditProjectionFixer,
    LeakageRegularizer,
    canonical_responsibility_trace,
)
from freq_hrl.core.leakage import causal_rolling_mean, high_pass
from freq_hrl.domains.mujoco import (
    CausalLowerActionRouter,
    lower_action_router_contract,
)


def _trace(upper, lower, *, alpha=0.25, strength=1.0):
    fixer = CausalGaugeFixer(alpha=alpha, strength=strength)
    fixer.reset(upper.shape[1])
    return [fixer.split(u, l) for u, l in zip(upper, lower, strict=True)]


def test_full_strength_gauge_is_factorization_invariant_and_exact():
    rng = np.random.default_rng(42)
    upper = rng.normal(size=(32, 3))
    lower = rng.normal(size=(32, 3))
    transfer = rng.normal(size=(32, 3))
    left = _trace(upper, lower)
    right = _trace(upper + transfer, lower - transfer)
    for lhs, rhs, total in zip(left, right, upper + lower, strict=True):
        np.testing.assert_allclose(lhs["upper"], rhs["upper"], atol=1e-7)
        np.testing.assert_allclose(lhs["lower"], rhs["lower"], atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(lhs["upper"]) + np.asarray(lhs["lower"]),
            total,
            atol=2e-7,
        )
        np.testing.assert_allclose(lhs["reconstruction_error"], 0.0, atol=1e-12)


def test_gauge_is_causal_under_future_changes():
    prefix = np.asarray([[0.2], [0.4], [-0.1]], dtype=np.float64)
    left = np.concatenate([prefix, np.asarray([[10.0]])])
    right = np.concatenate([prefix, np.asarray([[-10.0]])])
    left_upper, left_lower = canonical_responsibility_trace(left, alpha=0.2)
    right_upper, right_lower = canonical_responsibility_trace(right, alpha=0.2)
    np.testing.assert_allclose(left_upper[:3], right_upper[:3])
    np.testing.assert_allclose(left_lower[:3], right_lower[:3])


def test_partial_gauge_preserves_total_without_claiming_invariance():
    fixer = CausalGaugeFixer(alpha=0.5, strength=0.25)
    fixer.reset(2)
    row = fixer.split([0.3, -0.2], [0.4, 0.1], lower_limit=1.0)
    np.testing.assert_allclose(
        np.asarray(row["upper"]) + np.asarray(row["lower"]),
        [0.7, -0.1],
        atol=1e-7,
    )
    assert row["gauge_fixed"] == 0.0


def test_mujoco_total_action_gauge_delegates_to_shared_core():
    router = CausalLowerActionRouter(
        mode="causal_total_action_gauge",
        alpha=0.5,
        strength=1.0,
    )
    router.reset(1)
    first = router.route(
        np.asarray([0.6]), upper_action=np.asarray([0.2]), action_limit=1.0
    )
    second = router.route(
        np.asarray([0.4]), upper_action=np.asarray([0.4]), action_limit=1.0
    )
    np.testing.assert_allclose(
        np.asarray(first["upper_transfer"]) + np.asarray(first["effective"]),
        [0.6],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(second["upper_transfer"]) + np.asarray(second["effective"]),
        [0.4],
        atol=1e-7,
    )
    np.testing.assert_allclose(first["transfer_reconstruction_error"], 0.0)


def test_total_action_gauge_has_an_explicit_runtime_contract():
    assert lower_action_router_contract("causal_total_action_gauge") == (
        "causal_total_action_ema_gauge_fixed_responsibility_with_exact_"
        "pre_split_action_execution_v1"
    )
    with pytest.raises(ValueError, match="unknown"):
        lower_action_router_contract("missing")


def test_audit_aligned_gauge_is_factorization_invariant_and_exact():
    rng = np.random.default_rng(101)
    upper = rng.normal(size=(48, 2))
    lower = rng.normal(size=(48, 2))
    transfer = rng.normal(size=(48, 2))

    def trace(left, right):
        fixer = CausalAuditAlignedGaugeFixer(strength=1.0)
        fixer.reset(2)
        return [fixer.split(u, l) for u, l in zip(left, right, strict=True)]

    original = trace(upper, lower)
    transformed = trace(upper + transfer, lower - transfer)
    for lhs, rhs, total in zip(
        original, transformed, upper + lower, strict=True
    ):
        np.testing.assert_allclose(lhs["upper"], rhs["upper"], atol=1e-6)
        np.testing.assert_allclose(lhs["lower"], rhs["lower"], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(lhs["upper"]) + np.asarray(lhs["lower"]),
            total,
            atol=2e-6,
        )


def test_audit_aligned_feedback_improves_a_fixed_multiband_trace():
    steps = np.arange(256, dtype=np.float64)
    total = (
        0.35 * np.sin(2.0 * np.pi * steps / 96.0)
        + 0.10 * np.sin(2.0 * np.pi * steps / 5.0)
        + 0.20 * (steps >= 128)
    ).reshape(-1, 1)

    def split_trace(fixer):
        fixer.reset(1)
        rows = [fixer.split([0.0], value) for value in total]
        upper = np.asarray([row["upper"] for row in rows]).reshape(-1, 1)
        lower = np.asarray([row["lower"] for row in rows]).reshape(-1, 1)
        return rows, LeakageRegularizer(
            upper_hf_window=8,
            lower_lf_window=32,
        ).compute(upper, lower)

    _, fixed_metrics = split_trace(CausalGaugeFixer(alpha=0.04))
    adaptive_rows, adaptive_metrics = split_trace(
        CausalAuditAlignedGaugeFixer(
            initial_alpha=0.20,
            adaptation_rate=0.03,
        )
    )
    fixed_merit = (
        fixed_metrics["UpperHFPowerAbs"] / 0.075 ** 2
        + fixed_metrics["LowerLFDriftAbs"] / 0.0475 ** 2
    )
    adaptive_merit = (
        adaptive_metrics["UpperHFPowerAbs"] / 0.075 ** 2
        + adaptive_metrics["LowerLFDriftAbs"] / 0.0475 ** 2
    )

    assert adaptive_merit < 0.20 * fixed_merit
    assert adaptive_metrics["LowerLFDriftAbs"] < fixed_metrics["LowerLFDriftAbs"]
    assert adaptive_rows[-1]["alpha_after"] != pytest.approx(0.20)
    np.testing.assert_allclose(
        np.asarray([row["reconstruction_error"] for row in adaptive_rows]),
        0.0,
        atol=1e-12,
    )


def test_mujoco_audit_aligned_gauge_uses_the_shared_projection():
    router = CausalLowerActionRouter(
        mode="causal_audit_aligned_gauge",
        alpha=0.04,
        strength=1.0,
    )
    router.reset(1)
    row = router.route(
        np.asarray([0.4]), upper_action=np.asarray([0.3]), action_limit=1.0
    )

    np.testing.assert_allclose(
        np.asarray(row["upper_transfer"]) + np.asarray(row["effective"]),
        [0.4],
        atol=1e-7,
    )
    np.testing.assert_allclose(row["transfer_reconstruction_error"], 0.0)
    assert lower_action_router_contract("causal_audit_aligned_gauge") == (
        "causal_total_action_gauge_fixed_adaptive_lpf32_hpf8_feedback_with_"
        "exact_pre_split_action_execution_v1"
    )


def test_macro_hold_gauge_is_exact_invariant_and_upper_rate_compatible():
    rng = np.random.default_rng(303)
    upper = rng.normal(scale=0.15, size=(48, 2))
    lower = rng.normal(scale=0.20, size=(48, 2))
    transfer = rng.normal(scale=0.10, size=(48, 2))
    boundaries = [index % 8 == 0 for index in range(48)]

    def trace(left, right):
        fixer = CausalMacroHoldAuditGaugeFixer(strength=1.0)
        fixer.reset(2)
        return [
            fixer.split(
                u,
                l,
                macro_boundary=boundary,
                lower_limit=None,
            )
            for u, l, boundary in zip(
                left, right, boundaries, strict=True
            )
        ]

    original = trace(upper, lower)
    transformed = trace(upper + transfer, lower - transfer)
    for index, (lhs, rhs, total) in enumerate(zip(
        original, transformed, upper + lower, strict=True
    )):
        np.testing.assert_allclose(lhs["upper"], rhs["upper"], atol=1e-6)
        np.testing.assert_allclose(lhs["lower"], rhs["lower"], atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(lhs["upper"]) + np.asarray(lhs["lower"]),
            total,
            atol=2e-6,
        )
        if index % 8:
            np.testing.assert_allclose(
                lhs["upper"], original[index - 1]["upper"], atol=1e-7
            )


def test_macro_hold_gauge_preserves_a_synthetic_upper_hf_budget():
    steps = np.arange(512, dtype=np.float64)
    total = (
        0.35 * np.sin(2.0 * np.pi * steps / 160.0)
        + 0.08 * np.sin(2.0 * np.pi * steps / 5.0)
        + 0.15 * (steps >= 256)
    ).reshape(-1, 1)
    boundary_steps = np.arange(0, 512, 16, dtype=np.float64)
    latent_upper = np.repeat(
        0.30 * np.sin(2.0 * np.pi * boundary_steps / 160.0)
        + 0.07 * np.sin(2.0 * np.pi * boundary_steps / 17.0),
        16,
    )[:512].reshape(-1, 1)
    latent_lower = total - latent_upper
    fixer = CausalMacroHoldAuditGaugeFixer()
    fixer.reset(1)
    rows = [
        fixer.split(
            upper,
            lower,
            macro_boundary=index % 16 == 0,
            lower_limit=None,
        )
        for index, (upper, lower) in enumerate(zip(
            latent_upper, latent_lower, strict=True
        ))
    ]
    fixed_upper = np.asarray([row["upper"] for row in rows]).reshape(-1, 1)
    fixed_lower = np.asarray([row["lower"] for row in rows]).reshape(-1, 1)
    metrics = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(fixed_upper, fixed_lower)
    latent_metrics = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(latent_upper, latent_lower)

    assert metrics["UpperHFPowerAbs"] < 0.075 ** 2
    assert metrics["LowerLFDriftAbs"] < latent_metrics["LowerLFDriftAbs"]


def test_mujoco_macro_hold_gauge_requires_and_records_macro_boundaries():
    router = CausalLowerActionRouter(
        mode="causal_macro_hold_audit_gauge",
        alpha=0.20,
        strength=1.0,
    )
    router.reset(1)
    with pytest.raises(RuntimeError, match="boundary"):
        router.route(
            np.asarray([0.4]),
            upper_action=np.asarray([0.3]),
            action_limit=1.0,
        )
    first = router.route(
        np.asarray([0.4]),
        upper_action=np.asarray([0.3]),
        action_limit=1.0,
        macro_boundary=True,
    )
    second = router.route(
        np.asarray([0.2]),
        upper_action=np.asarray([0.3]),
        action_limit=1.0,
        macro_boundary=False,
    )
    np.testing.assert_allclose(first["transfer_reconstruction_error"], 0.0)
    np.testing.assert_allclose(second["transfer_reconstruction_error"], 0.0)
    assert lower_action_router_contract("causal_macro_hold_audit_gauge") == (
        "causal_total_action_gauge_fixed_at_upper_macro_boundaries_with_"
        "adaptive_lpf32_hpf8_feedback_and_exact_pre_split_action_execution_v1"
    )


def test_streaming_audit_projection_matches_batch_filters_and_budgets():
    steps = np.arange(512, dtype=np.float64)
    total = (
        0.35 * np.sin(2.0 * np.pi * steps / 160.0)
        + 0.08 * np.sin(2.0 * np.pi * steps / 5.0)
        + 0.15 * (steps >= 256)
    ).reshape(-1, 1)
    fixer = CausalStreamingAuditProjectionFixer(
        planning_horizon=16,
        upper_rms_budget=0.075,
        lower_rms_budget=0.0475,
    )
    fixer.reset(1)
    rows = [fixer.split([0.0], value) for value in total]
    upper = np.asarray([row["upper"] for row in rows]).reshape(-1, 1)
    lower = np.asarray([row["lower"] for row in rows]).reshape(-1, 1)
    online_upper = np.asarray([
        row["streaming_upper_high"] for row in rows
    ]).reshape(-1, 1)
    online_lower = np.asarray([
        row["streaming_lower_low"] for row in rows
    ]).reshape(-1, 1)
    metrics = LeakageRegularizer(
        upper_hf_window=8, lower_lf_window=32
    ).compute(upper, lower)

    np.testing.assert_allclose(online_upper, high_pass(upper, 8), atol=1e-7)
    np.testing.assert_allclose(
        online_lower, causal_rolling_mean(lower, 32), atol=1e-7
    )
    np.testing.assert_allclose(upper + lower, total, atol=1e-7)
    assert metrics["UpperHFPowerAbs"] <= 0.075 ** 2
    assert metrics["LowerLFDriftAbs"] <= 0.0475 ** 2
    assert all(row["upper_budget_feasible_rate"] == 1.0 for row in rows)
    action_blocks, scalars = fixer.policy_context
    assert len(action_blocks) == 38
    assert scalars == (1.0, 1.0)


def test_streaming_audit_projection_is_factorization_invariant_and_causal():
    rng = np.random.default_rng(17401)
    prefix = rng.normal(scale=0.15, size=(48, 2))
    suffix_left = rng.normal(scale=0.15, size=(16, 2))
    suffix_right = rng.normal(scale=0.15, size=(16, 2))
    total_left = np.concatenate((prefix, suffix_left), axis=0)
    total_right = np.concatenate((prefix, suffix_right), axis=0)

    def trace(total, gauge_shift):
        fixer = CausalStreamingAuditProjectionFixer(strength=1.0)
        fixer.reset(2)
        rows = [
            fixer.split(shift, value - shift)
            for value, shift in zip(total, gauge_shift, strict=True)
        ]
        return np.asarray([row["upper"] for row in rows]), rows

    shift_a = rng.normal(scale=0.10, size=total_left.shape)
    shift_b = rng.normal(scale=0.10, size=total_left.shape)
    factor_a, rows_a = trace(total_left, shift_a)
    factor_b, _ = trace(total_left, shift_b)
    future_changed, _ = trace(total_right, shift_a)

    np.testing.assert_allclose(factor_a, factor_b, atol=1e-7)
    np.testing.assert_allclose(factor_a[:48], future_changed[:48], atol=1e-7)
    np.testing.assert_allclose(
        np.asarray([row["reconstruction_error"] for row in rows_a]),
        0.0,
        atol=1e-12,
    )


def test_streaming_audit_projection_state_is_strength_invariant():
    rng = np.random.default_rng(17402)
    upper = rng.uniform(-0.35, 0.35, size=(64, 3))
    lower = rng.uniform(-0.35, 0.35, size=(64, 3))
    control = CausalStreamingAuditProjectionFixer(strength=0.0)
    projected = CausalStreamingAuditProjectionFixer(strength=1.0)
    control.reset(3)
    projected.reset(3)

    for upper_row, lower_row in zip(upper, lower, strict=True):
        control_blocks, control_scalars = control.policy_context
        projected_blocks, projected_scalars = projected.policy_context
        np.testing.assert_allclose(control_blocks, projected_blocks, atol=0.0)
        np.testing.assert_allclose(control_scalars, projected_scalars, atol=0.0)
        control_row = control.split(upper_row, lower_row)
        projected_row = projected.split(upper_row, lower_row)
        np.testing.assert_allclose(
            control_row["canonical_upper"],
            projected_row["canonical_upper"],
            atol=0.0,
        )
        np.testing.assert_allclose(
            control_row["canonical_lower"],
            projected_row["canonical_lower"],
            atol=0.0,
        )


def test_streaming_audit_projection_reports_physical_budget_infeasibility():
    fixer = CausalStreamingAuditProjectionFixer(upper_rms_budget=0.075)
    fixer.reset(1)
    for _ in range(7):
        fixer.split([0.0], [0.0])
    row = fixer.split([1.0], [1.0], upper_limit=1.0, lower_limit=1.0)

    assert row["upper_budget_feasible_rate"] == 0.0
    assert row["upper_budget_violation_rms"] > 0.0
    np.testing.assert_allclose(row["canonical_upper"], [1.0], atol=1e-12)
    np.testing.assert_allclose(row["canonical_lower"], [1.0], atol=1e-12)
    np.testing.assert_allclose(row["reconstruction_error"], 0.0, atol=1e-12)


def test_gauge_rejects_uninitialized_or_misaligned_inputs():
    fixer = CausalGaugeFixer()
    with pytest.raises(RuntimeError, match="reset"):
        fixer.split([0.0], [0.0])
    fixer.reset(2)
    with pytest.raises(ValueError, match="align"):
        fixer.split([0.0], [0.0])
