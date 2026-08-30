import numpy as np
import pytest

from freq_hrl.core import (
    CausalAuditAlignedGaugeFixer,
    CausalGaugeFixer,
    LeakageRegularizer,
    canonical_responsibility_trace,
)
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


def test_gauge_rejects_uninitialized_or_misaligned_inputs():
    fixer = CausalGaugeFixer()
    with pytest.raises(RuntimeError, match="reset"):
        fixer.split([0.0], [0.0])
    fixer.reset(2)
    with pytest.raises(ValueError, match="align"):
        fixer.split([0.0], [0.0])
