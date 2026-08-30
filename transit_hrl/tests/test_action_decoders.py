import numpy as np
import pytest

from freq_hrl.core import (
    CausalMacroHoldAuditGaugeFixer,
    CausalSmoothMacroGaugeFixer,
    CausalSmoothstepMacroPlan,
    CausalZeroDCMacroProjector,
    LeakageRegularizer,
)
from freq_hrl.domains.mujoco import (
    CausalLowerActionRouter,
    lower_action_router_contract,
)


def test_smoothstep_macro_plan_is_continuous_and_boundary_frozen():
    plan = CausalSmoothstepMacroPlan(macro_steps=8)
    plan.reset(1)
    np.testing.assert_allclose(plan.activate([0.2]), [0.2])
    first_macro = [plan.current.copy()]
    first_macro.extend(plan.advance() for _ in range(7))
    before_boundary = first_macro[-1].copy()
    np.testing.assert_allclose(before_boundary, [0.2], atol=0.0)
    at_boundary = plan.activate([0.8])

    assert float(at_boundary[0]) >= float(before_boundary[0])
    assert float(at_boundary[0] - before_boundary[0]) < 0.02
    np.testing.assert_allclose(plan.target, [0.8])
    assert plan.progress == 0.0
    second_macro = [at_boundary]
    second_macro.extend(plan.advance() for _ in range(7))
    assert np.all(np.diff(np.asarray(second_macro).reshape(-1)) >= -1e-12)
    np.testing.assert_allclose(second_macro[-1], [0.8], atol=1e-7)


def test_smoothstep_plan_reduces_upper_hpf_against_macro_hold():
    targets = np.asarray([0.0, 0.8, -0.5, 0.6, -0.2], dtype=np.float64)
    held = np.repeat(targets, 16).reshape(-1, 1)
    plan = CausalSmoothstepMacroPlan(macro_steps=16)
    plan.reset(1)
    smooth = []
    for target in targets:
        smooth.append(plan.activate([target]))
        smooth.extend(plan.advance() for _ in range(15))
    smooth_values = np.asarray(smooth, dtype=np.float64).reshape(-1, 1)
    zeros = np.zeros_like(held)
    metric = LeakageRegularizer(upper_hf_window=8, lower_lf_window=32)
    held_hf = metric.compute(held, zeros)["UpperHFPowerAbs"]
    smooth_hf = metric.compute(smooth_values, zeros)["UpperHFPowerAbs"]
    assert smooth_hf < 0.45 * held_hf


def test_smoothstep_plan_exposes_only_the_frozen_macro_suffix():
    plan = CausalSmoothstepMacroPlan(macro_steps=4)
    plan.reset(1)
    plan.activate([0.4])
    np.testing.assert_allclose(plan.future_values(), [[0.4], [0.4], [0.4]])
    plan.activate([0.8])
    future = plan.future_values().reshape(-1)
    assert future.shape == (3,)
    assert np.all(np.diff(future) > 0.0)
    plan.advance()
    np.testing.assert_allclose(plan.future_values().reshape(-1), future[1:])


def test_smooth_macro_gauge_is_function_preserving_and_identifiable():
    totals = np.asarray(
        [[0.2], [0.5], [-0.1], [0.4], [0.7], [0.0], [-0.4], [0.1]],
        dtype=np.float64,
    )
    supplied_upper = 0.35 * totals
    supplied_lower = totals - supplied_upper
    gauge_shift = np.asarray(
        [[0.15], [-0.2], [0.1], [0.05], [-0.1], [0.2], [-0.15], [0.0]],
        dtype=np.float64,
    )

    def run(strength, upper, lower):
        fixer = CausalSmoothMacroGaugeFixer(
            macro_steps=4, alpha=0.2, strength=strength
        )
        fixer.reset(1)
        rows = []
        contexts = []
        for index, (upper_row, lower_row) in enumerate(zip(upper, lower)):
            rows.append(fixer.split(
                upper_row,
                lower_row,
                macro_boundary=index % 4 == 0,
                upper_limit=1.0,
                lower_limit=1.0,
            ))
            contexts.append(fixer.context)
        return rows, np.asarray(contexts)

    control, control_context = run(0.0, supplied_upper, supplied_lower)
    fixed, fixed_context = run(1.0, supplied_upper, supplied_lower)
    shifted, shifted_context = run(
        1.0,
        supplied_upper + gauge_shift,
        supplied_lower - gauge_shift,
    )

    control_upper = np.asarray([row["upper"] for row in control])
    control_lower = np.asarray([row["lower"] for row in control])
    fixed_upper = np.asarray([row["upper"] for row in fixed])
    fixed_lower = np.asarray([row["lower"] for row in fixed])
    shifted_upper = np.asarray([row["upper"] for row in shifted])
    shifted_lower = np.asarray([row["lower"] for row in shifted])
    np.testing.assert_allclose(control_upper, supplied_upper, atol=0.0)
    np.testing.assert_allclose(control_lower, supplied_lower, atol=0.0)
    np.testing.assert_allclose(control_context, fixed_context, atol=0.0)
    np.testing.assert_allclose(fixed_context, shifted_context, atol=0.0)
    np.testing.assert_allclose(fixed_upper, shifted_upper, atol=0.0)
    np.testing.assert_allclose(fixed_lower, shifted_lower, atol=0.0)
    np.testing.assert_allclose(fixed_upper + fixed_lower, totals, atol=1e-7)
    assert np.max(np.abs(fixed_upper)) <= 1.0
    assert np.max(np.abs(fixed_lower)) <= 1.0
    assert max(float(row["canonical_component_clip_rate"]) for row in fixed) == 0.0
    np.testing.assert_allclose(
        fixed[3]["smooth_requested"], fixed[3]["smooth_target"], atol=0.0
    )


def test_smooth_macro_gauge_reduces_upper_hpf_against_macro_hold_gauge():
    totals = np.repeat(
        np.asarray([0.0, 0.75, -0.5, 0.65, -0.25], dtype=np.float64), 16
    ).reshape(-1, 1)
    zeros = np.zeros_like(totals)
    held = CausalMacroHoldAuditGaugeFixer(
        initial_alpha=0.2, adaptation_rate=0.0, strength=1.0
    )
    smooth = CausalSmoothMacroGaugeFixer(
        macro_steps=16, alpha=0.2, strength=1.0
    )
    held.reset(1)
    smooth.reset(1)
    held_upper = []
    smooth_upper = []
    for index, total in enumerate(totals):
        boundary = index % 16 == 0
        held_upper.append(held.split(
            zeros[index], total, macro_boundary=boundary, lower_limit=1.0
        )["upper"])
        smooth_upper.append(smooth.split(
            zeros[index],
            total,
            macro_boundary=boundary,
            upper_limit=1.0,
            lower_limit=1.0,
        )["upper"])
    metric = LeakageRegularizer(upper_hf_window=8, lower_lf_window=32)
    held_hf = metric.compute(
        np.asarray(held_upper), zeros
    )["UpperHFPowerAbs"]
    smooth_hf = metric.compute(
        np.asarray(smooth_upper), zeros
    )["UpperHFPowerAbs"]
    assert smooth_hf < held_hf


def test_mujoco_smooth_macro_gauge_router_preserves_additive_action():
    upper = np.asarray([[0.2], [0.3], [0.4], [0.5], [-0.2], [-0.1], [0.0], [0.1]])
    lower = np.asarray([[0.1], [-0.2], [0.15], [-0.1], [0.2], [0.3], [-0.2], [0.0]])

    def run(strength):
        router = CausalLowerActionRouter(
            mode="causal_smooth_macro_gauge",
            alpha=0.2,
            strength=strength,
            macro_steps=4,
        )
        router.reset(1)
        rows = []
        contexts = []
        for index, (upper_row, lower_row) in enumerate(zip(upper, lower)):
            rows.append(router.route(
                lower_row,
                upper_action=upper_row,
                action_limit=1.0,
                macro_boundary=index % 4 == 0,
            ))
            contexts.append(router.context)
        return rows, np.asarray(contexts)

    control, control_context = run(0.0)
    fixed, fixed_context = run(1.0)
    control_lower = np.asarray([row["effective"] for row in control])
    fixed_lower = np.asarray([row["effective"] for row in fixed])
    fixed_transfer = np.asarray([row["upper_transfer"] for row in fixed])
    np.testing.assert_allclose(control_lower, lower, atol=0.0)
    np.testing.assert_allclose(control_context, fixed_context, atol=0.0)
    np.testing.assert_allclose(
        upper + fixed_transfer + fixed_lower, upper + lower, atol=1e-7
    )
    assert max(float(row["headroom_clip_rate"]) for row in fixed) == 0.0
    assert lower_action_router_contract("causal_smooth_macro_gauge") == (
        "causal_prior_total_low_pass_macro_target_with_frozen_smooth_curve_"
        "bounded_components_and_exact_pre_split_action_execution_v1"
    )


def test_zero_dc_projector_is_causal_bounded_and_exact_per_macro():
    rng = np.random.default_rng(71)
    proposals = rng.normal(scale=0.9, size=(48, 3))
    projector = CausalZeroDCMacroProjector(macro_steps=8)
    projector.reset(3)
    effective = []
    completion_errors = []
    for index, proposal in enumerate(proposals):
        row = projector.project(
            proposal,
            macro_boundary=index % 8 == 0,
            action_limit=1.0,
        )
        effective.append(row["effective"])
        if row["macro_completed"]:
            completion_errors.append(row["macro_completion_error_rms"])
    actions = np.asarray(effective, dtype=np.float64)

    assert np.max(np.abs(actions)) <= 1.0
    np.testing.assert_allclose(actions.reshape(6, 8, 3).sum(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(completion_errors, 0.0, atol=1e-7)


def test_zero_dc_projector_does_not_use_future_proposals():
    prefix = np.asarray([[0.7], [0.6], [-0.4], [0.2]], dtype=np.float64)
    suffix_a = np.asarray([[0.9], [0.9], [0.9], [0.9]], dtype=np.float64)
    suffix_b = -suffix_a

    def run(values):
        projector = CausalZeroDCMacroProjector(macro_steps=8)
        projector.reset(1)
        return np.asarray([
            projector.project(
                value,
                macro_boundary=index == 0,
                action_limit=1.0,
            )["effective"]
            for index, value in enumerate(values)
        ])

    left = run(np.concatenate([prefix, suffix_a]))
    right = run(np.concatenate([prefix, suffix_b]))
    np.testing.assert_allclose(left[:4], right[:4], atol=0.0)


def test_zero_dc_projector_reserves_frozen_upper_headroom_and_future_repayment():
    macro_steps = 8
    upper = np.full((macro_steps, 1), 0.8, dtype=np.float64)
    proposals = np.full((macro_steps, 1), 0.9, dtype=np.float64)
    projector = CausalZeroDCMacroProjector(macro_steps=macro_steps)
    projector.reset(1)
    effective = []
    direct = []
    for index, proposal in enumerate(proposals):
        row = projector.project(
            proposal,
            macro_boundary=index == 0,
            action_limit=1.0,
            current_upper_action=upper[index],
            future_upper_actions=upper[index + 1:],
            total_action_limit=1.0,
        )
        effective.append(row["effective"])
        direct.append(row["direct_feasible"])
    lower = np.asarray(effective, dtype=np.float64).reshape(-1)
    direct_lower = np.asarray(direct, dtype=np.float64).reshape(-1)

    np.testing.assert_allclose(np.sum(lower), 0.0, atol=1e-7)
    assert np.max(np.abs(upper.reshape(-1) + lower)) <= 1.0 + 1e-7
    np.testing.assert_allclose(direct_lower, 0.2, atol=1e-7)
    np.testing.assert_allclose(
        upper.reshape(-1) + direct_lower,
        np.clip(upper.reshape(-1) + proposals.reshape(-1), -1.0, 1.0),
        atol=1e-7,
    )


def test_zero_dc_projector_requires_explicit_macro_boundaries():
    projector = CausalZeroDCMacroProjector(macro_steps=4)
    projector.reset(1)
    with pytest.raises(RuntimeError, match="boundary"):
        projector.project([0.2], macro_boundary=False, action_limit=1.0)


def test_mujoco_zero_dc_router_exposes_debt_and_closes_each_macro():
    router = CausalLowerActionRouter(
        mode="causal_macro_zero_dc",
        strength=1.0,
        macro_steps=4,
    )
    router.reset(1)
    rows = [
        router.route(
            np.asarray([value]),
            action_limit=1.0,
            macro_boundary=index % 4 == 0,
        )
        for index, value in enumerate([0.8, 0.8, 0.8, 0.8] * 2)
    ]
    actions = np.asarray([row["effective"] for row in rows]).reshape(2, 4)

    np.testing.assert_allclose(actions.sum(axis=1), 0.0, atol=1e-7)
    assert max(row["macro_projection_rate"] for row in rows) > 0.0
    assert max(row["macro_completion_error_rms"] for row in rows) <= 1e-7
    assert lower_action_router_contract("causal_macro_zero_dc") == (
        "causal_bounded_lower_projection_with_exact_zero_sum_on_each_complete_"
        "upper_macro_interval_v1"
    )


def test_headroom_zero_dc_router_is_function_continuous_and_exact_at_full_strength():
    upper = np.full((4, 1), 0.8, dtype=np.float64)
    proposals = np.full((4, 1), 0.9, dtype=np.float64)

    def run(strength):
        router = CausalLowerActionRouter(
            mode="causal_macro_zero_dc_headroom",
            strength=strength,
            macro_steps=4,
        )
        router.reset(1)
        rows = [
            router.route(
                proposal,
                upper_action=upper[index],
                future_upper_actions=upper[index + 1:],
                action_limit=1.0,
                macro_boundary=index == 0,
            )
            for index, proposal in enumerate(proposals)
        ]
        return router, rows

    zero_router, zero_rows = run(0.0)
    full_router, full_rows = run(1.0)
    zero_lower = np.asarray([row["effective"] for row in zero_rows]).reshape(-1)
    full_lower = np.asarray([row["effective"] for row in full_rows]).reshape(-1)

    np.testing.assert_allclose(
        upper.reshape(-1) + zero_lower,
        np.clip(upper.reshape(-1) + proposals.reshape(-1), -1.0, 1.0),
        atol=1e-7,
    )
    np.testing.assert_allclose(np.sum(full_lower), 0.0, atol=1e-7)
    assert np.max(np.abs(upper.reshape(-1) + full_lower)) <= 1.0 + 1e-7
    assert full_rows[-1]["macro_completion_error_rms"] <= 1e-7
    assert zero_rows[-1]["macro_completion_error_rms"] > 0.1
    np.testing.assert_allclose(zero_router.promotion_context, [0.9])
    np.testing.assert_allclose(full_router.promotion_context, [0.9])
    assert lower_action_router_contract("causal_macro_zero_dc_headroom") == (
        "causal_upper_plan_headroom_feasible_lower_homotopy_with_exact_zero_"
        "sum_at_full_strength_and_function_continuity_at_zero_strength_v1"
    )
