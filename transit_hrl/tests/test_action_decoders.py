import numpy as np
import pytest

from freq_hrl.core import (
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
    at_boundary = plan.activate([0.8])

    assert float(at_boundary[0]) >= float(before_boundary[0])
    assert float(at_boundary[0] - before_boundary[0]) < 0.02
    np.testing.assert_allclose(plan.target, [0.8])
    assert plan.progress == 0.0
    second_macro = [at_boundary]
    second_macro.extend(plan.advance() for _ in range(8))
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
