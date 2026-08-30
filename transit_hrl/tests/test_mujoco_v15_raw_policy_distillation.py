import copy

from scripts import mujoco_v15_raw_policy_distillation_preflight_spec as spec
from scripts.probe_mujoco_raw_policy_distillation import (
    FREQUENCY_ENDPOINTS,
    _complete_endpoint_diagnostics,
    _paths_for_roots,
)


def _rows(value: float) -> list[dict[str, float | int | str]]:
    return [
        {
            "disturbance_mode": mode,
            "seed": index + 1,
            "reward_mean": 10.0,
            **{endpoint: value for endpoint in FREQUENCY_ENDPOINTS},
        }
        for index, mode in enumerate(spec.DISTURBANCE_MODES)
    ]


def _summary() -> dict[str, float]:
    return {
        "lower_deployment_frequency_reference_reduction_fraction": 0.05,
        "upper_deployment_frequency_reference_reduction_fraction": 0.05,
        "lower_deployment_frequency_rms_budget": 1e-3,
        "upper_deployment_frequency_rms_budget": 1e-3,
    }


def test_v15_seed_roles_and_design_folds_are_disjoint():
    roles = (spec.DISTILL_ROOTS, spec.DESIGN_ROOTS, spec.VALIDATION_ROOTS)
    flattened = tuple(root for role in roles for root in role)

    assert len(flattened) == len(set(flattened)) == 20
    assert len(spec.DESIGN_ROOTS) % spec.DESIGN_FOLD_COUNT == 0
    assert len(spec.CANDIDATES) == 18


def test_crossed_paths_keep_each_root_with_all_disturbance_modes():
    paths = _paths_for_roots("HalfCheetah-v5", (17, 29))
    mode_count = len(spec.DISTURBANCE_MODES)

    assert len(paths) == 2 * mode_count
    assert len({int(path["seed"]) for path in paths}) == len(paths)
    assert tuple(path["disturbance_mode"] for path in paths[:mode_count]) == (
        spec.DISTURBANCE_MODES
    )
    assert tuple(path["disturbance_mode"] for path in paths[mode_count:]) == (
        spec.DISTURBANCE_MODES
    )


def test_complete_endpoint_gate_rejects_any_single_frequency_failure():
    baseline = _rows(1.0)
    candidate = _rows(0.90)
    passed = _complete_endpoint_diagnostics(
        candidate,
        baseline,
        _summary(),
        risk_mode="mode_mean",
        cvar_alpha=0.5,
    )
    assert passed["complete"] is True
    assert all(
        value == 0.0
        for value in passed[
            "frequency_endpoint_maximum_normalized_violations"
        ].values()
    )

    for endpoint in FREQUENCY_ENDPOINTS:
        failed_rows = copy.deepcopy(candidate)
        failed_rows[0][endpoint] = 0.96
        failed = _complete_endpoint_diagnostics(
            failed_rows,
            baseline,
            _summary(),
            risk_mode="mode_mean",
            cvar_alpha=0.5,
        )
        assert failed["complete"] is False
        assert failed[
            "frequency_endpoint_maximum_normalized_violations"
        ][endpoint] > 0.0


def test_complete_endpoint_gate_rejects_reward_floor_failure():
    baseline = _rows(1.0)
    candidate = _rows(0.90)
    candidate[0]["reward_mean"] = 9.0

    failed = _complete_endpoint_diagnostics(
        candidate,
        baseline,
        _summary(),
        risk_mode="mode_mean",
        cvar_alpha=0.5,
    )

    assert failed["complete"] is False
    assert failed["reward_maximum_normalized_violation"] > 0.0
