import copy
from argparse import Namespace

import pytest

from scripts import mujoco_v15_raw_policy_distillation_preflight_spec as spec
from scripts.analyze_mujoco_v15_raw_policy_distillation import analyze_payloads
from scripts.probe_mujoco_raw_policy_distillation import (
    FREQUENCY_ENDPOINTS,
    PROBE_VERSION,
)
from scripts.submit_mujoco_v15_raw_policy_distillation_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


def _payload(environment: str, seed: int, *, supported: bool) -> dict:
    gate = {
        "complete": supported,
        "reward_maximum_normalized_violation": 0.0 if supported else 0.1,
        "frequency_endpoint_maximum_normalized_violations": {
            endpoint: 0.0 for endpoint in FREQUENCY_ENDPOINTS
        },
    }
    validation = ({
        "supported": True,
        "merit_gate": True,
        "complete_endpoint_gate": gate,
        "candidate_snapshot": {
            "frequency_violation_merit": 0.0,
            "worst_frequency_violation": 0.0,
        },
    } if supported else None)
    selected_index = 0 if supported else None
    selected = ({
        "config": dict(spec.CANDIDATES[0]),
    } if supported else None)
    return {
        "probe_version": PROBE_VERSION,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "environment": environment,
        "optimizer_seed": seed,
        "distill_roots": list(spec.DISTILL_ROOTS),
        "design_roots": list(spec.DESIGN_ROOTS),
        "validation_roots": list(spec.VALIDATION_ROOTS),
        "candidate_count": len(spec.CANDIDATES),
        "candidates": [
            {"candidate_index": index}
            for index in range(len(spec.CANDIDATES))
        ],
        "design_eligible_candidate_count": int(supported),
        "selected_index": selected_index,
        "selected_candidate": selected,
        "validation": validation,
        "validation_supported": supported,
        "status": (
            "raw_policy_distillation_preflight_supported"
            if supported else "no_complete_raw_policy_candidate"
        ),
    }


def _all_payloads(*, supported: bool):
    return [
        (environment, seed, _payload(environment, seed, supported=supported))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]


def test_v15_analysis_requires_all_environments_for_support():
    result = analyze_payloads(_all_payloads(supported=True))

    assert result["cell_count"] == spec.EXPECTED_CELL_COUNT
    assert result["validation_supported_count"] == spec.EXPECTED_CELL_COUNT
    assert result["status"] == (
        "raw_policy_distillation_preflight_supported_all_environments"
    )


def test_v15_analysis_preserves_a_negative_environment():
    payloads = _all_payloads(supported=True)
    environment, seed, _ = payloads[-1]
    payloads[-1] = (environment, seed, _payload(
        environment, seed, supported=False
    ))

    result = analyze_payloads(payloads)

    assert result["validation_supported_count"] == spec.EXPECTED_CELL_COUNT - 1
    assert result["status"] == "raw_policy_distillation_preflight_not_supported"


def test_v15_analysis_rejects_root_role_drift():
    payloads = _all_payloads(supported=True)
    environment, seed, payload = payloads[0]
    changed = copy.deepcopy(payload)
    changed["validation_roots"][0] += 1
    payloads[0] = (environment, seed, changed)

    with pytest.raises(ValueError, match="validation_roots drifted"):
        analyze_payloads(payloads)


def _launcher_args() -> Namespace:
    return Namespace(
        run_name="v15_unit",
        anchor_run_name=spec.ANCHOR_RUN_NAME,
        python_executable="python3",
        priority="normal",
        nodes=[f"node00{index}" for index in range(1, 7)],
    )


def test_v15_launcher_is_dynamic_and_resources_match_workers():
    assert len(selected_cells()) == spec.EXPECTED_CELL_COUNT
    environment, seed = selected_cells()[0]
    payload = build_scheduler_spec(_launcher_args(), environment, seed)

    assert payload["require_node"] is None
    assert payload["cpu"] == spec.CPU_PER_TASK == spec.WORKERS
    assert payload["ram_mb"] == spec.RAM_MB_PER_TASK
    assert set(payload["allowed_nodes"]) == set(_launcher_args().nodes)
    assert len(payload["wait_for_files"]) == 2
    assert payload["reroute_on_node_down"] is True


def test_v15_launcher_command_freezes_root_roles_and_worker_count():
    environment, seed = selected_cells()[0]
    command = build_probe_command(_launcher_args(), environment, seed)

    assert f"--distill-roots {','.join(map(str, spec.DISTILL_ROOTS))}" in command
    assert f"--design-roots {','.join(map(str, spec.DESIGN_ROOTS))}" in command
    assert f"--validation-roots {','.join(map(str, spec.VALIDATION_ROOTS))}" in command
    assert f"--workers {spec.WORKERS}" in command
    assert command.endswith("&& echo DONE")
