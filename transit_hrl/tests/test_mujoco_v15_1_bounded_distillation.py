from argparse import Namespace

from scripts import mujoco_v15_1_bounded_distillation_preflight_spec as spec
from scripts import mujoco_v15_raw_policy_distillation_preflight_spec as previous
from scripts.analyze_mujoco_v15_1_bounded_distillation import analyze_payloads
from scripts.probe_mujoco_raw_policy_distillation import FREQUENCY_ENDPOINTS
from scripts.submit_mujoco_v15_1_bounded_distillation_scheduleurm import (
    build_probe_command,
    build_scheduler_spec,
    selected_cells,
)


def _payload(environment: str, seed: int) -> dict:
    gate = {
        "complete": True,
        "reward_maximum_normalized_violation": 0.0,
        "frequency_endpoint_maximum_normalized_violations": {
            endpoint: 0.0 for endpoint in FREQUENCY_ENDPOINTS
        },
    }
    return {
        "probe_version": spec.PROBE_VERSION,
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
        "design_eligible_candidate_count": 1,
        "selected_index": 0,
        "selected_candidate": {"config": dict(spec.CANDIDATES[0])},
        "validation": {
            "supported": True,
            "merit_gate": True,
            "complete_endpoint_gate": gate,
            "candidate_snapshot": {
                "frequency_violation_merit": 0.0,
                "worst_frequency_violation": 0.0,
            },
        },
        "validation_supported": True,
        "status": "raw_policy_distillation_preflight_supported",
    }


def _args() -> Namespace:
    return Namespace(
        run_name="v15_1_unit",
        anchor_run_name=spec.ANCHOR_RUN_NAME,
        python_executable="python3",
        priority="normal",
        nodes=[f"node00{index}" for index in range(1, 7)],
    )


def test_v15_1_roots_are_fresh_and_candidate_grid_is_complete():
    current = set(spec.DISTILL_ROOTS + spec.DESIGN_ROOTS + spec.VALIDATION_ROOTS)
    consumed = set(
        previous.DISTILL_ROOTS
        + previous.DESIGN_ROOTS
        + previous.VALIDATION_ROOTS
    )

    assert len(current) == 24
    assert not current & consumed
    assert len(spec.CANDIDATES) == 108
    assert {candidate["raw_target_limit"] for candidate in spec.CANDIDATES} == {
        2.5, 3.5
    }
    assert {
        candidate["head_delta_rms_limit"] for candidate in spec.CANDIDATES
    } == {0.02, 0.05, 0.1}
    assert {candidate["router_strength"] for candidate in spec.CANDIDATES} == {
        0.5, 0.75, 1.0
    }


def test_v15_1_analysis_uses_its_own_protocol_status():
    payloads = [
        (environment, seed, _payload(environment, seed))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]

    result = analyze_payloads(payloads)

    assert result["analysis_version"] == spec.ANALYSIS_VERSION
    assert result["status"] == spec.SUPPORTED_ANALYSIS_STATUS


def test_v15_1_launcher_is_dynamic_and_uses_one_core_per_candidate():
    assert len(selected_cells()) == spec.EXPECTED_CELL_COUNT
    environment, seed = selected_cells()[0]
    scheduler = build_scheduler_spec(_args(), environment, seed)

    assert scheduler["require_node"] is None
    assert scheduler["cpu"] == spec.WORKERS == len(spec.CANDIDATES)
    assert scheduler["ram_mb"] == spec.RAM_MB_PER_TASK
    assert set(scheduler["allowed_nodes"]) == set(_args().nodes)


def test_v15_1_command_freezes_new_roots_and_versioned_probe():
    environment, seed = selected_cells()[0]
    command = build_probe_command(_args(), environment, seed)

    assert "scripts/probe_mujoco_v15_1_bounded_distillation.py" in command
    assert f"--distill-roots {','.join(map(str, spec.DISTILL_ROOTS))}" in command
    assert f"--design-roots {','.join(map(str, spec.DESIGN_ROOTS))}" in command
    assert f"--validation-roots {','.join(map(str, spec.VALIDATION_ROOTS))}" in command
    assert f"--workers {spec.WORKERS}" in command
    assert command.endswith("&& echo DONE")
