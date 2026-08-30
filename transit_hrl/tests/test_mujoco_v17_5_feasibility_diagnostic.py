from types import SimpleNamespace

from scripts.run_mujoco_v17_5_feasibility_diagnostic import (
    ROUTER_MODES,
    diagnose_projection_limit,
)
from scripts.submit_mujoco_v17_5_feasibility_diagnostic_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)


def _aggregate(*, regret: float, upper_floor: float, lower_floor: float):
    return {
        "LowerRouterBudgetExcessRegretRMSMax": regret,
        "LowerRouterUnavoidableUpperBudgetViolationRMSMax": upper_floor,
        "LowerRouterUnavoidableLowerBudgetViolationRMSMax": lower_floor,
    }


def test_diagnosis_selects_policy_work_when_projection_is_already_optimal():
    result = diagnose_projection_limit(
        {"exact_legacy_replay": True},
        {
            ROUTER_MODES[0]: _aggregate(
                regret=0.0, upper_floor=0.0, lower_floor=0.02
            ),
            ROUTER_MODES[1]: _aggregate(
                regret=0.0, upper_floor=0.0, lower_floor=0.02
            ),
        },
    )
    assert result == "learned_policy_limited_unavoidable_physical_budget_floor"


def test_diagnosis_selects_projection_work_only_for_avoidable_regret():
    result = diagnose_projection_limit(
        {"exact_legacy_replay": True},
        {
            ROUTER_MODES[0]: _aggregate(
                regret=0.02, upper_floor=0.0, lower_floor=0.01
            ),
            ROUTER_MODES[1]: _aggregate(
                regret=0.0, upper_floor=0.0, lower_floor=0.01
            ),
        },
    )
    assert result == "projection_limited_avoidable_budget_regret"


def test_diagnosis_stops_on_legacy_replay_regression():
    result = diagnose_projection_limit(
        {"exact_legacy_replay": False},
        {
            ROUTER_MODES[0]: _aggregate(
                regret=0.0, upper_floor=0.0, lower_floor=0.0
            ),
            ROUTER_MODES[1]: _aggregate(
                regret=0.0, upper_floor=0.0, lower_floor=0.0
            ),
        },
    )
    assert result == "invalid_due_to_v17_4_replay_regression"


def test_scheduler_binding_is_explicitly_data_local():
    args = SimpleNamespace(
        python_executable="python3",
        source_run_name="source",
        run_name="diagnostic",
        code_revision="a" * 40,
        source_manifest_sha256="b" * 64,
        priority="normal",
    )
    task = build_scheduler_spec(args, "Hopper-v5", 3105897127)
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert task["cpu"] == 1
    assert task["ram_mb"] == 1024
    assert ".server_artifacts" in task["stage_excludes"]
    assert "checkpoint.pt" in task["ckpt_glob"]
