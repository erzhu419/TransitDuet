import copy

import pytest

from freq_hrl.experiments import pointmaze_native_selection_replication as experiment


def synthetic_cells(*, no_update=False, calibration_radial=False):
    spec, cells = experiment.spec, []
    for root in spec.ROOTS:
        groups = {}
        for p in spec.PERIODS:
            scores = {"two_step": 0., "refresh": -1. if no_update else .3, "stale": -.1 if no_update else .1}
            chosen = max(scores, key=scores.get)
            returns = {"source_flat": -.1, "source_forecast": 0., "two_step": 1., "refresh": 1.4,
                "stale": 1.2, "refresh_descent": .6, "refresh_blinded": 0.}
            returns["selected"] = returns[chosen]
            effects = {f"{a}_minus_{b}": {"mean": returns[a]-returns[b],
                "paired_differences": [returns[a]-returns[b]]*spec.EVALUATION_EPISODES} for a,b in spec.CONTRASTS}
            groups[str(p)] = {"training_roles": spec.training_roles(root), "validation_roles": spec.validation_roles(root),
                "evaluation_seeds": spec.evaluation_seeds(root), "lower_and_critics_frozen": "passed",
                "selection": {"method": chosen, "mean_incremental_returns": scores,
                    "paired_differences": {m:[scores[m]]*(2*spec.VALIDATION_SCENARIOS) for m in spec.METHODS}},
                "training_native_return_fits": {"refresh": {"pooled": {"predicted_gain": .3}},
                    "stale": {"pooled": {"predicted_gain": .5 if calibration_radial else .1}}},
                "mean_metrics": {m:{"episode_return": r} for m,r in returns.items()}, "effects": effects}
        cells.append({"status": "complete", "protocol": spec.PROTOCOL, "root": root, "cost": spec.budget(),
            "source_protocol": spec.source.PROTOCOL, "inherited_source_cost": spec.source.budget(),
            "kind": spec.EVIDENCE_ROLE, "groups": groups})
    return cells


def test_material_selection_and_validation_claims_are_separate():
    summary = experiment.aggregate(synthetic_cells())
    assert summary["material_continuation_gate"] == "supported_both_periods"
    assert summary["fixed_method_selection_gate"] == "not_closed"
    assert summary["validation_added_value_gate"] == "not_closed"
    assert summary["endpoints"]["50/selected_minus_calibration_choice"]["ci"] == [0., 0.]
    assert summary["statistics"]["n_independent"] == 6 and summary["statistics"]["bonferroni_family_size"] == 10
    assert experiment.aggregate(synthetic_cells(no_update=True))["material_continuation_gate"] == "not_closed"


def test_cheaper_comparator_follows_training_fit_not_evaluation_winner():
    summary = experiment.aggregate(synthetic_cells(calibration_radial=True))
    for root in summary["root_rows"]:
        assert all(r["selected_method"] == "refresh" and r["calibration_choice"] == "stale" for r in root["periods"].values())
    assert summary["endpoints"]["100/selected_minus_calibration_choice"]["mean"] == pytest.approx(.2)
    assert summary["validation_added_value_gate"] == "supported_both_periods"


def test_inference_rejects_wrong_root_source_validation_or_alias():
    cells = synthetic_cells()
    with pytest.raises(ValueError, match="exactly six"):
        experiment.aggregate(cells[:-1])
    wrong = copy.deepcopy(cells)
    wrong[0]["inherited_source_cost"]["native_steps"] += 1
    with pytest.raises(ValueError, match="source or budget"):
        experiment.aggregate(wrong)
    wrong = copy.deepcopy(cells)
    wrong[0]["groups"]["50"]["validation_roles"][0]["scenario_seed"] += 1
    with pytest.raises(ValueError, match="roster or freeze"):
        experiment.aggregate(wrong)
    wrong = copy.deepcopy(cells)
    wrong[0]["groups"]["50"]["selection"]["method"] = "two_step"
    with pytest.raises(ValueError, match="follows validation"):
        experiment.aggregate(wrong)
    wrong = copy.deepcopy(cells)
    wrong[0]["groups"]["50"]["mean_metrics"]["selected"]["episode_return"] += 1
    with pytest.raises(ValueError, match="alias changed"):
        experiment.aggregate(wrong)
