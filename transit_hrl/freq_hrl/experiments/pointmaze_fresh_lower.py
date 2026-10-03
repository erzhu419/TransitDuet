"""Rebuild matched lower policies with the existing paired-noise training core."""

import json

from . import pointmaze_upper_noise_replication as previous
from . import pointmaze_fresh_joint as fresh_joint
from scripts import pointmaze_fresh_lower_stage99_spec as spec

learning = previous.learning


def worker_pair(jobs):
    return previous.worker_pair(jobs, protocol=spec)


def qualify(cell, *, preflight):
    previous.qualify(cell, preflight=preflight, protocol=spec)
    if cell["source_initialization"] != spec.source.source_record(cell["root"]):
        raise ValueError("fresh lower source record changed")
    return cell


def run(root, *, preflight, output):
    training = json.loads(spec.donor_result(root, "joint_call").read_text())
    fresh_joint.qualify(training, preflight=False)
    donors = {}

    def initialize(period, models, cost):
        donors[period], metadata = previous.prepare_training(training, root, period, models, cost,
            protocol=spec, checkpoint_protocol=spec)
        return metadata

    return learning.run(root, preflight=preflight, output=output, protocol=spec, qualifier=qualify,
        source_loader=fresh_joint.load_source, initialize_models=initialize,
        evaluation_weights=lambda p, w, c: previous.prepare_evaluation(donors[p], p, w, c, protocol=spec),
        training_pair_worker=worker_pair,
        scenario_pair_check=lambda pairs, roster: previous.check_scenario_pair(pairs, roster, root=root, protocol=spec))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(performance_claim="new_teacher_fixed_upper_conditional_lower_MC_mean_learning_not_frequency_superiority",
        primary_endpoints=list(spec.PRIMARY_ENDPOINTS), conditioning_confirmation="mechanical_only" if preflight else
        ("supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"))
    return result
