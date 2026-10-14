"""Physical-mean checkpoint selection on existing joint-critic paths."""

import copy
import numpy as np

from freq_hrl.domains.transit import native_joint_advantage as learning


def repair_selection(cell, kind):
    selection = copy.deepcopy(cell["selection"][kind])
    roots = sorted({str(f["held_group"][0]) for f in cell["inner_fits"]})
    scales = {}
    for root in roots:
        folds = [f for f in cell["inner_fits"] if str(f["held_group"][0]) == root]
        scales[root] = np.average([f["fits"][kind]["scales"] for f in folds],
            axis=0, weights=[f["held_states"] for f in folds]).tolist()
    # A single positive scale per root preserves the pooled physical mean sign.
    for curve in selection["curves"].values():
        for root, stats in curve["by_root"].items():
            stats["mean_scaled_cost"] = stats["mean_delta_cost"]/scales[root][0]
            stats["mean_scaled_wait"] = stats["mean_delta_wait_min"]/scales[root][1]
        curve["joint_score"] = max(x[k] for x in curve["by_root"].values() for k in ("mean_scaled_cost", "mean_scaled_wait"))
    chosen = min(learning.CHECKPOINTS, key=lambda s: (selection["curves"][str(s)]["joint_score"], s))
    selection["joint_step"] = chosen if selection["curves"][str(chosen)]["joint_score"] < 0 else None
    selection["source_scales"] = scales
    return selection
