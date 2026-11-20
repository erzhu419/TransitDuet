"""Check Route A/B task registration without touching the live scheduler."""

from contextlib import nullcontext
import sys
import unittest
from unittest.mock import Mock, patch

with patch.dict(sys.modules, {"scheduler": Mock()}):
    from scripts import run_route_a_scheduleurm as route


class RouteSchedulerResumeTests(unittest.TestCase):
    def test_partial_cohort_submits_only_missing_and_dispatches_only_queued(self):
        jobs = [{"signature": signature} for signature in ("done", "queued", "running", "missing")]
        state = {"tasks": [
            {"signature": "done", "id": "t1", "status": "done"},
            {"signature": "queued", "id": "t2", "status": "queued"},
            {"signature": "running", "id": "t3", "status": "running"},
        ]}

        def register(specs, **kwargs):
            for spec in specs:
                state["tasks"].append({**spec, "id": "t4", "status": "queued"})

        with patch.object(route, "scheduler") as scheduler, \
                patch.object(route, "execute_bulk", side_effect=register) as submit, \
                patch.object(route.subprocess, "run") as dispatch:
            scheduler.load_state.return_value = state
            scheduler.state_lock.side_effect = nullcontext
            ids, signatures = route.submit_jobs(jobs, "test")
            submit.assert_called_once_with(jobs[-1:], dry_run=False, intent_label="test")
            scheduler.save_state.assert_called_once_with(state)
            self.assertEqual(dispatch.call_args.args[0][3:7], ["--task-id", "t2", "--task-id", "t4"])
        self.assertEqual(ids, ["t1", "t2", "t3", "t4"])
        self.assertEqual(signatures, {"done", "queued", "running", "missing"})
        self.assertTrue(state["tasks"][1]["allow_cpu_training"])
        self.assertTrue(state["tasks"][3]["protect_from_auto_eviction"])
        self.assertNotIn("allow_cpu_training", state["tasks"][0])
        self.assertNotIn("allow_cpu_training", state["tasks"][2])

    def test_completed_retry_does_not_submit_or_dispatch(self):
        state = {"tasks": [
            {"signature": "cell", "id": "t1", "status": "failed"},
            {"signature": "cell", "id": "t2", "status": "done"},
        ]}
        with patch.object(route, "scheduler") as scheduler, \
                patch.object(route, "execute_bulk") as submit, \
                patch.object(route.subprocess, "run") as dispatch:
            scheduler.load_state.return_value = state
            scheduler.state_lock.side_effect = nullcontext
            self.assertEqual(route.submit_jobs([{"signature": "cell"}], "test"), (["t2"], {"cell"}))
            submit.assert_not_called()
            dispatch.assert_not_called()
            scheduler.save_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
