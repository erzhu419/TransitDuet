"""Continue the frozen Stage9 study through scheduler, without queued waiters."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, SCHEDULER, execute_bulk

sys.path.insert(0, str(SCHEDULER.parent))
import scheduler


RUNNER = "scripts/run_stage9_strong_timing_confirmation.py"
REMOTE = Path("/home/zhengliang01/scheduleurm_work/TransitDuet_freqhrl_v3/transit_hrl")


def job_specs(plan, manifest, phase):
    jobs = []
    for i, cell in enumerate(plan["confirmation_reservation"]):
        root = cell["optimizer_seed"]
        job = deepcopy(plan["jobs"]["audit"][0])
        public = str(ROOT / "results" / plan["run_name"] / "confirmation" / str(root) / phase / "public")
        command = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "--manifest", str(manifest),
                   "--phase", phase, "--cell", str(i)]
        job.update(description=f"Frozen Stage9 {phase} {root}",
            signature=f"Freq-HRL/stage9-strong-timing/{plan['run_name']}/confirmation/{phase}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
            result_dir=public, local_result_dir=public, wait_for_files=[str(ROOT / manifest)])
        jobs.append(job)
    return jobs


def save(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2)
        out.write("\n")


def tasks_with(signatures, state=None):
    tasks = (state if state is not None else scheduler.load_state())["tasks"]
    return [t for t in (tasks.values() if isinstance(tasks, dict) else tasks) if t.get("signature") in signatures]


def submit(jobs, label):
    signatures = {j["signature"] for j in jobs}
    if tasks_with(signatures):
        raise RuntimeError("do not resubmit any registered confirmation root")
    execute_bulk(jobs, dry_run=False, intent_label=label)
    with scheduler.state_lock():
        state = scheduler.load_state()
        tasks = tasks_with(signatures, state)
        if len(tasks) != len(jobs):
            raise RuntimeError("scheduler did not register the complete cohort")
        for task in tasks:
            task.update(allow_cpu_training=True, protect_from_auto_eviction=True)
        scheduler.save_state(state)
    ids = [t["id"] for t in tasks]
    print(label, ids, flush=True)
    subprocess.run([sys.executable, str(SCHEDULER), "dispatch",
        *(s for task in ids for s in ("--task-id", task)), "--intent-label", label], check=True)
    return ids, signatures


def remote(command):
    shell = "cd " + shlex.quote(str(REMOTE)) + " && env PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    for i in range(3):
        rc, out, err = scheduler.run_on("node006", shell + shlex.join([DEFAULT_LINUX_PYTHON, *command]), timeout=60, check=False)
        if rc == 0:
            return json.loads(out), len(out.encode())
        if rc != 255 or i == 2:
            raise RuntimeError(err[-2000:])
        time.sleep(20)


def wait_reports(plan, phase, signatures, *, hours):
    paths = [str(Path("results") / plan["run_name"] / "confirmation" / str(c["optimizer_seed"]) / phase / "public/result.json")
             for c in plan["confirmation_reservation"]]
    probe = """import json
from pathlib import Path
paths = %r
rows = [json.loads(Path(p).read_text()) if Path(p).is_file() else None for p in paths]
print(json.dumps({"statuses": [r["status"] if r else None for r in rows],
                  "rows": rows if all(rows) else None}, separators=(",", ":")))
""" % paths
    cutoff, previous, next_queue_check = time.monotonic() + 3600 * hours, None, 0
    while time.monotonic() < cutoff:
        r, pulled = remote(["-c", probe])
        if r["statuses"] != previous:
            print(phase, r["statuses"], flush=True)
            previous = r["statuses"]
        if r["rows"] is not None:
            return r["rows"], pulled
        if time.monotonic() >= next_queue_check:
            next_queue_check = time.monotonic() + 600
            state = tasks_with(signatures)
            if len(state) == len(paths) and all(t.get("status") in {"done", "failed", "cancelled", "canceled"} for t in state):
                raise RuntimeError(f"{phase} terminated without complete worker reports: " +
                    str([(t["id"], t.get("status"), t.get("exit_code")) for t in state]))
        time.sleep(60)
    raise TimeoutError(f"{phase} reports did not close within the monitoring window; no next phase submitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve().relative_to(ROOT)
    plan = json.loads((ROOT / manifest).read_text())
    folder = ROOT / "results" / plan["run_name"]
    development = json.loads((folder / "development_summary.json").read_text())
    if not development["confirmation_builds_allowed"]:
        raise ValueError("development gate did not release new roots")
    precheck, _ = remote(["-c", f"import json; from pathlib import Path; "
        f"from scripts.run_stage9_strong_timing_confirmation import require_development; "
        f"p=json.loads(Path({str(manifest)!r}).read_text()); print(len(require_development(p)))"])
    if precheck != len(plan["development_cells"]):
        raise RuntimeError("remote development gate did not reproduce")

    build_ids, build_signatures = submit(job_specs(plan, manifest, "build"), plan["run_name"] + "_build")
    save(folder / "confirmation_release.json", {"time_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest), "build_task_ids": build_ids,
        "evaluation_rule": "all_eight_admitted_builds_frozen_then_submit_once",
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    builds, pulled = wait_reports(plan, "build", build_signatures, hours=6)
    save(folder / "confirmation_builds.json", {"task_ids": build_ids, "roots": builds, "retrieved_summary_bytes": pulled})
    if not all(r["status"] == "complete" and r["admitted"] and not r["test_accessed"] for r in builds):
        save(folder / "confirmation_decision.json", {"status": "BUILD_GATE_HOLD",
            "evaluation_task_ids": [], "supported": False})
        print("BUILD_GATE_HOLD: no evaluation released", flush=True)
        return

    remote(["-c", f"import json; from pathlib import Path; "
        f"from scripts.run_stage9_strong_timing_confirmation import require_global_freeze; "
        f"p=json.loads(Path({str(manifest)!r}).read_text()); require_global_freeze(p); print('true')"])
    ids, signatures = submit(job_specs(plan, manifest, "evaluate"), plan["run_name"] + "_evaluate")
    save(folder / "confirmation_evaluation_release.json", {"task_ids": ids, "global_freeze_verified": True})
    reports, pulled = wait_reports(plan, "evaluate", signatures, hours=2)
    if not all(r["status"] == "complete" for r in reports):
        save(folder / "confirmation_decision.json", {"status": "EVALUATION_FAILED", "roots": reports,
            "evaluation_task_ids": ids, "retrieved_summary_bytes": pulled, "supported": False})
        return
    result, merged_bytes = remote([RUNNER, "--manifest", str(manifest), "--phase", "readout"])
    save(folder / "confirmation_decision.json", {**result, "status": "COMPLETE",
        "evaluation_task_ids": ids, "retrieved_summary_bytes": pulled + merged_bytes})
    print("confirmation complete", result["pointmaze_strong_timing_supported"], flush=True)


if __name__ == "__main__":
    main()
