#!/usr/bin/env python3
"""Stage and dispatch V500 Path B Gate 1 onto node001-node006 via scheduler."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time


HERE = Path(__file__).resolve().parent
FREQDUET_CORE = HERE.parents[1] / "freqduet"
REMOTE_BASE = Path("/home/zhengliang01/scheduleurm_work/FreqDuet-v500-pathb-headroom")
SOURCE = REMOTE_BASE / "source03"
ROOT = REMOTE_BASE / "run03"
REVIEW_DIR = HERE / "review03"

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler  # noqa: E402


def deployment() -> dict:
    return json.loads((HERE.parent / "v400_independent_confirmation" / "deployment.json").read_text())


def stage_bundle() -> None:
    marker = HERE / "staged_source03.json"
    if marker.exists():
        print("Already staged:", marker.read_text().strip(), flush=True)
        return

    buf = io.BytesIO()
    core_packages = ("env", "frequency", "lower", "upper", "coupling")
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for fn in ("gate1_study.py", "protocol.json", "PROTOCOL.md"):
            p = HERE / fn
            if p.suffix == ".py":
                compile(p.read_bytes(), fn, "exec")
            archive.add(p, arcname=fn)
        for p in sorted((HERE / "corridors").rglob("*")):
            if p.is_file():
                archive.add(p, arcname=str(p.relative_to(HERE)))
        for fn in ("__init__.py", "randomness.py"):
            p = FREQDUET_CORE / fn
            if p.exists():
                archive.add(p, arcname=f"freqduet_src/{fn}")
        for pkg in core_packages:
            pkg_dir = FREQDUET_CORE / pkg
            for p in sorted(pkg_dir.rglob("*.py")):
                if "__pycache__" not in p.parts:
                    archive.add(p, arcname=f"freqduet_src/{p.relative_to(FREQDUET_CORE)}")

    raw_bytes = buf.getvalue()
    import hashlib
    expected_sha = hashlib.sha256(raw_bytes).hexdigest()
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    chunk_size = 75000  # strictly below Linux MAX_ARG_STRLEN (131072)
    chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
    print(f"Built tarball: {len(raw_bytes) / 1024:.1f} KB ({len(chunks)} chunks)", flush=True)

    py = deployment()["python"]
    tmp_b64 = str(REMOTE_BASE / "source03.b64")
    last_err = None
    for node in ("node004", "node003", "node002", "node001"):
        ok_node = True
        for idx, ch in enumerate(chunks):
            mode = "w" if idx == 0 else "a"
            c_write = (
                "from pathlib import Path\n"
                f"p = Path({tmp_b64!r})\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                f"with p.open({mode!r}) as f: f.write({ch!r})\n"
            )
            rc, out, err = scheduler.run_on(node, shlex.join([py, "-c", c_write]), timeout=90, check=False)
            if rc != 0:
                ok_node = False
                last_err = (node, f"chunk_{idx}", rc, out[-300:], err[-600:])
                break
        if not ok_node:
            continue

        c_extract = (
            "import base64, hashlib, io, shutil, tarfile\n"
            "from pathlib import Path\n"
            f"b64_p = Path({tmp_b64!r})\n"
            "raw = base64.b64decode(b64_p.read_text())\n"
            f"assert hashlib.sha256(raw).hexdigest() == {expected_sha!r}, 'sha256 mismatch'\n"
            f"p = Path({str(SOURCE)!r})\n"
            "if p.exists(): shutil.rmtree(p)\n"
            "p.mkdir(parents=True, exist_ok=True)\n"
            "tarfile.open(fileobj=io.BytesIO(raw)).extractall(p)\n"
            "b64_p.unlink(missing_ok=True)\n"
            "print('STAGED_OK', p)\n"
        )
        rc, out, err = scheduler.run_on(node, shlex.join([py, "-c", c_extract]), timeout=120, check=False)
        if rc == 0:
            print(f"Staged via {node}: {out.strip()}", flush=True)
            marker.write_text(json.dumps({"source": str(SOURCE), "root": str(ROOT), "node": node, "sha256": expected_sha}, indent=2) + "\n")
            return
        last_err = (node, "extract", rc, out[-500:], err[-1000:])
    raise RuntimeError(f"Failed to stage onto cluster NFS: {last_err}")


def make_task(label: str, phase_args: list[str], result_subdir: str, waits: tuple[str, ...] = ()) -> dict:
    d = deployment()
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
    }
    cmd = [d["python"], "-B", "-I", "-u", str(SOURCE / "gate1_study.py"), *phase_args, "--root", str(ROOT)]
    spec = {
        "description": f"FreqDuet V500 Gate1 Rev03 {label}",
        "project": "FreqDuet",
        "signature": f"FreqDuet/v500_gate1_rev03/{label}",
        "priority": "high",
        "allowed_nodes": ["node001", "node002", "node003", "node004", "node005", "node006"],
        "cpu": 1,
        "ram_mb": 4096,
        "vram": 0,
        "allow_cpu_training": True,
        "cpu_training_justification": "Frozen CPU transit simulation for V500 Gate 1 headroom audit; zero training.",
        "allow_no_ckpt": True,
        "no_auto_requeue": True,
        "skip_launch_staging": True,
        "cwd": str(SOURCE),
        "env": env,
        "cmd": shlex.join(cmd),
        "result_dir": str(ROOT / result_subdir / "summary"),
        "local_result_dir": str(REVIEW_DIR / label),
    }
    if waits:
        spec["wait_for_files"] = [str(REVIEW_DIR / w) for w in waits]
    return spec


def submit_and_dispatch(batch_name: str, requests: list[dict]) -> list[dict]:
    receipt = HERE / f"submission_rev03_{batch_name}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["tasks"]

    proc = subprocess.run(
        [sys.executable, scheduler.__file__, "submit-jsonl", "--stdin", "--trusted", "--json", "--lock-timeout", "600"],
        input=json.dumps(requests),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
    tasks = [{"id": t["id"], "signature": t["signature"]} for t in json.loads(proc.stdout)["submitted"]]
    receipt.write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")

    # Mandatory manual dispatch per AGENTS.md
    cmd = [sys.executable, scheduler.__file__, "dispatch", "--lock-timeout", "180"]
    for t in tasks:
        cmd += ["--task-id", t["id"]]
    dproc = subprocess.run(cmd, text=True, capture_output=True)
    print(dproc.stdout[-3500:], dproc.stderr[-1000:], flush=True)
    if dproc.returncode != 0:
        raise RuntimeError(f"Manual dispatch failed: {dproc.stderr[-1000:]}")
    return tasks


def sync_completed(tasks: list[dict], poll_s: int = 20) -> None:
    pending = {t["id"] for t in tasks}
    dispatched_waits: set[str] = set()
    prev_n = None
    while pending:
        state = {t["id"]: t for t in scheduler.load_state()["tasks"]}
        ready_to_dispatch = []
        for tid in list(pending):
            t = state[tid]
            st = t["status"]
            if st in ("failed", "cancelled"):
                raise RuntimeError(f"Task {tid} ({t.get('signature')}) ended as {st}: {t.get('last_error')}")
            if st == "done":
                ok, err = scheduler._sync_one_result(t)
                assert ok, (tid, err)
                pending.remove(tid)
            elif st == "queued" and tid not in dispatched_waits:
                wfiles = t.get("wait_for_files") or []
                if wfiles and all(Path(wf).exists() for wf in wfiles):
                    ready_to_dispatch.append(tid)
                    dispatched_waits.add(tid)
        if ready_to_dispatch:
            cmd = [sys.executable, scheduler.__file__, "dispatch", "--lock-timeout", "180"]
            for tid in ready_to_dispatch:
                cmd += ["--task-id", tid]
            subprocess.run(cmd, text=True, capture_output=True)
        done_n = len(tasks) - len(pending)
        if done_n != prev_n:
            print(f"V500 progress: {done_n}/{len(tasks)} tasks synced", flush=True)
            prev_n = done_n
        if pending:
            time.sleep(poll_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "submit_matrix", "wait_all", "run_all"))
    args = parser.parse_args()

    stage_bundle()
    if args.stage in ("preflight", "run_all"):
        tasks = submit_and_dispatch("preflight", [make_task("preflight", ["preflight"], "preflight")])
        sync_completed(tasks, poll_s=10)
        res = json.loads((REVIEW_DIR / "preflight" / "results.json").read_text())
        assert res["passed"] and res["physical_starts"] == 0
        print("PREFLIGHT_PASSED", json.dumps(res), flush=True)
    if args.stage in ("submit_matrix", "run_all"):
        pre = json.loads((REVIEW_DIR / "preflight" / "results.json").read_text())
        assert pre["passed"]
        proto = json.loads((HERE / "protocol.json").read_text())
        reqs = []
        waits = []
        for c in proto["corridors"]:
            for d in proto["days"]:
                label = f"shard_{c}_{d}"
                reqs.append(make_task(label, ["shard", "--corridor", c, "--day", str(d)], f"shards/{c}_{d}"))
                waits.append(f"{label}/results.json")
        reqs.append(make_task("aggregate", ["aggregate"], "aggregate", waits=tuple(waits)))
        tasks = submit_and_dispatch("matrix", reqs)
        print(f"SUBMITTED_AND_DISPATCHED {len(tasks)} tasks: {tasks[0]['id']}..{tasks[-1]['id']}", flush=True)
    if args.stage in ("wait_all", "run_all"):
        tasks = json.loads((HERE / "submission_rev03_matrix.json").read_text())["tasks"]
        sync_completed(tasks, poll_s=20)
        agg = json.loads((REVIEW_DIR / "aggregate" / "results.json").read_text())
        print("V500_GATE1_REV03_COMPLETE", agg["completed"], flush=True)


if __name__ == "__main__":
    main()
