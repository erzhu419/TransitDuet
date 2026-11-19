#!/usr/bin/env python3
"""Stage and dispatch V502 Gate 3 (Full TR-C Baselines, 4-Way Frequency Ablation & OOD Suite) onto node001-node006."""

from __future__ import annotations

import argparse
import base64
import hashlib
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
V500_CORRIDORS = HERE.parent / "v500_pathb_headroom_gate" / "corridors"
V501_DIR = HERE.parent / "v501_pathb_b1_stability"
REMOTE_BASE = Path("/home/zhengliang01/scheduleurm_work/FreqDuet-v502-pathb-trc")
SOURCE = REMOTE_BASE / "source02"
ROOT = REMOTE_BASE / "run01"
V501_REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/FreqDuet-v501-pathb-b1/run03")
REVIEW_DIR = HERE / "review"

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler  # noqa: E402


CLASSICAL_ARMS = (
    "B0_Fixed_NoHold",
    "B1_Daganzo2009",
    "B2_Berrebi2018",
    "B3_EconMPC",
)

LEARNED_ARMS = (
    "FreqDuet_Aligned",
    "B4_FlatSAC",
    "B5_TransitDuet_NoFreq",
    "Abl_RawHistory",
    "Abl_AllBand",
    "Abl_Swapped",
    "Abl_NoBudget",
    "Abl_NoPromotion",
)


def deployment() -> dict:
    return json.loads((HERE.parent / "v400_independent_confirmation" / "deployment.json").read_text())


def stage_bundle() -> None:
    marker = HERE / "staged_source02.json"
    if marker.exists():
        print("Already staged:", marker.read_text().strip(), flush=True)
        return

    buf = io.BytesIO()
    core_packages = ("env", "frequency", "lower", "upper", "coupling")
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for fn in ("trc_matrix_runner.py", "PROTOCOL.md"):
            p = HERE / fn
            if p.suffix == ".py":
                compile(p.read_bytes(), fn, "exec")
            archive.add(p, arcname=fn)
        v7_py = V501_DIR / "freqduet_v7_b1.py"
        compile(v7_py.read_bytes(), "freqduet_v7_b1.py", "exec")
        archive.add(v7_py, arcname="freqduet_v7_b1.py")

        for p in sorted(V500_CORRIDORS.rglob("*")):
            if p.is_file():
                archive.add(p, arcname=f"corridors/{p.relative_to(V500_CORRIDORS)}")
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
    expected_sha = hashlib.sha256(raw_bytes).hexdigest()
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    chunk_size = 75000
    chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
    print(f"Built V502 tarball: {len(raw_bytes) / 1024:.1f} KB ({len(chunks)} chunks)", flush=True)

    py = deployment()["python"]
    tmp_b64 = str(REMOTE_BASE / "source01.b64")
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
            f"r_root = Path({str(ROOT)!r})\n"
            "if r_root.exists():\n"
            "    for tp in r_root.rglob('*.tmp'): tp.unlink(missing_ok=True)\n"
            "b64_p.unlink(missing_ok=True)\n"
            "print('STAGED_OK', p)\n"
        )
        rc, out, err = scheduler.run_on(node, shlex.join([py, "-c", c_extract]), timeout=120, check=False)
        if rc == 0:
            print(f"Staged V502 via {node}: {out.strip()}", flush=True)
            marker.write_text(json.dumps({"source": str(SOURCE), "root": str(ROOT), "node": node, "sha256": expected_sha}, indent=2) + "\n")
            return
        last_err = (node, "extract", rc, out[-500:], err[-1000:])
    raise RuntimeError(f"Failed to stage V502 onto cluster NFS: {last_err}")


def make_task(
    label: str,
    phase_args: list[str],
    result_subdir: str,
    *,
    is_training: bool = False,
    waits_abs: tuple[str, ...] = (),
) -> dict:
    d = deployment()
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
        "FREQDUET_TORCH_THREADS": "1",
        "SDL_VIDEODRIVER": "dummy",
        "SDL_AUDIODRIVER": "dummy",
    }
    res_dir = ROOT / result_subdir / "summary"
    ckpt_dir = ROOT / result_subdir / "ckpt"
    cmd = [
        d["python"], "-B", "-I", "-u",
        str(SOURCE / "trc_matrix_runner.py"),
        *phase_args,
        "--corridors-root", str(SOURCE / "corridors"),
        "--root", str(ROOT),
        "--v501-root", str(V501_REMOTE_ROOT),
        "--result-dir", str(res_dir),
    ]
    if is_training:
        cmd += ["--ckpt-dir", str(ckpt_dir), "--resume"]

    spec = {
        "description": f"FreqDuet V502 R2 TRC {label}",
        "project": "FreqDuet",
        "signature": f"FreqDuet/v502_r2_trc/{label}",
        "priority": "high",
        "allowed_nodes": ["node001", "node002", "node003", "node004", "node005", "node006"],
        "cpu": 1,
        "ram_mb": 3072 if is_training else 2048,
        "vram": 0,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Discrete-event Python transit simulation accounts for >98% of step wall time; "
            "compact 96-unit categorical SAC updates run faster on CPU without GPU transfer latency."
        ),
        "skip_launch_staging": True,
        "cwd": str(SOURCE),
        "env": env,
        "cmd": shlex.join(cmd),
        "result_dir": str(res_dir),
        "local_result_dir": str(REVIEW_DIR / label),
    }
    if is_training:
        spec["ckpt_dir"] = str(ckpt_dir)
        spec["resume_flag"] = "--resume_from"
    else:
        spec["allow_no_ckpt"] = True
        spec["no_auto_requeue"] = True
    if waits_abs:
        spec["wait_for_files"] = list(waits_abs)
    return spec


def submit_and_dispatch(batch_name: str, requests: list[dict]) -> list[dict]:
    receipt = HERE / f"submission_r2_{batch_name}.json"
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

    cmd = [sys.executable, scheduler.__file__, "dispatch", "--lock-timeout", "600"]
    for t in tasks:
        cmd += ["--task-id", t["id"]]
    dproc = subprocess.run(cmd, text=True, capture_output=True)
    print(dproc.stdout[-3500:], dproc.stderr[-1000:], flush=True)
    if dproc.returncode != 0 and "SchedulerLockTimeout" not in dproc.stderr:
        raise RuntimeError(f"Manual dispatch failed: {dproc.stderr[-1000:]}")
    return tasks


def sync_completed(tasks: list[dict], poll_s: int = 30) -> None:
    pending = {t["id"] for t in tasks}
    prev_n = None
    tick = 0
    while pending:
        tick += 1
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
            elif st == "queued":
                wfiles = t.get("wait_for_files") or []
                if (not wfiles and tick % 3 == 1) or (wfiles and all(Path(wf).exists() for wf in wfiles)):
                    ready_to_dispatch.append(tid)
        if ready_to_dispatch:
            cmd = [sys.executable, scheduler.__file__, "dispatch", "--lock-timeout", "300"]
            for tid in ready_to_dispatch:
                cmd += ["--task-id", tid]
            subprocess.run(cmd, text=True, capture_output=True)
        done_n = len(tasks) - len(pending)
        if done_n != prev_n:
            print(f"V502 progress: {done_n}/{len(tasks)} tasks synced", flush=True)
            prev_n = done_n
        if pending:
            time.sleep(poll_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "submit_matrix", "wait_all", "run_all"))
    args = parser.parse_args()

    stage_bundle()
    corridors = ("tidal_bimodal", "mbta_route111_wkdy")
    seeds = tuple(range(101, 105))

    if args.stage in ("preflight", "run_all"):
        tasks = submit_and_dispatch("preflight", [make_task("preflight", ["preflight"], "preflight")])
        sync_completed(tasks, poll_s=10)
        res = json.loads((REVIEW_DIR / "preflight" / "results.json").read_text())
        assert res["passed"] and res["physical_starts"] == 0
        print("V502_PREFLIGHT_PASSED", json.dumps(res), flush=True)

    if args.stage in ("submit_matrix", "run_all"):
        pre = json.loads((REVIEW_DIR / "preflight" / "results.json").read_text())
        assert pre["passed"]
        reqs = []
        agg_waits = []

        for c in corridors:
            for m in CLASSICAL_ARMS:
                lbl = f"classical_{c}_{m}"
                reqs.append(make_task(lbl, ["classical", "--corridor", c, "--method", m], lbl))
                agg_waits.append(str(REVIEW_DIR / lbl / "results.json"))

            for m in LEARNED_ARMS:
                is_reused = m == "FreqDuet_Aligned"
                for s in seeds:
                    lbl = f"learned_{c}_{m}_s{s}"
                    w_abs = (str(V501_DIR / "review03" / f"train_{c}_s{s}" / "results.json"),) if is_reused else ()
                    reqs.append(make_task(
                        lbl,
                        ["learned", "--corridor", c, "--method", m, "--seed", str(s), "--episodes", "200"],
                        lbl,
                        is_training=(not is_reused),
                        waits_abs=w_abs,
                    ))
                    agg_waits.append(str(REVIEW_DIR / lbl / "results.json"))

        reqs.append(make_task("aggregate", ["aggregate"], "aggregate", waits_abs=tuple(agg_waits)))
        tasks = submit_and_dispatch("matrix", reqs)
        print(f"V502_SUBMITTED_AND_DISPATCHED {len(tasks)} tasks: {tasks[0]['id']}..{tasks[-1]['id']}", flush=True)

    if args.stage in ("wait_all", "run_all"):
        tasks = json.loads((HERE / "submission_r2_matrix.json").read_text())["tasks"]
        sync_completed(tasks, poll_s=30)
        agg = json.loads((REVIEW_DIR / "aggregate" / "results.json").read_text())
        print("V502_GATE3_COMPLETE", agg["completed"], flush=True)


if __name__ == "__main__":
    main()
