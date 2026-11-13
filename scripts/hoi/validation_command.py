"""Run one validation command under a cumulative wall-time GPU budget.

Counts the entire child lifetime (including imports/CPU work) conservatively.
Never discovers checkpoints, edits configs, or implements a training loop.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("logs/hoi_validation_20261114"))
    p.add_argument("--label", required=True)
    p.add_argument("--limit-seconds", type=float, default=600)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd or Path(a.label).name != a.label or a.limit_seconds <= 0:
        p.error("Need a command, a simple label and a positive time limit")
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / "budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = a.root / "autoresearch.jsonl"
        rows = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        used = sum(r["charged_seconds"] for r in rows)
        remaining = 3600 - used
        # Leave time to deliver SIGINT and then kill only our own child group.
        limit = min(a.limit_seconds, remaining - 10)
        if limit <= 0:
            raise RuntimeError(f"GPU validation budget exhausted: {used:.3f} seconds")
        meta = a.root / (a.label + ".json")
        if meta.exists():
            raise FileExistsError(meta)
        start = time.monotonic()
        record = {
            "label": a.label,
            "command": cmd,
            "started_unix": time.time(),
            "limit_seconds": limit,
            "budget_before_seconds": used,
        }
        meta.write_text(json.dumps(record, indent=2) + "\n")
        peak, samples = 0, []
        peak_rss_kib = 0
        with (a.root / (a.label + ".log")).open("x") as log:
            child = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            timed_out = False
            try:
                while child.poll() is None:
                    if time.monotonic() - start > limit:
                        timed_out = True
                        os.killpg(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    try:
                        status = Path(f"/proc/{child.pid}/status").read_text()
                        rss = next(
                            int(line.split()[1])
                            for line in status.splitlines()
                            if line.startswith("VmHWM:")
                        )
                        peak_rss_kib = max(peak_rss_kib, rss)
                    except (FileNotFoundError, StopIteration):
                        pass
                    probe = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    if probe.returncode == 0:
                        memory, util = map(int, probe.stdout.splitlines()[0].split(","))
                        peak = max(peak, memory)
                        samples.append([round(time.monotonic() - start, 3), memory, util])
                    time.sleep(1)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                record.update(
                    charged_seconds=time.monotonic() - start,
                    returncode=child.returncode,
                    timed_out=timed_out,
                    gpu_peak_total_mib=peak,
                    process_peak_rss_kib=peak_rss_kib,
                    gpu_samples=samples,
                )
                meta.write_text(json.dumps(record, indent=2) + "\n")
                entry = {k: v for k, v in record.items() if k != "gpu_samples"}
                with ledger.open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
        raise SystemExit(child.returncode if not timed_out else 124)


if __name__ == "__main__":
    main()
