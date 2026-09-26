"""Run every frozen (seed, arm) pair, two worker processes at a time. Resumable."""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lso.experiment import ARMS, SEEDS  # noqa: E402


def run(job):
    seed, arm = job
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_worker.py"), str(seed), arm],
                       cwd=ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr[-400:]).strip()


if __name__ == "__main__":
    jobs = [(s, a) for s in SEEDS for a in ARMS]
    with ThreadPoolExecutor(max_workers=2) as ex:
        for line in ex.map(run, jobs):
            print(line, flush=True)
