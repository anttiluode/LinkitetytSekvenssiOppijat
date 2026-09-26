"""Post-hoc v0.1 runner: one process per (alpha, seed, arm), two at a time. Resumable."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(alpha: float, seed: int, arm: str) -> None:
    from lso.social import develop
    out = ROOT / "results" / "posthoc" / f"a{alpha:.2f}_seed{seed:03d}_{arm}.json"
    if out.exists():
        print(f"skip {out.name}")
        return
    res = develop(seed, arm, alpha)
    res["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2))
    tmp.rename(out)
    print(f"done {out.name} SELF={res['final']['SELF']:.3f} PAIR23={res['final']['PAIR23']:.3f} in {res['seconds']}s", flush=True)


def run(job):
    a, s, arm = job
    r = subprocess.run([sys.executable, __file__, "worker", str(a), str(s), arm], cwd=ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr[-400:]).strip()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        from lso.experiment import SEEDS
        from lso.social import ALPHAS, ARMS
        jobs = [(a, s, arm) for a in ALPHAS for s in SEEDS for arm in ARMS]
        with ThreadPoolExecutor(max_workers=2) as ex:
            for line in ex.map(run, jobs):
                print(line, flush=True)
