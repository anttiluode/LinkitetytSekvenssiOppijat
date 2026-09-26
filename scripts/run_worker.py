"""Raise one infant (one seed, one arm) and write its receipt. One process per run."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lso.experiment import CONFIG, develop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(seed: int, arm: str) -> None:
    out = ROOT / "results" / "seeds" / f"seed{seed:03d}_{arm}.json"
    if out.exists():
        print(f"skip {out.name}")
        return
    res = develop(seed, arm)
    res["config"] = CONFIG
    res["config_digest"] = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    res["git_sha"] = git_sha()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2))
    tmp.rename(out)
    print(f"done {out.name} SELF={res['final']['SELF']:.3f} in {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
