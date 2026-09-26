"""Mechanical summary: read every receipt, apply the frozen gates, write results/summary.json.

Nothing here is tuned after seeing results; thresholds are copied from PREDICTIONS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lso.experiment import ARMS, SEEDS  # noqa: E402


def load():
    R = {}
    for s in SEEDS:
        for a in ARMS:
            p = ROOT / "results" / "seeds" / f"seed{s:03d}_{a}.json"
            if not p.exists():
                raise SystemExit(f"missing receipt {p.name}")
            R[(s, a)] = json.loads(p.read_text())
    return R


def diff(R, metric, a, b, curve_step=None):
    vals = []
    for s in SEEDS:
        if curve_step is None:
            va, vb = R[(s, a)]["final"][metric], R[(s, b)]["final"][metric]
        else:
            va, vb = R[(s, a)]["curve"][str(curve_step)], R[(s, b)]["curve"][str(curve_step)]
        vals.append(va - vb)
    v = np.array(vals)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "positive": int((v > 0).sum()),
            "per_seed": {str(s): float(x) for s, x in zip(SEEDS, v)}}


def main():
    R = load()
    arm_means = {}
    for a in ARMS:
        keys = R[(SEEDS[0], a)]["final"].keys()
        arm_means[a] = {k: float(np.mean([R[(s, a)]["final"][k] for s in SEEDS])) for k in keys}
        arm_means[a]["curve"] = {c: float(np.mean([R[(s, a)]["curve"][c] for s in SEEDS]))
                                 for c in R[(SEEDS[0], a)]["curve"]}

    g0 = all(arm_means[a]["SELF"] >= 0.40 for a in ARMS) and arm_means["FAITHFUL"]["SELF_MIRROR_ON"] >= 0.85

    h1a = diff(R, "SELF", "FAITHFUL", "YOKED")
    h1b = diff(R, "SELF", "FAITHFUL", "ALONE")
    h1 = h1a["mean"] >= 0.03 and h1a["positive"] >= 6 and h1b["mean"] >= 0.03 and h1b["positive"] >= 6

    h2a = diff(R, "PAIR23", "FAITHFUL", "BIASED")
    h2b = diff(R, "PAIR23", "ALONE", "BIASED")
    h2c = diff(R, "PAIR01", "FAITHFUL", "BIASED")
    h2 = h2a["mean"] >= 0.05 and h2a["positive"] >= 6 and h2b["mean"] >= 0.03 and abs(h2c["mean"]) < 0.03

    h3d = diff(R, "INTERO_DROP", "FAITHFUL", "ALONE")
    h3 = h3d["mean"] >= 0.02 and h3d["positive"] >= 6

    h4d = diff(R, "SELF", "COLEARN", "YOKED")
    h4 = h4d["mean"] >= 0.02 and h4d["positive"] >= 6

    h5d = diff(R, "SELF", "FAITHFUL", "ALONE", curve_step=500)
    h5 = h5d["mean"] >= 0.03

    summary = {
        "gates": {"G0": g0, "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5},
        "contrasts": {
            "H1_faithful_minus_yoked_SELF": h1a, "H1_faithful_minus_alone_SELF": h1b,
            "H2_faithful_minus_biased_PAIR23": h2a, "H2_alone_minus_biased_PAIR23": h2b,
            "H2_faithful_minus_biased_PAIR01": h2c,
            "H3_faithful_minus_alone_INTERO_DROP": h3d,
            "H4_colearn_minus_yoked_SELF": h4d,
            "H5_faithful_minus_alone_SELF_at500": h5d,
        },
        "arm_means": arm_means,
        "seeds": list(SEEDS),
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary["gates"], indent=2))
    for k, v in summary["contrasts"].items():
        print(f"{k:42s} mean {v['mean']:+.4f} ± {v['se']:.4f}  positive {v['positive']}/8")


if __name__ == "__main__":
    main()
