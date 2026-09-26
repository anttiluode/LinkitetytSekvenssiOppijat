"""Mechanical summary of post-hoc v0.1 against the predictions in POSTHOC.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lso.experiment import SEEDS  # noqa: E402
from lso.social import ALPHAS, ARMS  # noqa: E402


def load():
    R = {}
    for a in ALPHAS:
        for s in SEEDS:
            for arm in ARMS:
                p = ROOT / "results" / "posthoc" / f"a{a:.2f}_seed{s:03d}_{arm}.json"
                if not p.exists():
                    raise SystemExit(f"missing {p.name}")
                R[(a, s, arm)] = json.loads(p.read_text())["final"]
    return R


def diff(R, alpha, metric, x, y):
    v = np.array([R[(alpha, s, x)][metric] - R[(alpha, s, y)][metric] for s in SEEDS])
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "positive": int((v > 0).sum())}


def main():
    R = load()
    means = {f"{a:.2f}": {arm: {k: float(np.mean([R[(a, s, arm)][k] for s in SEEDS]))
                                  for k in R[(a, SEEDS[0], arm)]} for arm in ARMS} for a in ALPHAS}
    c = {}
    for a in ALPHAS:
        c[f"{a:.2f}"] = {
            "faithful_minus_alone_SELF": diff(R, a, "SELF", "FAITHFUL", "ALONE"),
            "faithful_minus_yoked_SELF": diff(R, a, "SELF", "FAITHFUL", "YOKED"),
            "yoked_minus_alone_SELF": diff(R, a, "SELF", "YOKED", "ALONE"),
            "faithful_minus_biased_PAIR23": diff(R, a, "PAIR23", "FAITHFUL", "BIASED"),
            "faithful_minus_biased_PAIR01": diff(R, a, "PAIR01", "FAITHFUL", "BIASED"),
        }
    z, m, h = c["0.00"], c["0.35"], c["0.70"]
    fa = [x["faithful_minus_alone_SELF"]["mean"] for x in (z, m, h)]
    gates = {
        "PH1": z["faithful_minus_alone_SELF"]["mean"] >= 0.05 and z["faithful_minus_alone_SELF"]["positive"] >= 6
               and abs(h["faithful_minus_alone_SELF"]["mean"]) < 0.02,
        "PH2": z["faithful_minus_yoked_SELF"]["mean"] >= 0.05 and z["faithful_minus_yoked_SELF"]["positive"] >= 6
               and abs(z["yoked_minus_alone_SELF"]["mean"]) < 0.02,
        "PH3": z["faithful_minus_biased_PAIR23"]["mean"] >= 0.05 and z["faithful_minus_biased_PAIR23"]["positive"] >= 6
               and abs(z["faithful_minus_biased_PAIR01"]["mean"]) < 0.03,
        "PH4": min(fa[0], fa[2]) <= fa[1] <= max(fa[0], fa[2]),
    }
    out = {"gates": gates, "contrasts": c, "means": means, "seeds": list(SEEDS)}
    (ROOT / "results" / "posthoc_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(gates, indent=2))
    for a in ALPHAS:
        k = f"{a:.2f}"
        print(f"alpha {k}: " + "  ".join(f"{arm} SELF {means[k][arm]['SELF']:.3f} P23 {means[k][arm]['PAIR23']:.3f} P01 {means[k][arm]['PAIR01']:.3f}" for arm in ARMS)
              + f"  ideal {means[k]['ALONE']['IDEAL_SELF']:.3f}")
        for name, v in c[k].items():
            print(f"   {name:32s} {v['mean']:+.4f} ± {v['se']:.4f}  positive {v['positive']}/8")


if __name__ == "__main__":
    main()
