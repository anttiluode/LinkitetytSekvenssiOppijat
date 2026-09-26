"""Figure: self-knowledge with the elder gone, by arm, as the drive matters less for the infant's own senses."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "posthoc_summary.json").read_text())
V0 = json.loads((ROOT / "results" / "summary.json").read_text())

alphas = ["0.00", "0.35", "0.70"]
x = [0.0, 0.35, 0.7]
arms = [("FAITHFUL", "faithful elder", "#2a6fdb"), ("BIASED", "elder blind to 2 vs 3", "#8e5bd8"),
        ("YOKED", "replayed (yoked) elder", "#e8702a"), ("ALONE", "raised alone", "#1d2f5e")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180, gridspec_kw={"width_ratios": [1.35, 1]})

ax = axes[0]
ideal = [S["means"][a]["ALONE"]["IDEAL_SELF"] for a in alphas]
ax.plot(x, ideal, color="#999", ls="--", lw=1.5)
ax.text(0.71, ideal[-1] + 0.012, "ideal self-knower", color="#666", fontsize=9, ha="right")
for key, label, c in arms:
    ys = [S["means"][a][key]["SELF"] for a in alphas]
    ax.plot(x, ys, color=c, lw=2.4, marker="o", ms=6, label=label)
ax.axhline(0.25, color="#ccc", lw=1)
ax.text(0.0, 0.26, "chance", color="#999", fontsize=9)
ax.set_xticks(x, ["0\n(drive only inside)", "0.35", "0.7\n(v0 world)"])
ax.set_xlabel("How strongly the drive shapes the infant's own sensations (alpha)")
ax.set_ylabel("Self-knowledge with the elder gone\n(linear probe accuracy for own drive)")
ax.set_ylim(0.2, 0.85)
ax.set_title("Post-hoc v0.1: an elder matters only where your own senses don't", loc="left", fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(axis="y", color="#eee")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
labels = ["ALONE", "YOKED", "FAITHFUL", "BIASED", "COLEARN"]
vals = [V0["arm_means"][a]["SELF"] for a in labels]
cols = ["#1d2f5e", "#e8702a", "#2a6fdb", "#8e5bd8", "#1fa67a"]
ax.bar(range(5), vals, color=cols)
ax.axhline(V0["arm_means"]["ALONE"]["IDEAL_SELF"], color="#999", ls="--", lw=1.5)
for i, v in enumerate(vals):
    ax.text(i, v - 0.06, f"{v:.3f}", ha="center", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(range(5), ["alone", "yoked", "faithful", "biased", "co-learning"], fontsize=9)
ax.set_ylim(0.0, 0.9)
ax.set_title("Frozen v0: no arm differs", loc="left", fontsize=12, fontweight="bold")
ax.set_ylabel("Self-knowledge, elder gone")
ax.text(4.4, V0["arm_means"]["ALONE"]["IDEAL_SELF"] + 0.02, "ideal", color="#666", fontsize=9, ha="right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.text(0.01, 0.01, "8 seeds per point. v0: mirror as input only. v0.1: infant also predicts the elder's reactions. "
         "Dashed: Bayes-optimal self-knower from the infant's own channels.", fontsize=8, color="#777")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(ROOT / "results" / "self_knowledge.png", facecolor="white")
print("wrote results/self_knowledge.png")
