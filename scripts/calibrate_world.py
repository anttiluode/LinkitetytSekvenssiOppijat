"""Calibrate the world against ideal observers only (no learner is trained here).

Targets, fixed before any experiment arm was run:
  C1 elder (face only) filtering accuracy          >= 0.93
  C2 ideal self-knower (sensations + intero)       0.75 .. 0.90   (room to learn)
  C3 interoception-only ideal                      0.50 .. 0.72   (weak but real)
  C4 sensations-only ideal                         0.50 .. 0.75
  C5 drive matters for self-prediction: marginal NLL - oracle NLL >= 0.12 nats
  C6 ideal self-knower separates drive 2 from 3 about as well as other pairs
"""
from __future__ import annotations

import json
import sys
import numpy as np

sys.path.insert(0, ".")
from lso.world import World, WorldParams, K  # noqa: E402


def pair_acc(post, d, a, b):
    mask = (d == a) | (d == b)
    pred = np.where(post[..., a] >= post[..., b], a, b)
    return float((pred[mask] == d[mask]).mean())


def calibrate(params: WorldParams, n: int = 4000, seed: int = 12345) -> dict:
    w = World(params)
    ep = w.sample(np.random.default_rng(seed), n)
    d = ep["d"]
    out = {}
    for name, flags in {
        "elder_face": (False, False, True),
        "self_s_intero": (True, True, False),
        "self_intero_only": (False, True, False),
        "self_s_only": (True, False, False),
    }.items():
        post = w.filter(ep, *flags)
        out[name] = float((post.argmax(-1) == d).mean())
        if name == "self_s_intero":
            out["self_pair_23"] = pair_acc(post, d, 2, 3)
            out["self_pair_01"] = pair_acc(post, d, 0, 1)
            out["self_pair_02"] = pair_acc(post, d, 0, 2)
            out["nll_self_ideal"] = w.next_token_nll(ep, post)
    out["nll_oracle"] = w.next_token_nll(ep, None)
    out["nll_marginal"] = w.marginal_nll(ep)
    out["nll_gap"] = out["nll_marginal"] - out["nll_oracle"]
    return out


if __name__ == "__main__":
    res = calibrate(WorldParams())
    print(json.dumps(res, indent=2))
