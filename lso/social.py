"""Post-hoc v0.1: the social world (see POSTHOC.md).

Differences from v0, and only these:
  * the elder reacts in sparse glances (probability Q_REACT per step), and the infant must
    predict the elder's next reaction from its own state (a second head, soft-target CE);
  * the world's alpha (how much the drive shapes the infant's own sensations) is a parameter.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .experiment import CONFIG, _flatten, probe_accuracy
from .world import K, V, World, WorldParams, mirror_biased, mirror_faithful

ARMS = ("ALONE", "YOKED", "FAITHFUL", "BIASED")
ALPHAS = (0.0, 0.35, 0.7)
Q_REACT = 0.15
SOCIAL_WEIGHT = 1.0
IN_DIM = V + K + K + 1


class SocialInfant(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(IN_DIM, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)          # next own sensation
        self.social = nn.Linear(hidden, K)        # next elder reaction

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h), self.social(h), h


def inputs(s, intero, display, react):
    """react: (B, T) 0/1 per step. The display is shown only on steps where the elder reacts."""
    onehot = F.one_hot(s, V).float()
    r = react.float().unsqueeze(-1)
    return torch.cat([onehot, intero, display * r, r], dim=-1)


def elder_display(arm: str, world: World, ep: dict) -> np.ndarray:
    if arm in ("FAITHFUL", "YOKED"):
        m = mirror_faithful(world, ep)
    elif arm == "BIASED":
        m = mirror_biased(world, ep)
    else:
        n, T = ep["d"].shape
        return np.zeros((n, T, K), dtype=np.float32)
    if arm == "YOKED":
        m = np.roll(m, shift=1, axis=0)
    return m


def evaluate(infant: SocialInfant, world: World, test: dict) -> dict:
    """Elder absent: no reactions at all."""
    n, T = test["d"].shape
    t0 = CONFIG["probe_t0"]
    s = torch.as_tensor(test["s"][:, :T])
    tgt = torch.as_tensor(test["s"][:, 1:T + 1])
    zeros = torch.zeros(n, T, K)
    none = torch.zeros(n, T)
    with torch.no_grad():
        logits, _, h = infant(inputs(s, torch.as_tensor(test["intero"]), zeros, none))
    feats, labels, _ = _flatten(h, test["d"], t0)
    half = (n // 2) * (T - t0)
    post = world.filter(test, use_s=True, use_intero=True, use_face=False)
    return {
        "SELF": probe_accuracy(feats, labels, half, [0, 1, 2, 3]),
        "PAIR23": probe_accuracy(feats, labels, half, [2, 3]),
        "PAIR01": probe_accuracy(feats, labels, half, [0, 1]),
        "NLL": float(F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))),
        "IDEAL_SELF": float((post[:, t0:].argmax(-1) == test["d"][:, t0:]).mean()),
    }


def develop(seed: int, arm: str, alpha: float, steps: int | None = None) -> dict:
    cfg = CONFIG
    steps = steps or cfg["steps"]
    torch.set_num_threads(1)
    world = World(WorldParams(alpha=alpha))
    data_rng = np.random.default_rng(seed)
    react_rng = np.random.default_rng(seed + 1)
    test = world.sample(np.random.default_rng(seed + 50_000), cfg["test_episodes"])
    torch.manual_seed(seed)
    infant = SocialInfant(cfg["hidden"])
    opt = torch.optim.Adam(infant.parameters(), lr=cfg["lr"])
    t_start = time.time()
    soc_losses, own_losses = [], []
    for _ in range(steps):
        ep = world.sample(data_rng, cfg["batch"])
        B, T = ep["d"].shape
        react = react_rng.random((B, T)) < Q_REACT
        if arm == "ALONE":
            react[:] = False
        elif arm == "YOKED":
            react = np.roll(react, shift=1, axis=0)    # timing from the same other episode as the display
        disp = torch.as_tensor(elder_display(arm, world, ep))
        r = torch.as_tensor(react)
        x = inputs(torch.as_tensor(ep["s"][:, :T]), torch.as_tensor(ep["intero"]), disp, r)
        logits, soc, _ = infant(x)
        own = F.cross_entropy(logits.reshape(-1, V), torch.as_tensor(ep["s"][:, 1:T + 1]).reshape(-1))
        loss = own
        if arm != "ALONE":
            nxt = r[:, 1:]                                   # a reaction at t+1 ...
            if nxt.any():
                logp = F.log_softmax(soc[:, :-1], dim=-1)    # ... predicted from the state at t
                ce = -(disp[:, 1:] * logp).sum(-1)
                soc_loss = ce[nxt].mean()
                loss = loss + SOCIAL_WEIGHT * soc_loss
                soc_losses.append(float(soc_loss.detach()))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(infant.parameters(), cfg["clip"])
        opt.step()
        own_losses.append(float(own.detach()))
    final = evaluate(infant, world, test)
    return {
        "seed": seed, "arm": arm, "alpha": alpha, "steps": steps, "final": final,
        "own_loss_last100": float(np.mean(own_losses[-100:])),
        "social_loss_last100": float(np.mean(soc_losses[-100:])) if soc_losses else None,
        "seconds": round(time.time() - t_start, 1),
        "q_react": Q_REACT, "social_weight": SOCIAL_WEIGHT,
    }
