"""Development and testing, exactly as specified in PREDICTIONS.md."""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn.functional as F

from .agents import CoLearnElder, Infant, infant_inputs
from .world import K, World, mirror_biased, mirror_faithful

ARMS = ("ALONE", "YOKED", "FAITHFUL", "BIASED", "COLEARN")
SEEDS = (3, 17, 29, 41, 59, 73, 97, 113)
PILOT_SEED = 1000

CONFIG = {
    "hidden": 32,
    "lr": 3e-3,
    "batch": 64,
    "steps": 2000,
    "clip": 1.0,
    "p_present": 0.5,
    "test_episodes": 1024,
    "probe_t0": 4,
    "curve_steps": [250, 500, 1000, 2000],
    "elder_lr": 3e-3,
}


# ----------------------------------------------------------------------------- mirrors
def make_mirror(arm: str, world: World, ep: dict, elder: CoLearnElder | None = None) -> np.ndarray | torch.Tensor:
    n, T = ep["d"].shape
    if arm == "ALONE":
        return np.zeros((n, T, K), dtype=np.float32)
    if arm == "FAITHFUL":
        return mirror_faithful(world, ep)
    if arm == "BIASED":
        return mirror_biased(world, ep)
    if arm == "YOKED":
        # the faithful elder's display for a different episode: same statistics, not contingent
        return np.roll(mirror_faithful(world, ep), shift=1, axis=0)
    if arm == "COLEARN":
        assert elder is not None
        with torch.no_grad():
            _, h = elder(torch.as_tensor(ep["face"]))
        return elder.show(h)
    raise ValueError(arm)


def _t(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype) if not isinstance(x, torch.Tensor) else x.to(dtype)


# ----------------------------------------------------------------------------- probes
def probe_accuracy(feats: np.ndarray, labels: np.ndarray, n_train: int, classes: list[int], steps: int = 300) -> float:
    """Linear (logistic) probe: fit on the first n_train rows, score on the rest."""
    mask = np.isin(labels, classes)
    idx = np.nonzero(mask)[0]
    tr = idx[idx < n_train]
    te = idx[idx >= n_train]
    if len(tr) < 10 or len(te) < 10:
        return float("nan")
    lab_map = {c: i for i, c in enumerate(classes)}
    y = np.array([lab_map[int(c)] for c in labels[mask]])
    ytr = torch.as_tensor(y[: len(tr)])
    yte = torch.as_tensor(y[len(tr):])
    X = torch.as_tensor(feats, dtype=torch.float32)
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    g = torch.Generator().manual_seed(0)
    W = torch.zeros(X.shape[1], len(classes), requires_grad=True)
    b = torch.zeros(len(classes), requires_grad=True)
    with torch.no_grad():
        W.normal_(0, 0.01, generator=g)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-4 * (W ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((Xte @ W + b).argmax(1) == yte).float().mean())


def _flatten(h: torch.Tensor, d: np.ndarray, t0: int):
    """(B, T, H) hidden states and (B, T) drives -> rows ordered episode-major, steps t0..T-1."""
    B, T, H = h.shape
    feats = h[:, t0:].reshape(B * (T - t0), H).numpy()
    labels = d[:, t0:].reshape(-1)
    return feats, labels, (T - t0)


# ----------------------------------------------------------------------------- evaluation
def evaluate(infant: Infant, world: World, test: dict, arm: str, elder: CoLearnElder | None, full: bool = True) -> dict:
    cfg = CONFIG
    n, T = test["d"].shape
    s = torch.as_tensor(test["s"][:, :T])
    tgt = torch.as_tensor(test["s"][:, 1:T + 1])
    intero = _t(test["intero"])
    zeros = torch.zeros(n, T, K)
    off = torch.zeros(n)
    half_rows = (n // 2) * (T - cfg["probe_t0"])
    out = {}
    with torch.no_grad():
        logits, h = infant(infant_inputs(s, intero, zeros, off))
    feats, labels, _ = _flatten(h, test["d"], cfg["probe_t0"])
    out["SELF"] = probe_accuracy(feats, labels, half_rows, [0, 1, 2, 3])
    if not full:
        return out
    out["NLL"] = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)))
    out["PAIR23"] = probe_accuracy(feats, labels, half_rows, [2, 3])
    out["PAIR01"] = probe_accuracy(feats, labels, half_rows, [0, 1])
    # interoception silenced: same episodes, signal-free noise in the inner channel
    rng = np.random.default_rng(777)
    silent = torch.as_tensor(rng.standard_normal((n, T, K)).astype(np.float32))
    with torch.no_grad():
        _, h_s = infant(infant_inputs(s, silent, zeros, off))
    f_s, l_s, _ = _flatten(h_s, test["d"], cfg["probe_t0"])
    out["SELF_INTERO_SILENT"] = probe_accuracy(f_s, l_s, half_rows, [0, 1, 2, 3])
    out["INTERO_DROP"] = out["SELF"] - out["SELF_INTERO_SILENT"]
    # descriptive: this arm's own mirror switched on at test
    if arm != "ALONE":
        m = _t(make_mirror(arm, world, test, elder))
        with torch.no_grad():
            _, h_m = infant(infant_inputs(s, intero, m, torch.ones(n)))
        f_m, l_m, _ = _flatten(h_m, test["d"], cfg["probe_t0"])
        out["SELF_MIRROR_ON"] = probe_accuracy(f_m, l_m, half_rows, [0, 1, 2, 3])
    # descriptive reference on the same test episodes: the ideal self-knower
    post = world.filter(test, use_s=True, use_intero=True, use_face=False)
    out["IDEAL_SELF"] = float((post[:, cfg["probe_t0"]:].argmax(-1) == test["d"][:, cfg["probe_t0"]:]).mean())
    return out


# ----------------------------------------------------------------------------- development
def develop(seed: int, arm: str, steps: int | None = None, log_every: int = 0) -> dict:
    """Raise one infant in one arm. Returns metrics at the curve checkpoints and the final test."""
    cfg = CONFIG
    steps = steps or cfg["steps"]
    torch.set_num_threads(1)
    world = World()
    data_rng = np.random.default_rng(seed)             # same episode stream in every arm
    presence_rng = np.random.default_rng(seed + 1)     # same on/off pattern in every mirror arm
    test = world.sample(np.random.default_rng(seed + 50_000), cfg["test_episodes"])

    torch.manual_seed(seed)                            # same infant initialisation in every arm
    infant = Infant(cfg["hidden"])
    opt = torch.optim.Adam(infant.parameters(), lr=cfg["lr"])

    elder = None
    elder_opt = None
    if arm == "COLEARN":
        torch.manual_seed(seed + 2)
        elder = CoLearnElder(cfg["hidden"], seed=seed + 3)
        elder_opt = torch.optim.Adam(elder.parameters(), lr=cfg["elder_lr"])

    curve = {}
    t_start = time.time()
    losses = []
    for step in range(1, steps + 1):
        ep = world.sample(data_rng, cfg["batch"])
        T = ep["d"].shape[1]
        present = presence_rng.random(cfg["batch"]) < cfg["p_present"]
        if arm == "ALONE":
            present[:] = False

        if elder is not None:
            # the elder learns from its own error only: predict the next face
            face = torch.as_tensor(ep["face"])
            pred, h_e = elder(face)
            e_loss = F.mse_loss(pred[:, :-1], face[:, 1:])
            elder_opt.zero_grad()
            e_loss.backward()
            elder_opt.step()
            mirror = elder.show(h_e)
        else:
            mirror = _t(make_mirror(arm, world, ep))

        x = infant_inputs(torch.as_tensor(ep["s"][:, :T]), _t(ep["intero"]), mirror.detach(), torch.as_tensor(present))
        logits, _ = infant(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.as_tensor(ep["s"][:, 1:T + 1]).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(infant.parameters(), cfg["clip"])
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and step % log_every == 0:
            print(f"  [{arm} seed {seed}] step {step} loss {np.mean(losses[-log_every:]):.4f}", flush=True)
        if step in cfg["curve_steps"] and step != steps:
            curve[str(step)] = evaluate(infant, world, test, arm, elder, full=False)["SELF"]

    final = evaluate(infant, world, test, arm, elder, full=True)
    curve[str(steps)] = final["SELF"]
    return {
        "seed": seed,
        "arm": arm,
        "steps": steps,
        "final": final,
        "curve": curve,
        "train_loss_first100": float(np.mean(losses[:100])),
        "train_loss_last100": float(np.mean(losses[-100:])),
        "seconds": round(time.time() - t_start, 1),
    }
