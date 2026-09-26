"""The world: an infant body with a hidden drive, and what that drive touches.

One hidden variable, the drive d_t in {0,1,2,3} (think of it as a mood), changes
slowly. It shapes three things:

  s_t  the infant's own sensation stream: a token sequence whose transition
       statistics depend on d_t. The infant learns by predicting s_{t+1}.
  i_t  interoception: a weak, noisy direct cue of d_t, visible only to the infant.
  x_t  the face: a clearer cue of d_t, visible only to the other (the elder).

You cannot see your own face; the other cannot feel your insides. That
asymmetry is the whole premise of the social-biofeedback idea being tested.

Everything is generated from a fixed WORLD_SEED so the world is the same for
every infant; experiment seeds only change which episodes are sampled and how
the infant is initialised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

K = 4          # drive states
V = 6          # sensation tokens
WORLD_SEED = 0


@dataclass(frozen=True)
class WorldParams:
    T: int = 64            # steps per episode
    p_stay: float = 0.96   # drive stickiness per step
    alpha: float = 0.70    # how strongly the drive shapes sensation transitions
    conc: float = 0.6      # Dirichlet concentration for transition rows
    a_intero: float = 0.55 # interoception signal amplitude (noise sd = 1)
    a_face: float = 2.0    # face signal amplitude (noise sd = 1)


@dataclass
class World:
    params: WorldParams = field(default_factory=WorldParams)

    def __post_init__(self):
        rng = np.random.default_rng(WORLD_SEED)
        p = self.params
        shared = rng.dirichlet(np.full(V, p.conc), size=V)             # V x V
        spec = rng.dirichlet(np.full(V, p.conc), size=(K, V))           # K x V x V
        self.trans = (1 - p.alpha) * shared[None] + p.alpha * spec      # K x V x V
        self.s0 = np.full(V, 1.0 / V)
        self.A = np.full((K, K), (1 - p.p_stay) / (K - 1))
        np.fill_diagonal(self.A, p.p_stay)
        self.d0 = np.full(K, 1.0 / K)

    # ------------------------------------------------------------------ sampling
    def sample(self, rng: np.random.Generator, n: int) -> dict:
        """Sample n episodes. Returns arrays with a leading batch axis."""
        p = self.params
        T = p.T
        d = np.empty((n, T), dtype=np.int64)
        s = np.empty((n, T + 1), dtype=np.int64)   # s[:, t+1] is the target at step t
        d[:, 0] = rng.integers(0, K, size=n)
        for t in range(1, T):
            stay = rng.random(n) < p.p_stay
            jump = (d[:, t - 1] + rng.integers(1, K, size=n)) % K
            d[:, t] = np.where(stay, d[:, t - 1], jump)
        s[:, 0] = rng.integers(0, V, size=n)
        for t in range(T):
            probs = self.trans[d[:, t], s[:, t]]            # n x V
            u = rng.random((n, 1))
            s[:, t + 1] = (probs.cumsum(axis=1) < u).sum(axis=1).clip(0, V - 1)
        eye = np.eye(K)
        intero = p.a_intero * eye[d] + rng.standard_normal((n, T, K))
        face = p.a_face * eye[d] + rng.standard_normal((n, T, K))
        return {"d": d, "s": s, "intero": intero.astype(np.float32), "face": face.astype(np.float32)}

    # ------------------------------------------------------------- ideal observers
    def _gauss_loglik(self, obs: np.ndarray, amp: float) -> np.ndarray:
        """log p(obs_t | d) up to a constant, for unit-variance Gaussian around amp*onehot(d)."""
        # ||o - a e_k||^2 = ||o||^2 - 2 a o_k + a^2  -> only -2 a o_k varies with k
        return amp * obs  # (n, T, K): log-lik differences across k

    def filter(self, episodes: dict, use_s: bool, use_intero: bool, use_face: bool) -> np.ndarray:
        """Forward (filtering) posterior p(d_t | channels up to t). Returns (n, T, K)."""
        n, T = episodes["d"].shape
        logl = np.zeros((n, T, K))
        if use_s:
            s = episodes["s"]
            for t in range(T):
                # transition s_t -> s_{t+1} is only observed after step t; for filtering at t
                # we can use transitions up to (s_{t-1} -> s_t).
                if t >= 1:
                    logl[:, t] += np.log(self.trans[:, s[:, t - 1], s[:, t]].T + 1e-12)
        if use_intero:
            logl += self._gauss_loglik(episodes["intero"], self.params.a_intero)
        if use_face:
            logl += self._gauss_loglik(episodes["face"], self.params.a_face)
        post = np.empty((n, T, K))
        prior = np.tile(self.d0, (n, 1))
        for t in range(T):
            lp = np.log(prior + 1e-300) + logl[:, t]
            lp -= lp.max(axis=1, keepdims=True)
            q = np.exp(lp)
            q /= q.sum(axis=1, keepdims=True)
            post[:, t] = q
            prior = q @ self.A
        return post

    def next_token_nll(self, episodes: dict, post_d: np.ndarray | None) -> float:
        """Mean NLL of s_{t+1} given s_t and a belief over d_t (None = oracle d)."""
        s = episodes["s"]
        n, T = episodes["d"].shape
        total = 0.0
        for t in range(T):
            if post_d is None:
                probs = self.trans[episodes["d"][:, t], s[:, t]]
            else:
                probs = np.einsum("nk,nkv->nv", post_d[:, t], self.trans[:, s[:, t]].transpose(1, 0, 2))
            total += -np.log(probs[np.arange(n), s[:, t + 1]] + 1e-12).sum()
        return total / (n * T)

    def marginal_nll(self, episodes: dict) -> float:
        """NLL of s_{t+1} with a flat belief over d (knows the world, ignores the drive)."""
        n, T = episodes["d"].shape
        flat = np.full((n, T, K), 1.0 / K)
        return self.next_token_nll(episodes, flat)


def mirror_faithful(world: World, ep: dict) -> np.ndarray:
    """The elder's display: its posterior over the infant's drive from the face, lagged one step."""
    post = world.filter(ep, use_s=False, use_intero=False, use_face=True)
    m = np.empty_like(post)
    m[:, 0] = 1.0 / K
    m[:, 1:] = post[:, :-1]
    return m.astype(np.float32)


def mirror_biased(world: World, ep: dict) -> np.ndarray:
    """An elder who cannot tell drive 2 from drive 3: it shows their average for both."""
    m = mirror_faithful(world, ep).copy()
    avg = 0.5 * (m[..., 2] + m[..., 3])
    m[..., 2] = avg
    m[..., 3] = avg
    return m
