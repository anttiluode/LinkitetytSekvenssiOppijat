"""The two kinds of learner: the infant, and an elder that learns alongside it.

Both learn only from their own prediction errors. Nothing here lets a gradient
cross from one learner to the other: everything the infant receives from an
elder is detached first.
"""
from __future__ import annotations

import torch
from torch import nn

from .world import K, V

MIRROR_DIM = K
INFANT_IN = V + K + MIRROR_DIM + 1   # sensation one-hot, interoception, mirror, mirror-present flag


class Infant(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(INFANT_IN, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)

    def forward(self, x: torch.Tensor):
        h, _ = self.gru(x)
        return self.head(h), h


class CoLearnElder(nn.Module):
    """Watches the infant's face and learns to predict the next face. Shows a fixed
    random projection of its own hidden state: it does not try to teach, it just shows."""

    def __init__(self, hidden: int = 32, seed: int = 0):
        super().__init__()
        self.gru = nn.GRU(K, hidden, batch_first=True)
        self.head = nn.Linear(hidden, K)
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(hidden, hidden, generator=g))
        self.register_buffer("display", q[:, :MIRROR_DIM] * 2.0)

    def forward(self, face: torch.Tensor):
        h, _ = self.gru(face)
        return self.head(h), h

    def show(self, h: torch.Tensor) -> torch.Tensor:
        """Display lagged one step, detached so nothing flows back into the elder."""
        m = torch.tanh(h.detach() @ self.display)
        out = torch.zeros_like(m)
        out[:, 1:] = m[:, :-1]
        return out


def infant_inputs(s: torch.Tensor, intero: torch.Tensor, mirror: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Build the infant's input. s: (B, T) tokens s_0..s_{T-1}; intero, mirror: (B, T, K);
    present: (B,) 0/1. Where present is 0, the mirror is zeroed and the flag is 0."""
    B, T = s.shape
    onehot = torch.nn.functional.one_hot(s, V).float()
    p = present.float().view(B, 1, 1)
    m = mirror * p
    flag = p.expand(B, T, 1)
    return torch.cat([onehot, intero, m, flag], dim=-1)
