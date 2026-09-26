import numpy as np
import torch

from lso.world import World, WorldParams, K
from lso.social import inputs, elder_display, SocialInfant


def test_display_hidden_when_elder_does_not_react():
    w = World(WorldParams(alpha=0.0))
    ep = w.sample(np.random.default_rng(0), 4)
    disp = torch.as_tensor(elder_display("FAITHFUL", w, ep))
    react = torch.zeros(4, 64)
    react[:, 10] = 1
    x = inputs(torch.as_tensor(ep["s"][:, :64]), torch.as_tensor(ep["intero"]), disp, react)
    assert torch.all(x[:, 9, -(K + 1):] == 0)
    assert torch.allclose(x[:, 10, -(K + 1):-1], disp[:, 10])


def test_alpha_zero_means_sensations_carry_no_drive():
    w = World(WorldParams(alpha=0.0))
    assert np.allclose(w.trans[0], w.trans[1]) and np.allclose(w.trans[2], w.trans[3])


def test_social_infant_shapes():
    net = SocialInfant(8)
    a, b, h = net(torch.zeros(2, 5, 15))
    assert a.shape == (2, 5, 6) and b.shape == (2, 5, 4) and h.shape == (2, 5, 8)
