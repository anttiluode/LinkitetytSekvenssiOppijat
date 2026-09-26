import numpy as np
import torch
import pytest

from lso.world import World, K, mirror_faithful, mirror_biased
from lso.agents import Infant, CoLearnElder, infant_inputs
from lso.experiment import make_mirror, probe_accuracy


@pytest.fixture(scope="module")
def world():
    return World()


@pytest.fixture(scope="module")
def ep(world):
    return world.sample(np.random.default_rng(1), 256)


def test_world_shapes_and_stickiness(world, ep):
    n, T = ep["d"].shape
    assert ep["s"].shape == (n, T + 1)
    assert ep["intero"].shape == ep["face"].shape == (n, T, K)
    stay = (ep["d"][:, 1:] == ep["d"][:, :-1]).mean()
    assert 0.93 < stay < 0.99


def test_faithful_mirror_reads_the_face(world, ep):
    m = mirror_faithful(world, ep)
    acc = (m[:, 1:].argmax(-1) == ep["d"][:, :-1]).mean()   # lagged one step
    assert acc > 0.9


def test_biased_mirror_cannot_tell_2_from_3(world, ep):
    m = mirror_biased(world, ep)
    assert np.allclose(m[..., 2], m[..., 3])
    assert np.allclose(m.sum(-1), 1.0, atol=1e-5)


def test_yoked_mirror_is_not_contingent(world, ep):
    y = make_mirror("YOKED", world, ep)
    acc = (y[:, 1:].argmax(-1) == ep["d"][:, :-1]).mean()
    assert acc < 0.4                                         # near chance, 0.25
    f = mirror_faithful(world, ep)
    assert np.allclose(np.sort(y.ravel()), np.sort(f.ravel()))  # same statistics


def test_mirror_off_means_zero_input(ep):
    s = torch.as_tensor(ep["s"][:4, :64])
    intero = torch.as_tensor(ep["intero"][:4])
    m = torch.ones(4, 64, K)
    x = infant_inputs(s, intero, m, torch.tensor([0, 1, 0, 1]))
    assert torch.all(x[0, :, -5:] == 0) and torch.all(x[2, :, -5:] == 0)
    assert torch.all(x[1, :, -1] == 1) and torch.all(x[1, :, -5:-1] == 1)


def test_no_gradient_crosses_from_infant_to_elder(ep):
    elder = CoLearnElder(16, seed=0)
    infant = Infant(16)
    _, h = elder(torch.as_tensor(ep["face"][:8]))
    m = elder.show(h)
    x = infant_inputs(torch.as_tensor(ep["s"][:8, :64]), torch.as_tensor(ep["intero"][:8]), m, torch.ones(8))
    logits, _ = infant(x)
    logits.sum().backward()
    assert all(p.grad is None for p in elder.parameters())


def test_probe_sanity():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 4000)
    feats = np.eye(4)[labels] * 3 + rng.standard_normal((4000, 4))
    assert probe_accuracy(feats, labels, 2000, [0, 1, 2, 3]) > 0.9
    noise = rng.standard_normal((4000, 8))
    assert probe_accuracy(noise, labels, 2000, [0, 1, 2, 3]) < 0.32
