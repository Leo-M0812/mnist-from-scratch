"""Tests for the network.

The important one is `test_gradients_match_numerical`: it compares the
analytic gradients from `backprop` against central-difference estimates for
every activation/cost combination.  A backprop bug that still trains -- the
usual kind -- shows up here and almost nowhere else.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import (  # noqa: E402
    ACTIVATIONS,
    COSTS,
    Network,
    Sigmoid,
    one_hot,
    random_shift,
    shift_images,
    softmax,
)


def toy(cost, activation, seed=0):
    net = Network([5, 4, 4, 3], hidden_activation=activation, cost=cost, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = rng.standard_normal((5, 7))
    Y = one_hot(rng.integers(0, 3, size=7), 3)
    return net, X, Y


@pytest.mark.parametrize("cost", list(COSTS))
@pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
def test_gradients_match_numerical(cost, activation):
    """Analytic and numerical gradients must agree to ~1e-7 relative."""
    net, X, Y = toy(cost, activation)
    nabla_w, nabla_b = net.backprop(X, Y)
    num_w, num_b = net.numerical_gradient(X, Y)

    for analytic, numeric in ((nabla_w, num_w), (nabla_b, num_b)):
        for a, n in zip(analytic, numeric):
            denom = np.maximum(np.abs(a) + np.abs(n), 1e-8)
            assert np.max(np.abs(a - n) / denom) < 1e-6


@pytest.mark.parametrize("cost", list(COSTS))
def test_gradients_match_numerical_relu(cost):
    """ReLU separately: the kink at 0 makes finite differences unreliable, so
    use a seed/scale where no pre-activation sits near zero."""
    net = Network([5, 6, 3], hidden_activation="relu", cost=cost, weight_init="he", seed=3)
    rng = np.random.default_rng(4)
    X = rng.standard_normal((5, 6))
    Y = one_hot(rng.integers(0, 3, size=6), 3)
    nabla_w, nabla_b = net.backprop(X, Y)
    num_w, num_b = net.numerical_gradient(X, Y, eps=1e-6)
    for analytic, numeric in ((nabla_w, num_w), (nabla_b, num_b)):
        for a, n in zip(analytic, numeric):
            denom = np.maximum(np.abs(a) + np.abs(n), 1e-8)
            assert np.max(np.abs(a - n) / denom) < 1e-5


def test_batch_gradient_is_mean_of_single_sample_gradients():
    """The mini-batch gradient must be the *average* of per-sample gradients.

    Summing instead of averaging silently rescales the learning rate by the
    batch size -- the classic mini-batch SGD bug.
    """
    net, X, Y = toy("cross-entropy", "sigmoid")
    batch_w, batch_b = net.backprop(X, Y)
    singles = [net.backprop(X[:, i : i + 1], Y[:, i : i + 1]) for i in range(X.shape[1])]
    mean_w = [np.mean([s[0][l] for s in singles], axis=0) for l in range(len(batch_w))]
    mean_b = [np.mean([s[1][l] for s in singles], axis=0) for l in range(len(batch_b))]
    for a, b in zip(batch_w, mean_w):
        assert np.allclose(a, b)
    for a, b in zip(batch_b, mean_b):
        assert np.allclose(a, b)


def test_cross_entropy_output_delta_equals_a_minus_y():
    """The sigma'(z) factor should cancel for the matched cost/output pairs."""
    a = np.array([[0.2], [0.7], [0.1]])
    y = np.array([[0.0], [1.0], [0.0]])
    assert np.allclose(COSTS["cross-entropy"].output_delta(a, y, Sigmoid), a - y)
    assert np.allclose(COSTS["log-likelihood"].output_delta(a, y, None), a - y)
    quad = COSTS["quadratic"].output_delta(a, y, Sigmoid)
    assert not np.allclose(quad, a - y)  # quadratic keeps the slowdown factor


def test_softmax_columns_sum_to_one_and_is_stable():
    z = np.array([[1000.0, -1000.0], [1000.0, -1000.0], [999.0, -1001.0]])
    p = softmax(z)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=0), 1.0)


def test_sigmoid_is_stable_at_extremes():
    z = np.array([[-800.0, 0.0, 800.0]])
    a = Sigmoid.f(z)
    assert np.all(np.isfinite(a))
    assert np.allclose(a, [[0.0, 0.5, 1.0]])


@pytest.mark.parametrize("name", list(ACTIVATIONS))
def test_activation_derivative_matches_numerical(name):
    act = ACTIVATIONS[name]
    z = np.array([[-2.0, -0.5, 0.5, 2.0]])
    eps = 1e-6
    numeric = (act.f(z + eps) - act.f(z - eps)) / (2 * eps)
    assert np.allclose(act.df(act.f(z)), numeric, atol=1e-5)


def test_one_hot():
    Y = one_hot([0, 3, 9], 10)
    assert Y.shape == (10, 3)
    assert Y.sum() == 3
    assert Y[3, 1] == 1.0


def test_forward_pass_shapes():
    net = Network([4, 3, 2], seed=0)
    acts = net.forward(np.zeros((4, 11)))
    assert [a.shape for a in acts] == [(4, 11), (3, 11), (2, 11)]


def test_training_reduces_cost_and_learns_a_toy_problem():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 300))
    labels = (X[0] + X[1] > 0).astype(int)
    Y = one_hot(labels, 2)
    net = Network([10, 12, 2], cost="cross-entropy", seed=0)
    before = net.total_cost(X, Y)
    net.fit(X, Y, epochs=40, batch_size=16, lr=0.5, monitor=False)
    assert net.total_cost(X, Y) < before
    assert net.accuracy(X, labels) > 0.95


def test_every_sample_is_used_when_batch_size_does_not_divide_n():
    """13 samples with batch size 5 must be 3 batches, not 2."""
    net = Network([3, 3, 2], seed=0)
    seen = []
    original = net.update_from_batch

    def spy(X, Y, *a, **k):
        seen.append(X.shape[1])
        return original(X, Y, *a, **k)

    net.update_from_batch = spy
    X = np.random.default_rng(0).standard_normal((3, 13))
    net.fit(X, one_hot(np.zeros(13), 2), epochs=1, batch_size=5, monitor=False)
    assert sum(seen) == 13
    assert seen == [5, 5, 3]


def test_l2_shrinks_weights():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 40))
    Y = one_hot(rng.integers(0, 3, 40), 3)
    norms = []
    for lmbda in (0.0, 20.0):
        net = Network([6, 8, 3], seed=1)
        net.fit(X, Y, epochs=30, batch_size=10, lr=0.3, lmbda=lmbda, monitor=False)
        norms.append(sum(np.sum(w ** 2) for w in net.weights))
    assert norms[1] < norms[0]


def test_save_load_round_trip(tmp_path):
    net = Network([4, 5, 3], hidden_activation="tanh", cost="log-likelihood", seed=2)
    X = np.random.default_rng(0).standard_normal((4, 5))
    before = net.predict(X)
    path = tmp_path / "model.npz"
    net.save(path, test_accuracy=0.99)
    loaded = Network.load(path)
    assert loaded.sizes == net.sizes
    assert loaded.cost.name == "log-likelihood"
    assert loaded.hidden_activation.name == "tanh"
    assert loaded.meta["test_accuracy"] == 0.99
    assert np.allclose(loaded.predict(X), before)


def test_shift_images_moves_pixels_and_zero_fills():
    img = np.zeros((28, 28))
    img[10, 10] = 1.0
    X = img.reshape(784, 1)
    out = shift_images(X, dx=2, dy=-3).reshape(28, 28)
    assert out[7, 12] == 1.0
    assert out.sum() == 1.0


def test_random_shift_preserves_shape_and_most_ink():
    rng = np.random.default_rng(0)
    X = np.zeros((784, 50))
    X[400, :] = 1.0
    out = random_shift(X, rng)
    assert out.shape == X.shape
    assert np.allclose(out.sum(), X.sum())


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        Network([2, 2], cost="hinge")
    with pytest.raises(ValueError):
        Network([2, 2], hidden_activation="elu")
    with pytest.raises(ValueError):
        Network([2])
