"""A feed-forward neural network written from scratch with NumPy.

Conventions
-----------
Data is stored **column-major**: an array of ``m`` samples with ``n`` features
has shape ``(n, m)``, so a single sample is a column vector.  This is the
convention used in Nielsen's *Neural Networks and Deep Learning* and it keeps
the forward pass as a plain matrix product ``W @ A + b``.

Every layer operation is expressed on a whole mini-batch at once, so there is
no Python-level loop over samples anywhere in training or inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Activation functions
# --------------------------------------------------------------------------- #
# Each activation exposes `f(z)` and `df(a)`, where `df` takes the *activation*
# a = f(z) rather than the pre-activation z.  For all three functions below the
# derivative can be written in terms of a, which saves keeping z around.


class Sigmoid:
    name = "sigmoid"

    @staticmethod
    def f(z):
        # Branch on the sign of z so neither exp() call can overflow:
        #   z >= 0:  1 / (1 + e^-z)
        #   z <  0:  e^z / (1 + e^z)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    @staticmethod
    def df(a):
        return a * (1.0 - a)


class Tanh:
    name = "tanh"

    @staticmethod
    def f(z):
        return np.tanh(z)

    @staticmethod
    def df(a):
        return 1.0 - a * a


class ReLU:
    name = "relu"

    @staticmethod
    def f(z):
        return np.maximum(z, 0.0)

    @staticmethod
    def df(a):
        return (a > 0.0).astype(float)


ACTIVATIONS = {c.name: c for c in (Sigmoid, Tanh, ReLU)}


def softmax(z):
    """Column-wise softmax, shifted by the column max for numerical stability."""
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


# --------------------------------------------------------------------------- #
# Cost functions
# --------------------------------------------------------------------------- #
# `output_delta` returns delta^L = dC/dz^L for one sample-column.  The whole
# point of pairing cross-entropy with a sigmoid output (or log-likelihood with
# softmax) is that the sigma'(z) factor cancels and delta^L collapses to (a - y).


class QuadraticCost:
    name = "quadratic"
    output_layer = "sigmoid"

    @staticmethod
    def loss(a, y):
        return 0.5 * np.sum((a - y) ** 2) / a.shape[1]

    @staticmethod
    def output_delta(a, y, output_activation):
        return (a - y) * output_activation.df(a)


class CrossEntropyCost:
    name = "cross-entropy"
    output_layer = "sigmoid"

    @staticmethod
    def loss(a, y):
        a = np.clip(a, 1e-12, 1.0 - 1e-12)
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / a.shape[1]

    @staticmethod
    def output_delta(a, y, output_activation):
        return a - y


class LogLikelihoodCost:
    name = "log-likelihood"
    output_layer = "softmax"

    @staticmethod
    def loss(a, y):
        a = np.clip(a, 1e-12, 1.0)
        return -np.sum(y * np.log(a)) / a.shape[1]

    @staticmethod
    def output_delta(a, y, output_activation):
        return a - y


COSTS = {c.name: c for c in (QuadraticCost, CrossEntropyCost, LogLikelihoodCost)}


# --------------------------------------------------------------------------- #
# Training history
# --------------------------------------------------------------------------- #


@dataclass
class History:
    train_cost: list = field(default_factory=list)
    train_accuracy: list = field(default_factory=list)
    eval_cost: list = field(default_factory=list)
    eval_accuracy: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    best_eval_accuracy: float = 0.0
    best_epoch: int = -1


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class Network:
    """A fully connected network of arbitrary depth.

    Parameters
    ----------
    sizes:
        Nodes per layer, e.g. ``[784, 100, 10]``.
    hidden_activation:
        ``"sigmoid"``, ``"tanh"`` or ``"relu"``.
    cost:
        ``"quadratic"``, ``"cross-entropy"`` (sigmoid output) or
        ``"log-likelihood"`` (softmax output).  The output layer is chosen by
        the cost, since only those pairings give delta^L = a - y.
    weight_init:
        ``"scaled"``  -> N(0, 1/n_in)  (sigmoid/tanh; Nielsen ch.3)
        ``"he"``      -> N(0, 2/n_in)  (the right choice for ReLU)
        ``"large"``   -> N(0, 1), the naive initialisation, kept for comparison.
        ``"auto"`` (or ``None``) picks "he" for ReLU and "scaled" otherwise.
    """

    def __init__(
        self,
        sizes,
        hidden_activation="sigmoid",
        cost="cross-entropy",
        weight_init="auto",
        seed=None,
    ):
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output layer")
        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {hidden_activation!r}")
        if cost not in COSTS:
            raise ValueError(f"unknown cost {cost!r}")

        self.sizes = list(sizes)
        self.num_layers = len(sizes)
        self.hidden_activation = ACTIVATIONS[hidden_activation]
        self.cost = COSTS[cost]
        self.output_layer = self.cost.output_layer  # "sigmoid" or "softmax"
        if weight_init in (None, "auto"):
            weight_init = "he" if hidden_activation == "relu" else "scaled"
        if weight_init not in ("scaled", "he", "large"):
            raise ValueError(f"unknown weight_init {weight_init!r}")
        self.weight_init = weight_init
        self.rng = np.random.default_rng(seed)
        self.initialise_parameters()

    # -- initialisation ---------------------------------------------------- #

    def initialise_parameters(self):
        scale = {
            "scaled": lambda n_in: 1.0 / np.sqrt(n_in),
            "he": lambda n_in: np.sqrt(2.0 / n_in),
            "large": lambda n_in: 1.0,
        }[self.weight_init]

        self.weights = [
            self.rng.standard_normal((n_out, n_in)) * scale(n_in)
            for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.biases = [np.zeros((n_out, 1)) for n_out in self.sizes[1:]]
        # Momentum buffers.
        self._vw = [np.zeros_like(w) for w in self.weights]
        self._vb = [np.zeros_like(b) for b in self.biases]

    # -- forward / backward ------------------------------------------------ #

    def forward(self, X):
        """Return the list of activations, one array per layer.

        ``X`` has shape ``(n_input, m)``; the returned arrays have shape
        ``(n_layer, m)``.
        """
        a = X
        activations = [a]
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ a + b
            if i == last:
                a = softmax(z) if self.output_layer == "softmax" else Sigmoid.f(z)
            else:
                a = self.hidden_activation.f(z)
            activations.append(a)
        return activations

    def predict(self, X):
        """Class probabilities (or sigmoid scores) for every column of X."""
        return self.forward(X)[-1]

    def backprop(self, X, Y):
        """Gradients of the *mean* cost over the batch.

        Returns ``(nabla_w, nabla_b)``, each a list matching ``self.weights`` /
        ``self.biases``.
        """
        m = X.shape[1]
        activations = self.forward(X)

        out_act = Sigmoid if self.output_layer == "sigmoid" else None
        delta = self.cost.output_delta(activations[-1], Y, out_act)

        nabla_w = [None] * len(self.weights)
        nabla_b = [None] * len(self.biases)
        nabla_w[-1] = delta @ activations[-2].T / m
        nabla_b[-1] = delta.sum(axis=1, keepdims=True) / m

        # Walk backwards: delta^l = ((W^{l+1})^T delta^{l+1}) * f'(a^l)
        for l in range(2, self.num_layers):
            delta = (self.weights[-l + 1].T @ delta) * self.hidden_activation.df(
                activations[-l]
            )
            nabla_w[-l] = delta @ activations[-l - 1].T / m
            nabla_b[-l] = delta.sum(axis=1, keepdims=True) / m

        return nabla_w, nabla_b

    # -- parameter update -------------------------------------------------- #

    def update_from_batch(self, X, Y, lr, lmbda=0.0, momentum=0.0, n_train=1):
        """One gradient step. L2 penalty is lmbda/(2n) * sum(w^2)."""
        nabla_w, nabla_b = self.backprop(X, Y)
        for i in range(len(self.weights)):
            grad_w = nabla_w[i] + (lmbda / n_train) * self.weights[i]
            self._vw[i] = momentum * self._vw[i] - lr * grad_w
            self._vb[i] = momentum * self._vb[i] - lr * nabla_b[i]
            self.weights[i] += self._vw[i]
            self.biases[i] += self._vb[i]

    # -- training ---------------------------------------------------------- #

    def fit(
        self,
        X,
        Y,
        epochs=30,
        batch_size=32,
        lr=0.5,
        lmbda=0.0,
        momentum=0.0,
        eval_data=None,
        augment=False,
        lr_decay=1.0,
        patience=None,
        early_stopping=None,
        monitor=True,
        progress=None,
        keep_best=True,
    ):
        """Mini-batch stochastic gradient descent.

        ``X``: ``(n_input, n_samples)``, ``Y``: ``(n_output, n_samples)`` one-hot.
        ``eval_data``: ``(X_eval, y_eval_labels)`` used **only** for monitoring,
        learning-rate decay and early stopping -- never for gradients.

        ``lr_decay``  multiplies the learning rate whenever the evaluation
        accuracy has not improved for ``patience`` epochs.
        ``early_stopping`` stops after that many epochs without improvement.
        ``keep_best`` restores the parameters from the best epoch at the end.
        """
        X = np.ascontiguousarray(X, dtype=float)
        Y = np.ascontiguousarray(Y, dtype=float)
        n = X.shape[1]
        history = History()
        best_params = None
        stale = 0

        for epoch in range(epochs):
            order = self.rng.permutation(n)
            Xs, Ys = X[:, order], Y[:, order]
            if augment:
                Xs = random_shift(Xs, self.rng)

            # Every sample is used, including the final short batch.
            n_batches = int(np.ceil(n / batch_size))
            batches = range(n_batches)
            if progress is not None:
                batches = progress(batches, desc=f"epoch {epoch + 1}/{epochs}")
            for k in batches:
                lo, hi = k * batch_size, min((k + 1) * batch_size, n)
                self.update_from_batch(
                    Xs[:, lo:hi], Ys[:, lo:hi], lr, lmbda, momentum, n_train=n
                )

            history.learning_rates.append(lr)
            if monitor:
                history.train_cost.append(self.total_cost(X, Y, lmbda))
                history.train_accuracy.append(self.accuracy(X, Y.argmax(axis=0)))

            if eval_data is not None:
                Xe, ye = eval_data
                acc = self.accuracy(Xe, ye)
                history.eval_accuracy.append(acc)
                if monitor:
                    history.eval_cost.append(
                        self.total_cost(Xe, one_hot(ye, self.sizes[-1]), lmbda)
                    )
                if acc > history.best_eval_accuracy:
                    history.best_eval_accuracy = acc
                    history.best_epoch = epoch
                    stale = 0
                    if keep_best:
                        best_params = (
                            [w.copy() for w in self.weights],
                            [b.copy() for b in self.biases],
                        )
                else:
                    stale += 1
                    if patience and stale and stale % patience == 0 and lr_decay != 1.0:
                        lr *= lr_decay
                    if early_stopping and stale >= early_stopping:
                        break

        if keep_best and best_params is not None:
            self.weights, self.biases = best_params
        return history

    # -- evaluation -------------------------------------------------------- #

    def accuracy(self, X, labels, batch_size=2048):
        correct = 0
        for lo in range(0, X.shape[1], batch_size):
            chunk = X[:, lo : lo + batch_size]
            correct += int(
                np.sum(self.predict(chunk).argmax(axis=0) == labels[lo : lo + batch_size])
            )
        return correct / X.shape[1]

    def total_cost(self, X, Y, lmbda=0.0, batch_size=2048):
        n = X.shape[1]
        total = 0.0
        for lo in range(0, n, batch_size):
            a = self.predict(X[:, lo : lo + batch_size])
            y = Y[:, lo : lo + batch_size]
            total += self.cost.loss(a, y) * a.shape[1]
        total /= n
        total += 0.5 * (lmbda / n) * sum(np.sum(w ** 2) for w in self.weights)
        return total

    # -- persistence ------------------------------------------------------- #

    def save(self, path, **extra):
        meta = {
            "sizes": self.sizes,
            "hidden_activation": self.hidden_activation.name,
            "cost": self.cost.name,
            "weight_init": self.weight_init,
            **extra,
        }
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        net = cls(
            meta["sizes"],
            hidden_activation=meta["hidden_activation"],
            cost=meta["cost"],
            weight_init=meta.get("weight_init", "scaled"),
        )
        net.weights = [data[f"w{i}"] for i in range(len(net.weights))]
        net.biases = [data[f"b{i}"] for i in range(len(net.biases))]
        net._vw = [np.zeros_like(w) for w in net.weights]
        net._vb = [np.zeros_like(b) for b in net.biases]
        net.meta = meta
        return net

    # -- gradient checking ------------------------------------------------- #

    def numerical_gradient(self, X, Y, eps=1e-5):
        """Central-difference gradients, for testing `backprop` only.

        Cost of this is O(number of parameters) forward passes, so use it on a
        toy network, never on the real one.
        """
        num_w, num_b = [], []
        for params, store in ((self.weights, num_w), (self.biases, num_b)):
            for p in params:
                g = np.zeros_like(p)
                it = np.nditer(p, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    original = p[idx]
                    p[idx] = original + eps
                    plus = self.cost.loss(self.predict(X), Y)
                    p[idx] = original - eps
                    minus = self.cost.loss(self.predict(X), Y)
                    p[idx] = original
                    g[idx] = (plus - minus) / (2 * eps)
                    it.iternext()
                store.append(g)
        return num_w, num_b


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def one_hot(labels, n_classes=10):
    """(m,) integer labels -> (n_classes, m) one-hot columns."""
    labels = np.asarray(labels).astype(int).ravel()
    Y = np.zeros((n_classes, labels.size))
    Y[labels, np.arange(labels.size)] = 1.0
    return Y


def shift_images(X, dx, dy, side=28):
    """Translate every column of X (flattened images) by (dx, dy), zero-filled."""
    m = X.shape[1]
    imgs = X.T.reshape(m, side, side)
    out = np.zeros_like(imgs)
    ys, yd = (slice(0, side - dy), slice(dy, side)) if dy >= 0 else (
        slice(-dy, side),
        slice(0, side + dy),
    )
    xs, xd = (slice(0, side - dx), slice(dx, side)) if dx >= 0 else (
        slice(-dx, side),
        slice(0, side + dx),
    )
    out[:, yd, xd] = imgs[:, ys, xs]
    return out.reshape(m, side * side).T


def random_shift(X, rng, max_shift=1, side=28):
    """Randomly translate each image by up to `max_shift` pixels in x and y.

    Nielsen ch.3 notes that expanding MNIST with one-pixel translations is one
    of the cheapest accuracy wins available; doing it per epoch gives the same
    effect without holding a 5x copy of the data in memory.
    """
    m = X.shape[1]
    out = np.empty_like(X)
    shifts = rng.integers(-max_shift, max_shift + 1, size=(m, 2))
    keys = (shifts[:, 0] + max_shift) * (2 * max_shift + 1) + (shifts[:, 1] + max_shift)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        dx, dy = shifts[idx[0]]
        out[:, idx] = (
            X[:, idx] if (dx == 0 and dy == 0) else shift_images(X[:, idx], int(dx), int(dy), side)
        )
    return out
