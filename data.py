"""Loading and splitting MNIST.

Accepts the usual CSV layout (first column = label, next 784 columns = pixel
intensities 0-255) as well as a pre-packed ``.npz``.  Everything is returned
column-major -- shape ``(784, n_samples)`` -- scaled to [0, 1].
"""

from __future__ import annotations

import os

import numpy as np


def _has_header(path):
    """True if the first line is a header row rather than data."""
    with open(path) as f:
        first = f.readline().split(",")
    try:
        [float(x) for x in first]
        return False
    except ValueError:
        return True


def load_csv(path):
    """Return ``(X, y)`` with ``X`` of shape ``(784, n)`` in [0, 1]."""
    import pandas as pd

    cache = os.path.splitext(path)[0] + ".npz"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        cached = np.load(cache)
        return cached["X"], cached["y"]

    frame = pd.read_csv(path, header=0 if _has_header(path) else None)
    raw = frame.to_numpy()
    y = raw[:, 0].astype(np.int64)
    X = (raw[:, 1:].astype(np.float32) / 255.0).T
    np.savez_compressed(cache, X=X, y=y)  # parsing 60k rows of CSV is slow; cache it
    return X, y


def load_npz(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def split_validation(X, y, validation_size=10000, seed=0):
    """Hold out the last `validation_size` samples of a shuffled copy."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[1])
    X, y = X[:, order], y[order]
    cut = X.shape[1] - validation_size
    return (X[:, :cut], y[:cut]), (X[:, cut:], y[cut:])


def load_mnist(train_file, test_file, validation_size=10000, seed=0):
    """Load train/validation/test.

    The validation set is carved out of the training data.  The test set is
    touched **once**, at the very end -- tuning against it is how you end up
    reporting an accuracy you cannot reproduce on new data.
    """
    if train_file.endswith(".npz") and test_file == train_file:
        d = load_npz(train_file)
        return {
            "train": (d["Xtr"], d["ytr"]),
            "validation": (d["Xva"], d["yva"]),
            "test": (d["Xte"], d["yte"]),
        }

    X, y = load_csv(train_file)
    (Xtr, ytr), (Xva, yva) = split_validation(X, y, validation_size, seed)
    Xte, yte = load_csv(test_file)
    return {"train": (Xtr, ytr), "validation": (Xva, yva), "test": (Xte, yte)}
