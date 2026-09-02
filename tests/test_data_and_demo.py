"""Tests for CSV loading, the validation split, and the demo preprocessing."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import _has_header, load_csv, load_mnist, split_validation  # noqa: E402


def write_csv(path, n, header, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, n)
    pixels = rng.integers(0, 256, (n, 784))
    with open(path, "w") as f:
        if header:
            f.write("label," + ",".join(f"px{i}" for i in range(784)) + "\n")
        for lab, row in zip(labels, pixels):
            f.write(f"{lab}," + ",".join(map(str, row)) + "\n")
    return labels


@pytest.mark.parametrize("header", [True, False])
def test_csv_loads_every_row_with_or_without_a_header(tmp_path, header):
    """A header row must not be silently eaten as a training example."""
    path = tmp_path / "d.csv"
    labels = write_csv(path, 20, header)
    assert _has_header(str(path)) is header
    X, y = load_csv(str(path))
    assert X.shape == (784, 20)
    assert np.array_equal(y, labels)
    assert 0.0 <= X.min() and X.max() <= 1.0


def test_csv_cache_round_trips(tmp_path):
    path = tmp_path / "d.csv"
    write_csv(path, 10, True)
    X1, y1 = load_csv(str(path))
    assert (tmp_path / "d.npz").exists()
    X2, y2 = load_csv(str(path))  # second call hits the cache
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_validation_split_is_disjoint_and_the_right_size():
    X = np.arange(784 * 50, dtype=float).reshape(784, 50)
    y = np.arange(50)
    (Xtr, ytr), (Xva, yva) = split_validation(X, y, validation_size=10)
    assert Xtr.shape[1] == 40 and Xva.shape[1] == 10
    assert set(ytr).isdisjoint(set(yva))
    assert set(ytr) | set(yva) == set(range(50))


def test_load_mnist_keeps_test_separate(tmp_path):
    train, test = tmp_path / "tr.csv", tmp_path / "te.csv"
    write_csv(train, 30, True, seed=1)
    write_csv(test, 10, True, seed=2)
    sets = load_mnist(str(train), str(test), validation_size=10)
    assert sets["train"][0].shape[1] == 20
    assert sets["validation"][0].shape[1] == 10
    assert sets["test"][0].shape[1] == 10


def test_demo_preprocessing_centres_the_digit():
    """A stroke drawn in the corner must come out centred and 28x28."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    from draw_demo import preprocess

    img = Image.new("L", (280, 280), 0)
    ImageDraw.Draw(img).line([(20, 20), (20, 90)], fill=255, width=14)
    x = preprocess(img)
    assert x.shape == (784, 1)
    assert 0.0 <= x.min() and x.max() <= 1.0

    pixels = x.reshape(28, 28)
    rows, cols = np.indices((28, 28))
    total = pixels.sum()
    cy = (pixels * rows).sum() / total
    cx = (pixels * cols).sum() / total
    assert abs(cy - 13.5) < 1.0 and abs(cx - 13.5) < 1.0


def test_demo_preprocessing_handles_an_empty_canvas():
    pytest.importorskip("PIL")
    from PIL import Image

    from draw_demo import preprocess

    assert preprocess(Image.new("L", (280, 280), 0)) is None
