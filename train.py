"""Command-line training / evaluation for the from-scratch MNIST network.

Examples
--------
Train with the defaults (the tuned configuration from the README)::

    python train.py --train mnist_train.csv --test mnist_test.csv

Reproduce the textbook baseline::

    python train.py --layers 784 30 10 --cost quadratic --lr 3.0 --epochs 30

Load a saved model, score it and try it on your own handwriting::

    python train.py --load model.npz --test mnist_test.csv --draw
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from data import load_mnist
from network import Network, one_hot


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Neural network from scratch (NumPy only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--train", default="mnist_train.csv", help="training CSV, or an .npz")
    data.add_argument("--test", default="mnist_test.csv", help="test CSV")
    data.add_argument("--validation-size", type=int, default=10000,
                      help="samples held out of the training set for tuning")
    data.add_argument("--no-test", action="store_true",
                      help="skip the test set entirely (use while tuning)")

    arch = p.add_argument_group("architecture")
    arch.add_argument("--layers", type=int, nargs="+", default=[784, 256, 128, 10],
                      help="nodes per layer; first must be 784 and last 10")
    arch.add_argument("--activation", default="tanh", choices=["sigmoid", "tanh", "relu"])
    arch.add_argument("--cost", default="log-likelihood",
                      choices=["quadratic", "cross-entropy", "log-likelihood"])
    arch.add_argument("--init", default="auto", choices=["auto", "scaled", "he", "large"],
                      help="auto = 'he' for relu, 'scaled' otherwise")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=100)
    opt.add_argument("--batch-size", type=int, default=32)
    opt.add_argument("--lr", type=float, default=0.3)
    opt.add_argument("--lmbda", type=float, default=1.0, help="L2 regularisation strength")
    opt.add_argument("--momentum", type=float, default=0.0)
    opt.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                     help="randomly translate each image by up to 1px each epoch")
    opt.add_argument("--lr-decay", type=float, default=0.5,
                     help="multiply lr by this after --patience stale epochs")
    opt.add_argument("--patience", type=int, default=6)
    opt.add_argument("--early-stopping", type=int, default=20,
                     help="stop after this many epochs with no validation improvement")
    opt.add_argument("--seed", type=int, default=None)

    io = p.add_argument_group("model i/o")
    io.add_argument("--save", default="model.npz", help="where to write the trained model")
    io.add_argument("--load", default=None, help="skip training and load this model")
    io.add_argument("--draw", action="store_true",
                    help="open the draw-a-digit demo when finished")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.load:
        net = Network.load(args.load)
        print(f"loaded {args.load}: {net.sizes}, {net.hidden_activation.name}, {net.cost.name}")
        if not args.no_test:
            from data import load_csv
            Xte, yte = load_csv(args.test)
            print(f"test accuracy: {net.accuracy(Xte, yte) * 100:.2f}%")
    else:
        if args.layers[0] != 784 or args.layers[-1] != 10:
            raise SystemExit("--layers must start at 784 (pixels) and end at 10 (digits)")

        sets = load_mnist(args.train, args.test, args.validation_size, seed=0)
        Xtr, ytr = sets["train"]
        Xva, yva = sets["validation"]
        print(f"train {Xtr.shape[1]}  validation {Xva.shape[1]}  test {sets['test'][1].size}")

        net = Network(args.layers, hidden_activation=args.activation, cost=args.cost,
                      weight_init=args.init, seed=args.seed)
        print(f"{args.layers}  {args.activation}  {args.cost}  "
              f"lr={args.lr} lambda={args.lmbda} momentum={args.momentum} "
              f"batch={args.batch_size} augment={args.augment}")

        started = time.perf_counter()
        history = net.fit(
            Xtr, one_hot(ytr, 10),
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            lmbda=args.lmbda, momentum=args.momentum, augment=args.augment,
            eval_data=(Xva, yva), lr_decay=args.lr_decay, patience=args.patience,
            early_stopping=args.early_stopping, monitor=True, progress=_progress,
        )
        for i, (c, a) in enumerate(zip(history.train_cost, history.eval_accuracy)):
            print(f"  epoch {i + 1:3d}  train cost {c:.4f}  validation {a * 100:.2f}%")
        print(f"best validation {history.best_eval_accuracy * 100:.2f}% "
              f"at epoch {history.best_epoch + 1}  ({time.perf_counter() - started:.1f}s)")

        test_accuracy = None
        if not args.no_test:
            Xte, yte = sets["test"]
            test_accuracy = net.accuracy(Xte, yte)
            print(f"test accuracy: {test_accuracy * 100:.2f}%")

        net.save(args.save, validation_accuracy=history.best_eval_accuracy,
                 test_accuracy=test_accuracy, epochs_run=len(history.eval_accuracy))
        print(f"saved {args.save}")

    if args.draw:
        from draw_demo import run_paint_demo
        run_paint_demo(net)


def _progress(iterable, desc=""):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, unit="batch", leave=False)


if __name__ == "__main__":
    main()
