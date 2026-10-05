"""Load a trained model, report its test accuracy and show test images
with the network's guesses (green = right, red = wrong).

    python test.py            # random test images
    python test.py --wrong    # only the ones the network got wrong
"""
import argparse
import math

import numpy as np
from matplotlib import pyplot as plt

import network
from mnist import load_mnist


def main():
    parser = argparse.ArgumentParser(description="Inspect a trained MNIST model on the test set.")
    parser.add_argument("--model", default="model.npz", help="model saved by train.py")
    parser.add_argument("--num", type=int, default=25, help="how many images to show")
    parser.add_argument("--wrong", action="store_true", help="only show misclassified images")
    args = parser.parse_args()

    params = dict(np.load(args.model))
    _, _, (X, Y) = load_mnist()

    probs, _ = network.forward(params, X)
    guesses = probs.argmax(axis=0)
    confidence = probs.max(axis=0)
    wrong = np.flatnonzero(guesses != Y)
    print(f"Test accuracy: {100 * (1 - wrong.size / Y.size):.2f}%  ({wrong.size} of {Y.size} wrong)")

    pool = wrong if args.wrong else np.arange(Y.size)
    shown = np.random.default_rng().choice(pool, size=min(args.num, pool.size), replace=False)
    if shown.size == 0:
        return

    # One figure with a grid of images, rather than one window per image
    cols = math.ceil(math.sqrt(shown.size))
    rows = math.ceil(shown.size / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 2.1 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, shown):
        ax.imshow(X[:, i].reshape(28, 28), cmap="gray")
        ax.set_title(f"guess {guesses[i]} ({confidence[i]:.0%})\nlabel {Y[i]}",
                     fontsize=9, color="green" if guesses[i] == Y[i] else "red")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
