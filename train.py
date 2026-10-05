"""Train the network on MNIST, report its test accuracy and save it.

    python train.py
    python train.py --epochs 5 --hidden-size 64
"""
import argparse
import time

import numpy as np

import network
from mnist import load_mnist


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST network and save it to a .npz file.")
    parser.add_argument("--hidden-size", type=int, default=512, help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=20, help="passes over the training set")
    parser.add_argument("--lr", type=float, default=0.05, help="starting learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum (0 = plain SGD)")
    parser.add_argument("--dropout", type=float, default=0.2, help="fraction of hidden units dropped while training")
    parser.add_argument("--batch-size", type=int, default=64, help="images per gradient step")
    parser.add_argument("--seed", type=int, default=0, help="random seed (same seed = same result)")
    parser.add_argument("--out", default="model.npz", help="where to save the trained model")
    args = parser.parse_args()

    (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test) = load_mnist()
    print(f"{X_train.shape[1]} train / {X_dev.shape[1]} dev / {X_test.shape[1]} test images")

    start = time.time()
    params = network.train(
        X_train, Y_train, X_dev, Y_dev,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        batch_size=args.batch_size,
        drop_rate=args.dropout,
        seed=args.seed,
    )
    print(f"Training took {time.time() - start:.0f}s")

    # The test set is used exactly once, at the very end, for the final score.
    print(f"Test accuracy: {network.accuracy(params, X_test, Y_test) * 100:.2f}%")
    np.savez(args.out, **params)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
