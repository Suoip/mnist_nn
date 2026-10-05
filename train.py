"""Train the network on MNIST, report its test accuracy and save it.

    python train.py
    python train.py --epochs 5 --hidden-size 64
"""
import argparse
import math
import time

import numpy as np

import network
from mnist import load_mnist


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST network and save it to a .npz file.")
    parser.add_argument("--hidden-size", type=int, default=512, help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=30, help="passes over the training set")
    parser.add_argument("--lr", type=float, default=0.05, help="starting learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum (0 = plain SGD)")
    parser.add_argument("--dropout", type=float, default=0.2, help="fraction of hidden units dropped while training")
    parser.add_argument("--no-augment", action="store_true", help="don't randomly shift the training images")
    parser.add_argument("--batch-size", type=int, default=64, help="images per gradient step")
    parser.add_argument("--seed", type=int, default=0, help="random seed (same seed = same result)")
    parser.add_argument("--out", default="model.npz", help="where to save the trained model")
    parser.add_argument("--history", default="history.npz",
                        help="where to save snapshots taken during training (for export_web.py)")
    args = parser.parse_args()

    (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test) = load_mnist()
    print(f"{X_train.shape[1]} train / {X_dev.shape[1]} dev / {X_test.shape[1]} test images")

    # Snapshots of the network while it learns, for the web page's training replay.
    # Taken often at the very start, where it learns fastest, then after every epoch.
    m = X_train.shape[1]
    steps_per_epoch = math.ceil(m / args.batch_size)
    snapshot_steps = {0, 1, 2, 5, 10, 20, 50, 100, 200, steps_per_epoch // 2}
    snapshot_steps |= {steps_per_epoch * e for e in range(1, args.epochs + 1)}
    snapshots = []

    def take_snapshot(step, params):
        if step in snapshot_steps:
            epochs_done, steps_into_epoch = divmod(step, steps_per_epoch)
            # float16 keeps the file small; plenty for drawing pictures of the weights
            snapshots.append({"images_seen": epochs_done * m + steps_into_epoch * args.batch_size,
                              **{k: v.astype(np.float16) for k, v in params.items()}})

    start = time.time()
    params = network.train(
        X_train, Y_train, X_dev, Y_dev,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        batch_size=args.batch_size,
        drop_rate=args.dropout,
        augment=not args.no_augment,
        seed=args.seed,
        on_step=take_snapshot,
    )
    print(f"Training took {time.time() - start:.0f}s")

    # The test set is used exactly once, at the very end, for the final score.
    print(f"Test accuracy: {network.accuracy(params, X_test, Y_test) * 100:.2f}%")
    np.savez(args.out, **params)
    print(f"Model saved to {args.out}")
    np.savez(args.history, **{k: np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
    print(f"{len(snapshots)} training snapshots saved to {args.history}")


if __name__ == "__main__":
    main()
