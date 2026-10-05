"""Load MNIST from the .gz files that download_mnist.py fetches.

MNIST is 70,000 grayscale 28x28 images of handwritten digits, split by its
creators into 60,000 training images and 10,000 test images.

The files use the simple IDX format: a big-endian header, then raw bytes.
    images: magic 2051, count, rows, cols, then count*rows*cols pixels (0-255)
    labels: magic 2049, count, then count labels (0-9)
"""
import gzip
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


def _read_gz(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python download_mnist.py` first")
    with gzip.open(path, "rb") as f:
        return f.read()


def read_images(name):
    """Return the images as an array of shape (784, count), values in [0, 1]."""
    data = _read_gz(name)
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])  # ">" big-endian, "I" 4-byte uint
    assert magic == 2051, f"{name} is not an IDX image file"
    pixels = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows * cols)
    # Transpose so each image is a column (see the shape convention in network.py),
    # and scale 0..255 down to 0..1 so the inputs are small numbers.
    return pixels.T.astype(np.float64) / 255


def read_labels(name):
    """Return the labels as an int array of shape (count,)."""
    data = _read_gz(name)
    magic, count = struct.unpack(">II", data[:8])
    assert magic == 2049, f"{name} is not an IDX label file"
    return np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int64)


def load_mnist(dev_size=5000):
    """Return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test).

    X arrays have shape (784, m), Y arrays have shape (m,).

    Three sets, three jobs:
      train - the images the network learns from.
      dev   - held out of training and checked after every epoch, to watch
              progress and to compare settings (hidden size, lr, ...).
      test  - the official 10,000 test images. Only used for the final score,
              so that score stays an honest measure of unseen digits.
    The dev set is always the *last* dev_size training images, so every script
    agrees on which images were held out.
    """
    X = read_images("train-images-idx3-ubyte.gz")
    Y = read_labels("train-labels-idx1-ubyte.gz")
    X_test = read_images("t10k-images-idx3-ubyte.gz")
    Y_test = read_labels("t10k-labels-idx1-ubyte.gz")
    split = X.shape[1] - dev_size
    return (X[:, :split], Y[:split]), (X[:, split:], Y[split:]), (X_test, Y_test)
