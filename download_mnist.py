"""Download the four MNIST files into the data/ folder next to this script.

The files stay as .gz archives: mnist.py reads them directly, so there is
nothing to extract or rename afterwards.

    python download_mnist.py
"""
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = [
    "train-images-idx3-ubyte.gz",  # 60,000 training images
    "train-labels-idx1-ubyte.gz",  # their labels (0-9)
    "t10k-images-idx3-ubyte.gz",   # 10,000 test images
    "t10k-labels-idx1-ubyte.gz",   # their labels
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        path = DATA_DIR / name
        if path.exists():
            print(f"[ok]       {name}")
            continue
        print(f"[download] {name} ...")
        # Download to a temporary name and only rename once it is complete, so an
        # interrupted download is never mistaken for a finished file next time.
        partial = path.with_name(name + ".part")
        urllib.request.urlretrieve(BASE_URL + name, partial)
        partial.replace(path)
    print(f"MNIST is ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
