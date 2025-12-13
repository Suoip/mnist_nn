# MNIST from scratch (minimal)

This repository contains a minimal-from-scratch 2-layer fully-connected neural network for MNIST.
It focuses on clarity and teaching: all code is plain NumPy and small enough to study.

Project layout
-`mnist.py` - dataset utilities: `load_data()` and `display_image()` (IDX and CSV support)
-`cnn.py` - neural network primitives and training utilities (forward/backward/train/lr_finder)
-`train.py` - CLI to train and save a model (`.npz`)
-`test.py` - CLI to load a model and inspect predictions (includes `--show-wrongs`)
-`download_mnist.py` - helper to download and extract MNIST IDX files into `data/`

Setup

Install dependencies in PowerShell:

```powershell
pip install -r requirements.txt
```

Put your data files in the `data/` folder. Supported input formats (checked in this order):
-Original MNIST IDX files: `train-images.idx3-ubyte` and `train-labels.idx1-ubyte` (recommended).
-CSV file named `mnist_train.csv` or `train.csv` where the first column is the label and the following 784 columns are pixels (0-255).

If you don't have IDX files, `download_mnist.py` can fetch and extract them for you.

Scripts & CLI reference

Below are the available commands, flags, descriptions and copy/paste PowerShell examples.

`download_mnist.py` - Download and extract MNIST IDX files into `data/`.

Options:
-`--delete-gz`: delete the downloaded `.gz` files after extraction.

Examples:

```powershell
python .\download_mnist.py
```

```powershell
python .\download_mnist.py --delete-gz
```

`train.py` - Train a model and save parameters to a `.npz` file.

Common flags:
-`--data-dir`: directory containing your data (default: `data`)
-`--dev-size`: number of examples reserved for the dev/validation set (default: `1000`)
-`--hidden-size`: hidden layer width (default: `128`)
-`--lr`: learning rate (default: `0.1`)
-`--epochs`: number of training epochs (default: `10`)
-`--batch-size`: mini-batch size (default: `64`)
-`--print-every`: print metrics every N epochs (default: `1`)
-`--out`: output `.npz` path (default: `model.npz`)
-`--no-tqdm`: disable progress bars
-`--lr-finder`: run a quick LR range test and exit

LR finder flags:
-`--lr-start` (default: `1e-7`)
-`--lr-end` (default: `1.0`)
-`--lr-iters` (default: `100`)
-`--lr-batch-size` (default: `128`)

Example:

```powershell
python .\train.py --epochs 10 --batch-size 64 --lr 0.1 --out model.npz
```

`test.py` - Load a saved `.npz` model and inspect predictions.

Flags:
-`--data-dir`: data directory (default `data`)
-`--dev-size`: dev split size used by `load_data()` (default `1000`)
-`--model`: path to the saved `.npz` model (default `model.npz`)
-`--num`: number of random samples to show (default `5`)
-`--use-dev`: use the dev set instead of the training set
-`--show-wrongs`: show misclassified examples from the chosen set
-`--limit`: max number of wrong examples to display when `--show-wrongs` is used (default `20`)

Examples:

```powershell
python .\test.py --model model.npz --num 5
```

```powershell
python .\test.py --model model.npz --use-dev --show-wrongs --limit 20
```

`mnist.py` utilities:
-`load_data(data_dir='data', dev_size=1000, shuffle=True)` — returns `(X_train, Y_train, X_dev, Y_dev)` with `X` shaped `(features, m)` (i.e., `(784, m)`).
-`display_image(image_vector, cmap='gray')` — show a 28x28 image using `matplotlib`.

Troubleshooting
-If the loader picks up `sample_submission.csv`, move it out of `data/` and add the proper IDX files or a CSV with 784 pixel columns.
-If images do not show, ensure `matplotlib` is installed and GUI is available (or use a notebook).
-If training is slow, reduce `--hidden-size` or `--epochs`, or increase `--batch-size`.

License: small teaching project - reuse freely for learning and experimentation.
