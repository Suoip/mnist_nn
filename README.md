# MNIST neural network from scratch

A small fully-connected neural network that learns to read handwritten digits,
written with nothing but NumPy: no PyTorch, no TensorFlow. Forward pass,
backpropagation and training loop are all in `network.py` (about 200 lines,
about half of them comments).

It reaches **about 99% accuracy** on the official MNIST test set in about a
minute of training on a CPU.

## Quick start

Works the same on Windows, macOS and Linux (on macOS/Linux you may need to type
`python3` instead of `python`).

```
pip install -r requirements.txt
python download_mnist.py      # fetches the 4 MNIST files into data/
python train.py               # trains, prints test accuracy, saves model.npz
python test.py                # shows random test images with the network's guesses
python test.py --wrong        # shows only the ones it got wrong
python export_web.py          # updates the web page with the newly trained model
```

## Files

| File | What it does |
|---|---|
| `download_mnist.py` | Downloads the four MNIST `.gz` files into `data/`. |
| `mnist.py` | Reads those files into NumPy arrays and splits them into train / dev / test. |
| `network.py` | The neural network: initialization, forward pass, loss, backpropagation, training. |
| `train.py` | Command line script: trains a network, reports test accuracy, saves it to `model.npz`. |
| `test.py` | Command line script: loads `model.npz` and shows test images with its guesses. |
| `export_web.py` | Packs `model.npz` into `docs/model.js` for the web page. |
| `docs/` | The web page: `index.html`, `style.css`, `app.js` and the generated `model.js`. |

## The web page

`docs/` is an interactive page made for people who have never heard of a neural
network. Visitors draw a digit and see:

1. **The guess**, live while drawing, with how sure the network is about each digit.
2. **What the computer sees**: the drawing cut out, shrunk to 28 × 28 and turned into 784 numbers.
3. **Inside the network**: all 512 hidden neurons lighting up, the 10 output scores, and
   the neurons that voted hardest for the answer, with the pattern each one looks for.

Training stays in Python. The page only runs the finished network: `app.js` repeats
`network.forward()` in JavaScript (about 20 lines) on the weights exported to
`model.js`. To try it, open `docs/index.html` in a browser; it needs no server.

Two details make it work on real drawings:

- **The drawing is prepared exactly like the MNIST images**: cropped to the ink,
  scaled so its longer side is 20 pixels, then centered by its center of mass in a
  28 × 28 grid (`preprocess()` in `app.js`). Without this, a digit drawn small or in
  a corner looks nothing like what the network learned from.
- **W1 is stored as 8-bit integers** with one scale per neuron, which makes
  `model.js` 4x smaller (0.7 MB) without changing the test accuracy.

## How it works

### The data

MNIST is 70,000 grayscale 28x28 images of handwritten digits: 60,000 for
training and 10,000 for testing. Each image is flattened into a column of 784
numbers between 0 (black) and 1 (white). The data is split three ways:

- **train** (55,000 images): what the network learns from.
- **dev** (the last 5,000 training images): never trained on. Checked after
  every epoch to watch progress and to compare settings.
- **test** (the official 10,000): used only once, for the final score. If you
  tuned settings on it, the score would no longer be honest.

### The network

```
x (784) -> Linear -> ReLU -> Dropout -> Linear -> Softmax -> p (10)
```

Images are stored as **columns**: a batch of `m` images is a `(784, m)` matrix,
so each layer is a single matrix product for the whole batch.

```
Z1 = W1 @ X + b1        (hidden, m)   W1 is (hidden, 784), b1 is (hidden, 1)
A1 = max(Z1, 0)         (hidden, m)   ReLU
Z2 = W2 @ A1 + b2       (10, m)       W2 is (10, hidden),  b2 is (10, 1)
P  = softmax(Z2)        (10, m)       each column: 10 probabilities that sum to 1
```

The guess is the digit with the highest probability.

### The loss

**Cross-entropy**: the average of `-log(probability given to the correct digit)`.
It is 0 when the network is 100% sure of the right answer and grows quickly as
that probability drops.

### Backpropagation

To improve, the network needs the gradient of the loss with respect to every
weight: which direction to nudge each one so the loss goes down. Backprop gets
them by applying the chain rule backwards through the forward pass:

```
dZ2 = (P - one_hot(Y)) / m      softmax + cross-entropy combined: just "probabilities minus answers"
dW2 = dZ2 @ A1.T                db2 = sum of dZ2 over the batch
dA1 = W2.T @ dZ2
dZ1 = dA1 * (Z1 > 0)            ReLU only lets gradient through where its input was positive
dW1 = dZ1 @ X.T                 db1 = sum of dZ1 over the batch
```

### Training

Repeat for each epoch (one full pass over the training set, in a fresh random
order): take a batch of 64 images, run forward, run backward, nudge every
weight against its gradient. Then print the dev accuracy.

## From 97% to 99%: what each improvement does

The original version (small random weights, plain SGD, 128 hidden units, 10
epochs) scored 96.9% on the test set. Each row below is one commit, so
`git log` and `git show` let you see exactly what changed.

| Step | Test accuracy | Training time |
|---|---|---|
| Original algorithm | 96.94% | 7s |
| **float32** instead of float64: half the memory, about 2x faster math, same accuracy | 97.27% | 3s |
| **He initialization**: start weights at a scale suited to ReLU (`std = sqrt(2 / inputs)`) | 97.51% | 4s |
| **Momentum + cosine learning-rate schedule**: step along a running average of gradients, and shrink the step size smoothly to 0 by the end | 98.11% | 6s |
| **Dropout + bigger network**: randomly switch off 20% of hidden units while training so the network can't memorize; 512 hidden units, 20 epochs | 98.42% | 41s |
| **Data augmentation**: shift each training batch by up to 2 pixels so position stops mattering; 30 epochs | **99.01%** | 58s |

Every number is a single run with seed 0. The final version scores 99.01%,
98.96% and 98.94% with seeds 0, 1 and 2, and earlier rows vary by similar
amounts, so differences of about 0.1% or less are noise.

About 99% is close to the limit for this kind of network. Going clearly beyond
it takes a convolutional network, which looks at small patches of the image
instead of all 784 pixels at once. That would be a different project.

## Options

`python train.py --help` and `python test.py --help` list every option.

| `train.py` option | Default | Meaning |
|---|---|---|
| `--hidden-size` | 512 | hidden units |
| `--epochs` | 30 | passes over the training set |
| `--lr` | 0.05 | starting learning rate (decays to 0) |
| `--momentum` | 0.9 | momentum, 0 = plain SGD |
| `--dropout` | 0.2 | fraction of hidden units dropped while training |
| `--no-augment` | off | don't shift the training images |
| `--batch-size` | 64 | images per gradient step |
| `--seed` | 0 | random seed; the same seed gives the same result |
| `--out` | `model.npz` | where to save the model |

| `test.py` option | Default | Meaning |
|---|---|---|
| `--model` | `model.npz` | model to load |
| `--num` | 25 | how many images to show |
| `--wrong` | off | only show misclassified images |

For a quick run while experimenting: `python train.py --epochs 5 --no-augment`
(about 10 seconds, ~98%).
