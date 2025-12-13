import numpy as np
from math import ceil

"""Simple educational 2-layer neural network (fully connected).

Highlights:
- Input vectors `X` have shape (features, m) where `m` is the batch size or number
  of examples. This column-oriented shape is convenient for vectorized operations.
- The network uses: Linear -> ReLU -> Linear -> Softmax.
- Functions are intentionally clear and annotated with shapes to help learning.
"""

try:
    from tqdm import trange, tqdm
except Exception:
    # fallback no-op progress iterators (if tqdm is not installed)
    def trange(x, **kwargs):
        return range(x)

    def tqdm(it, **kwargs):
        return it


def init_params(input_size=784, hidden_size=128, output_size=10):
    """Initialize weights and biases.

    Returns:
      W1, b1, W2, b2 with shapes:
        W1: (hidden_size, input_size)
        b1: (hidden_size, 1)
        W2: (output_size, hidden_size)
        b2: (output_size, 1)
    """
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2


def ReLU(Z):
    """Rectified Linear Unit activation (element-wise)."""
    return np.maximum(0, Z)


def ReLU_deriv(Z):
    """Derivative of ReLU used in backprop: 1 where Z>0 else 0."""
    return (Z > 0).astype(float)


def softmax(Z):
    """Numerically-stable softmax over columns.

    Z shape: (num_classes, m)
    """
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def one_hot(Y, num_classes=None):
    """Convert integer labels vector (m,) to one-hot matrix (num_classes, m)."""
    Y = np.array(Y).reshape(-1)
    if num_classes is None:
        num_classes = int(Y.max()) + 1
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


def forward_prop(W1, b1, W2, b2, X):
    """Perform forward propagation for inputs X.

    Returns intermediate values (Z1, A1, Z2, A2) needed for backprop.
    """
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def compute_loss(A2, Y):
    """Cross-entropy loss averaged over the batch.

    A2: softmax outputs (num_classes, m)
    Y: integer labels (m,)
    """
    m = Y.size
    Y_oh = one_hot(Y, num_classes=A2.shape[0])
    log_probs = np.log(A2 + 1e-15)
    loss = -1.0 / m * np.sum(Y_oh * log_probs)
    return loss


def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    """Vectorized backpropagation.

    Returns gradients (dW1, db1, dW2, db2) with the same shapes as the params.
    """
    m = X.shape[1]
    Y_oh = one_hot(Y, num_classes=A2.shape[0])
    dZ2 = A2 - Y_oh
    dW2 = (1.0 / m) * dZ2.dot(A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1.0 / m) * dZ1.dot(X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """Simple SGD parameter update.

    For clarity we return new parameter arrays rather than updating in-place.
    """
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W1, b1, W2, b2


def predict(W1, b1, W2, b2, X):
    """Return predicted class indices for inputs X."""
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    preds = np.argmax(A2, axis=0)
    return preds


def predict_with_probs(W1, b1, W2, b2, X):
    """Return (preds, probs) where `probs` is the softmax output (num_classes, m)."""
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    preds = np.argmax(A2, axis=0)
    return preds, A2


def accuracy(preds, Y):
    preds = np.array(preds).reshape(-1)
    Y = np.array(Y).reshape(-1)
    return np.mean(preds == Y)


def top_confidences(probs):
    """Given softmax probs (num_classes, m) return array of top confidence per example."""
    return np.max(probs, axis=0)


def train(X, Y, hidden_size=128, lr=0.1, epochs=10, batch_size=64, print_every=1, use_tqdm=True):
    """Train a simple 2-layer NN using mini-batch SGD with a tqdm progress bar.

    Args:
      X: input matrix (features, m)
      Y: labels (m,)
      hidden_size: number of hidden units
      lr: learning rate for SGD
      epochs: number of passes through the dataset
      batch_size: mini-batch size
      print_every: how often (in epochs) to print training status
      use_tqdm: show progress bars when True

    Returns:
      Trained parameters (W1, b1, W2, b2)
    """
    input_size = X.shape[0]
    m = X.shape[1]
    output_size = int(np.max(Y) + 1)
    W1, b1, W2, b2 = init_params(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    steps_per_epoch = max(1, ceil(m / batch_size))
    total_steps = epochs * steps_per_epoch
    step = 0

    epoch_iter = trange(epochs, desc='Epochs') if use_tqdm else range(epochs)
    for e in epoch_iter:
        # shuffle the dataset each epoch for better SGD behavior
        perm = np.random.permutation(m)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[perm]

        for i in range(0, m, batch_size):
            step += 1
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        # end epoch -- optionally print metrics on the full training set
        if (e + 1) % print_every == 0 or e == 0 or e == epochs - 1:
            _, _, _, A2_full = forward_prop(W1, b1, W2, b2, X)
            loss = compute_loss(A2_full, Y)
            preds = np.argmax(A2_full, axis=0)
            acc = accuracy(preds, Y)
            print(f"Epoch {e+1}/{epochs} — loss: {loss:.4f} — acc: {acc:.4f}")

    return W1, b1, W2, b2


def lr_finder(X, Y, hidden_size=128, start_lr=1e-7, end_lr=1, num_iters=100, batch_size=128, beta=0.98, stop_factor=4.0, use_tqdm=True):
    """Run a learning-rate range test (as in Leslie Smith's LR Finder).

    This gradually increases the learning rate and records a smoothed loss curve.
    The returned `best_lr` is the lr at minimum smoothed loss and `suggested_lr`
    is conservative (best_lr / 10).
    """
    import math
    m = X.shape[1]
    W1, b1, W2, b2 = init_params(input_size=X.shape[0], hidden_size=hidden_size, output_size=int(np.max(Y) + 1))

    lrs = []
    losses = []
    avg_loss = 0.0
    best_loss = float('inf')

    def get_lr(i):
        return start_lr * (end_lr / start_lr) ** (i / max(1, num_iters - 1))

    iterator = trange(num_iters, desc='LR finder') if use_tqdm else range(num_iters)
    for i in iterator:
        lr = get_lr(i)
        idx = np.random.choice(m, size=min(batch_size, m), replace=False)
        Xb = X[:, idx]
        Yb = Y[idx]

        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, Xb)
        loss = compute_loss(A2, Yb)

        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed = avg_loss / (1 - beta ** (i + 1))

        lrs.append(lr)
        losses.append(smoothed)
        if smoothed < best_loss:
            best_loss = smoothed
            best_lr = lr

        if i > 0 and smoothed > stop_factor * best_loss:
            break

        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, Xb, Yb)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

    suggested = float(best_lr) / 10.0
    return {'lrs': np.array(lrs), 'losses': np.array(losses), 'best_lr': float(best_lr), 'suggested_lr': suggested}
