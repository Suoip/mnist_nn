"""A 2-layer fully-connected neural network, written from scratch with NumPy.

    input (784 pixels) -> Linear -> ReLU -> Linear -> Softmax -> 10 probabilities

Shape convention: examples are stored as *columns*. A batch of m images is an
array X of shape (784, m) and the network's output for it is (10, m). That way
a whole layer is a single matrix product (W @ X) with no loop over examples.

The parameters live in a dict so the training loop can treat them all the same:
    W1 (hidden, 784)   b1 (hidden, 1)   W2 (10, hidden)   b2 (10, 1)
"""
import time

import numpy as np


def init_params(hidden_size, rng):
    """Small random weights and zero biases.

    The weights must be random: if they all started equal, every hidden unit
    would compute the same thing and get the same update, forever.
    """
    return {
        "W1": rng.standard_normal((hidden_size, 784)) * 0.01,
        "b1": np.zeros((hidden_size, 1)),
        "W2": rng.standard_normal((10, hidden_size)) * 0.01,
        "b2": np.zeros((10, 1)),
    }


def softmax(Z):
    """Turn each column of scores into probabilities that sum to 1.

    Subtracting the column max first does not change the result (it cancels out
    in the division) but stops exp() from overflowing on large scores.
    """
    E = np.exp(Z - Z.max(axis=0, keepdims=True))
    return E / E.sum(axis=0, keepdims=True)


def forward(params, X):
    """Run the network on X (784, m).

    Returns the output probabilities (10, m) and a cache of the intermediate
    values that backward() needs.
    """
    Z1 = params["W1"] @ X + params["b1"]   # (hidden, m)  b1 is broadcast across the m columns
    A1 = np.maximum(Z1, 0)                 # (hidden, m)  ReLU: negative values become 0
    Z2 = params["W2"] @ A1 + params["b2"]  # (10, m)      one score per digit
    probs = softmax(Z2)                    # (10, m)      each column sums to 1
    return probs, (X, Z1, A1)


def cross_entropy(probs, Y):
    """The loss: average of -log(probability given to the correct digit).

    It is 0 when the network is 100% sure of the right answer, and grows
    quickly the less probability the right answer gets.
    """
    correct = probs[Y, np.arange(Y.size)]  # probs[Y[i], i] for every example i
    return float(-np.log(correct + 1e-12).mean())


def backward(params, probs, cache, Y):
    """Backpropagation: the gradient of the loss with respect to every parameter.

    Walks the forward pass in reverse, applying the chain rule one step at a
    time. Every gradient has the same shape as the thing it is the gradient of.
    """
    X, Z1, A1 = cache
    m = Y.size

    # Softmax + cross-entropy together have a famously simple gradient:
    #   dLoss/dZ2 = probs - one_hot(Y)
    # i.e. subtract 1 from the probability of the correct digit. Divide by m
    # because the loss is an average over the batch.
    dZ2 = probs.copy()
    dZ2[Y, np.arange(m)] -= 1
    dZ2 /= m                                # (10, m)

    # Z2 = W2 @ A1 + b2
    dW2 = dZ2 @ A1.T                        # (10, hidden)
    db2 = dZ2.sum(axis=1, keepdims=True)    # (10, 1)  b2 was added to all m columns, so sum over them
    dA1 = params["W2"].T @ dZ2              # (hidden, m)

    # A1 = ReLU(Z1): the gradient only flows back where Z1 was positive
    dZ1 = dA1 * (Z1 > 0)                    # (hidden, m)

    # Z1 = W1 @ X + b1
    dW1 = dZ1 @ X.T                         # (hidden, 784)
    db1 = dZ1.sum(axis=1, keepdims=True)    # (hidden, 1)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def accuracy(params, X, Y):
    """Fraction of the images in X whose most likely digit is the right one."""
    probs, _ = forward(params, X)
    return float((probs.argmax(axis=0) == Y).mean())


def train(X, Y, X_dev, Y_dev, hidden_size=128, epochs=10, lr=0.1, batch_size=64, seed=0):
    """Train with mini-batch stochastic gradient descent (SGD).

    One epoch = one pass over all training images, in a fresh random order,
    taking one small step downhill on the loss after every batch.
    The same seed always gives the same result.
    """
    rng = np.random.default_rng(seed)
    params = init_params(hidden_size, rng)
    m = X.shape[1]

    for epoch in range(epochs):
        start = time.time()
        order = rng.permutation(m)
        losses = []
        for i in range(0, m, batch_size):
            batch = order[i:i + batch_size]
            Xb, Yb = X[:, batch], Y[batch]

            probs, cache = forward(params, Xb)
            grads = backward(params, probs, cache, Yb)
            losses.append(cross_entropy(probs, Yb))

            # SGD: move every parameter a small step against its gradient
            for k in params:
                params[k] -= lr * grads[k]

        print(f"epoch {epoch + 1:2d}/{epochs}  train loss {np.mean(losses):.4f}  "
              f"dev acc {accuracy(params, X_dev, Y_dev) * 100:.2f}%  ({time.time() - start:.1f}s)")
    return params
