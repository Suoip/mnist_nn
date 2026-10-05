"""A 2-layer fully-connected neural network, written from scratch with NumPy.

    input (784 pixels) -> Linear -> ReLU -> Dropout -> Linear -> Softmax -> 10 probabilities

Shape convention: examples are stored as *columns*. A batch of m images is an
array X of shape (784, m) and the network's output for it is (10, m). That way
a whole layer is a single matrix product (W @ X) with no loop over examples.

The parameters live in a dict so the training loop can treat them all the same:
    W1 (hidden, 784)   b1 (hidden, 1)   W2 (10, hidden)   b2 (10, 1)

Everything is float32. Careful when adding code: mixing in a NumPy float64
*scalar* (e.g. the result of np.sqrt(2.0) or np.cos(x)) silently turns the
arrays back into float64. Plain Python floats and the math module are safe.
"""
import math
import time

import numpy as np


def init_params(hidden_size, rng):
    """Random weights (He initialization) and zero biases.

    The weights must be random: if they all started equal, every hidden unit
    would compute the same thing and get the same update, forever.

    Their scale matters too. Too small and the signal shrinks a little at every
    layer (so do the gradients, and learning starts slowly); too large and it
    blows up. He initialization draws each weight with standard deviation
    sqrt(2 / fan_in), fan_in being the number of inputs to the layer. That keeps
    the size of the activations about the same from layer to layer. The 2 is
    there because ReLU zeroes about half of its inputs.
    """
    return {
        "W1": rng.standard_normal((hidden_size, 784), dtype=np.float32) * (2 / 784) ** 0.5,
        "b1": np.zeros((hidden_size, 1), dtype=np.float32),
        "W2": rng.standard_normal((10, hidden_size), dtype=np.float32) * (2 / hidden_size) ** 0.5,
        "b2": np.zeros((10, 1), dtype=np.float32),
    }


def softmax(Z):
    """Turn each column of scores into probabilities that sum to 1.

    Subtracting the column max first does not change the result (it cancels out
    in the division) but stops exp() from overflowing on large scores.
    """
    E = np.exp(Z - Z.max(axis=0, keepdims=True))
    return E / E.sum(axis=0, keepdims=True)


def forward(params, X, drop_rate=0.0, rng=None):
    """Run the network on X (784, m).

    Returns the output probabilities (10, m) and a cache of the intermediate
    values that backward() needs. drop_rate > 0 turns on dropout, which is only
    used during training.
    """
    Z1 = params["W1"] @ X + params["b1"]   # (hidden, m)  b1 is broadcast across the m columns
    A1 = np.maximum(Z1, 0)                 # (hidden, m)  ReLU: negative values become 0

    mask = None
    if drop_rate > 0:
        # Dropout: switch off each hidden unit at random (a different set for
        # every image and every batch). The network can't rely on any single
        # unit, so it learns redundant, more general features and overfits less.
        # The survivors are scaled up by 1 / (1 - drop_rate) so the average size
        # of A1 is unchanged, which means nothing has to change at test time.
        keep = rng.random(A1.shape, dtype=np.float32) >= drop_rate
        mask = keep.astype(np.float32) / (1 - drop_rate)
        A1 = A1 * mask

    Z2 = params["W2"] @ A1 + params["b2"]  # (10, m)      one score per digit
    probs = softmax(Z2)                    # (10, m)      each column sums to 1
    return probs, (X, Z1, A1, mask)


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
    X, Z1, A1, mask = cache
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

    # Dropped units sent nothing forward, so they get no gradient back
    if mask is not None:
        dA1 *= mask

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


def train(X, Y, X_dev, Y_dev, hidden_size=512, epochs=20, lr=0.05, momentum=0.9,
          batch_size=64, drop_rate=0.2, seed=0):
    """Train with mini-batch stochastic gradient descent (SGD) with momentum
    and a cosine learning-rate schedule.

    One epoch = one pass over all training images, in a fresh random order,
    taking one step downhill on the loss after every batch.
    The same seed always gives the same result.
    """
    rng = np.random.default_rng(seed)
    params = init_params(hidden_size, rng)
    # One "velocity" per parameter, same shape, starting at rest
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    m = X.shape[1]
    total_steps = epochs * math.ceil(m / batch_size)
    step = 0

    for epoch in range(epochs):
        start = time.time()
        order = rng.permutation(m)
        losses = []
        for i in range(0, m, batch_size):
            batch = order[i:i + batch_size]
            Xb, Yb = X[:, batch], Y[batch]

            probs, cache = forward(params, Xb, drop_rate, rng)
            grads = backward(params, probs, cache, Yb)
            losses.append(cross_entropy(probs, Yb))

            # Cosine schedule: the learning rate starts at lr and glides down to 0
            # along half a cosine wave. Big steps early to make fast progress,
            # tiny steps at the end to settle into the bottom of the valley.
            step_lr = lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            step += 1

            # Momentum: instead of stepping along this batch's gradient alone,
            # step along a running average of recent gradients (the "velocity").
            # Directions that agree from batch to batch build up speed, while
            # noisy back-and-forth directions cancel out. momentum=0 is plain SGD.
            for k in params:
                velocity[k] = momentum * velocity[k] + grads[k]
                params[k] -= step_lr * velocity[k]

        print(f"epoch {epoch + 1:2d}/{epochs}  train loss {np.mean(losses):.4f}  "
              f"dev acc {accuracy(params, X_dev, Y_dev) * 100:.2f}%  "
              f"lr {step_lr:.4f}  ({time.time() - start:.1f}s)")
    return params
