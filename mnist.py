
import os
import struct
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _read_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid IDX image file magic {magic} for {path}')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
    return data


def _read_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid IDX label file magic {magic} for {path}')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def _find_data_sources(data_dir='data'):
    # Prefer IDX files (original MNIST). Then look for CSV train files.
    img_idx = os.path.join(data_dir, 'train-images.idx3-ubyte')
    lbl_idx = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    if os.path.exists(img_idx) and os.path.exists(lbl_idx):
        return ('idx', img_idx, lbl_idx)

    # CSV candidates
    for name in ('mnist_train.csv', 'train.csv'):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return ('csv', p)

    # fallback: do not use sample_submission.csv as training data
    sample = os.path.join(data_dir, 'sample_submission.csv')
    if os.path.exists(sample):
        return ('sample_submission', sample)

    return (None,)


def load_data(data_dir='data', dev_size=1000, shuffle=True):
    """Load MNIST training data from IDX files or a CSV.

    Returns X_train, Y_train, X_dev, Y_dev where X has shape (features, m).
    - If IDX files are present (`train-images.idx3-ubyte` and `train-labels.idx1-ubyte`),
      they are loaded (recommended).
    - If a CSV file is present, the loader expects the first column to be the label
      and the rest to be pixel columns (784 columns). If the CSV looks like
      `sample_submission.csv` (only `ImageId,Label`), the loader will raise an error
      asking for proper training data.
    """
    src = _find_data_sources(data_dir)
    if src[0] == 'idx':
        _, img_path, lbl_path = src
        images = _read_idx_images(img_path)  # shape (N, 784)
        labels = _read_idx_labels(lbl_path)  # shape (N,)
        # convert to expected shapes: X (features, m), Y (m,)
        X_all = images.T.astype(float) / 255.0
        Y_all = labels.astype(int)
    elif src[0] == 'csv':
        _, csv_path = src
        df = pd.read_csv(csv_path)
        # If CSV looks like sample_submission (ImageId,Label) but not training, reject
        cols = [c.lower() for c in df.columns]
        if 'imageid' in cols and 'label' in cols and df.shape[1] == 2:
            raise ValueError(f"Found '{os.path.basename(csv_path)}' which looks like a submission file.\n"
                             "Please provide a training CSV (with label column and 784 pixel columns) "
                             "or the original IDX files in the data/ folder.")

        data = df.to_numpy()
        # Heuristic: if there are 785 or more columns, assume first column is label
        if data.shape[1] >= 785:
            labels = data[:, 0].astype(int)
            images = data[:, 1:785].astype(float)
        else:
            raise ValueError(f'CSV at {csv_path} does not appear to contain 784 pixel columns')

        X_all = images.T / 255.0
        Y_all = labels.astype(int)
    elif src[0] == 'sample_submission':
        _, sample = src
        raise ValueError(f"Only found '{os.path.basename(sample)}' in {data_dir}. This is a submission template, not training data.\n"
                         "Please put 'mnist_train.csv' or the IDX files in the data/ folder.")
    else:
        raise FileNotFoundError(f'No suitable MNIST data found in {data_dir}.')

    # shuffle and split into dev/train
    m_total = X_all.shape[1]
    indices = np.arange(m_total)
    if shuffle:
        np.random.shuffle(indices)
    X_all = X_all[:, indices]
    Y_all = Y_all[indices]

    dev_end = min(dev_size, m_total)
    X_dev = X_all[:, :dev_end]
    Y_dev = Y_all[:dev_end]
    X_train = X_all[:, dev_end:]
    Y_train = Y_all[dev_end:]

    return X_train, Y_train, X_dev, Y_dev


def display_image(image_vector, cmap='gray'):
    """Display a single 784-vector image (0-1 floats) as 28x28."""
    img = np.array(image_vector).reshape((28, 28))
    plt.figure()
    plt.imshow(img, cmap=cmap, interpolation='nearest')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print('mnist.py: utility module — call load_data() from your scripts')