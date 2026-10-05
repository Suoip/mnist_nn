import argparse
import numpy as np
import random
from mnist import load_data, display_image
import cnn


def main(args):
    X_train, Y_train, X_dev, Y_dev = load_data(data_dir=args.data_dir, dev_size=args.dev_size)

    model = np.load(args.model)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # choose source set
    if args.use_dev:
        X, Y = X_dev, Y_dev
    else:
        X, Y = X_train, Y_train

    # If user requested misclassified examples, compute predictions for the
    # whole chosen set and display the wrong ones (up to `args.limit`). This is
    # useful for debugging model errors and seeing what it confuses.
    if args.show_wrongs:
        preds, probs = cnn.predict_with_probs(W1, b1, W2, b2, X)
        # ensure arrays are 1D for comparison
        preds = np.array(preds).reshape(-1)
        Y_arr = np.array(Y).reshape(-1)
        wrong_idx = np.where(preds != Y_arr)[0]
        if wrong_idx.size == 0:
            print("No misclassified examples found in the chosen set.")
        else:
            print(f"Found {wrong_idx.size} misclassified examples; showing up to {args.limit} of them.")
            show_idx = wrong_idx[:args.limit]
            for idx in show_idx:
                x = X[:, idx:idx+1]
                pred = int(preds[idx])
                label = int(Y_arr[idx])
                conf = float(np.max(probs[:, idx]))
                print(f'Index {idx} — predicted: {pred} (confidence: {conf*100:.2f}%) — actual: {label}')
                x_flat = x.reshape(-1)
                if x_flat.size == 784:
                    display_image(x_flat)
                else:
                    try:
                        side = int(np.sqrt(x_flat.size))
                        if side * side == x_flat.size:
                            display_image(x_flat.reshape(side*side))
                        else:
                            print(f"Cannot display image: unexpected vector size {x_flat.size}")
                    except Exception:
                        print(f"Cannot display image: unexpected vector size {x_flat.size}")

        # summary for the chosen set
        overall_acc = np.mean(preds == Y_arr)
        avg_conf = np.mean(np.max(probs, axis=0))
        print(f"Overall accuracy on chosen set: {overall_acc*100:.2f}% — average confidence: {avg_conf*100:.2f}%")
        return

    # Default behavior: show a small random sample (backwards-compatible)
    num = args.num
    indices = random.sample(range(X.shape[1]), num)

    sampled_preds = []
    sampled_labels = []
    sampled_confs = []

    for idx in indices:
        x = X[:, idx:idx+1]
        pred, probs = cnn.predict_with_probs(W1, b1, W2, b2, x)
        pred = int(pred[0])
        label = int(Y[idx])
        conf = float(np.max(probs))
        print(f'Index {idx} — predicted: {pred} (confidence: {conf*100:.2f}%) — actual: {label}')

        x_flat = x.reshape(-1)
        # If input size is not 784 (e.g., sample_submission fallback), try best-effort display
        if x_flat.size == 784:
            display_image(x_flat)
        else:
            try:
                side = int(np.sqrt(x_flat.size))
                if side * side == x_flat.size:
                    display_image(x_flat.reshape(side*side))
                else:
                    print(f"Cannot display image: unexpected vector size {x_flat.size}")
            except Exception:
                print(f"Cannot display image: unexpected vector size {x_flat.size}")

        sampled_preds.append(pred)
        sampled_labels.append(label)
        sampled_confs.append(conf)

    # summary
    sampled_preds = np.array(sampled_preds)
    sampled_labels = np.array(sampled_labels)
    sampled_confs = np.array(sampled_confs)
    overall_acc = np.mean(sampled_preds == sampled_labels)
    avg_conf = sampled_confs.mean()
    print(f"Sampled accuracy: {overall_acc*100:.2f}% — average confidence: {avg_conf*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--dev-size', type=int, default=1000)
    parser.add_argument('--model', default='model.npz', help='Path to saved model .npz')
    parser.add_argument('--num', type=int, default=5, help='Number of random samples to test')
    parser.add_argument('--use-dev', action='store_true', help='Use dev set instead of train set')
    parser.add_argument('--show-wrongs', action='store_true', help='Show misclassified examples from the chosen set')
    parser.add_argument('--limit', type=int, default=20, help='Max misclassified examples to show when --show-wrongs is used')
    args = parser.parse_args()
    main(args)
