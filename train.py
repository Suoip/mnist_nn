import argparse
import time
import numpy as np
from mnist import load_data
import cnn


def main(args):
    print('Loading data...')
    X_train, Y_train, X_dev, Y_dev = load_data(data_dir=args.data_dir, dev_size=args.dev_size)
    print(f'Training on {X_train.shape[1]} examples; validation {X_dev.shape[1]} examples')

    start = time.time()
    W1, b1, W2, b2 = cnn.train(
        X_train,
        Y_train,
        hidden_size=args.hidden_size,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        print_every=args.print_every,
        use_tqdm=not args.no_tqdm,
    )
    duration = time.time() - start
    print(f'Training completed in {duration:.1f}s')

    out_path = args.out or 'model.npz'
    np.savez(out_path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f'Model saved to: {out_path}')
    # Evaluate on dev set and report accuracy + average confidence
    try:
        _, _, X_dev, Y_dev = load_data(data_dir=args.data_dir, dev_size=args.dev_size)
        import cnn as _cnn
        preds, probs = _cnn.predict_with_probs(W1, b1, W2, b2, X_dev)
        acc = _cnn.accuracy(preds, Y_dev)
        avg_conf = float(_cnn.top_confidences(probs).mean())
        print(f'Dev accuracy: {acc*100:.2f}% — average confidence: {avg_conf*100:.2f}%')
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train simple MNIST NN')
    parser.add_argument('--data-dir', default='data', help='Directory containing training CSV')
    parser.add_argument('--dev-size', type=int, default=1000, help='Number of dev examples')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--print-every', type=int, default=1, help='Print metrics every N epochs')
    parser.add_argument('--out', default='model.npz')
    parser.add_argument('--no-tqdm', action='store_true', help='Disable tqdm progress bars')
    parser.add_argument('--lr-finder', action='store_true', help='Run an LR finder range test and exit')
    parser.add_argument('--lr-start', type=float, default=1e-7, help='LR finder start lr')
    parser.add_argument('--lr-end', type=float, default=1.0, help='LR finder end lr')
    parser.add_argument('--lr-iters', type=int, default=100, help='LR finder number of updates')
    parser.add_argument('--lr-batch-size', type=int, default=128, help='LR finder batch size')
    args = parser.parse_args()
    if args.lr_finder:
        print('Running LR finder...')
        X_train, Y_train, X_dev, Y_dev = load_data(data_dir=args.data_dir, dev_size=args.dev_size)
        res = cnn.lr_finder(
            X_train,
            Y_train,
            hidden_size=args.hidden_size,
            start_lr=args.lr_start,
            end_lr=args.lr_end,
            num_iters=args.lr_iters,
            batch_size=args.lr_batch_size,
            use_tqdm=not args.no_tqdm,
        )
        print(f"LR finder best lr: {res['best_lr']:.3e}")
        print(f"Suggested lr (conservative): {res['suggested_lr']:.3e}")
    else:
        main(args)
