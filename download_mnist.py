import os
import urllib.request
import gzip
import shutil
import argparse
import zipfile

MNIST_FILES = {
    "train-images-idx3-ubyte.gz":
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz":
        "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":
        "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz"
}

def ensure_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")

def download_file(file_name, url):
    path = os.path.join("data", file_name)
    # download if missing
    if not os.path.exists(path):
        print(f"[Downloading] {file_name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[Done] Saved to {path}")
    else:
        print(f"[OK] {file_name} already exists.")

    # handle .gz extraction
    if path.endswith('.gz'):
        out_path = path[:-3]
        if os.path.exists(out_path):
            print(f"[Extracted] {out_path} already exists.")
        else:
            try:
                with gzip.open(path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                print(f"[Extracted] {out_path}")
            except Exception as e:
                print(f"[Warning] could not extract {path}: {e}")

    # handle .zip extraction
    if path.endswith('.zip'):
        try:
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(os.path.dirname(path))
            print(f"[Unzipped] {path}")
        except Exception as e:
            print(f"[Warning] could not unzip {path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download and extract MNIST IDX files into data/')
    parser.add_argument('--delete-gz', action='store_true', help='Delete .gz files after extraction')
    args = parser.parse_args()

    ensure_data_dir()

    for file_name, url in MNIST_FILES.items():
        download_file(file_name, url)

    if args.delete_gz:
        for gz in [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.gz')]:
            try:
                os.remove(gz)
                print(f"[Removed] {gz}")
            except Exception as e:
                print(f"[Warning] could not remove {gz}: {e}")

    print("\nAll MNIST files downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
