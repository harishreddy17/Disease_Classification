import hashlib
import os

DATA_DIR = "dataset/train"
HASH_FILE = "dataset/data_version.txt"

def hash_dir(directory):
    sha = hashlib.sha256()
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
    return sha.hexdigest()

current_hash = hash_dir(DATA_DIR)

try:
    with open(HASH_FILE, "r") as f:
        saved_hash = f.read().strip()
except FileNotFoundError:
    saved_hash = ""

if current_hash != saved_hash:
    print("Dataset changed. Needs retraining.")
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)
    exit(1)  # Exit code 1 signals dataset has changed
else:
    print("Dataset unchanged.")
    exit(0)
