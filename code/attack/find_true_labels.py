"""
Match competition images against CIFAR-10 test set to recover true labels.
Uses exact pixel-level comparison (numpy array equality).
"""
import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

IMAGES_DIR = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\datasets\images")
LABEL_FILE = Path(r"C:\Users\admin\Desktop\AISafety\_vera\work\eval\label.txt")
OUT_FILE = Path(r"C:\Users\admin\Desktop\AISafety\_vera\work\true_labels.txt")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def load_cifar10_test():
    """Load CIFAR-10 test batch from local binary file."""
    candidates = [
        Path(r"C:\Users\admin\Desktop\AISafety\_vera\work\cifar10_data\cifar-10-batches-py\test_batch"),
        Path(r"C:\Users\admin\Desktop\AISafety\_vera\work\data\cifar-10-batches-py\test_batch"),
    ]
    batch_path = None
    for p in candidates:
        if p.exists():
            batch_path = p
            break
    if batch_path is None:
        raise FileNotFoundError(f"CIFAR-10 test_batch not found. Tried: {candidates}")
    print(f"Loading CIFAR-10 test set from {batch_path} ...")
    with open(batch_path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # d[b'data']: [10000, 3072] uint8, row-major CHW
    data_flat = d[b"data"]  # [10000, 3072]
    targets = d[b"labels"]  # list of 10000
    # reshape to [10000, 3, 32, 32] then transpose to [10000, 32, 32, 3]
    data = data_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    return data, targets

def load_competition_images():
    """Load all 500 competition images as uint8 numpy arrays."""
    imgs = {}
    for i in range(500):
        p = IMAGES_DIR / f"{i}.png"
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        imgs[f"{i}.png"] = arr
    return imgs

def load_pseudo_labels():
    labels = {}
    with open(LABEL_FILE) as f:
        for line in f:
            name, cls = line.strip().split()
            labels[name] = int(cls)
    return labels

def main():
    cifar_data, cifar_targets = load_cifar10_test()
    comp_imgs = load_competition_images()
    pseudo_labels = load_pseudo_labels()

    # Build index: bytes -> (cifar_idx, label) for fast lookup
    print("Building CIFAR-10 lookup index...")
    cifar_index = {}
    for i, arr in enumerate(cifar_data):
        key = arr.tobytes()
        cifar_index[key] = (i, cifar_targets[i])

    print(f"CIFAR-10 test set: {len(cifar_data)} images indexed")
    print(f"Competition images: {len(comp_imgs)}")
    print()

    results = []
    not_found = []
    mismatch = []

    for i in range(500):
        name = f"{i}.png"
        arr = comp_imgs[name]
        key = arr.tobytes()
        pseudo = pseudo_labels.get(name, -1)

        if key in cifar_index:
            cifar_idx, true_label = cifar_index[key]
            match = (pseudo == true_label)
            results.append((name, true_label, pseudo, cifar_idx, match))
            if not match:
                mismatch.append((name, true_label, pseudo))
        else:
            not_found.append(name)
            results.append((name, -1, pseudo, -1, None))

    # Print summary
    found = [r for r in results if r[3] >= 0]
    print(f"Matched: {len(found)}/500")
    print(f"Not found in CIFAR-10 test set: {len(not_found)}")
    if not_found:
        print(f"  First 10 not found: {not_found[:10]}")

    if found:
        match_rate = sum(1 for r in found if r[4]) / len(found)
        print(f"Pseudo-label accuracy (vs true): {match_rate:.4f} ({sum(1 for r in found if r[4])}/{len(found)})")

    if mismatch:
        print(f"\nMismatches ({len(mismatch)}):")
        for name, true_l, pseudo_l in mismatch[:20]:
            print(f"  {name}: true={CIFAR10_CLASSES[true_l]}({true_l})  pseudo={CIFAR10_CLASSES[pseudo_l]}({pseudo_l})")

    # Write output: name true_label pseudo_label match cifar_idx
    lines = []
    for name, true_l, pseudo_l, cidx, match in results:
        true_str = str(true_l)
        match_str = str(match) if match is not None else "NOT_FOUND"
        lines.append(f"{name} {true_str} {pseudo_l} {match_str} {cidx}")

    with open(OUT_FILE, "w") as f:
        f.write("# name  true_label  pseudo_label  match  cifar_test_idx\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nFull results written to {OUT_FILE}")

if __name__ == "__main__":
    main()
