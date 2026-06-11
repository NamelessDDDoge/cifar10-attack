# -*- coding: utf-8 -*-
"""
Unified label provider for CIFAR-10 competition images.
Two modes:
  - "pseudo": majority vote across top-N models (default all 20)
  - "true":   exact pixel match against CIFAR-10 test set

Usage:
    provider = LabelProvider(mode="true")
    labels = provider.get_labels()  # {"0.png": 3, "1.png": 8, ...}
"""
import os
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Project paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = _PROJECT_ROOT / "data" / "images"
CACHE_DIR  = _PROJECT_ROOT / "cache"
TRUE_LABELS_FILE = CACHE_DIR / "true_labels.txt"
# CIFAR-10 test batch — try workspace cache first, then common locations
_CANDIDATES = [
    CACHE_DIR / "cifar10_data" / "cifar-10-batches-py" / "test_batch",
    Path(r"C:\Users\admin\Desktop\AISafety\_vera\work\cifar10_data\cifar-10-batches-py\test_batch"),
    Path.home() / ".cache" / "torch" / "datasets" / "cifar-10-batches-py" / "test_batch",
]
CIFAR10_TEST_BATCH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

try:
    from .models import TOP20_MODELS
except ImportError:
    from models import TOP20_MODELS


class LabelProvider:
    """Provides labels for competition images. Two modes: pseudo (voting) or true (CIFAR-10 match)."""

    def __init__(self, mode: str = "true", models: list = None, device: str = None):
        if mode not in ("pseudo", "true"):
            raise ValueError(f"mode must be 'pseudo' or 'true', got '{mode}'")
        self.mode = mode
        self.models = models or TOP20_MODELS
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._labels = None

    def get_labels(self, force_refresh: bool = False) -> dict:
        """Returns {filename: class_int} dict. Cached after first call."""
        if self._labels is not None and not force_refresh:
            return self._labels
        if self.mode == "true":
            self._labels = self._load_true_labels()
        else:
            self._labels = self._generate_pseudo_labels()
        return self._labels

    def label_array(self) -> np.ndarray:
        """Return labels as numpy array in image order (0.png..499.png)."""
        labels = self.get_labels()
        return np.array([labels[f"{i}.png"] for i in range(500)], dtype=np.int32)

    # ── True labels ──

    def _load_true_labels(self) -> dict:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if TRUE_LABELS_FILE.exists():
            print(f"[LabelProvider] Loading cached true labels from {TRUE_LABELS_FILE}", flush=True)
            return self._read_label_file(TRUE_LABELS_FILE)
        print("[LabelProvider] Matching against CIFAR-10 test set...", flush=True)
        labels = self._match_cifar10_test()
        self._write_label_file(TRUE_LABELS_FILE, labels)
        return labels

    def _match_cifar10_test(self) -> dict:
        if not CIFAR10_TEST_BATCH.exists():
            raise FileNotFoundError(f"CIFAR-10 test_batch not found at {CIFAR10_TEST_BATCH}")
        with open(CIFAR10_TEST_BATCH, "rb") as fb:
            d = pickle.load(fb, encoding="bytes")
        cifar_data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
        cifar_targets = d[b"labels"]
        lookup = {}
        for i, arr in enumerate(cifar_data):
            lookup[arr.tobytes()] = cifar_targets[i]

        labels = {}
        not_found = []
        for idx in range(500):
            name = f"{idx}.png"
            img_path = IMAGES_DIR / name
            arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            key = arr.tobytes()
            if key in lookup:
                labels[name] = int(lookup[key])
            else:
                not_found.append(name)
        if not_found:
            raise RuntimeError(f"{len(not_found)} images not in CIFAR-10 test set: {not_found[:10]}")
        print(f"[LabelProvider] True labels matched: {len(labels)}/500", flush=True)
        return labels

    # ── Pseudo labels ──

    def _generate_pseudo_labels(self) -> dict:
        print(f"[LabelProvider] Majority vote over {len(self.models)} models...", flush=True)
        from pytorchcv.model_provider import get_model

        names, inputs = self._load_images()
        vote_matrix = np.zeros((500, 10), dtype=np.int32)

        for mname in self.models:
            print(f"  {mname} ...", end=" ", flush=True)
            model = get_model(mname, pretrained=True).to(self.device).eval()
            preds = self._predict_batch(model, inputs)
            for i, p in enumerate(preds):
                vote_matrix[i, p] += 1
            del model
            print("ok", flush=True)

        majority = vote_matrix.argmax(axis=1)
        labels = {names[i]: int(majority[i]) for i in range(500)}

        # Check accuracy vs true if available
        if TRUE_LABELS_FILE.exists():
            true_labels = self._read_label_file(TRUE_LABELS_FILE)
            correct = sum(labels[n] == true_labels[n] for n in labels)
            print(f"[LabelProvider] Pseudo vs true accuracy: {correct}/500 = {correct/500:.4f}", flush=True)
        return labels

    def _load_images(self):
        mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
        names, inputs = [], []
        for i in range(500):
            name = f"{i}.png"
            arr = np.array(Image.open(IMAGES_DIR / name).convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr.transpose(2, 0, 1))
            inputs.append(((t - mean) / std).unsqueeze(0).to(self.device))
            names.append(name)
        return names, inputs

    @torch.no_grad()
    def _predict_batch(self, model, inputs, batch_size=64):
        preds = []
        for i in range(0, len(inputs), batch_size):
            batch = torch.cat(inputs[i:i+batch_size], dim=0)
            preds.extend(model(batch).argmax(dim=1).cpu().tolist())
        return preds

    # ── I/O ──

    @staticmethod
    def _read_label_file(path) -> dict:
        labels = {}
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                labels[parts[0]] = int(parts[1])
        return labels

    @staticmethod
    def _write_label_file(path, labels):
        with open(path, "w") as f:
            f.write("# name class_int\n")
            for i in range(500):
                name = f"{i}.png"
                f.write(f"{name} {labels[name]}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pseudo", "true"], default="true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    provider = LabelProvider(mode=args.mode, device=args.device)
    labels = provider.get_labels()
    print(f"Generated {len(labels)} labels in '{args.mode}' mode")
    from collections import Counter
    dist = Counter(labels.values())
    for cls_id in range(10):
        print(f"  {CIFAR10_CLASSES[cls_id]}: {dist.get(cls_id, 0)}")
