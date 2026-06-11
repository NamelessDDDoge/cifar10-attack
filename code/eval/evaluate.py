# -*- coding: utf-8 -*-
"""
CIFAR-10 Adversarial Attack Evaluation — official-metric aligned.

Competition scoring: Score_M = 100 * Score_ASR * Score_SSIM
  Score_ASR  = |{x' : F(x') != y}| / n   (y = original/true label)
  Score_SSIM = mean SSIM(x, x') over ALL n submitted images

Local protocol:
  - Labels: workspace/data/label.txt (TRUE labels, pixel-matched to CIFAR-10 test set).
  - Two disjoint model pools, both from the accuracy-screened TOP20 (models.py):
      surrogate (6)  = white-box models the attack used  -> sanity only
      holdout  (14)  = never seen by attack              -> proxy for hidden judges
  - Pool ASR = mean of per-model ASRs in the pool.
  - SSIM is model-independent: mean over all 500 (image, adv) pairs.
  - Decision metric = holdout score. Surrogate score only shows white-box headroom.

Usage:
    python evaluate.py --adv-dir <dir> [--models surrogate|holdout|all] [--json-out <path>]
    python evaluate.py --adv-dir <dir> --models vit --limit 100  # pilot subset
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from models import SURROGATE_MODELS, HOLDOUT_MODELS
from external_models import (
    EXTERNAL_MODEL_SPECS, EXTERNAL_POOLS,
    load_external_model, resolve_external_names,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGES_DIR, LABEL_FILE, CIFAR10_MEAN, CIFAR10_STD

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

N_IMAGES = 500


def load_labels():
    """Read true labels from data/label.txt: `<name>.png <class_int>` per line."""
    labels = {}
    with open(LABEL_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n, c = line.split()[:2]
            labels[n] = int(c)
    if len(labels) != N_IMAGES:
        raise RuntimeError(f"label.txt has {len(labels)} entries, expected {N_IMAGES}")
    return labels


class Evaluator:
    """Evaluate adversarial images on disjoint surrogate/holdout pools."""

    def __init__(self, pools="all", device=None, batch_size=16, limit=N_IMAGES):
        valid = ("surrogate", "holdout", "all", "external", "vit", "robust", "chenyaofo")
        if pools not in valid:
            raise ValueError(f"pools must be {'|'.join(valid)}, got '{pools}'")
        if not 1 <= limit <= N_IMAGES:
            raise ValueError(f"limit must be in [1, {N_IMAGES}], got {limit}")
        self.pools = pools
        self.device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
        self.batch_size = batch_size
        self.limit = limit

    def evaluate(self, adv_dir, json_out=None):
        """Run full evaluation. Returns results dict."""
        adv_dir = Path(adv_dir)
        labels = load_labels()
        clean_items = self._load_clean_images()
        adv_items, n_missing = self._load_adv_images(adv_dir, clean_items)

        name_list = [n for n, _, _ in adv_items]
        true_labels = [labels[n] for n in name_list]

        # ── SSIM over ALL images (official: model-independent) ──
        clean_arrs = {n: a for n, a, _ in clean_items}
        adv_arrs = {n: a for n, a, _ in adv_items}
        ssim_vals = [
            float(ssim(clean_arrs[n], adv_arrs[n], channel_axis=2, data_range=1.0))
            for n in name_list
        ]
        mean_ssim = float(np.mean(ssim_vals))

        print(f"\n{'='*60}")
        print(f"Evaluating pools='{self.pools}' on {len(adv_items)} images")
        if self.limit != N_IMAGES:
            print(f"Subset limit: first {self.limit}/{N_IMAGES} images")
        print(f"Labels: {LABEL_FILE} (true labels)")
        print(f"Mean SSIM (all {len(ssim_vals)} images): {mean_ssim:.4f}")
        if n_missing:
            print(f"WARNING: {n_missing} adv images missing, clean fallback used")
        print(f"Device: {self.device}")
        print(f"{'='*60}")

        result = {
            "adv_dir": str(adv_dir),
            "label_file": str(LABEL_FILE),
            "limit": self.limit,
            "n_total": len(name_list),
            "n_missing_adv": n_missing,
            "mean_ssim": mean_ssim,
            "surrogate_models": SURROGATE_MODELS,
            "holdout_models": HOLDOUT_MODELS,
            "external_models": [s.name for s in EXTERNAL_MODEL_SPECS],
            "external_pools": EXTERNAL_POOLS,
        }

        pool_defs = []
        if self.pools in ("surrogate", "all"):
            pool_defs.append(("surrogate", SURROGATE_MODELS))
        if self.pools in ("holdout", "all"):
            pool_defs.append(("holdout", HOLDOUT_MODELS))
        if self.pools in ("external", "vit", "robust", "chenyaofo"):
            pool_defs.append((self.pools, resolve_external_names(self.pools)))

        for pool_name, model_names in pool_defs:
            if pool_name in ("surrogate", "holdout"):
                result[pool_name] = self._eval_pool(
                    pool_name, model_names, adv_items, true_labels, mean_ssim)
            else:
                result[pool_name] = self._eval_external_pool(
                    pool_name, model_names, adv_items, true_labels, mean_ssim)

        # Summary
        print(f"\n{'='*60}")
        print("SCORING (Score = 100 x pool_ASR x mean_SSIM_all)")
        for pool_name, _ in pool_defs:
            r = result[pool_name]
            tag = "DECISION METRIC" if pool_name == "holdout" else "white-box sanity"
            if pool_name not in ("surrogate", "holdout"):
                tag = "external proxy"
            print(f"  [{pool_name:9s}] ASR={r['pool_asr']:.4f}  "
                  f"SSIM={mean_ssim:.4f}  Score={r['pool_score']:.4f}  ({tag})")
        print(f"{'='*60}")

        if json_out:
            json_out = Path(json_out)
            json_out.parent.mkdir(parents=True, exist_ok=True)
            with open(json_out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults written to {json_out}")

        return result

    def _eval_pool(self, pool_name, model_names, adv_items, true_labels, mean_ssim):
        """Evaluate one model pool: per-model ASR, pool ASR (mean), score."""
        per_model = {}
        all_preds = []

        print(f"\n-- pool '{pool_name}' ({len(model_names)} models) --")
        for mname in model_names:
            print(f"  {mname} ...", end=" ", flush=True)
            from pytorchcv.model_provider import get_model
            model = get_model(mname, pretrained=True).to(self.device).eval()
            preds = self._predict_batch(model, [inp for _, _, inp in adv_items],
                                        batch_size=self.batch_size)
            all_preds.append(preds)
            del model
            if self.device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()

            n_attacked = sum(p != t for p, t in zip(preds, true_labels))
            asr = n_attacked / len(true_labels)
            per_model[mname] = {
                "asr": asr,
                "n_attacked": n_attacked,
                "n_total": len(true_labels),
            }
            print(f"ASR={asr:.4f}", flush=True)

        all_preds_arr = np.array(all_preds)            # [n_models, n_images]
        true_arr = np.array(true_labels)
        misclassified = (all_preds_arr != true_arr[None, :])
        n_universal = int(misclassified.all(axis=0).sum())

        pool_asr = float(np.mean([per_model[m]["asr"] for m in model_names]))
        pool_score = 100.0 * pool_asr * mean_ssim

        return {
            "models": list(model_names),
            "per_model": per_model,
            "pool_asr": pool_asr,
            "pool_ssim": mean_ssim,
            "pool_score": pool_score,
            "n_universal_success": n_universal,
        }

    def _eval_external_pool(self, pool_name, model_names, adv_items, true_labels, mean_ssim):
        """Evaluate external pools. External wrappers expect raw [0,1] tensors."""
        per_model = {}
        all_preds = []
        raw_inputs = [
            torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            for _, arr, _ in adv_items
        ]

        print(f"\n-- external pool '{pool_name}' ({len(model_names)} models) --")
        for mname in model_names:
            print(f"  {mname} ...", end=" ", flush=True)
            model = load_external_model(mname, device=self.device)
            preds = self._predict_batch(model, raw_inputs, batch_size=self.batch_size)
            all_preds.append(preds)
            del model
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            n_attacked = sum(p != t for p, t in zip(preds, true_labels))
            asr = n_attacked / len(true_labels)
            per_model[mname] = {
                "asr": asr,
                "n_attacked": n_attacked,
                "n_total": len(true_labels),
            }
            print(f"ASR={asr:.4f}", flush=True)

        all_preds_arr = np.array(all_preds)
        true_arr = np.array(true_labels)
        misclassified = (all_preds_arr != true_arr[None, :])
        n_universal = int(misclassified.all(axis=0).sum())

        pool_asr = float(np.mean([per_model[m]["asr"] for m in model_names]))
        pool_score = 100.0 * pool_asr * mean_ssim

        return {
            "models": list(model_names),
            "per_model": per_model,
            "pool_asr": pool_asr,
            "pool_ssim": mean_ssim,
            "pool_score": pool_score,
            "n_universal_success": n_universal,
        }

    def _load_clean_images(self):
        mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
        items = []
        for i in range(self.limit):
            name = f"{i}.png"
            arr = np.array(Image.open(IMAGES_DIR / name).convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr.transpose(2, 0, 1))
            inp = ((t - mean) / std).unsqueeze(0).to(self.device)
            items.append((name, arr, inp))
        return items

    def _load_adv_images(self, adv_dir, clean_items):
        mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
        adv_dir = Path(adv_dir)
        items = []
        n_missing = 0
        for name, clean_arr, _ in clean_items:
            p = adv_dir / name
            if p.exists():
                arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            else:
                arr = clean_arr
                n_missing += 1
            t = torch.from_numpy(arr.transpose(2, 0, 1))
            inp = ((t - mean) / std).unsqueeze(0).to(self.device)
            items.append((name, arr, inp))
        return items, n_missing

    @torch.no_grad()
    def _predict_batch(self, model, inputs, batch_size=64):
        preds = []
        for i in range(0, len(inputs), batch_size):
            batch = torch.cat(inputs[i:i+batch_size], dim=0)
            preds.extend(model(batch).argmax(dim=1).cpu().tolist())
        return preds


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Adversarial Evaluation (official-aligned)")
    parser.add_argument("--adv-dir", required=True, help="Directory of adversarial PNGs")
    parser.add_argument("--models",
                        choices=["surrogate", "holdout", "all", "external", "vit", "robust", "chenyaofo"],
                        default="all",
                        help="Which pool(s) to evaluate (external pools are stronger hidden-model proxies)")
    parser.add_argument("--json-out", default=None, help="Write results JSON")
    parser.add_argument("--device", default=None, help="torch device")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size (lower if CUDA OOM)")
    parser.add_argument("--limit", type=int, default=N_IMAGES,
                        help="Evaluate only the first N images; default keeps full official-style 500-image evaluation")
    args = parser.parse_args()

    evaluator = Evaluator(pools=args.models, device=args.device, batch_size=args.batch_size, limit=args.limit)
    evaluator.evaluate(args.adv_dir, args.json_out)


if __name__ == "__main__":
    main()
