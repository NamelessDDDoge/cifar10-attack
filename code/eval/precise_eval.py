"""
Precise competition-accurate evaluation.

Score_M = 100 * Score_ASR * Score_SSIM

  Score_ASR  = |{x' : F(x') != y}| / n    (true labels y from true_labels.txt)
  Score_SSIM = mean SSIM(x, x') over ALL n images

Usage:
    python precise_eval.py --adv-dir <dir> [--models surrogate|holdout|all] [--json-out <path>]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim

_WORKSPACE = Path(__file__).resolve().parents[2]
IMAGES_DIR = _WORKSPACE / "data" / "images"
TRUE_LABELS_FILE = _WORKSPACE / "data" / "label.txt"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Attack surrogates (from config.py SURROGATE_NAMES)
SURROGATE_MODELS = [
    "pyramidnet164_a270_bn_cifar10",
    "wrn16_10_cifar10",
    "densenet190_k40_bc_cifar10",
    "seresnet110_cifar10",
    "resnext29_16x64d_cifar10",
    "diaresnet56_cifar10",
]

# Full holdout pool — TOP20 minus surrogates (14 models, unbiased eval)
HOLDOUT_MODELS = [
    "pyramidnet236_a220_bn_cifar10",
    "pyramidnet110_a270_cifar10",
    "pyramidnet272_a200_bn_cifar10",
    "wrn20_10_1bit_cifar10",
    "wrn28_10_cifar10",
    "wrn20_10_32bit_cifar10",
    "wrn40_8_cifar10",
    "densenet250_k24_bc_cifar10",
    "pyramidnet110_a84_cifar10",
    "resnet272bn_cifar10",
    "pyramidnet200_a240_bn_cifar10",
    "resnext272_1x64d_cifar10",
    "seresnet164bn_cifar10",
    "preresnet1001_cifar10",
]


def load_true_labels():
    """Parse true_labels.txt: name true_label pseudo_label match idx"""
    labels = {}
    with open(TRUE_LABELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            labels[parts[0]] = int(parts[1])
    return labels


def load_image_pair(clean_path, adv_path):
    """Returns (arr_clean, arr_adv, inp_adv) where arrs are [H,W,3] float32 [0,1]."""
    def _load(p):
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    c = _load(clean_path)
    a = _load(adv_path)
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1, 1, 3)
    # normalized tensor for model
    t = torch.from_numpy(((a - mean) / std).transpose(2, 0, 1)).unsqueeze(0)
    return c, a, t


def load_model(name, device):
    from pytorchcv.model_provider import get_model
    print(f"  Loading {name} ...", end=" ", flush=True)
    m = get_model(name, pretrained=True).to(device).eval()
    print("ok")
    return m


@torch.no_grad()
def predict_batch(model, tensors, device, bs=128):
    preds = []
    for i in range(0, len(tensors), bs):
        batch = torch.cat(tensors[i:i + bs], dim=0).to(device)
        preds.extend(model(batch).argmax(1).cpu().tolist())
    return preds


def eval_pool(pool_names, true_labels, clean_arrs, adv_arrs, adv_tensors, names, device):
    """
    Returns per-model and pool metrics.
    Score_SSIM is mean over ALL n images (not just successful).
    """
    n = len(names)
    true = [true_labels[name] for name in names]

    # SSIM over ALL images — independent of model
    ssim_vals = [
        float(ssim(clean_arrs[name], adv_arrs[name], channel_axis=2, data_range=1.0))
        for name in names
    ]
    mean_ssim = float(np.mean(ssim_vals))

    per_model = {}
    pool_asr_sum = 0.0
    for mname in pool_names:
        model = load_model(mname, device)
        preds = predict_batch(model, adv_tensors, device)
        del model

        n_attacked = sum(p != t for p, t in zip(preds, true))
        asr = n_attacked / n
        pool_asr_sum += asr
        per_model[mname] = {
            "asr": round(asr, 6),
            "n_attacked": n_attacked,
            "n_total": n,
        }

    pool_asr = pool_asr_sum / len(pool_names) if pool_names else 0.0
    score_m = 100.0 * pool_asr * mean_ssim

    return {
        "models": per_model,
        "pool_asr": round(pool_asr, 6),
        "pool_ssim_all": round(mean_ssim, 6),
        "score_m": round(score_m, 6),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-dir", required=True)
    parser.add_argument("--models", choices=["surrogate", "holdout", "all"], default="all")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    adv_dir = Path(args.adv_dir)
    print(f"[eval] adv-dir={adv_dir}  models={args.models}  device={device}")

    true_labels = load_true_labels()
    names = sorted(true_labels.keys(), key=lambda s: int(Path(s).stem))
    print(f"[eval] Loaded {len(true_labels)} true labels from {TRUE_LABELS_FILE}")

    print("[eval] Loading image pairs ...")
    clean_arrs, adv_arrs, adv_tensors = {}, {}, []
    for name in names:
        ca, aa, at = load_image_pair(IMAGES_DIR / name, adv_dir / name)
        clean_arrs[name] = ca
        adv_arrs[name] = aa
        adv_tensors.append(at)
    print(f"[eval] Loaded {len(names)} pairs.")

    results = {"adv_dir": str(adv_dir)}

    if args.models in ("surrogate", "all"):
        print("\n[eval] SURROGATE pool ...")
        results["surrogate"] = eval_pool(
            SURROGATE_MODELS, true_labels, clean_arrs, adv_arrs, adv_tensors, names, device
        )

    if args.models in ("holdout", "all"):
        print("\n[eval] HOLDOUT pool ...")
        results["holdout"] = eval_pool(
            HOLDOUT_MODELS, true_labels, clean_arrs, adv_arrs, adv_tensors, names, device
        )

    print("\n" + "=" * 60)
    for pool in ("surrogate", "holdout"):
        if pool not in results:
            continue
        r = results[pool]
        print(f"[{pool.upper():9}]  ASR={r['pool_asr']:.4f}  SSIM(all)={r['pool_ssim_all']:.4f}  Score_M={r['score_m']:.4f}")
        for mname, mr in r["models"].items():
            print(f"  {mname}: asr={mr['asr']:.4f}")
    print("=" * 60)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] Written to {args.json_out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
