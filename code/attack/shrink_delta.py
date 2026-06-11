"""
SSIM-shrinkage post-processor for adversarial images.

For each successfully misclassified image (against surrogate ensemble),
binary-search for the minimum perturbation scale that still causes
misclassification. Reduces delta → increases SSIM → improves Score.

Score = 100 * pool_ASR * mean_SSIM; maximising mean_SSIM for fixed ASR raises score.

Usage:
    python shrink_delta.py --adv-dir results/adv_ilpd --out-dir results/adv_ilpd_shrunk
    python shrink_delta.py --adv-dir results/adv_pgn  --out-dir results/adv_pgn_shrunk
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGES_DIR, LABEL_FILE, CIFAR10_MEAN, CIFAR10_STD, SURROGATE_NAMES, BATCH_SIZE

BISECT_STEPS = 12   # binary search iterations per image
MIN_SCALE = 0.0     # allow zeroing perturbation if model still fools


class CifarEnsemble(nn.Module):
    def __init__(self, models, device):
        super().__init__()
        self.models = nn.ModuleList(models)
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std  = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x):
        xn = (x - self.mean) / self.std
        return sum(m(xn) for m in self.models) / len(self.models)


def build_ensemble(device):
    from pytorchcv.model_provider import get_model
    models = []
    for name in SURROGATE_NAMES:
        print(f"  load {name} ...", end=" ", flush=True)
        m = get_model(name, pretrained=True).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
        print("ok")
    return CifarEnsemble(models, device).to(device)


def load_images(dirpath):
    names = [f"{i}.png" for i in range(500)]
    to_tensor = transforms.ToTensor()
    arr = []
    for name in names:
        img = Image.open(dirpath / name).convert("RGB")
        arr.append(to_tensor(img))
    return names, torch.stack(arr, 0)


def load_labels():
    labels = {}
    with open(LABEL_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nm, c = line.split()
            labels[nm] = int(c)
    return labels


def predict(model, x, device, bs=50):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i+bs].to(device)).argmax(1).cpu())
    return torch.cat(out)


def save_images(images, names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for name, img in zip(names, images):
        to_pil(img.clamp(0, 1)).save(out_dir / name, format="PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-dir",  required=True,  type=Path)
    parser.add_argument("--out-dir",  required=True,  type=Path)
    parser.add_argument("--bisect",   type=int,   default=BISECT_STEPS,
                        help="Binary-search steps (default 12)")
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE,
                        help="Minimum allowed scale (0=allow zero delta)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shrink] device={device}  adv_dir={args.adv_dir}")

    print("[shrink] Loading surrogate ensemble ...")
    model = build_ensemble(device)

    print("[shrink] Loading images ...")
    names, clean = load_images(IMAGES_DIR)
    _, adv   = load_images(args.adv_dir)
    delta = adv - clean           # raw float32 delta in [0,1] space

    label_map = load_labels()
    labels = torch.tensor([label_map[n] for n in names], dtype=torch.long)

    clean_pred = predict(model, clean, device)
    adv_pred   = predict(model, adv,   device)

    initial_success = (adv_pred != labels)
    n_success = initial_success.sum().item()
    print(f"[shrink] Initial surrogate ASR: {n_success}/500 = {n_success/500:.3f}")

    # Per-image binary search
    scales = torch.ones(500)            # scale=1.0 → original delta
    for idx in range(500):
        if not initial_success[idx]:
            # Already fails → keep original (scale=1) for consistency
            continue

        lo, hi = args.min_scale, 1.0
        for _ in range(args.bisect):
            mid = (lo + hi) / 2.0
            candidate = (clean[idx:idx+1] + mid * delta[idx:idx+1]).clamp(0, 1)
            with torch.no_grad():
                pred = model(candidate.to(device)).argmax(1).item()
            if pred != labels[idx].item():
                hi = mid    # still fools → shrink further
            else:
                lo = mid    # no longer fools → need more perturbation
        scales[idx] = hi    # smallest scale that still fools

        if idx % 50 == 0:
            print(f"  [{idx}/500] scale={hi:.4f}")

    # Build shrunk adversarial images
    scales_t = scales.view(500, 1, 1, 1)
    adv_shrunk = (clean + scales_t * delta).clamp(0, 1)

    # Verify ASR preserved
    shrunk_pred = predict(model, adv_shrunk, device)
    final_success = (shrunk_pred != labels)
    n_final = final_success.sum().item()
    print(f"[shrink] Final surrogate ASR:   {n_final}/500 = {n_final/500:.3f}")

    # Compute SSIM improvement
    try:
        from skimage.metrics import structural_similarity as ssim
        clean_np  = (clean.numpy().transpose(0,2,3,1) * 255).round().clip(0,255).astype(np.uint8)
        adv_np    = (adv.numpy().transpose(0,2,3,1)   * 255).round().clip(0,255).astype(np.uint8)
        shrunk_np = (adv_shrunk.numpy().transpose(0,2,3,1) * 255).round().clip(0,255).astype(np.uint8)
        ssim_before = np.mean([float(ssim(clean_np[i], adv_np[i],   channel_axis=2, data_range=255)) for i in range(500)])
        ssim_after  = np.mean([float(ssim(clean_np[i], shrunk_np[i], channel_axis=2, data_range=255)) for i in range(500)])
        print(f"[shrink] mean_SSIM: {ssim_before:.4f} → {ssim_after:.4f}  (+{ssim_after-ssim_before:.4f})")
        print(f"[shrink] Score estimate: {100*n_success/500*ssim_before:.2f} → {100*n_final/500*ssim_after:.2f}")
    except ImportError:
        print("[shrink] scikit-image not found, skipping SSIM comparison")

    save_images(adv_shrunk, names, args.out_dir)
    print(f"[shrink] Saved {len(names)} shrunk images to {args.out_dir}")


if __name__ == "__main__":
    main()
