import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from skimage.metrics import structural_similarity as ssim
except Exception as e:
    raise RuntimeError("scikit-image is required. Install with: pip install scikit-image") from e

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Default surrogates: pytorchcv CIFAR-10 trained (prefix "ptcv:")
# + HuggingFace diverse architectures (prefix "hf:")
DEFAULT_ENSEMBLE = ",".join([
    "ptcv:pyramidnet164_a270_bn_cifar10",
    "ptcv:wrn16_10_cifar10",
    "ptcv:densenet190_k40_bc_cifar10",
    "ptcv:seresnet110_cifar10",
    "ptcv:resnext29_16x64d_cifar10",
    "ptcv:diaresnet56_cifar10",
    "hf:aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
    "hf:ahsanjavid/convnext-tiny-finetuned-cifar10",
])


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class WrappedModel(nn.Module):
    """Wraps pytorchcv models: [0,1] input -> CIFAR10 normalize -> backbone."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.norm = Normalize(CIFAR10_MEAN, CIFAR10_STD)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(self.norm(x))


class HFWrapper(nn.Module):
    """Wraps HuggingFace image classifiers: [0,1] 32x32 -> upsample 224 -> ImageNet normalize -> model."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.norm(x)
        out = self.model(pixel_values=x)
        return out.logits


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images(images_dir: Path) -> Tuple[List[str], torch.Tensor]:
    names = sorted([p.name for p in images_dir.glob("*.png")], key=lambda s: int(Path(s).stem))
    to_tensor = transforms.ToTensor()
    arr = []
    for name in names:
        im = Image.open(images_dir / name).convert("RGB")
        if im.size != (32, 32):
            raise ValueError(f"Expected 32x32, got {im.size} for {name}")
        arr.append(to_tensor(im))
    return names, torch.stack(arr, dim=0)


def load_ptcv_model(name: str, device):
    from pytorchcv.model_provider import get_model
    m = get_model(name, pretrained=True)
    return WrappedModel(m).to(device).eval()


def load_hf_model(model_id: str, device):
    from transformers import AutoModelForImageClassification
    m = AutoModelForImageClassification.from_pretrained(model_id)
    return HFWrapper(m).to(device).eval()


def build_surrogates(device, requested: List[str]):
    models, logs = [], []
    for spec in requested:
        try:
            if spec.startswith("ptcv:"):
                name = spec[5:]
                wm = load_ptcv_model(name, device)
                models.append((spec, wm))
                logs.append(f"[ok] ptcv {name}")
            elif spec.startswith("hf:"):
                model_id = spec[3:]
                wm = load_hf_model(model_id, device)
                models.append((spec, wm))
                logs.append(f"[ok] hf {model_id}")
            else:
                logs.append(f"[skip] Unknown prefix (use ptcv: or hf:): {spec}")
        except Exception as e:
            logs.append(f"[fail] {spec}: {e}")
    return models, logs


def load_true_labels(path: Path, names: List[str], device) -> torch.Tensor:
    """Load from label.txt (format: filename label). Falls back to None."""
    if not path.exists():
        return None
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # support both "name label" and "name true_label pseudo_label ..."
            mapping[parts[0]] = int(parts[1])
    if all(n in mapping for n in names):
        lbls = [mapping[n] for n in names]
        print(f"[labels] Loaded {len(lbls)} true labels from {path}", flush=True)
        return torch.tensor(lbls, dtype=torch.long, device=device)
    print(f"[warn] label.txt missing some entries, will predict labels", flush=True)
    return None


def predict_labels(models, x, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            xb = x[i:i + bs]
            logits = 0
            for _, m in models:
                logits = logits + m(xb)
            logits = logits / len(models)
            out.append(logits.argmax(1).cpu())
    return torch.cat(out)


def get_labels(label_file: Path, names: List[str], models, clean, device):
    labels = load_true_labels(label_file, names, device)
    if labels is not None:
        return labels
    print("[labels] Predicting labels via ensemble (no canonical label.txt found)...", flush=True)
    preds = predict_labels(models, clean.to(device))
    return preds.to(device)


def diverse_input(x, p):
    if random.random() > p:
        return x
    rnd = random.randint(28, 32)
    resized = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pad_h = 32 - rnd
    pad_w = 32 - rnd
    top  = random.randint(0, pad_h)
    left = random.randint(0, pad_w)
    return F.pad(resized, (left, pad_w - left, top, pad_h - top), value=0.0)


def gaussian_kernel(k=5, sigma=1.0, channels=3, device="cpu"):
    ax = torch.arange(-k // 2 + 1.0, k // 2 + 1.0, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    ker = (ker / ker.sum()).view(1, 1, k, k).repeat(channels, 1, 1, 1)
    return ker


def tim_filter(grad, kernel):
    return F.conv2d(grad, kernel, stride=1, padding=kernel.shape[-1] // 2, groups=grad.shape[1])


def admix_batch(x, eta=0.2):
    idx = torch.randperm(x.size(0), device=x.device)
    return torch.clamp(x + eta * x[idx], 0.0, 1.0)


def run_attack_batch(models, x0_b, y_b, eps, steps, step_size, momentum, p_div, eta, m2, device, ker, ce):
    """Attack a single mini-batch. Returns adversarial images on CPU."""
    x0 = x0_b.to(device)
    y  = y_b.to(device)
    delta = torch.zeros_like(x0, requires_grad=True)
    vel   = torch.zeros_like(x0)

    for _ in range(steps):
        grad_acc = torch.zeros_like(x0)
        for _ in range(m2):
            adv = torch.clamp(x0 + delta, 0.0, 1.0)
            mix = admix_batch(adv, eta)
            inp = diverse_input(mix, p_div)
            logits = 0
            for _, m in models:
                logits = logits + m(inp)
            logits = logits / len(models)
            loss = ce(logits, y)
            g = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            grad_acc += g

        grad = grad_acc / float(m2)
        grad = tim_filter(grad, ker)
        denom = torch.clamp(torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True), min=1e-12)
        grad  = grad / denom
        vel   = momentum * vel + grad
        delta = (delta + step_size * torch.sign(vel)).clamp(-eps, eps)
        delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0
        delta = delta.detach().requires_grad_(True)

    return torch.clamp(x0 + delta.detach(), 0.0, 1.0).cpu()


def run_attack(models, clean, labels, eps, steps, step_size, momentum, p_div, eta, m2, device, batch_size=32):
    ker = gaussian_kernel(5, 1.0, 3, device)
    ce  = nn.CrossEntropyLoss()
    n   = clean.size(0)
    results = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x0_b = clean[start:end]
        y_b  = labels[start:end] if labels.device.type == "cpu" else labels[start:end].cpu()
        adv_b = run_attack_batch(models, x0_b, y_b, eps, steps, step_size,
                                  momentum, p_div, eta, m2, device, ker, ce)
        results.append(adv_b)
        print(f"[attack] batch {end}/{n} done", flush=True)

    return torch.cat(results, dim=0)


def save_adv(out_dir: Path, names: List[str], adv: torch.Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for name, tensor in zip(names, adv.cpu()):
        to_pil(tensor).save(out_dir / name, format="PNG")


def compute_metrics(models, clean, adv, labels, names):
    device = next(models[0][1].parameters()).device
    labels = labels.to(device)
    adv_pred = predict_labels(models, adv.to(device)).to(device)
    success = adv_pred != labels
    asr = success.float().mean().item()

    clean_np = (clean.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    adv_np   = (adv.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    vals = []
    for i in range(len(names)):
        vals.append(float(ssim(clean_np[i], adv_np[i], channel_axis=2, data_range=255)))
    mean_ssim = float(np.mean(vals)) if vals else 0.0
    score = 100.0 * asr * mean_ssim
    return asr, mean_ssim, score


def main():
    parser = argparse.ArgumentParser()
    _root = Path(__file__).resolve().parents[2]  # workspace/
    parser.add_argument("--images-dir",   default=os.environ.get("IMAGES_DIR",  str(_root / "data" / "images")))
    parser.add_argument("--work-dir",     default=os.environ.get("WORK_DIR",    str(_root / "results" / "work")))
    parser.add_argument("--label-file",   default=os.environ.get("LABEL_FILE",  str(_root / "data" / "label.txt")))
    parser.add_argument("--results-json", default=os.environ.get("RESULTS_JSON",
        r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\runs\lit-bridge-e2e\artifacts\quick-experiment\results.json"))
    parser.add_argument("--eps",          type=float, default=float(os.environ.get("EPS",       8.0 / 255.0)))
    parser.add_argument("--steps",        type=int,   default=int  (os.environ.get("STEPS",     50)))
    parser.add_argument("--step-size",    type=float, default=float(os.environ.get("STEP_SIZE", 2.0 / 255.0)))
    parser.add_argument("--momentum",     type=float, default=float(os.environ.get("MOMENTUM",  1.0)))
    parser.add_argument("--diversity-prob", type=float, default=float(os.environ.get("DIVERSITY_PROB", 0.7)))
    parser.add_argument("--ensemble",     default=os.environ.get("ENSEMBLE", DEFAULT_ENSEMBLE))
    parser.add_argument("--admix-eta",    type=float, default=float(os.environ.get("ADMIX_ETA", 0.2)))
    parser.add_argument("--admix-m2",     type=int,   default=int  (os.environ.get("ADMIX_M2",  3)))
    parser.add_argument("--seed",         type=int,   default=int  (os.environ.get("SEED",      42)))
    parser.add_argument("--batch-size",   type=int,   default=int  (os.environ.get("BATCH_SIZE",  32)))
    parser.add_argument("--force-rerun",  action="store_true", help="Ignore cached adv images")
    args = parser.parse_args()

    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}", flush=True)
    print(f"[info] eps={args.eps:.5f} steps={args.steps} step_size={args.step_size:.5f} momentum={args.momentum}", flush=True)

    names, clean = load_images(Path(args.images_dir))
    print(f"[info] loaded {len(names)} images", flush=True)

    surrogate_specs = [x.strip() for x in args.ensemble.split(",") if x.strip()]
    models, logs = build_surrogates(device, surrogate_specs)
    for line in logs:
        print(line, flush=True)

    if not models:
        raise RuntimeError("No surrogates loaded. Check ensemble spec.")

    labels = get_labels(Path(args.label_file), names, models, clean, device)

    out_dir = work_dir / "adv_images"
    if not args.force_rerun and out_dir.exists() and len(list(out_dir.glob("*.png"))) == len(names):
        print(f"[cache] Reusing existing adversarial images in {out_dir}", flush=True)
        to_tensor = transforms.ToTensor()
        adv = torch.stack([to_tensor(Image.open(out_dir / n).convert("RGB")) for n in names], dim=0)
    else:
        adv = run_attack(models, clean, labels, args.eps, args.steps, args.step_size,
                         args.momentum, args.diversity_prob, args.admix_eta, args.admix_m2,
                         device, batch_size=args.batch_size)
        save_adv(out_dir, names, adv)
        print(f"[write] Saved adversarial images to {out_dir}", flush=True)

    asr, mean_ssim, score = compute_metrics(models, clean, adv, labels, names)
    metrics = {
        "eps": args.eps,
        "steps": args.steps,
        "asr": asr,
        "mean_ssim": mean_ssim,
        "score": score,
        "n": len(names),
        "ensemble": surrogate_specs,
    }

    metrics_path = work_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[write] metrics={metrics}", flush=True)

    out_results = Path(args.results_json)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metrics_path, out_results)
    print(f"[write] copied metrics to {out_results}", flush=True)


if __name__ == "__main__":
    main()
