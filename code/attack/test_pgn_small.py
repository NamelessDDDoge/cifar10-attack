"""Quick test: run PGN on first 50 images to verify the pipeline works."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGES_DIR, LABEL_FILE, CIFAR10_MEAN, CIFAR10_STD, SURROGATE_NAMES, EPS, ALPHA, STEPS

# PGN params
N_SAMPLES = 5       # reduced for quick test
MOMENTUM = 1.0
ZETA = 3.0
DELTA_COEF = 0.5


class CifarEnsemble(nn.Module):
    def __init__(self, models, device):
        super().__init__()
        self.models = nn.ModuleList(models)
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x_n = (x - self.mean) / self.std
        return sum(m(x_n) for m in self.models) / len(self.models)


def build_ensemble(device):
    from pytorchcv.model_provider import get_model
    models = []
    for name in SURROGATE_NAMES:
        print(f"  Loading {name} ...", end=" ", flush=True)
        m = get_model(name, pretrained=True).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
        print("ok")
    return CifarEnsemble(models, device).to(device)


def pgn_attack(model, images, labels, device):
    x0 = images.clone()
    x = x0.clone()
    grad_acc = torch.zeros_like(x)

    for step in range(STEPS):
        avg_grad = torch.zeros_like(x)
        for _ in range(N_SAMPLES):
            noise = torch.empty_like(x).uniform_(-EPS * ZETA, EPS * ZETA)
            x_near = (x + noise).detach().requires_grad_(True)
            loss = F.cross_entropy(model(x_near), labels)
            g1 = torch.autograd.grad(loss, x_near)[0].detach()
            x_star = x_near.detach() + ALPHA * (-g1) / (torch.abs(g1).mean([1, 2, 3], keepdim=True) + 1e-12)
            x_star = x_star.detach().requires_grad_(True)
            loss2 = F.cross_entropy(model(x_star), labels)
            g2 = torch.autograd.grad(loss2, x_star)[0].detach()
            avg_grad = avg_grad + (1 - DELTA_COEF) * g1 + DELTA_COEF * g2

        noise_norm = avg_grad / (torch.abs(avg_grad).mean([1, 2, 3], keepdim=True) + 1e-12)
        grad_acc = MOMENTUM * grad_acc + noise_norm
        x = x + ALPHA * torch.sign(grad_acc)
        x = torch.min(torch.max(x, x0 - EPS), x0 + EPS).clamp(0, 1)
        print(f"  step {step+1}/{STEPS}")

    return x.detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building surrogate ensemble ...")
    ensemble = build_ensemble(device)

    # Load only first 50 images
    names = [f"{i}.png" for i in range(50)]
    images = []
    for name in names:
        img = Image.open(IMAGES_DIR / name).convert("RGB")
        images.append(np.array(img, dtype=np.float32) / 255.0)
    images = torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2)

    labels = {}
    with open(LABEL_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            n, c = line.split()
            labels[n] = int(c)
    label_tensor = torch.tensor([labels[n] for n in names], dtype=torch.long)

    images = images.to(device)
    label_tensor = label_tensor.to(device)

    print(f"Running PGN on {len(names)} images ...")
    adv = pgn_attack(ensemble, images, label_tensor, device)

    # Check output
    adv_np = adv.cpu().numpy().transpose(0, 2, 3, 1)
    print(f"Output shape: {adv_np.shape}")
    print(f"Output range: [{adv_np.min():.4f}, {adv_np.max():.4f}]")
    print("Test PASSED!")


if __name__ == "__main__":
    main()
