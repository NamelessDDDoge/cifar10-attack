"""Quick sanity check: load surrogates, verify they can classify CIFAR-10 images."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from attack import build_surrogates, load_images, load_true_labels, predict_labels

IMAGES_DIR = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\data\images")
LABEL_FILE  = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\data\label.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# --- Test 1: ptcv surrogates ---
print("\n=== Test 1: pytorchcv surrogates ===")
ptcv_specs = [
    "ptcv:wrn16_10_cifar10",
    "ptcv:pyramidnet164_a270_bn_cifar10",
    "ptcv:densenet190_k40_bc_cifar10",
]
models, logs = build_surrogates(device, ptcv_specs)
for l in logs:
    print(l)
print(f"Loaded {len(models)} ptcv models")

# --- Test 2: accuracy check on first 20 images ---
print("\n=== Test 2: accuracy on first 20 images ===")
names, clean = load_images(IMAGES_DIR)
names20, clean20 = names[:20], clean[:20]

labels = load_true_labels(LABEL_FILE, names20, device)
if labels is None:
    print("WARN: no label file, skipping accuracy check")
else:
    for spec, m in models:
        with torch.no_grad():
            logits = m(clean20.to(device))
        preds = logits.argmax(1).cpu()
        acc = (preds == labels.cpu()).float().mean().item()
        print(f"  {spec}: acc on 20 images = {acc:.2f}")

# --- Test 3: HF model (ViT) ---
print("\n=== Test 3: HuggingFace ViT ===")
hf_specs = ["hf:aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"]
hf_models, hf_logs = build_surrogates(device, hf_specs)
for l in hf_logs:
    print(l)
if hf_models and labels is not None:
    for spec, m in hf_models:
        with torch.no_grad():
            logits = m(clean20.to(device))
        preds = logits.argmax(1).cpu()
        acc = (preds == labels.cpu()).float().mean().item()
        print(f"  {spec}: acc on 20 images = {acc:.2f}")

# --- Test 4: gradient flows through ptcv model ---
print("\n=== Test 4: gradient flow check ===")
if models:
    x = clean20[:4].to(device).requires_grad_(False)
    delta = torch.zeros_like(x, requires_grad=True)
    adv = torch.clamp(x + delta, 0, 1)
    logits = 0
    for _, m in models[:2]:
        logits = logits + m(adv)
    loss = F.cross_entropy(logits / 2, labels[:4].to(device))
    loss.backward()
    print(f"  grad norm = {delta.grad.abs().mean().item():.6f}  (should be > 0)")

print("\nDone.")
