"""
Quick smoke test: build each attacker with 2 images, run 2 steps, check output.
Run: python smoke_test_all.py
"""
import sys, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGES_DIR, LABEL_FILE, RESULTS_DIR, REPOS_DIR, CIFAR10_MEAN, CIFAR10_STD, SURROGATE_NAMES, EPS, ALPHA

N_IMAGES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEVICE}  eps={EPS:.5f}  alpha={ALPHA:.5f}")


class CifarEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        mean = torch.tensor(CIFAR10_MEAN).view(1,3,1,1).to(DEVICE)
        std  = torch.tensor(CIFAR10_STD).view(1,3,1,1).to(DEVICE)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)
    def forward(self, x):
        xn = (x - self.mean) / self.std
        return sum(m(xn) for m in self.models) / len(self.models)


def build_ensemble(n=2):
    from pytorchcv.model_provider import get_model
    models = []
    for name in SURROGATE_NAMES[:n]:
        print(f"  load {name} ...", end=" ", flush=True)
        m = get_model(name, pretrained=True).to(DEVICE).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
        print("ok")
    return CifarEnsemble(models).to(DEVICE)


def load_small():
    names = [f"{i}.png" for i in range(N_IMAGES)]
    imgs = []
    for n in names:
        img = Image.open(IMAGES_DIR / n).convert("RGB")
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
    imgs = torch.tensor(np.stack(imgs)).permute(0,3,1,2).to(DEVICE)
    labels = {}
    with open(LABEL_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            nm, c = line.split()
            labels[nm] = int(c)
    lbls = torch.tensor([labels[n] for n in names], dtype=torch.long).to(DEVICE)
    return imgs, lbls


RESULTS = {}

def run_test(name, fn):
    print(f"\n{'='*50}\n[TEST] {name}")
    try:
        fn()
        RESULTS[name] = "PASS"
        print(f"[PASS] {name}")
    except Exception:
        RESULTS[name] = "FAIL"
        traceback.print_exc()
        print(f"[FAIL] {name}")


# ── Load shared resources once ──────────────────────────────────────────
print("\n=== Loading models (2 surrogates for speed) ===")
ensemble = build_ensemble(n=2)
images, labels = load_small()
print(f"images={images.shape}  labels={labels}")


# ── PGN ─────────────────────────────────────────────────────────────────
def test_pgn():
    import torch.nn.functional as F
    EPS_T, ALPHA_T, STEPS = EPS, ALPHA, 2
    x0 = images.clone()
    x = x0.clone()
    grad_acc = torch.zeros_like(x)
    for step in range(STEPS):
        avg_grad = torch.zeros_like(x)
        for _ in range(2):
            noise = torch.empty_like(x).uniform_(-EPS_T*3, EPS_T*3)
            x_near = (x + noise).detach().requires_grad_(True)
            loss = F.cross_entropy(ensemble(x_near), labels)
            g1 = torch.autograd.grad(loss, x_near)[0].detach()
            x_star = (x_near.detach() + ALPHA_T * (-g1)/(torch.abs(g1).mean([1,2,3],keepdim=True)+1e-12)).detach().requires_grad_(True)
            loss2 = F.cross_entropy(ensemble(x_star), labels)
            g2 = torch.autograd.grad(loss2, x_star)[0].detach()
            avg_grad += 0.5*g1 + 0.5*g2
        noise_norm = avg_grad / (torch.abs(avg_grad).mean([1,2,3],keepdim=True)+1e-12)
        grad_acc = 1.0*grad_acc + noise_norm
        x = x + ALPHA_T*torch.sign(grad_acc)
        x = torch.min(torch.max(x, x0-EPS_T), x0+EPS_T).clamp(0,1)
    delta = (x-x0).abs().max().item()
    print(f"  delta_max={delta:.5f}  (eps={EPS_T:.5f})")
    assert delta <= EPS_T + 1e-6

run_test("PGN", test_pgn)


# ── AWT ─────────────────────────────────────────────────────────────────
def test_awt():
    sys.path.insert(0, str(REPOS_DIR / "AWT"))
    from transferattack.awt import AWT
    class CifarAWT(AWT):
        def load_model(self, model_name):
            return ensemble
    att = CifarAWT("cifar10_ensemble", epsilon=EPS, alpha=ALPHA, beta=3.0, gamma=0.5,
                   num_neighbor=2, epoch=2, decay=1.0, targeted=False, random_start=False,
                   norm="linfty", loss="crossentropy", device=DEVICE)
    delta = att(images, labels)
    adv = torch.clamp(images + delta, 0, 1)
    d = (adv-images).abs().max().item()
    print(f"  delta_max={d:.5f}")
    assert d <= EPS + 1e-6

run_test("AWT", test_awt)


# ── BSR ─────────────────────────────────────────────────────────────────
def test_bsr():
    sys.path.insert(0, str(REPOS_DIR / "TransferAttack"))
    from transferattack.input_transformation.bsr import BSR
    class CifarBSR(BSR):
        def load_model(self, model_name):
            return ensemble
    att = CifarBSR("cifar10_ensemble", epsilon=EPS, alpha=ALPHA, epoch=2, decay=1.0,
                   num_scale=3, num_block=3, targeted=False, random_start=False,
                   norm="linfty", loss="crossentropy", device=DEVICE)
    delta = att(images, labels)
    adv = torch.clamp(images + delta, 0, 1)
    d = (adv-images).abs().max().item()
    print(f"  delta_max={d:.5f}")
    assert d <= EPS + 1e-6

run_test("BSR", test_bsr)


# ── SIT ─────────────────────────────────────────────────────────────────
def test_sit():
    sys.path.insert(0, str(REPOS_DIR / "SIT"))
    from attack import SIA
    att = SIA(model=ensemble, epsilon=EPS, alpha=ALPHA, epoch=2, decay=1.0,
              num_copies=2, num_block=4, targeted=False, random_start=False,
              norm="linfty", loss="crossentropy", device=DEVICE)
    delta = att(images, labels)
    adv = torch.clamp(images + delta, 0, 1)
    d = (adv-images).abs().max().item()
    print(f"  delta_max={d:.5f}")
    assert d <= EPS + 1e-6

run_test("SIT", test_sit)


# ── L2T ─────────────────────────────────────────────────────────────────
def test_l2t():
    sys.path.insert(0, str(REPOS_DIR / "L2T"))
    import l2t as l2t_module
    print(f"  op_list total={len(l2t_module.op_list)}, truncating to 71")
    l2t_module.op_list = l2t_module.op_list[:71]
    class CifarL2T(l2t_module.L2T):
        def load_model(self, model_name):
            return ensemble
    att = CifarL2T("cifar10_ensemble", epsilon=EPS, alpha=ALPHA, epoch=2, decay=1.0,
                   targeted=False, random_start=False, norm="linfty", loss="crossentropy",
                   device=DEVICE, num_scale=2)
    delta = att(images, labels)
    adv = torch.clamp(images + delta, 0, 1)
    d = (adv-images).abs().max().item()
    print(f"  delta_max={d:.5f}")
    assert d <= EPS + 1e-6

run_test("L2T", test_l2t)


# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("SUMMARY:")
for k, v in RESULTS.items():
    print(f"  {k}: {v}")
