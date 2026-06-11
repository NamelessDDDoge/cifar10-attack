"""
Generic CIFAR-10 attack runner for TransferAttack methods.
Usage: python run_transferattack.py <method> [--eps 8/255] [--alpha 2/255] [--epoch 10] [--batch-size 50]

Methods: mifgsm, nifgsm, dim, tim, admix, bsr, sia, sim, ssm, pgn, l2t, awt, etc.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Use shared config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGES_DIR, LABEL_FILE, RESULTS_DIR, REPOS_DIR, CIFAR10_MEAN, CIFAR10_STD, SURROGATE_NAMES, EPS, ALPHA, BATCH_SIZE

REPO_DIR = REPOS_DIR / "TransferAttack"
sys.path.insert(0, str(REPO_DIR))


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


def load_data(device):
    names = [f"{i}.png" for i in range(500)]
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
    return names, images.to(device), label_tensor.to(device)


def save_adv(adv, names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    adv_np = (adv.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    for i, name in enumerate(names):
        Image.fromarray(adv_np[i]).save(out_dir / name)


_ATTACK_REGISTRY = {}


def register(name):
    _ATTACK_REGISTRY[name] = name


# Register known methods
METHODS_WITH_PARAMS = {
    # gradient-based
    "mifgsm":  {},
    "nifgsm":  {},
    "vmifgsm": {},
    "vnifgsm": {},
    "emifgsm": {},
    # input transformation
    "dim":     {},
    "tim":     {},
    "admix":   {},
    "bsr":     {"num_scale": 10, "num_block": 3},
    "sim":     {},
    "ssm":     {},
    "sia":     {},
    "l2t":     {"num_scale": 3},
    # generation
    "diffattack": {},
    "ttp":        {},
    "cdtp":       {},
    # gradient optimization
    "pgn":       {},
    "pifgsm":    {},
    "gifgsm":    {},
    "gnp":        {},
    "aifgtm":     {},
    "anda":       {},
    "foolmix":    {},
    "mig":        {},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, choices=list(METHODS_WITH_PARAMS.keys()))
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--norm", type=str, default="linfty", choices=["linfty", "l2"])
    parser.add_argument("--loss", type=str, default="crossentropy")
    args = parser.parse_args()

    method = args.method
    adv_dir = RESULTS_DIR / f"adv_{method}"
    if adv_dir.exists() and len(list(adv_dir.glob("*.png"))) == 500:
        print(f"[skip] {adv_dir} already complete (500 files).")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{method}] device={device}")

    print(f"[{method}] Building surrogate ensemble ...")
    ensemble = build_ensemble(device)

    print(f"[{method}] Loading data ...")
    names, images, labels = load_data(device)

    # Dynamically import attack class
    module_map = {
        "mifgsm":  "transferattack.gradient.mifgsm",
        "nifgsm":  "transferattack.gradient.nifgsm",
        "vmifgsm": "transferattack.gradient.vmifgsm",
        "vnifgsm": "transferattack.gradient.vnifgsm",
        "emifgsm": "transferattack.gradient.emifgsm",
        "dim":     "transferattack.input_transformation.dim",
        "tim":     None,  # TIM is in utils, not a standalone class
        "admix":   "transferattack.input_transformation.admix",
        "bsr":     "transferattack.input_transformation.bsr",
        "sim":     "transferattack.input_transformation.sim",
        "ssm":     "transferattack.input_transformation.ssm",
        "sia":     "transferattack.input_transformation.sia",
        "l2t":     "transferattack.input_transformation.l2t",
        "diffattack": "transferattack.generation.diffattack",
        "ttp":        "transferattack.generation.ttp",
        "cdtp":       "transferattack.generation.cdtp",
        "pgn":        "transferattack.gradient.pgn",
        "pifgsm":     "transferattack.gradient.pifgsm",
        "gifgsm":     "transferattack.gradient.gifgsm",
        "gnp":        "transferattack.gradient.gnp",
        "aifgtm":     "transferattack.gradient.aifgtm",
        "anda":       "transferattack.gradient.anda",
        "foolmix":    "transferattack.gradient.foolmix",
        "mig":        "transferattack.gradient.mig",
    }

    import importlib
    module_path = module_map[method]
    if module_path is None:
        print(f"[error] {method} not yet supported as standalone attack class.")
        return

    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        print(f"[error] Failed to import {module_path}: {e}")
        return

    # Find attack class
    AttackCls = None
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if isinstance(obj, type) and "Attack" in obj.__name__ and issubclass(obj, object):
            if attr_name.lower() == method:
                AttackCls = obj
                break
            if AttackCls is None:
                AttackCls = obj

    if AttackCls is None:
        print(f"[error] No attack class found in {module_path}")
        return

    # Override load_model
    class CifarAttack(AttackCls):
        def load_model(self, model_name):
            return ensemble

    extra_params = METHODS_WITH_PARAMS[method]

    attacker = CifarAttack(
        model_name=f"cifar10_ensemble_{method}",
        epsilon=args.eps,
        alpha=args.alpha,
        epoch=args.epoch,
        decay=args.decay,
        targeted=args.targeted,
        random_start=False,
        norm=args.norm,
        loss=args.loss,
        device=device,
        **extra_params,
    )

    adv_dir.mkdir(parents=True, exist_ok=True)
    all_adv = []
    n = len(names)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        img_b = images[start:end]
        lbl_b = labels[start:end]
        try:
            delta = attacker(img_b, lbl_b)
        except Exception as e:
            print(f"[error] batch {start}-{end}: {e}")
            traceback.print_exc()
            continue
        adv_b = torch.clamp(img_b + delta, 0, 1).detach().cpu()
        all_adv.append(adv_b)
        print(f"[{method}] batch {start}-{end} done ({start+len(img_b)}/{n})")

    if len(all_adv) * args.batch_size >= 500:
        all_adv = torch.cat(all_adv, dim=0)
        save_adv(all_adv, names, adv_dir)
        print(f"[{method}] Saved adversarial images to {adv_dir}")
    else:
        print(f"[error] Only {sum(a.size(0) for a in all_adv)}/{n} images generated")


if __name__ == "__main__":
    main()
