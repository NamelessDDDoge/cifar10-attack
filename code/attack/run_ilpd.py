"""
ILPD (Intermediate-Level Perturbation Decay) attack adapter for CIFAR-10
Origin: https://github.com/qizhangli/ILPD-attack
Paper: Li et al., NeurIPS 2023

Adaptations from original attacks/ilpd.py:
  - Replaced timm models with pytorchcv CIFAR-10 surrogate ensemble
  - Custom _select_pos: hooks into features.stage2 (analog of layer2 in standard ResNets)
  - eps=8/255, alpha=2/255, steps=10
  - ilpd_coef=0.1 (1/gamma), ilpd_sigma=0.05, ilpd_N=1
  - PNG load/save for 32x32 images
  - No DataParallel; CPU-compatible
  - The paper reports +3.88% on CIFAR-10 - only candidate with explicit CIFAR-10 numbers
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RESULTS_DIR, CIFAR10_MEAN, CIFAR10_STD,
    SURROGATE_NAMES, EPS, ALPHA, STEPS, BATCH_SIZE, EVAL_DIR,
)
from run_utils import (
    setup_logger, load_data, adv_to_numpy,
    save_adv_batch, get_done_set, image_pbar, step_pbar,
    iter_todo_batches, log_progress_event,
)

TAG = "ILPD"
ADV_DIR = RESULTS_DIR / "adv_ilpd"

ILPD_COEF  = 0.1    # 1/gamma
ILPD_SIGMA = 0.05
ILPD_N     = 1


class CifarNormModel(nn.Module):
    def __init__(self, backbone, device):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std  = torch.tensor(CIFAR10_STD,  dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def _hook_ilout(module, input, output):
    module.ilpd_output = output


def _hook_pd(ori_ilout, gamma):
    def hook(module, input, output):
        return gamma * output + (1 - gamma) * ori_ilout
    return hook


class ILPDAttacker:
    def __init__(self, primary_model, il_module, extra_models=None,
                 coef=0.1, sigma=0.05, N=1,
                 epsilon=EPS, step_size=ALPHA, steps=STEPS):
        self.model       = primary_model
        self.il_module   = il_module
        self.extra_models = extra_models or []
        self.coef        = coef
        self.sigma       = sigma
        self.N           = N
        self.epsilon     = epsilon
        self.step_size   = step_size
        self.steps       = steps
        self.hook        = self.il_module.register_forward_hook(_hook_pd(0, 1))

    def _prep_hook(self, ori_img, iteration):
        if self.sigma == 0 and iteration > 0:
            return
        self.hook.remove()
        with torch.no_grad():
            h = self.il_module.register_forward_hook(_hook_ilout)
            self.model(ori_img + self.sigma * torch.randn_like(ori_img))
            ori_ilout = self.il_module.ilpd_output.clone()
            h.remove()
        self.hook = self.il_module.register_forward_hook(_hook_pd(ori_ilout, self.coef))

    def attack(self, ori_img, label, init_img=None, sbar=None):
        adv_img = project_initial_images(ori_img, init_img, self.epsilon) if init_img is not None else ori_img.clone()
        momentum = torch.zeros_like(ori_img)

        for i in range(self.steps):
            input_grad = torch.zeros_like(ori_img)

            for _ in range(self.N):
                self._prep_hook(ori_img, i)
                adv_img = adv_img.detach().requires_grad_(True)
                logits  = self.model(adv_img)
                loss    = F.cross_entropy(logits, label)
                g       = torch.autograd.grad(loss, adv_img)[0].detach()
                input_grad = input_grad + g

            for em in self.extra_models:
                xr  = adv_img.detach().requires_grad_(True)
                lgt = em(xr)
                lss = F.cross_entropy(lgt, label)
                g_e = torch.autograd.grad(lss, xr)[0].detach()
                input_grad = input_grad + g_e

            n_models   = self.N + len(self.extra_models)
            input_grad = input_grad / n_models

            grad_norm  = torch.abs(input_grad).mean(dim=(1, 2, 3), keepdim=True) + 1e-12
            momentum   = 1.0 * momentum + input_grad / grad_norm
            adv_img    = adv_img.detach() + self.step_size * torch.sign(momentum)
            adv_img    = torch.min(torch.max(adv_img, ori_img - self.epsilon),
                                   ori_img + self.epsilon).clamp(0, 1)

            if sbar is not None:
                sbar.update(1)
                sbar.set_postfix(loss=f"{loss.item():.4f}")

        self.hook.remove()
        return adv_img.detach()


def project_initial_images(clean: torch.Tensor, init: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.min(torch.max(init.detach(), clean - epsilon), clean + epsilon).clamp(0, 1)


def load_init_data(init_dir: Path, names: list[str], device: torch.device) -> torch.Tensor:
    images = []
    missing = []
    for name in names:
        path = init_dir / name
        if not path.exists():
            missing.append(name)
            continue
        img = Image.open(path).convert("RGB")
        images.append(np.array(img, dtype=np.float32) / 255.0)
    if missing:
        raise FileNotFoundError(f"{init_dir} missing initial image(s): {missing[:5]}")
    return torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2).to(device)


def _select_external_names(selection: str, specs) -> list[str]:
    names = [s.name for s in specs]
    if selection == "none":
        return []
    if selection == "all":
        return names
    requested = [part.strip() for part in selection.split(",") if part.strip()]
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"unknown external surrogate(s): {sorted(unknown)}")
    return requested


def build_models(device, logger, cnn_count=None, vit_surrogates="none", robust_surrogates="none"):
    from pytorchcv.model_provider import get_model
    sys.path.insert(0, str(EVAL_DIR))
    from external_models import VIT_SPECS, ROBUST_SPECS, load_external_model

    primary_name = SURROGATE_NAMES[0]
    logger.info(f"  Loading {primary_name} (primary) ...")
    backbone = get_model(primary_name, pretrained=True).to(device).eval()
    primary  = CifarNormModel(backbone, device).to(device)
    il_module = backbone.features.stage2

    extra = []
    selected_cnn = SURROGATE_NAMES[1:] if cnn_count is None else SURROGATE_NAMES[1:cnn_count]
    for name in selected_cnn:
        logger.info(f"  Loading {name} ...")
        m = get_model(name, pretrained=True).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        wrapped = CifarNormModel(m, device).to(device)
        for p in wrapped.parameters():
            p.requires_grad_(False)
        extra.append(wrapped)

    for name in _select_external_names(vit_surrogates, VIT_SPECS):
        logger.info(f"  Loading ViT surrogate {name} ...")
        m = load_external_model(name, device=device).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        extra.append(m)

    for name in _select_external_names(robust_surrogates, ROBUST_SPECS):
        logger.info(f"  Loading robust surrogate {name} ...")
        m = load_external_model(name, device=device).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        extra.append(m)

    logger.info(f"  Models ready: 1 primary + {len(extra)} extra")
    return primary, il_module, extra


def main():
    parser = argparse.ArgumentParser(description="ILPD CIFAR-10 attack")
    parser.add_argument("--out-dir", default=str(ADV_DIR))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cnn-count", type=int, default=None)
    parser.add_argument("--vit-surrogates", default="none")
    parser.add_argument("--robust-surrogates", default="none")
    parser.add_argument("--init-dir", default=None,
                        help="Optional adversarial image directory used to initialize ILPD before projection")
    args = parser.parse_args()

    adv_dir = Path(args.out_dir)
    logger = setup_logger(TAG)
    done_set = get_done_set(adv_dir)

    limit = args.max_images if args.max_images is not None else 500
    if args.max_images is None and len(done_set) == 500:
        logger.info(f"Already complete (500 files). Skip.")
        return

    logger.info(f"Resume: {len(done_set)}/{limit} done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info("Building models ...")
    primary, il_module, extra = build_models(
        device, logger,
        cnn_count=args.cnn_count,
        vit_surrogates=args.vit_surrogates,
        robust_surrogates=args.robust_surrogates,
    )

    attacker = ILPDAttacker(
        primary_model=primary, il_module=il_module, extra_models=extra,
        coef=ILPD_COEF, sigma=ILPD_SIGMA, N=ILPD_N,
        epsilon=EPS, step_size=ALPHA, steps=args.steps,
    )

    logger.info("Loading data ...")
    names, images, labels = load_data(device)
    init_images = None
    if args.init_dir:
        init_dir = Path(args.init_dir)
        logger.info(f"Loading initial images from {init_dir} ...")
        init_images = load_init_data(init_dir, names[:limit], device)

    adv_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, n) for i, n in enumerate(names[:limit]) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}")

    with image_pbar(limit, min(len(done_set), limit), TAG) as ibar:
        for chunk in iter_todo_batches(todo, args.batch_size):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)

            img_b = images[idxs]
            lbl_b = labels[idxs]
            init_b = init_images[idxs] if init_images is not None else None

            batch_desc = f"[{idxs[0]}:{idxs[-1]+1}]"
            with step_pbar(args.steps, batch_desc) as sbar:
                adv_b = attacker.attack(img_b, lbl_b, init_img=init_b, sbar=sbar)

            save_adv_batch(adv_to_numpy(adv_b), list(names_b), adv_dir)
            ibar.update(len(chunk))
            log_progress_event(logger, TAG, adv_dir, len(get_done_set(adv_dir)),
                               total=limit, batch=[int(idxs[0]), int(idxs[-1])])
            logger.debug(f"Saved batch {batch_desc} ({len(chunk)} imgs)")

    logger.info(f"Done. {len(list(adv_dir.glob('*.png')))} images in {adv_dir}")


if __name__ == "__main__":
    main()
