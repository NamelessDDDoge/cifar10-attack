"""Diverse shortfall-aware transfer attack for CIFAR-10.

This runner is a stronger baseline than the original isolated attacks:
  - mixed CNN + external ViT surrogate ensemble
  - untargeted logit-margin objective
  - per-step shortfall-aware model weights
  - DI/SI/EOT-style input diversity

The ViTs are intentionally named "surrogates" here because they contribute
gradients. Do not interpret same-model ViT eval numbers as clean holdout
estimates when all ViTs are included.
"""
import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RESULTS_DIR, CIFAR10_MEAN, CIFAR10_STD, SURROGATE_NAMES,
    EPS, ALPHA, BATCH_SIZE, EVAL_DIR,
)
sys.path.insert(0, str(EVAL_DIR))
from external_models import VIT_SPECS, ROBUST_SPECS, load_external_model
from run_utils import (
    setup_logger, load_data, adv_to_numpy, save_adv_batch,
    get_done_set, image_pbar, step_pbar,
    iter_todo_batches, log_progress_event,
)


TAG = "DIVERSE_VIT"
DEFAULT_ADV_DIR = RESULTS_DIR / "adv_diverse_vit"
DEFAULT_STEPS = 50
DEFAULT_BATCH_SIZE = min(2, BATCH_SIZE)
DEFAULT_CNN_COUNT = 4
DEFAULT_EOT = 2
DEFAULT_WEIGHT_FLOOR = 0.08
DEFAULT_WEIGHT_TEMP = 5.0
MOMENTUM = 1.0


@dataclass
class AttackModel:
    name: str
    family: str
    model: nn.Module


class CifarNormModel(nn.Module):
    def __init__(self, backbone: nn.Module, device: torch.device):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def untargeted_margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return mean(max_non_true_logit - true_logit), maximized by the attack."""
    true = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), -torch.inf)
    other = masked.max(dim=1).values
    return (other - true).mean()


def per_model_asr_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) != labels).float().mean()


def compute_shortfall_weights(
    asr: torch.Tensor,
    floor: float = DEFAULT_WEIGHT_FLOOR,
    temperature: float = DEFAULT_WEIGHT_TEMP,
) -> torch.Tensor:
    """Weight lower-ASR models more while keeping every model represented."""
    if asr.ndim != 1:
        raise ValueError("asr must be a 1D tensor")
    n = asr.numel()
    if n == 0:
        raise ValueError("asr must not be empty")
    if not 0 <= floor < 1 / n:
        raise ValueError(f"floor must satisfy 0 <= floor < 1/{n}")

    shortfall = (1.0 - asr.float()).clamp(0.0, 1.0)
    adaptive = torch.softmax(shortfall * temperature, dim=0)
    return floor + (1.0 - floor * n) * adaptive


def input_diversity(x: torch.Tensor, prob: float = 0.7) -> torch.Tensor:
    """Small-image DI: random resize down/up and translation-preserving padding."""
    if prob <= 0 or random.random() > prob:
        return x
    size = x.shape[-1]
    min_size = max(24, int(size * 0.75))
    rnd = random.randint(min_size, size)
    out = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pad_total = size - rnd
    pad_left = random.randint(0, pad_total)
    pad_top = random.randint(0, pad_total)
    pad_right = pad_total - pad_left
    pad_bottom = pad_total - pad_top
    out = F.pad(out, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return out


def scaled_inputs(x: torch.Tensor, scales: tuple[float, ...]) -> list[torch.Tensor]:
    outs = []
    size = x.shape[-1]
    for scale in scales:
        if math.isclose(scale, 1.0):
            outs.append(x)
            continue
        scaled = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        outs.append(F.interpolate(scaled, size=(size, size), mode="bilinear", align_corners=False))
    return outs


def make_ti_kernel(device: torch.device, channels: int = 3, kernel_size: int = 5) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma = 1.0
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def smooth_gradient(grad: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    return F.conv2d(grad, kernel.to(device=grad.device, dtype=grad.dtype), padding=pad, groups=grad.shape[1])


def select_vit_names(selection: str) -> list[str]:
    names = [s.name for s in VIT_SPECS]
    if selection == "all":
        return names
    if selection == "weak2":
        return ["vit_hf_nateraw", "vit_timm_edadaltocg"]
    if selection == "weak1":
        return ["vit_timm_edadaltocg"]
    if selection == "none":
        return []
    requested = [part.strip() for part in selection.split(",") if part.strip()]
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"unknown ViT surrogate(s): {sorted(unknown)}")
    return requested


def select_robust_names(selection: str) -> list[str]:
    names = [s.name for s in ROBUST_SPECS]
    if selection == "none":
        return []
    if selection == "light3":
        return ["robust_engstrom", "robust_rade_r18_extra", "robust_xcit_s12"]
    if selection == "all":
        return names
    requested = [part.strip() for part in selection.split(",") if part.strip()]
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"unknown robust surrogate(s): {sorted(unknown)}")
    return requested


def build_attack_models(
    device: torch.device,
    logger,
    cnn_count: int,
    vit_selection: str,
    robust_selection: str,
) -> list[AttackModel]:
    from pytorchcv.model_provider import get_model

    models: list[AttackModel] = []
    cnn_names = SURROGATE_NAMES[:cnn_count]
    for name in cnn_names:
        logger.info(f"  Loading CNN surrogate {name} ...")
        backbone = get_model(name, pretrained=True).to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        models.append(AttackModel(name=name, family="cnn", model=CifarNormModel(backbone, device).to(device)))

    for name in select_vit_names(vit_selection):
        logger.info(f"  Loading ViT surrogate {name} ...")
        model = load_external_model(name, device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        models.append(AttackModel(name=name, family="vit", model=model.to(device).eval()))

    for name in select_robust_names(robust_selection):
        logger.info(f"  Loading robust surrogate {name} ...")
        model = load_external_model(name, device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        models.append(AttackModel(name=name, family="robust", model=model.to(device).eval()))

    if not models:
        raise RuntimeError("no attack models selected")
    logger.info(f"Attack pool ready: {len(models)} models ({', '.join(m.name for m in models)})")
    return models


def estimate_asr(models: list[AttackModel], x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vals = []
    with torch.no_grad():
        for am in models:
            vals.append(per_model_asr_from_logits(am.model(x), labels))
    return torch.stack(vals)


def weighted_margin_loss(
    models: list[AttackModel],
    x: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    eot: int,
    scales: tuple[float, ...],
) -> torch.Tensor:
    loss = x.new_tensor(0.0)
    norm = max(1, eot) * len(scales)
    for _ in range(max(1, eot)):
        diverse = input_diversity(x)
        for sx in scaled_inputs(diverse, scales):
            for idx, am in enumerate(models):
                loss = loss + weights[idx].to(x.device) * untargeted_margin_loss(am.model(sx), labels) / norm
    return loss


def diverse_attack(
    models: list[AttackModel],
    images: torch.Tensor,
    labels: torch.Tensor,
    steps: int = DEFAULT_STEPS,
    step_size: float = ALPHA,
    epsilon: float = EPS,
    eot: int = DEFAULT_EOT,
    weight_floor: float = DEFAULT_WEIGHT_FLOOR,
    weight_temp: float = DEFAULT_WEIGHT_TEMP,
    scales: tuple[float, ...] = (1.0, 0.75, 0.5),
    sbar=None,
) -> tuple[torch.Tensor, list[dict]]:
    x0 = images.detach()
    x = x0.clone()
    momentum = torch.zeros_like(x)
    ti_kernel = make_ti_kernel(x.device, channels=x.shape[1])
    history = []

    for step in range(steps):
        asr = estimate_asr(models, x, labels)
        weights = compute_shortfall_weights(asr.detach().cpu(), floor=weight_floor, temperature=weight_temp).to(x.device)

        x_req = x.detach().requires_grad_(True)
        loss = weighted_margin_loss(models, x_req, labels, weights, eot=eot, scales=scales)
        grad = torch.autograd.grad(loss, x_req)[0].detach()
        grad = smooth_gradient(grad, ti_kernel)
        grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        momentum = MOMENTUM * momentum + grad
        x = x.detach() + step_size * torch.sign(momentum)
        x = torch.min(torch.max(x, x0 - epsilon), x0 + epsilon).clamp(0, 1)

        rec = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "asr": [float(v) for v in asr.detach().cpu()],
            "weights": [float(v) for v in weights.detach().cpu()],
        }
        history.append(rec)
        if sbar is not None:
            weak_idx = int(torch.argmin(asr).item())
            sbar.update(1)
            sbar.set_postfix(
                loss=f"{rec['loss']:.3f}",
                weak=models[weak_idx].name[:10],
                w=f"{weights[weak_idx].item():.2f}",
            )

    return x.detach(), history


def parse_args():
    parser = argparse.ArgumentParser(description="Diverse shortfall-aware attack with ViT surrogate support")
    parser.add_argument("--out-dir", default=str(DEFAULT_ADV_DIR))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cnn-count", type=int, default=DEFAULT_CNN_COUNT)
    parser.add_argument("--vit-surrogates", default="all",
                        help="all|weak2|weak1|none or comma-separated ViT names")
    parser.add_argument("--robust-surrogates", default="none",
                        help="none|light3|all or comma-separated robust model names")
    parser.add_argument("--eot", type=int, default=DEFAULT_EOT)
    parser.add_argument("--weight-floor", type=float, default=DEFAULT_WEIGHT_FLOOR)
    parser.add_argument("--weight-temp", type=float, default=DEFAULT_WEIGHT_TEMP)
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    adv_dir = Path(args.out_dir)
    logger = setup_logger(TAG)
    done_set = get_done_set(adv_dir)
    if args.max_images is None and len(done_set) == 500:
        logger.info("Already complete (500 files). Skip.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")
    logger.info(f"out_dir={adv_dir}")
    logger.info(f"steps={args.steps} batch_size={args.batch_size} eot={args.eot}")
    logger.info(
        f"cnn_count={args.cnn_count} vit_surrogates={args.vit_surrogates} "
        f"robust_surrogates={args.robust_surrogates}"
    )
    logger.info(f"weight_floor={args.weight_floor} weight_temp={args.weight_temp}")

    models = build_attack_models(
        device,
        logger,
        cnn_count=args.cnn_count,
        vit_selection=args.vit_surrogates,
        robust_selection=args.robust_surrogates,
    )
    names, images, labels = load_data(device)

    limit = args.max_images if args.max_images is not None else 500
    todo = [(i, n) for i, n in enumerate(names[:limit]) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}/{limit}")
    adv_dir.mkdir(parents=True, exist_ok=True)

    history_path = adv_dir / "attack_history.jsonl"
    with image_pbar(limit, min(len(done_set), limit), TAG) as ibar:
        for chunk in iter_todo_batches(todo, args.batch_size):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)
            batch_desc = f"[{idxs[0]}:{idxs[-1] + 1}]"
            with step_pbar(args.steps, batch_desc) as sbar:
                adv_b, hist = diverse_attack(
                    models=models,
                    images=images[idxs],
                    labels=labels[idxs],
                    steps=args.steps,
                    eot=args.eot,
                    weight_floor=args.weight_floor,
                    weight_temp=args.weight_temp,
                    sbar=sbar,
                )
            save_adv_batch(adv_to_numpy(adv_b), list(names_b), adv_dir)
            with history_path.open("a", encoding="utf-8") as f:
                for rec in hist:
                    rec["batch"] = [int(idxs[0]), int(idxs[-1])]
                    rec["models"] = [m.name for m in models]
                    f.write(json.dumps(rec) + "\n")
            ibar.update(len(chunk))
            log_progress_event(logger, TAG, adv_dir, len(get_done_set(adv_dir)),
                               total=limit, batch=[int(idxs[0]), int(idxs[-1])])
            logger.info(f"Saved batch {batch_desc}")

    logger.info(f"Done. {len(list(adv_dir.glob('*.png')))} PNGs in {adv_dir}")


if __name__ == "__main__":
    main()
