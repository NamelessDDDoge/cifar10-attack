"""
Shared output/logging interface for all attack runners.

Provides:
  - setup_logger     : file + stdout logger
  - load_data        : load 500 clean images + labels
  - build_ensemble   : load surrogate ensemble (AWT/BSR/L2T/PGN/SIT)
  - adv_to_numpy     : tensor [N,3,H,W] float -> uint8 [N,H,W,3]
  - save_adv_batch   : write a batch of PNG files immediately (checkpoint)
  - get_done_set     : filenames already saved (for resume)
  - image_pbar       : tqdm wrapper for the outer image loop

CHECKPOINT_EVERY controls how many images are saved per flush (matches BATCH_SIZE).
All attack runners call save_adv_batch after every batch so progress survives crashes.
"""
import logging
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    IMAGES_DIR, LABEL_FILE, RESULTS_DIR, CIFAR10_MEAN, CIFAR10_STD,
    SURROGATE_NAMES, BATCH_SIZE, EVAL_DIR,
)

CHECKPOINT_EVERY = BATCH_SIZE   # save every batch (= every BATCH_SIZE images)
LOG_DIR = RESULTS_DIR / "logs"
PROGRESS_DIR = RESULTS_DIR / "progress"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(tag: str) -> logging.Logger:
    """Return a logger that writes to stdout AND a timestamped log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{tag}_{ts}.log"

    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s  %(message)s",
                             datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Log file: {log_path}")
    return logger


def progress_file(tag: str, adv_dir: Path) -> Path:
    safe_dir = adv_dir.name if adv_dir else "unknown"
    return PROGRESS_DIR / f"{tag.lower()}_{safe_dir}.jsonl"


def iter_todo_batches(todo: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for chunk_start in range(0, len(todo), batch_size):
        yield todo[chunk_start: chunk_start + batch_size]


def log_progress_event(logger: logging.Logger, tag: str, adv_dir: Path,
                       processed: int, total: int = 500,
                       progress_path: Path | None = None,
                       **extra):
    """Append a machine-readable progress row and mirror it to the logger."""
    if progress_path is None:
        progress_path = progress_file(tag, adv_dir)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    percent = round(100.0 * processed / total, 4) if total else 0.0
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": tag,
        "adv_dir": str(adv_dir),
        "processed": int(processed),
        "total": int(total),
        "percent": percent,
    }
    event.update(extra)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    logger.info(f"progress {processed}/{total} ({percent:.2f}%) -> {progress_path}")
    return event


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(device: torch.device):
    """Load 500 clean images and true labels. Returns (names, images, labels)."""
    names = [f"{i}.png" for i in range(500)]
    images = []
    for name in names:
        img = Image.open(IMAGES_DIR / name).convert("RGB")
        images.append(np.array(img, dtype=np.float32) / 255.0)
    images = torch.tensor(np.stack(images), dtype=torch.float32).permute(0, 3, 1, 2)

    label_map: dict = {}
    with open(LABEL_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            n, c = line.split()
            label_map[n] = int(c)
    labels = torch.tensor([label_map[n] for n in names], dtype=torch.long)
    return names, images.to(device), labels.to(device)


# ---------------------------------------------------------------------------
# Surrogate ensemble (shared by AWT / BSR / L2T / PGN / SIT)
# ---------------------------------------------------------------------------

class CifarEnsemble(nn.Module):
    def __init__(self, models, device, use_checkpointing: bool = False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.use_checkpointing = use_checkpointing
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std  = torch.tensor(CIFAR10_STD,  dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x):
        x_n = (x - self.mean) / self.std
        if self.use_checkpointing:
            from torch.utils.checkpoint import checkpoint
            # Checkpoint each model independently: only one model's activations
            # live in memory at a time during backward. Trades compute for memory.
            return sum(
                checkpoint(m, x_n, use_reentrant=False) for m in self.models
            ) / len(self.models)
        return sum(m(x_n) for m in self.models) / len(self.models)


class SingleCifarNormModel(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def build_ensemble(device: torch.device, tag: str = "",
                   logger: logging.Logger = None) -> CifarEnsemble:
    from pytorchcv.model_provider import get_model
    log = logger.info if logger else print
    models = []
    for name in SURROGATE_NAMES:
        log(f"  [{tag}] Loading {name} ...")
        m = get_model(name, pretrained=True).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    log(f"  [{tag}] Ensemble ready ({len(models)} models)")
    return CifarEnsemble(models, device).to(device)


class _EnsembleWithPrimaryParams(CifarEnsemble):
    """AWT-specific: forward through all models, but parameters() exposes only
    the primary model so SAM tunes one model's weights, avoiding OOM."""
    def parameters(self, recurse=True):
        return self.models[0].parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self.models[0].named_parameters(prefix=prefix, recurse=recurse)


class MixedLogitEnsemble(nn.Module):
    """Average logits from normalized CNN, ViT, and robust CIFAR-10 models."""

    def __init__(self, models: list[nn.Module], primary_param_model: nn.Module | None = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.primary_param_model = primary_param_model

    def forward(self, x):
        return sum(model(x) for model in self.models) / len(self.models)

    def parameters(self, recurse=True):
        if self.primary_param_model is not None:
            return self.primary_param_model.parameters(recurse=recurse)
        return super().parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        if self.primary_param_model is not None:
            return self.primary_param_model.named_parameters(
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
            )
        return super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


L2T_SURROGATE_NAMES = [
    # Subset that avoids OOM on 8GB VRAM with L2T's dual-forward (search + attack).
    # Excluded: resnext29_16x64d (68M params, huge activations),
    #           pyramidnet164_a270_bn (27M, deep),
    #           densenet190_k40_bc (25M but 190 dense-concat layers → O(L²) backprop memory).
    # Use densenet40_k12 (small DenseNet, 40 layers, manageable gradient memory).
    "wrn16_10_cifar10",
    "seresnet110_cifar10",
    "diaresnet56_cifar10",
    "densenet40_k12_cifar10",
]


def build_l2t_ensemble(device: torch.device, tag: str = "L2T",
                       logger: logging.Logger = None) -> CifarEnsemble:
    """L2T-specific ensemble: light models + gradient checkpointing to fit 8GB VRAM.
    L2T's dual-forward (search + attack) accumulates activations from all models
    simultaneously; checkpointing recomputes each model's forward during backward
    instead of retaining all activations, trading ~2× compute for ~4× memory savings."""
    from pytorchcv.model_provider import get_model
    log = logger.info if logger else print
    models = []
    for name in L2T_SURROGATE_NAMES:
        log(f"  [{tag}] Loading {name} ...")
        m = get_model(name, pretrained=True).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    log(f"  [{tag}] L2T ensemble ready ({len(models)} light models, gradient checkpointing ON)")
    return CifarEnsemble(models, device, use_checkpointing=True).to(device)


def build_awt_ensemble(device: torch.device, tag: str = "",
                       logger: logging.Logger = None) -> "_EnsembleWithPrimaryParams":
    """Like build_ensemble but primary model's params are unfrozen for SAM weight tuning."""
    from pytorchcv.model_provider import get_model
    log = logger.info if logger else print
    models = []
    for i, name in enumerate(SURROGATE_NAMES):
        log(f"  [{tag}] Loading {name} ...")
        m = get_model(name, pretrained=True).to(device).eval()
        if i > 0:
            for p in m.parameters():
                p.requires_grad_(False)
        models.append(m)
    log(f"  [{tag}] AWT ensemble ready ({len(models)} models, primary unfrozen for SAM)")
    return _EnsembleWithPrimaryParams(models, device).to(device)


def _select_external_names(selection: str, specs, none_ok: bool = True) -> list[str]:
    names = [s.name for s in specs]
    if selection == "all":
        return names
    if selection == "none" and none_ok:
        return []
    requested = [part.strip() for part in selection.split(",") if part.strip()]
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"unknown external surrogate(s): {sorted(unknown)}")
    return requested


def build_mixed_attack_ensemble(
    device: torch.device,
    tag: str,
    logger: logging.Logger = None,
    cnn_count: int = 2,
    vit_surrogates: str = "none",
    robust_surrogates: str = "none",
    expose_primary_params: bool = False,
) -> MixedLogitEnsemble:
    """Build the mixed surrogate pool used by the mixed attack scripts."""
    from pytorchcv.model_provider import get_model

    sys.path.insert(0, str(EVAL_DIR))
    from external_models import VIT_SPECS, ROBUST_SPECS, load_external_model

    log = logger.info if logger else print
    models: list[nn.Module] = []
    primary_param_model: nn.Module | None = None

    for idx, name in enumerate(SURROGATE_NAMES[:cnn_count]):
        log(f"  [{tag}] Loading CNN surrogate {name} ...")
        backbone = get_model(name, pretrained=True).to(device).eval()
        if expose_primary_params and idx == 0:
            primary_param_model = backbone
        else:
            for p in backbone.parameters():
                p.requires_grad_(False)
        models.append(SingleCifarNormModel(backbone, device).to(device))

    for name in _select_external_names(vit_surrogates, VIT_SPECS):
        log(f"  [{tag}] Loading ViT surrogate {name} ...")
        model = load_external_model(name, device=device).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models.append(model)

    for name in _select_external_names(robust_surrogates, ROBUST_SPECS):
        log(f"  [{tag}] Loading robust surrogate {name} ...")
        model = load_external_model(name, device=device).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models.append(model)

    if not models:
        raise RuntimeError("no attack models selected")
    log(f"  [{tag}] Mixed ensemble ready ({len(models)} models)")
    return MixedLogitEnsemble(models, primary_param_model=primary_param_model).to(device)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def adv_to_numpy(adv: torch.Tensor) -> np.ndarray:
    """[N,3,H,W] float [0,1] -> uint8 [N,H,W,3]."""
    return (adv.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            ).round().clip(0, 255).astype(np.uint8)


def save_adv_batch(adv_np: np.ndarray, names_batch: list, out_dir: Path):
    """Write a batch of adversarial PNGs to out_dir immediately (checkpoint)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names_batch):
        Image.fromarray(adv_np[i]).save(out_dir / name)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def get_done_set(adv_dir: Path) -> set:
    """Return set of PNG filenames already saved in adv_dir."""
    if not adv_dir.exists():
        return set()
    return {p.name for p in adv_dir.glob("*.png")}


# ---------------------------------------------------------------------------
# tqdm helpers
# ---------------------------------------------------------------------------

def image_pbar(total: int, done: int, tag: str) -> tqdm:
    """Outer progress bar over all 500 images."""
    return tqdm(total=total, initial=done, desc=f"[{tag}] images",
                unit="img", dynamic_ncols=True)


def step_pbar(total_steps: int, batch_desc: str) -> tqdm:
    """Inner progress bar over attack steps for one batch."""
    return tqdm(total=total_steps, desc=f"  steps {batch_desc}",
                unit="step", leave=False, dynamic_ncols=True)
