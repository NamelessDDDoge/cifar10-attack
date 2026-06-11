"""
PGN (Penalizing Gradient Norm) attack adapter for CIFAR-10
Origin: https://github.com/Trustworthy-AI-Group/PGN
Paper: Ge et al., NeurIPS 2023

Adaptations from original Incv3_PGN_Attack.py:
  - Replaced pretrainedmodels.inceptionv3 with pytorchcv CIFAR-10 surrogate ensemble (5 models)
  - eps=8/255 (was 16/255), alpha=2/255
  - Removed hardcoded CUDA; CPU fallback
  - PNG load/save for 32x32 images
  - Batch processing; no ImageNet CSV loader
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, EPS, ALPHA, STEPS, BATCH_SIZE
from run_utils import (
    setup_logger, load_data, build_ensemble, build_mixed_attack_ensemble, adv_to_numpy,
    save_adv_batch, get_done_set, image_pbar, step_pbar,
    iter_todo_batches, log_progress_event,
)

TAG     = "PGN"
ADV_DIR = RESULTS_DIR / "adv_pgn"

N_SAMPLES   = 20
MOMENTUM    = 1.0
ZETA        = 3.0
DELTA_COEF  = 0.5


def pgn_attack(model, images, labels, steps=STEPS, sbar=None):
    x0       = images.clone()
    x        = x0.clone()
    grad_acc = torch.zeros_like(x)

    for step in range(steps):
        avg_grad = torch.zeros_like(x)

        for _ in range(N_SAMPLES):
            noise  = torch.empty_like(x).uniform_(-EPS * ZETA, EPS * ZETA)
            x_near = (x + noise).detach().requires_grad_(True)
            loss   = F.cross_entropy(model(x_near), labels)
            g1     = torch.autograd.grad(loss, x_near)[0].detach()

            x_star = (x_near.detach() + ALPHA * (-g1) /
                      (torch.abs(g1).mean([1, 2, 3], keepdim=True) + 1e-12)
                      ).detach().requires_grad_(True)
            loss2  = F.cross_entropy(model(x_star), labels)
            g2     = torch.autograd.grad(loss2, x_star)[0].detach()

            avg_grad = avg_grad + (1 - DELTA_COEF) * g1 + DELTA_COEF * g2

        noise_norm = avg_grad / (torch.abs(avg_grad).mean([1, 2, 3], keepdim=True) + 1e-12)
        grad_acc   = MOMENTUM * grad_acc + noise_norm
        x          = x + ALPHA * torch.sign(grad_acc)
        x          = torch.min(torch.max(x, x0 - EPS), x0 + EPS).clamp(0, 1)

        if sbar is not None:
            sbar.update(1)

    return x.detach()


def main():
    parser = argparse.ArgumentParser(description="PGN CIFAR-10 attack")
    parser.add_argument("--out-dir", default=str(ADV_DIR))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cnn-count", type=int, default=None)
    parser.add_argument("--vit-surrogates", default="none")
    parser.add_argument("--robust-surrogates", default="none")
    args = parser.parse_args()

    adv_dir = Path(args.out_dir)
    logger   = setup_logger(TAG)
    done_set = get_done_set(adv_dir)

    limit = args.max_images if args.max_images is not None else 500
    if args.max_images is None and len(done_set) == 500:
        logger.info("Already complete (500 files). Skip.")
        return

    logger.info(f"Resume: {len(done_set)}/{limit} done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info("Building surrogate ensemble ...")
    if args.cnn_count is None and args.vit_surrogates == "none" and args.robust_surrogates == "none":
        ensemble = build_ensemble(device, TAG, logger)
    else:
        ensemble = build_mixed_attack_ensemble(
            device, TAG, logger,
            cnn_count=args.cnn_count or 0,
            vit_surrogates=args.vit_surrogates,
            robust_surrogates=args.robust_surrogates,
        )

    logger.info("Loading data ...")
    names, images, labels = load_data(device)

    adv_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, n) for i, n in enumerate(names[:limit]) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}")

    with image_pbar(limit, min(len(done_set), limit), TAG) as ibar:
        for chunk in iter_todo_batches(todo, args.batch_size):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)

            img_b = images[idxs]
            lbl_b = labels[idxs]

            batch_desc = f"[{idxs[0]}:{idxs[-1]+1}]"
            with step_pbar(args.steps, batch_desc) as sbar:
                adv_b = pgn_attack(ensemble, img_b, lbl_b, steps=args.steps, sbar=sbar)

            save_adv_batch(adv_to_numpy(adv_b), list(names_b), adv_dir)
            ibar.update(len(chunk))
            log_progress_event(logger, TAG, adv_dir, len(get_done_set(adv_dir)),
                               total=limit, batch=[int(idxs[0]), int(idxs[-1])])
            logger.debug(f"Saved batch {batch_desc} ({len(chunk)} imgs)")

    logger.info(f"Done. {len(list(adv_dir.glob('*.png')))} images in {adv_dir}")


if __name__ == "__main__":
    main()
