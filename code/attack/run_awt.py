"""
AWT (Adversarial Weight Tuning) attack adapter for CIFAR-10
Origin: https://github.com/xaddwell/AWT
Paper: Chen et al., AAAI 2025

Adaptations from original transferattack/awt.py:
  - Override load_model in Attack base class to use pytorchcv CIFAR-10 surrogate ensemble
  - eps=8/255, alpha=2/255, epoch=10
  - num_neighbor=5 (reduced from 20 for CPU speed)
  - PNG load/save for 32x32 images
  - No CUDA requirement
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, REPOS_DIR, EPS, ALPHA, BATCH_SIZE
from run_utils import (
    setup_logger, load_data, build_awt_ensemble, build_mixed_attack_ensemble, adv_to_numpy,
    save_adv_batch, get_done_set, image_pbar,
    iter_todo_batches, log_progress_event,
)

REPO_DIR = REPOS_DIR / "AWT"
sys.path.insert(0, str(REPO_DIR))

TAG     = "AWT"
ADV_DIR = RESULTS_DIR / "adv_awt"

EPOCH        = 10
NUM_NEIGHBOR = 20


def main():
    parser = argparse.ArgumentParser(description="AWT CIFAR-10 attack")
    parser.add_argument("--out-dir", default=str(ADV_DIR))
    parser.add_argument("--steps", type=int, default=EPOCH)
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

    logger.info("Building AWT ensemble (primary unfrozen for SAM) ...")
    if args.cnn_count is None and args.vit_surrogates == "none" and args.robust_surrogates == "none":
        ensemble = build_awt_ensemble(device, TAG, logger)
    else:
        ensemble = build_mixed_attack_ensemble(
            device, TAG, logger,
            cnn_count=args.cnn_count or 0,
            vit_surrogates=args.vit_surrogates,
            robust_surrogates=args.robust_surrogates,
            expose_primary_params=True,
        )

    logger.info("Loading data ...")
    names, images, labels = load_data(device)

    from transferattack.awt import AWT

    class CifarAWT(AWT):
        def load_model(self, model_name):
            return ensemble

    attacker = CifarAWT(
        model_name="cifar10_ensemble",
        epsilon=EPS, alpha=ALPHA,
        beta=3.0, gamma=0.5,
        num_neighbor=NUM_NEIGHBOR, epoch=args.steps,
        decay=1.0, targeted=False, random_start=False,
        norm="linfty", loss="crossentropy", device=device,
    )

    adv_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, n) for i, n in enumerate(names[:limit]) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}")

    with image_pbar(limit, min(len(done_set), limit), TAG) as ibar:
        for chunk in iter_todo_batches(todo, args.batch_size):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)

            img_b  = images[idxs]
            lbl_b  = labels[idxs]
            delta  = attacker(img_b, lbl_b)
            adv_b  = torch.clamp(img_b + delta, 0, 1).detach()

            save_adv_batch(adv_to_numpy(adv_b), list(names_b), adv_dir)
            ibar.update(len(chunk))
            log_progress_event(logger, TAG, adv_dir, len(get_done_set(adv_dir)),
                               total=limit, batch=[int(idxs[0]), int(idxs[-1])])
            logger.debug(f"Saved batch [{idxs[0]}:{idxs[-1]+1}] ({len(chunk)} imgs)")

    logger.info(f"Done. {len(list(adv_dir.glob('*.png')))} images in {adv_dir}")


if __name__ == "__main__":
    main()
